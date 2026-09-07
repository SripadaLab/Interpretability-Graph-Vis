"""Verify manual supernodes via GemmaScope transcoder inflow/outflow cosine.

For each graph that ships manual clusters (``qParams.supernodes``), we build a
per-feature direction by concatenating, left to right, its **inflow** (encoder
``W_enc[:, f]``) and **outflow** (decoder ``W_dec[f, :]``) directions — so the
similarity reflects both how a feature reads from and writes to the residual
stream. We then check that the *in-group* cosine similarity (members of the same
cluster) is higher than the *out-group* similarity (members of different
clusters). If that holds, the manual grouping is consistent with feature
geometry.

Decoder weights come from the public GemmaScope transcoders
(``google/gemma-scope-2b-pt-transcoders``, width 16k). Only the layers referenced
by a graph's clusters are downloaded, and files are cached by HuggingFace.

The result is written back into ``metadata.supernode_similarity`` so the frontend
cluster panel can display it:

    {
      "in_group_mean": 0.42,
      "out_group_mean": 0.08,
      "passed": true,
      "n_pairs_in": 37,
      "n_pairs_out": 210,
      "per_cluster": {"capital": {"in": 0.4, "out": 0.05, "n": 5}, ...}
    }
"""

from __future__ import annotations

import json
import logging
from itertools import combinations
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TRANSCODER_REPO = "google/gemma-scope-2b-pt-transcoders"
FEATURE_LAYER_STRIDE = 100_000  # graph 'feature' = layer * stride + feature_index


def _load_transcoder_config() -> dict[int, str]:
    """Map layer index → HF filename for the width-16k transcoder params.npz."""
    from huggingface_hub import hf_hub_download

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("verify_supernodes needs pyyaml: pip install pyyaml") from exc

    cfg_path = hf_hub_download("mntss/gemma-scope-transcoders", "config.yaml")
    cfg = yaml.safe_load(Path(cfg_path).read_text())

    layer_to_file: dict[int, str] = {}
    for entry in cfg["transcoders"]:
        # e.g. hf://google/gemma-scope-2b-pt-transcoders/layer_6/width_16k/average_l0_95/params.npz
        rel = entry.split(TRANSCODER_REPO + "/", 1)[1]
        layer = int(rel.split("/", 1)[0].removeprefix("layer_"))
        layer_to_file[layer] = rel
    return layer_to_file


def _decoder_matrix_cache() -> dict[int, Any]:
    return {}


def _get_feature_vectors(
    layer_feature_pairs: set[tuple[int, int]],
    layer_to_file: dict[int, str],
    cache: dict[int, Any],
):
    """Return {(layer, feat_idx): unit vector} where each vector concatenates the
    feature's inflow (encoder) and outflow (decoder) directions, left to right.

    inflow  = W_enc[:, f]   (how the feature reads from the residual stream)
    outflow = W_dec[f, :]   (how the feature writes to the residual stream)

    Each half is unit-normalized before concatenation so inflow and outflow
    contribute equally; the concatenated vector is then unit-normalized so that
    cosine(a, b) = ½·(cos_inflow + cos_outflow).
    """
    import numpy as np
    from huggingface_hub import hf_hub_download

    needed_layers = sorted({layer for layer, _ in layer_feature_pairs})
    for layer in needed_layers:
        if layer in cache:
            continue
        if layer not in layer_to_file:
            logger.warning("No transcoder file for layer %d; skipping", layer)
            continue
        logger.info("Loading enc+dec weights for layer %d …", layer)
        npz_path = hf_hub_download(TRANSCODER_REPO, layer_to_file[layer])
        with np.load(npz_path) as d:
            cache[layer] = {
                "W_enc": d["W_enc"],  # (d_model, n_features)
                "W_dec": d["W_dec"],  # (n_features, d_model)
            }

    def _unit(v):
        n = float((v @ v) ** 0.5)
        return v / n if n > 0 else None

    vecs: dict[tuple[int, int], Any] = {}
    for layer, feat_idx in layer_feature_pairs:
        w = cache.get(layer)
        if w is None:
            continue
        W_enc, W_dec = w["W_enc"], w["W_dec"]
        if feat_idx >= W_dec.shape[0] or feat_idx >= W_enc.shape[1]:
            continue
        inflow = _unit(W_enc[:, feat_idx].astype("float64"))
        outflow = _unit(W_dec[feat_idx].astype("float64"))
        if inflow is None or outflow is None:
            continue
        combined = np.concatenate([inflow, outflow])
        combined = _unit(combined)
        if combined is not None:
            vecs[(layer, feat_idx)] = combined
    return vecs


def _member_features(node: dict) -> tuple[int, int] | None:
    """(layer, feature_index) for a cross-layer transcoder node, else None.

    The node_id is the reliable source: ``{layer}_{feature_index}_{ctx}`` (the
    middle segment is the transcoder feature index within that layer). The
    ``feature`` field is not consistent across graphs — on some it's the raw
    feature index, on others a global id — so we parse node_id first and only
    fall back to the ``feature`` stride encoding when node_id can't be read.
    """
    if "transcoder" not in (node.get("feature_type") or ""):
        return None

    node_id = node.get("node_id") or ""
    parts = node_id.split("_")
    if len(parts) >= 2:
        try:
            return (int(parts[0]), int(parts[1]))
        except ValueError:
            pass

    feat = node.get("feature")
    if feat is None:
        return None
    layer = int(feat) // FEATURE_LAYER_STRIDE
    feat_idx = int(feat) % FEATURE_LAYER_STRIDE
    node_layer = node.get("layer")
    try:
        if node_layer not in (None, "", "E") and int(node_layer) != layer:
            layer = int(node_layer)
    except (TypeError, ValueError):
        pass
    return (layer, feat_idx)


def verify_graph_supernodes(
    json_path: str | Path,
    *,
    write: bool = True,
    cache: dict[int, Any] | None = None,
    layer_to_file: dict[int, str] | None = None,
) -> dict[str, Any] | None:
    """Compute in/out-group decoder cosine similarity for a graph's supernodes."""
    import numpy as np

    json_path = Path(json_path)
    data = json.loads(json_path.read_text())
    supernodes = (data.get("qParams") or {}).get("supernodes") or []
    if not supernodes:
        logger.info("%s has no manual supernodes; skipping", json_path.name)
        return None

    # Synthetic demos don't use real GemmaScope feature indices — skip them and
    # strip any stale result so the frontend doesn't show a meaningless check.
    md = data.get("metadata", {})
    transcoders = " ".join(md.get("transcoder_list") or []).lower()
    if "synthetic" in transcoders or "demo" in transcoders:
        if md.pop("supernode_similarity", None) is not None and write:
            json_path.write_text(json.dumps(data))
        logger.info("%s uses synthetic features; skipping cosine check", json_path.name)
        return None

    id_to_node = {n["node_id"]: n for n in data.get("nodes", [])}

    # cluster label → list of (layer, feat_idx)
    clusters: dict[str, list[tuple[int, int]]] = {}
    all_pairs: set[tuple[int, int]] = set()
    for sn in supernodes:
        label = sn[0]
        members = []
        for nid in sn[1:]:
            node = id_to_node.get(nid)
            if not node:
                continue
            lf = _member_features(node)
            if lf is not None:
                members.append(lf)
                all_pairs.add(lf)
        if members:
            clusters[label] = members

    if not all_pairs:
        logger.info("%s: no transcoder-feature members to compare", json_path.name)
        return None

    if layer_to_file is None:
        layer_to_file = _load_transcoder_config()
    if cache is None:
        cache = _decoder_matrix_cache()

    vecs = _get_feature_vectors(all_pairs, layer_to_file, cache)

    def cos(a, b):
        return float(np.dot(vecs[a], vecs[b]))

    # Assign each cluster the vectors we actually resolved.
    resolved = {
        label: [lf for lf in members if lf in vecs]
        for label, members in clusters.items()
    }

    in_sims: list[float] = []
    per_cluster: dict[str, dict[str, float]] = {}
    for label, members in resolved.items():
        pair_sims = [cos(a, b) for a, b in combinations(members, 2)]
        cluster_in = float(np.mean(pair_sims)) if pair_sims else float("nan")

        out_sims_cluster = []
        for a in members:
            for other_label, others in resolved.items():
                if other_label == label:
                    continue
                for b in others:
                    out_sims_cluster.append(cos(a, b))
        cluster_out = float(np.mean(out_sims_cluster)) if out_sims_cluster else float("nan")

        per_cluster[label] = {
            "in": cluster_in,
            "out": cluster_out,
            "n": len(members),
        }
        in_sims.extend(pair_sims)

    # Global out-group: all cross-cluster member pairs (each unordered pair once).
    out_sims: list[float] = []
    labels = list(resolved)
    for i, la in enumerate(labels):
        for lb in labels[i + 1 :]:
            for a in resolved[la]:
                for b in resolved[lb]:
                    out_sims.append(cos(a, b))

    in_group_mean = float(np.mean(in_sims)) if in_sims else float("nan")
    out_group_mean = float(np.mean(out_sims)) if out_sims else float("nan")
    passed = bool(in_sims and out_sims and in_group_mean > out_group_mean)

    # Cluster × cluster mean-cosine matrix. Diagonal = in-group pairwise mean;
    # off-diagonal[i][j] = mean cosine between members of cluster i and cluster j.
    # A healthy grouping has a bright diagonal (high) vs dim off-diagonal (low).
    # Only clusters we actually resolved vectors for are included. Empty cells
    # (e.g. a singleton cluster's diagonal) are stored as null, not NaN, so the
    # JSON stays valid for the frontend.
    matrix_labels = [label for label in resolved if resolved[label]]

    def _mean_cos(members_a, members_b, same):
        if same:
            sims = [cos(a, b) for a, b in combinations(members_a, 2)]
        else:
            sims = [cos(a, b) for a in members_a for b in members_b]
        return float(np.mean(sims)) if sims else None

    matrix_values = [
        [_mean_cos(resolved[la], resolved[lb], la == lb) for lb in matrix_labels]
        for la in matrix_labels
    ]

    result = {
        "in_group_mean": in_group_mean,
        "out_group_mean": out_group_mean,
        "passed": passed,
        "n_pairs_in": len(in_sims),
        "n_pairs_out": len(out_sims),
        "transcoder": TRANSCODER_REPO,
        "method": "cosine of concat[unit(W_enc inflow) || unit(W_dec outflow)]",
        "per_cluster": per_cluster,
        "matrix": {
            "labels": matrix_labels,
            "values": matrix_values,
            "counts": [len(resolved[label]) for label in matrix_labels],
        },
    }

    if write:
        data.setdefault("metadata", {})["supernode_similarity"] = result
        json_path.write_text(json.dumps(data))
        logger.info("Wrote supernode_similarity into %s", json_path.name)

    return result


def _agglomerative_labels_by_k(D, ks: set[int]) -> dict[int, list[int]]:
    """Average-linkage agglomerative clustering on a precomputed distance matrix
    ``D`` (here cosine distance = 1 − cosine). Returns {k: labels} for each k in
    ``ks``. Because clusters merge bottom-up, the groupings are *nested*: raising
    k splits an existing group, which makes a k-slider behave coherently.
    """
    import numpy as np

    m = D.shape[0]
    members: dict[int, list[int]] = {i: [i] for i in range(m)}
    active: list[int] = list(range(m))
    result: dict[int, list[int]] = {}

    def snapshot():
        count = len(active)
        if count in ks:
            labels = [0] * m
            for ci, c in enumerate(active):
                for idx in members[c]:
                    labels[idx] = ci
            result[count] = labels

    snapshot()
    next_id = m
    while len(active) > 1:
        best = None  # (distance, cluster_a, cluster_b)
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                a, b = active[i], active[j]
                d = float(D[np.ix_(members[a], members[b])].mean())
                if best is None or d < best[0]:
                    best = (d, a, b)
        _, a, b = best
        members[next_id] = members[a] + members[b]
        active.remove(a)
        active.remove(b)
        active.append(next_id)
        del members[a], members[b]
        next_id += 1
        snapshot()
    return result


def _link_neighborhood_vectors(
    data: dict[str, Any],
    node_ids: list[str],
):
    """Unit vectors from abs attribution to/from other candidates (concat in||out).

    Used as a fallback when GemmaScope transcoder weights aren't available
    (synthetic demos) or when too few feature vectors resolve.
    """
    import numpy as np

    idx = {nid: i for i, nid in enumerate(node_ids)}
    m = len(node_ids)
    inflow = np.zeros((m, m), dtype=float)
    outflow = np.zeros((m, m), dtype=float)
    for link in data.get("links") or data.get("edges") or []:
        s, t = link.get("source"), link.get("target")
        if s not in idx or t not in idx:
            continue
        w = abs(float(link.get("weight") or link.get("pctInput") or 0.0))
        if w <= 0:
            continue
        inflow[idx[t], idx[s]] += w
        outflow[idx[s], idx[t]] += w

    def _row_unit(M):
        out = np.zeros_like(M)
        for i in range(m):
            n = float(np.linalg.norm(M[i]))
            out[i] = M[i] / n if n > 0 else 0.0
        return out

    combined = np.concatenate([_row_unit(inflow), _row_unit(outflow)], axis=1)
    # Re-normalize rows; zero rows stay zero (isolated nodes).
    for i in range(m):
        n = float(np.linalg.norm(combined[i]))
        if n > 0:
            combined[i] /= n
        else:
            # Give isolates a unique one-hot so they don't all collapse together.
            combined[i, i % combined.shape[1]] = 1.0
    return combined


def compute_auto_clusters(
    json_path: str | Path,
    *,
    write: bool = True,
    cache: dict[int, Any] | None = None,
    layer_to_file: dict[int, str] | None = None,
    max_nodes: int = 40,
    k: int | None = None,  # kept for CLI compat; sets default_k when given
    k_max: int = 12,
    seed: int = 0,
) -> dict[str, Any] | None:
    """Cluster a graph's strongest features by cosine distance, producing auto
    supernodes for *every* group count from 2..k_max so the frontend can offer
    a live k-slider.

    Prefers GemmaScope [inflow||outflow] directions; falls back to attribution
    neighborhood vectors for synthetic demos (or when weights don't resolve).

        metadata.auto_clusters = {
          "method": ..., "node_ids": [...], "max_nodes": 40, "m": 40,
          "k_min": 2, "k_max": 12, "default_k": 6,
          "by_k": {"2": [[label, id, ...], ...], "3": [...], ...}
        }
    """
    import numpy as np

    json_path = Path(json_path)
    data = json.loads(json_path.read_text())

    md = data.get("metadata", {})
    transcoders = " ".join(md.get("transcoder_list") or []).lower()
    is_synthetic = "synthetic" in transcoders or "demo" in transcoders

    # Candidate feature nodes ranked by |influence|.
    # Never include MLP reconstruction error (or other error nodes) — they are
    # residual leftovers, not transcoder features with W_enc/W_dec directions.
    candidates: list[tuple[float, str, tuple[int, int] | None]] = []
    for node in data.get("nodes", []):
        ft = (node.get("feature_type") or "").lower()
        if "error" in ft or ft == "logit":
            continue
        if "transcoder" not in ft:
            # Synthetic demos may only have a handful of feature nodes — allow
            # non-error embeddings so clustering still runs.
            if not is_synthetic:
                continue
        lf = _member_features(node) if "transcoder" in ft else None
        infl = abs(float(node.get("influence") or 0.0))
        candidates.append((infl, node["node_id"], lf))
    candidates.sort(key=lambda t: -t[0])
    candidates = candidates[:max_nodes]
    if len(candidates) < 2:
        return None

    method = (
        "average-linkage agglomerative clustering on cosine distance of "
        "concat[unit(W_enc) || unit(W_dec)]"
    )
    items: list[tuple[str, Any]] = []
    X = None

    if not is_synthetic:
        if layer_to_file is None:
            layer_to_file = _load_transcoder_config()
        if cache is None:
            cache = _decoder_matrix_cache()
        pair_set = {lf for _, _, lf in candidates if lf is not None}
        vecs = _get_feature_vectors(pair_set, layer_to_file, cache) if pair_set else {}
        items = [(nid, lf) for _, nid, lf in candidates if lf is not None and lf in vecs]
        if len(items) >= 2:
            X = np.array([vecs[lf] for _, lf in items])

    if X is None or len(items) < 2:
        # Fallback: attribution neighborhood vectors (works for every graph,
        # including synthetic demos with no GemmaScope weights).
        node_ids_fb = [nid for _, nid, _ in candidates]
        X = _link_neighborhood_vectors(data, node_ids_fb)
        items = [(nid, None) for nid in node_ids_fb]
        method = (
            "average-linkage agglomerative clustering on cosine distance of "
            "concat[unit(in-attribution) || unit(out-attribution)] neighborhood vectors"
        )

    node_ids = [nid for nid, _ in items]
    m = len(items)

    # Cosine distance matrix (clip tiny negatives from float error).
    D = np.clip(1.0 - (X @ X.T), 0.0, 2.0)

    k_hi = max(2, min(k_max, m))
    ks = set(range(2, k_hi + 1))
    labels_by_k = _agglomerative_labels_by_k(D, ks)

    by_k: dict[str, list[list[str]]] = {}
    for kk, labels in labels_by_k.items():
        grouped: dict[int, list[str]] = {}
        for nid, c in zip(node_ids, labels):
            grouped.setdefault(int(c), []).append(nid)
        supernodes = [
            [f"group {gi + 1}", *grouped[cid]]
            for gi, cid in enumerate(sorted(grouped))
            if grouped[cid]
        ]
        by_k[str(kk)] = supernodes

    default_k = k if k is not None else max(2, min(k_hi, round(m / 5)))
    default_k = max(2, min(k_hi, default_k))

    result = {
        "method": method,
        "node_ids": node_ids,
        "max_nodes": max_nodes,
        "m": m,
        "k_min": 2,
        "k_max": k_hi,
        "default_k": default_k,
        "by_k": by_k,
    }

    # Cosine similarity matrix for the SVD panel (same node order). Compact
    # enough to ship in JSON; the frontend takes a prune-filtered submatrix.
    C = X @ X.T
    # Numerical clip into [-1, 1]
    C = np.clip(C, -1.0, 1.0)
    svd_geom = {
        "method": method.replace("distance of", "similarity of").replace(
            "agglomerative clustering on cosine distance of",
            "cosine similarity of",
        ),
        "node_ids": node_ids,
        "cosine": [[float(C[i, j]) for j in range(m)] for i in range(m)],
    }

    if write:
        data.setdefault("metadata", {})["auto_clusters"] = result
        data.setdefault("metadata", {})["svd_geom"] = svd_geom
        json_path.write_text(json.dumps(data))
        logger.info(
            "Wrote auto_clusters (%d nodes, k=2..%d, default %d) + svd_geom into %s",
            m, k_hi, default_k, json_path.name,
        )
    return result


def verify_all(
    graph_dir: str | Path = "graph_files",
    *,
    write: bool = True,
    max_nodes: int = 40,
    k: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Verify manual supernodes (where present) and compute auto k-means clusters
    for every real graph in a directory."""
    graph_dir = Path(graph_dir)
    cache: dict[int, Any] = {}
    layer_to_file = _load_transcoder_config()

    report: dict[str, dict[str, Any]] = {}
    for path in sorted(graph_dir.glob("*.json")):
        if path.name == "graph-metadata.json":
            continue
        res = verify_graph_supernodes(
            path, write=write, cache=cache, layer_to_file=layer_to_file
        )
        auto = compute_auto_clusters(
            path, write=write, cache=cache, layer_to_file=layer_to_file,
            max_nodes=max_nodes, k=k,
        )
        entry: dict[str, Any] = {}
        if res is not None:
            entry.update(res)
        if auto is not None:
            entry["auto_clusters"] = auto
        if entry:
            report[path.stem] = entry
    return report


def _feature_candidates(data: dict[str, Any]) -> tuple[list[tuple[float, str, tuple[int, int] | None]], bool]:
    """Ranked transcoder (or synthetic) feature nodes: (influence, node_id, layer/feat)."""
    md = data.get("metadata", {})
    transcoders = " ".join(md.get("transcoder_list") or []).lower()
    is_synthetic = "synthetic" in transcoders or "demo" in transcoders

    candidates: list[tuple[float, str, tuple[int, int] | None]] = []
    for node in data.get("nodes", []):
        ft = (node.get("feature_type") or "").lower()
        if "error" in ft or ft == "logit":
            continue
        if "transcoder" not in ft:
            if not is_synthetic:
                continue
        lf = _member_features(node) if "transcoder" in ft else None
        infl = abs(float(node.get("influence") or 0.0))
        candidates.append((infl, node["node_id"], lf))
    candidates.sort(key=lambda t: -t[0])
    return candidates, is_synthetic


def build_feature_cosine(
    json_path: str | Path,
    *,
    max_nodes: int | None = None,
    cache: dict[int, Any] | None = None,
    layer_to_file: dict[int, str] | None = None,
) -> tuple[Any, list[str], str]:
    """Build the node×node cosine matrix for transcoder features.

    ``max_nodes=None`` (or ≤0) = **all** feature nodes in the graph JSON
    (no top-N influence cut). Returns ``(C, node_ids, method)``.
    """
    import numpy as np

    json_path = Path(json_path)
    data = json.loads(json_path.read_text())
    candidates, is_synthetic = _feature_candidates(data)
    if max_nodes is not None and max_nodes > 0:
        candidates = candidates[:max_nodes]
    if len(candidates) < 2:
        raise ValueError(f"Need ≥2 feature nodes in {json_path.name}")

    method = "cosine similarity of concat[unit(W_enc) || unit(W_dec)]"
    items: list[tuple[str, Any]] = []
    X = None

    if not is_synthetic:
        if layer_to_file is None:
            layer_to_file = _load_transcoder_config()
        if cache is None:
            cache = _decoder_matrix_cache()
        pair_set = {lf for _, _, lf in candidates if lf is not None}
        logger.info(
            "Building cosine for %s: %d candidates, %d unique (layer, feat)…",
            json_path.name,
            len(candidates),
            len(pair_set),
        )
        vecs = _get_feature_vectors(pair_set, layer_to_file, cache) if pair_set else {}
        items = [(nid, lf) for _, nid, lf in candidates if lf is not None and lf in vecs]
        if len(items) >= 2:
            X = np.array([vecs[lf] for _, lf in items], dtype="float64")

    if X is None or len(items) < 2:
        node_ids_fb = [nid for _, nid, _ in candidates]
        X = _link_neighborhood_vectors(data, node_ids_fb)
        items = [(nid, None) for nid in node_ids_fb]
        method = (
            "cosine similarity of concat[unit(in-attribution) || unit(out-attribution)] "
            "neighborhood vectors"
        )

    node_ids = [nid for nid, _ in items]
    C = np.clip(X @ X.T, -1.0, 1.0)
    return C, node_ids, method


def export_cosine_matrix(
    json_path: str | Path,
    out_dir: str | Path,
    *,
    recompute: bool = True,
    max_nodes: int | None = None,
    cache: dict[int, Any] | None = None,
    layer_to_file: dict[int, str] | None = None,
) -> list[Path]:
    """Write node×node cosine CSV (+ labels JSON, .npy) for one graph.

    By default recomputes over **all** transcoder features in the graph
    (``max_nodes=None``). Does not run agglomerative clustering.
    """
    import csv

    import numpy as np

    json_path = Path(json_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(json_path.read_text())
    slug = data.get("metadata", {}).get("slug") or json_path.stem

    # Prefer a fresh full (or capped) compute; fall back to shipped svd_geom only
    # when recompute=False and a matching geom already exists.
    C = None
    node_ids: list[str] = []
    method = ""
    if not recompute:
        geom = (data.get("metadata") or {}).get("svd_geom") or {}
        if geom.get("cosine") and geom.get("node_ids"):
            node_ids = list(geom["node_ids"])
            C = np.asarray(geom["cosine"], dtype=float)
            method = geom.get("method") or ""
            if max_nodes is not None and max_nodes > 0:
                node_ids = node_ids[:max_nodes]
                C = C[: len(node_ids), : len(node_ids)]

    if C is None:
        C, node_ids, method = build_feature_cosine(
            json_path,
            max_nodes=max_nodes,
            cache=cache,
            layer_to_file=layer_to_file,
        )

    if C.shape != (len(node_ids), len(node_ids)):
        raise ValueError(
            f"cosine shape {C.shape} != ({len(node_ids)}, {len(node_ids)}) in {json_path.name}"
        )

    csv_path = out_dir / f"{slug}-cosine.csv"
    labels_path = out_dir / f"{slug}-cosine-labels.json"
    npy_path = out_dir / f"{slug}-cosine.npy"
    meta_path = out_dir / f"{slug}-cosine-meta.json"

    by_id = {n.get("node_id"): n for n in data.get("nodes", []) if n.get("node_id")}
    labels = []
    for nid in node_ids:
        n = by_id.get(nid) or {}
        labels.append(
            {
                "node_id": nid,
                "layer": n.get("layer"),
                "feature": n.get("feature"),
                "ctx_idx": n.get("ctx_idx"),
                "influence": n.get("influence"),
                "clerp": n.get("clerp") or n.get("ppClerp") or n.get("description"),
                "feature_type": n.get("feature_type"),
            }
        )

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node_id", *node_ids])
        for i, nid in enumerate(node_ids):
            w.writerow([nid, *[f"{float(C[i, j]):.8f}" for j in range(len(node_ids))]])

    payload = {
        "slug": slug,
        "prompt": (data.get("metadata") or {}).get("prompt"),
        "method": method,
        "n": len(node_ids),
        "max_nodes": max_nodes if (max_nodes is not None and max_nodes > 0) else "all",
        "unthresholded": max_nodes is None or max_nodes <= 0,
        "nodes": labels,
    }
    labels_path.write_text(json.dumps(payload, indent=2) + "\n")
    meta_path.write_text(
        json.dumps({k: v for k, v in payload.items() if k != "nodes"}, indent=2) + "\n"
    )
    np.save(npy_path, np.asarray(C, dtype="float32"))

    # Point the graph at the full export (do not embed the n×n matrix in graph JSON).
    data.setdefault("metadata", {})["svd_geom_full"] = {
        "method": method,
        "n": len(node_ids),
        "max_nodes": payload["max_nodes"],
        "csv": f"cosine_exports/{csv_path.name}",
        "npy": f"cosine_exports/{npy_path.name}",
        "labels": f"cosine_exports/{labels_path.name}",
        "meta": f"cosine_exports/{meta_path.name}",
    }
    json_path.write_text(json.dumps(data))

    logger.info(
        "Exported %s cosine %dx%d (unthresholded=%s) → %s",
        slug,
        C.shape[0],
        C.shape[1],
        payload["unthresholded"],
        csv_path,
    )
    return [csv_path, labels_path, npy_path, meta_path]


def export_cosine_matrices(
    graph_dir: str | Path = "graph_files",
    out_dir: str | Path | None = None,
    *,
    slugs: list[str] | None = None,
    recompute: bool = True,
    max_nodes: int | None = None,
) -> list[Path]:
    """Export cosine matrices for selected (or all) graphs.

    ``max_nodes=None`` / ≤0 means every transcoder feature in the graph JSON.
    """
    graph_dir = Path(graph_dir)
    out_dir = Path(out_dir) if out_dir else graph_dir / "cosine_exports"
    if max_nodes is not None and max_nodes <= 0:
        max_nodes = None

    if slugs:
        paths = []
        for slug in slugs:
            p = graph_dir / f"{slug}.json"
            if not p.exists():
                matches = list(graph_dir.glob(f"*{slug}*.json"))
                matches = [m for m in matches if m.name != "graph-metadata.json"]
                if not matches:
                    raise FileNotFoundError(f"No graph matching slug {slug!r} in {graph_dir}")
                p = matches[0]
            paths.append(p)
    else:
        paths = sorted(
            p for p in graph_dir.glob("*.json") if p.name != "graph-metadata.json"
        )

    cache: dict[int, Any] = {}
    layer_to_file = _load_transcoder_config() if recompute else None

    written: list[Path] = []
    for path in paths:
        written.extend(
            export_cosine_matrix(
                path,
                out_dir,
                recompute=recompute,
                max_nodes=max_nodes,
                cache=cache,
                layer_to_file=layer_to_file,
            )
        )
    return written


def _format_report(report: dict[str, dict[str, Any]]) -> str:
    lines = []
    for slug, r in report.items():
        if "in_group_mean" in r:
            status = "PASS" if r["passed"] else "FAIL"
            lines.append(
                f"[{status}] {slug}: in-group {r['in_group_mean']:.3f} "
                f"vs out-group {r['out_group_mean']:.3f} "
                f"(in pairs={r['n_pairs_in']}, out pairs={r['n_pairs_out']})"
            )
            for label, pc in r["per_cluster"].items():
                lines.append(
                    f"    - {label:<38} in={pc['in']:.3f}  out={pc['out']:.3f}  n={pc['n']}"
                )
        auto = r.get("auto_clusters")
        if auto:
            if "in_group_mean" not in r:
                lines.append(f"[----] {slug}: (no manual supernodes)")
            lines.append(
                f"    auto clusters: {auto['m']} nodes, k={auto['k_min']}..{auto['k_max']} "
                f"(default {auto['default_k']}) from top {auto['max_nodes']} by influence"
            )
    return "\n".join(lines) if lines else "No real graphs found."


if __name__ == "__main__":  # pragma: no cover
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph-dir", default="graph_files")
    ap.add_argument("--no-write", action="store_true", help="don't modify graph JSON")
    args = ap.parse_args()

    rep = verify_all(args.graph_dir, write=not args.no_write)
    print(_format_report(rep))
