"""Spectral clustering on feature cosine similarity.

Pipeline (matches the UI / README):
  1. All transcoder feature nodes (or top-N by influence)
  2. Cosine C from enc∥dec (or neighborhood fallback)
  3. Friendships W = max(C, 0)  — drop negatives entirely
  4. Mutual kNN sparsify — keep strongest positive links, zero the rest
  5. Symmetric normalized Laplacian L_sym = I − D^{-1/2} W D^{-1/2}
  6. Smallest eigenvectors → k-means for each k (raw rows; no NJW row-normalize)
  7. Write metadata.spectral_clusters + svd_exports/{slug}-affinity.npy (W)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _kmeans(
    X: Any,
    k: int,
    *,
    seed: int = 0,
    n_init: int = 8,
    max_iter: int = 100,
) -> list[int]:
    """Simple NumPy k-means (no sklearn dependency). Returns labels 0..k-1."""
    import numpy as np

    rng = np.random.default_rng(seed)
    n, d = X.shape
    k = max(1, min(int(k), n))
    best_labels = None
    best_inertia = np.inf

    for init in range(n_init):
        # k-means++ style seed
        centers = np.empty((k, d), dtype=np.float64)
        centers[0] = X[rng.integers(0, n)]
        closest = np.full(n, np.inf)
        for j in range(1, k):
            dist = np.sum((X - centers[j - 1]) ** 2, axis=1)
            closest = np.minimum(closest, dist)
            probs = closest / (closest.sum() + 1e-30)
            centers[j] = X[rng.choice(n, p=probs)]

        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            # assign
            d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = np.argmin(d2, axis=1)
            if np.array_equal(new_labels, labels):
                labels = new_labels
                break
            labels = new_labels
            for j in range(k):
                mask = labels == j
                if mask.any():
                    centers[j] = X[mask].mean(axis=0)
                else:
                    centers[j] = X[rng.integers(0, n)]

        inertia = float(((X - centers[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()

    assert best_labels is not None
    return [int(x) for x in best_labels]


def _mutual_knn(W: Any, knn: int) -> Any:
    """Keep each row's top-`knn` positive affinities; symmetrize by averaging."""
    import numpy as np

    m = W.shape[0]
    knn = max(1, min(int(knn), m - 1))
    out = np.zeros_like(W)
    for i in range(m):
        row = W[i].copy()
        row[i] = 0.0
        if knn >= m - 1:
            idx = np.where(row > 0)[0]
        else:
            # argpartition for top-k; ignore zeros
            if not np.any(row > 0):
                continue
            idx = np.argpartition(row, -knn)[-knn:]
            idx = idx[row[idx] > 0]
        out[i, idx] = row[idx]
    # Symmetrize
    out = 0.5 * (out + out.T)
    # Rescue isolates: connect to best positive partner in original W
    deg = out.sum(axis=1)
    for i in np.where(deg <= 0)[0]:
        row = W[i].copy()
        row[i] = 0.0
        j = int(np.argmax(row))
        if row[j] > 0:
            out[i, j] = out[j, i] = row[j]
    return out


def compute_spectral_clusters(
    json_path: str | Path,
    *,
    write: bool = True,
    max_nodes: int | None = None,
    k_max: int = 20,
    knn: int = 10,
    seed: int = 0,
    cache: dict[int, Any] | None = None,
    layer_to_file: dict[int, str] | None = None,
    cosine: Any | None = None,
    node_ids: list[str] | None = None,
) -> dict[str, Any] | None:
    """Run spectral clustering and optionally write ``metadata.spectral_clusters``."""
    import numpy as np

    from interpretability_graph.verify_supernodes import build_feature_cosine

    json_path = Path(json_path)
    data = json.loads(json_path.read_text())

    if cosine is None or node_ids is None:
        C, node_ids, cos_method = build_feature_cosine(
            json_path,
            max_nodes=max_nodes,
            cache=cache,
            layer_to_file=layer_to_file,
        )
    else:
        C = np.asarray(cosine, dtype=np.float64)
        node_ids = list(node_ids)
        cos_method = "precomputed cosine"

    m = len(node_ids)
    if m < 2:
        return None
    C = np.clip(np.asarray(C, dtype=np.float64), -1.0, 1.0)
    if C.shape != (m, m):
        raise ValueError(f"cosine shape {C.shape} != ({m},{m})")

    # 3) Friendships: drop negatives altogether
    W = np.maximum(C, 0.0)
    np.fill_diagonal(W, 0.0)

    # 4) kNN sparsify
    W = _mutual_knn(W, knn)

    # 5) Symmetric normalized Laplacian
    deg = W.sum(axis=1)
    deg_safe = np.maximum(deg, 1e-12)
    d_inv_sqrt = 1.0 / np.sqrt(deg_safe)
    L = np.eye(m) - (d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :])

    # 6) Smallest eigenvectors
    k_hi = max(2, min(int(k_max), m))
    evals, evecs = np.linalg.eigh(L)
    # eigh returns ascending eigenvalues
    U_full = evecs[:, :k_hi].copy()
    evals_kept = [float(x) for x in evals[:k_hi]]

    by_k: dict[str, list[list[str]]] = {}

    for kk in range(2, k_hi + 1):
        # Cluster in the raw Laplacian embedding (Shi–Malik style).
        # Do not row-normalize: |u_i| (how strongly node i participates in
        # the cut) is part of the geometry k-means sees.
        U = U_full[:, :kk]
        labels = _kmeans(U, kk, seed=seed)
        grouped: dict[int, list[str]] = {}
        for nid, c in zip(node_ids, labels):
            grouped.setdefault(int(c), []).append(nid)
        # Order groups by total |influence| if available
        infl = {
            n.get("node_id"): abs(float(n.get("influence") or 0.0))
            for n in data.get("nodes") or []
        }
        order = sorted(
            grouped.keys(),
            key=lambda c: -sum(infl.get(nid, 0.0) for nid in grouped[c]),
        )
        supernodes = [
            [f"group {gi + 1}", *grouped[cid]]
            for gi, cid in enumerate(order)
            if grouped[cid]
        ]
        by_k[str(kk)] = supernodes

    default_k = max(2, min(k_hi, round(m / 5) if m > 10 else 2))
    # Large graphs hit n/5 >> 20. Start at 12 so the slider can go higher.
    if k_hi > 12:
        default_k = min(default_k, 12)
    # nnz friendships for diagnostics
    nnz = int(np.count_nonzero(W))

    # Persist dense W for the UI affinity heatmap (same dir as cosine / SVD).
    affinity_path = None
    if write:
        export_dir = json_path.parent / "svd_exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        aff_file = export_dir / f"{json_path.stem}-affinity.npy"
        np.save(aff_file, np.asarray(W, dtype=np.float32))
        affinity_path = f"svd_exports/{aff_file.name}"
        logger.info("Wrote affinity W %s shape=%s nnz=%d", aff_file.name, W.shape, nnz)

    result: dict[str, Any] = {
        "method": (
            "spectral clustering on max(cosine,0) mutual-kNN affinity; "
            "L_sym smallest evecs → k-means (no row-normalize); "
            f"cosine = {cos_method}"
        ),
        "node_ids": node_ids,
        "max_nodes": max_nodes if max_nodes else m,
        "m": m,
        "knn": int(knn),
        "affinity": "max_cosine_mutual_knn",
        "laplacian": "symmetric_normalized",
        "embedding": "raw_smallest_evecs",
        "row_normalize": False,
        "k_min": 2,
        "k_max": k_hi,
        "default_k": default_k,
        "eigenvalues": evals_kept,
        "nnz_affinity": nnz,
        "affinity_path": affinity_path,
        "by_k": by_k,
    }

    if write:
        data.setdefault("metadata", {})["spectral_clusters"] = result
        json_path.write_text(json.dumps(data))
        logger.info(
            "Wrote spectral_clusters (%d nodes, knn=%d, k=2..%d, nnz=%d) → %s",
            m,
            knn,
            k_hi,
            nnz,
            json_path.name,
        )
    return result


def export_spectral_clusters(
    graph_dir: str | Path = "graph_files",
    *,
    slugs: list[str] | None = None,
    max_nodes: int | None = None,
    k_max: int = 20,
    knn: int = 10,
    seed: int = 0,
) -> list[Path]:
    """Run spectral clustering for selected (or all) graphs."""
    from interpretability_graph.verify_supernodes import _load_transcoder_config

    graph_dir = Path(graph_dir)
    if slugs:
        paths = []
        for slug in slugs:
            p = graph_dir / f"{slug}.json"
            if not p.exists():
                matches = [
                    m
                    for m in graph_dir.glob(f"*{slug}*.json")
                    if m.name != "graph-metadata.json"
                ]
                if not matches:
                    raise FileNotFoundError(f"No graph matching {slug!r}")
                p = matches[0]
            paths.append(p)
    else:
        paths = sorted(
            p for p in graph_dir.glob("*.json") if p.name != "graph-metadata.json"
        )

    cache: dict[int, Any] = {}
    layer_to_file = _load_transcoder_config()
    written: list[Path] = []
    for path in paths:
        # Prefer existing full-graph cosine .npy from export-svd when clustering all nodes
        cosine = None
        node_ids = None
        if max_nodes is None or max_nodes <= 0:
            slug = path.stem
            npy = graph_dir / "svd_exports" / f"{slug}-cosine.npy"
            bundle = graph_dir / "svd_exports" / f"{slug}-svd-bundle.json"
            if npy.exists() and bundle.exists():
                import numpy as np

                cosine = np.load(npy)
                node_ids = json.loads(bundle.read_text()).get("node_ids")
                logger.info("Using precomputed cosine %s (%s)", npy.name, cosine.shape)

        result = compute_spectral_clusters(
            path,
            write=True,
            max_nodes=max_nodes if max_nodes and max_nodes > 0 else None,
            k_max=k_max,
            knn=knn,
            seed=seed,
            cache=cache,
            layer_to_file=layer_to_file,
            cosine=cosine,
            node_ids=node_ids,
        )
        if result:
            written.append(path)
    return written
