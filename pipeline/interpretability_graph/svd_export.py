"""Backend full-graph SVD for the attribution-graph UI.

Builds cosine + attribution adjacency matrices over **all** transcoder nodes,
computes truncated SVD (default top-32), and writes a compact JSON bundle the
browser can load instead of running Jacobi in-page.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _attribution_matrices(
    data: dict[str, Any], node_ids: list[str]
) -> dict[str, Any]:
    import numpy as np

    idx = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)
    unsigned = np.zeros((n, n), dtype=np.float64)
    signed = np.zeros((n, n), dtype=np.float64)
    symmetric = np.zeros((n, n), dtype=np.float64)

    for link in data.get("links") or data.get("edges") or []:
        s = link.get("source")
        t = link.get("target")
        if s not in idx or t not in idx:
            continue
        w = float(link.get("weight") or link.get("pctInput") or 0.0)
        if not w:
            continue
        i, j = idx[s], idx[t]
        unsigned[i, j] += abs(w)
        signed[i, j] += w
        symmetric[i, j] += abs(w)
        symmetric[j, i] += abs(w)

    return {
        "unsigned": unsigned,
        "signed": signed,
        "symmetric": symmetric,
    }


def _truncated_svd(M: Any, k: int) -> dict[str, Any]:
    import numpy as np

    M = np.asarray(M, dtype=np.float64)
    n = M.shape[0]
    k = max(1, min(int(k), n))
    frob2 = float(np.sum(M * M))

    # Economy SVD; for wide/tall still fine (matrices are square here).
    try:
        # Full SVD then truncate — stable for n ≲ 2k.
        U, S, _Vt = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        # Fallback: eig of M Mᵀ
        G = M @ M.T
        evals, evecs = np.linalg.eigh(G)
        order = np.argsort(evals)[::-1]
        S = np.sqrt(np.clip(evals[order], 0.0, None))
        U = evecs[:, order]

    U = U[:, :k]
    S = S[:k]
    # Orient: largest-magnitude entry of each left vector positive.
    for r in range(U.shape[1]):
        i = int(np.argmax(np.abs(U[:, r])))
        if U[i, r] < 0:
            U[:, r] *= -1

    return {
        "sigmas": [float(x) for x in S],
        "vectors": [[float(x) for x in U[:, r]] for r in range(U.shape[1])],
        "frob2": frob2,
        "k": int(k),
        "truncated": k < n,
    }


def export_svd_bundle(
    json_path: str | Path,
    out_dir: str | Path,
    *,
    k: int = 32,
    cache: dict[int, Any] | None = None,
    layer_to_file: dict[int, str] | None = None,
) -> Path:
    """Write ``{slug}-svd-bundle.json`` with top-k SVD for all four matrices."""
    from interpretability_graph.verify_supernodes import build_feature_cosine

    import numpy as np

    json_path = Path(json_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(json_path.read_text())
    slug = (data.get("metadata") or {}).get("slug") or json_path.stem

    logger.info("Building full-graph cosine for %s …", json_path.name)
    C, node_ids, cos_method = build_feature_cosine(
        json_path,
        max_nodes=None,
        cache=cache,
        layer_to_file=layer_to_file,
    )
    n = len(node_ids)
    logger.info("Cosine %dx%d — building adjacency matrices …", n, n)
    adj = _attribution_matrices(data, node_ids)

    charts_spec = [
        ("cosine", C, cos_method, "Cosine similarity"),
        ("unsigned", adj["unsigned"], "A_ij = |w(i→j)|", "Unsigned adjacency"),
        ("signed", adj["signed"], "A_ij = w(i→j)", "Signed adjacency"),
        (
            "symmetric",
            adj["symmetric"],
            "A_ij = |w(i→j)| + |w(j→i)|",
            "Symmetric |A|",
        ),
    ]

    charts: dict[str, Any] = {}
    for key, M, method, title in charts_spec:
        logger.info("SVD %s (%dx%d, k=%d) …", key, n, n, min(k, n))
        piece = _truncated_svd(M, k)
        piece["title"] = title
        piece["method"] = method
        piece["equation"] = {
            "cosine": (
                "C_{ij} = \\cos(\\mathbf{v}_i, \\mathbf{v}_j),\\quad"
                "\\mathbf{v}_f = \\operatorname{unit}["
                "\\widehat{W_{\\mathrm{enc}}}[:,f]\\,\\|\\,"
                "\\widehat{W_{\\mathrm{dec}}}[f,:]]"
            ),
            "unsigned": "A_{ij} = \\lvert w(i \\rightarrow j)\\rvert",
            "signed": "A_{ij} = w(i \\rightarrow j)",
            "symmetric": (
                "A_{ij} = \\lvert w(i \\rightarrow j)\\rvert"
                " + \\lvert w(j \\rightarrow i)\\rvert"
            ),
        }.get(key, "")
        piece["note"] = {
            "cosine": "inflow ∥ outflow (backend full graph)",
            "unsigned": "w = attribution edge weight; direction i → j only",
            "signed": "keeps positive / negative attribution sign",
            "symmetric": "undirected strength between i and j",
        }.get(key, "")
        charts[key] = piece

        # Also stash float32 matrix for optional client submatrix / heatmap.
        np.save(out_dir / f"{slug}-{key}.npy", np.asarray(M, dtype=np.float32))

    bundle = {
        "slug": slug,
        "prompt": (data.get("metadata") or {}).get("prompt"),
        "n": n,
        "k": min(k, n),
        "node_ids": node_ids,
        "source": "interpretability_graph.svd_export",
        "charts": charts,
        "matrices": {
            key: f"svd_exports/{slug}-{key}.npy"
            for key in ("cosine", "unsigned", "signed", "symmetric")
        },
    }

    out_path = out_dir / f"{slug}-svd-bundle.json"
    # Compact JSON (no indent) — vectors dominate size.
    out_path.write_text(json.dumps(bundle, separators=(",", ":")))

    # Point the graph at the bundle for the UI.
    data.setdefault("metadata", {})["svd_bundle"] = {
        "n": n,
        "k": min(k, n),
        "path": f"svd_exports/{out_path.name}",
        "matrices": bundle["matrices"],
    }
    json_path.write_text(json.dumps(data))

    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info("Wrote %s (%.1f MB, n=%d, k=%d)", out_path, size_mb, n, min(k, n))
    return out_path


def export_svd_bundles(
    graph_dir: str | Path = "graph_files",
    out_dir: str | Path | None = None,
    *,
    slugs: list[str] | None = None,
    k: int = 32,
) -> list[Path]:
    from interpretability_graph.verify_supernodes import _load_transcoder_config

    graph_dir = Path(graph_dir)
    out_dir = Path(out_dir) if out_dir else graph_dir / "svd_exports"

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
        written.append(
            export_svd_bundle(
                path,
                out_dir,
                k=k,
                cache=cache,
                layer_to_file=layer_to_file,
            )
        )
    return written
