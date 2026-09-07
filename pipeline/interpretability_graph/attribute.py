"""Run Gemma-2B attribution via circuit-tracer and annotate edges with 2-colors."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from interpretability_graph.architecture import GEMMA_PRESETS
from interpretability_graph.edge_coloring import (
    annotate_graph,
    assign_edge_2colors,
    conflict_rate,
)

logger = logging.getLogger(__name__)


def resolve_preset(name: str) -> dict[str, str]:
    key = name.lower().strip()
    if key in GEMMA_PRESETS:
        return dict(GEMMA_PRESETS[key])
    return {
        "model": "google/gemma-2-2b",
        "transcoders": name,
        "description": f"Custom transcoder set: {name}",
    }


def attribute_prompt(
    prompt: str,
    *,
    preset: str = "gemma",
    slug: str = "gemma-run",
    graph_file_dir: str | Path = "graph_files",
    graph_output_path: str | Path | None = None,
    dtype: str | None = None,
    batch_size: int | None = None,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    max_feature_nodes: int = 7500,
    node_threshold: float = 1.0,
    edge_threshold: float = 0.98,
    offload: str | None = None,
    color_mode: str = "greedy4",
    verbose: bool = True,
) -> Path:
    """Attribute a prompt on Gemma-2B, prune, export JSON, and 2-color edges.

    Uses pretrained transcoder weights from HuggingFace (GemmaScope PLTs or
    published CLTs). Requires HuggingFace login with access to google/gemma-2-2b
    (accept the model license at https://huggingface.co/google/gemma-2-2b).
    """
    try:
        import torch
        from circuit_tracer import ReplacementModel, attribute
        from circuit_tracer.utils.create_graph_files import create_graph_files
    except ImportError as exc:
        raise ImportError(
            "Gemma attribution requires the optional gemma extra.\n"
            "  pip install -e '.[gemma]'"
        ) from exc

    # Mac MPS: prefer float32 (bf16 support is uneven). CUDA: bfloat16.
    if dtype is None:
        if torch.cuda.is_available():
            dtype = "bfloat16"
        else:
            dtype = "float32"
    if batch_size is None:
        batch_size = 64 if torch.backends.mps.is_available() and not torch.cuda.is_available() else 128

    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)

    cfg = resolve_preset(preset)
    graph_file_dir = Path(graph_file_dir)
    graph_file_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s with transcoders %s (dtype=%s)", cfg["model"], cfg["transcoders"], dtype)
    try:
        model = ReplacementModel.from_pretrained(
            cfg["model"],
            cfg["transcoders"],
            dtype=torch_dtype,
        )
    except Exception as exc:
        msg = str(exc).lower()
        if "gated" in msg or "401" in msg or "403" in msg or "authorized" in msg:
            raise RuntimeError(
                "Cannot download google/gemma-2-2b — HuggingFace auth / license required.\n"
                "  1. Visit https://huggingface.co/google/gemma-2-2b and accept the license\n"
                "  2. Run: hf auth login\n"
                "  3. Re-run attribution\n"
                "Meanwhile you can load pretrained graphs with:\n"
                "  python scripts/fetch_pretrained_graphs.py"
            ) from exc
        raise

    logger.info("Attributing prompt (%d chars)…", len(prompt))
    graph = attribute(
        prompt,
        model,
        max_n_logits=max_n_logits,
        desired_logit_prob=desired_logit_prob,
        batch_size=batch_size,
        max_feature_nodes=max_feature_nodes,
        offload=offload,
        verbose=verbose,
    )

    if graph_output_path is not None:
        out_pt = Path(graph_output_path)
        out_pt.parent.mkdir(parents=True, exist_ok=True)
        graph.to_pt(str(out_pt))
        logger.info("Saved raw graph → %s", out_pt)

    create_graph_files(
        graph,
        slug=slug,
        output_path=str(graph_file_dir),
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
    )

    json_path = graph_file_dir / f"{slug}.json"
    annotate_graph_json(json_path, mode=color_mode)
    return json_path


def annotate_graph_json(
    path: str | Path,
    *,
    mode: str = "greedy4",
    seed_subgraph: bool = True,
) -> dict[str, Any]:
    """Load a visualization JSON, attach k-color fields on links + nodes, rewrite."""
    from interpretability_graph.edge_coloring import HUE_LABELS, HUES, seed_subgraph_from_logits

    path = Path(path)
    with path.open() as f:
        data = json.load(f)

    # Preserve any manual clustering / pins that shipped with the graph
    # (e.g. Neuronpedia supernodes for Dallas / French opposite).
    prev_qp = data.get("qParams") or {}
    prev_supernodes = list(prev_qp.get("supernodes") or [])
    prev_pinned = list(prev_qp.get("pinnedIds") or [])
    prev_clicked = prev_qp.get("clickedId") or ""

    links = data.get("links") or data.get("edges") or []
    nodes = data.get("nodes") or []
    colored_links, colored_nodes = annotate_graph(links, nodes, mode=mode)  # type: ignore[arg-type]
    data["links"] = colored_links
    data["nodes"] = colored_nodes

    qparams = {
        "pinnedIds": prev_pinned,
        "supernodes": prev_supernodes,
        "linkType": prev_qp.get("linkType") or "either",
        "clickedId": prev_clicked,
        "sg_pos": prev_qp.get("sg_pos") or "",
    }

    # Only auto-seed when the graph has no manual clusters and no pins.
    if seed_subgraph and not prev_supernodes and not prev_pinned:
        pinned, supernodes = seed_subgraph_from_logits(colored_nodes, colored_links)
        qparams["pinnedIds"] = pinned
        qparams["supernodes"] = supernodes
        if pinned:
            qparams["clickedId"] = pinned[0]
    elif prev_supernodes and not prev_pinned:
        # Ensure every supernode member is pinned so the subgraph can show them
        member_ids = []
        for sn in prev_supernodes:
            member_ids.extend(sn[1:])
        qparams["pinnedIds"] = list(dict.fromkeys(member_ids))

    data["qParams"] = qparams

    assignments = assign_edge_2colors(
        [
            {
                "id": e.get("id") or f"{e['source']}__{e['target']}",
                "source": e["source"],
                "target": e["target"],
                "weight": e.get("weight", 0.0),
                "source_layer": e.get("source_layer"),
            }
            for e in colored_links
        ],
        mode=mode,  # type: ignore[arg-type]
    )
    rate = conflict_rate(assignments, colored_links)
    data.setdefault("metadata", {})
    data["metadata"]["edge_coloring"] = {
        "mode": mode,
        "conflict_rate": round(rate, 4),
        "n_edges": len(colored_links),
        "n_nodes": len(colored_nodes),
        "hues": {HUE_LABELS[i]: f"#{HUES[i][0]:02X}{HUES[i][1]:02X}{HUES[i][2]:02X}" for i in range(4)},
        "intensity": "|attribution| → color saturation + line thickness",
        "thickness": "sqrt(|pctInput| / p95) → stroke width (thicker = stronger)",
        "nodes": "weighted-majority of incident edge classes",
        "manual_clusters": len(prev_supernodes),
        "subgraph_seeded": bool(qparams["pinnedIds"]) and not prev_supernodes,
    }

    with path.open("w") as f:
        json.dump(data, f)
    logger.info(
        "Annotated %s with %s coloring (conflict_rate=%.3f, %d edges, %d nodes, "
        "%d pinned, %d clusters)",
        path,
        mode,
        rate,
        len(colored_links),
        len(colored_nodes),
        len(qparams["pinnedIds"]),
        len(qparams["supernodes"]),
    )
    return data
