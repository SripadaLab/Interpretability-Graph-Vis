"""Canonical local demo graphs: DAG acronym, 3+5 addition, Dallas→Austin fact.

The DAG acronym case is a compact synthetic graph (paper structure).
Addition and Dallas graphs come from pretrained GemmaScope attributions
(see `interpretability-graph fetch`).
"""

from __future__ import annotations

import json
from pathlib import Path

from interpretability_graph.edge_coloring import (
    HUE_LABELS,
    HUES,
    annotate_graph,
    assign_edge_2colors,
    conflict_rate,
)


# Paper case study: "The National Digital Analytics Group (N" → "DAG".
PROMPT = "The National Digital Analytics Group ("
TOKENS = ["The", " National", " Digital", " Analytics", " Group", " ("]

# Featured set shown in the UI dropdown (order matters).
CANONICAL_SLUGS = (
    "gemma-dag-demo",
    "gemma-addition",
    "gemma-fact-dallas-austin",
    "gemma-michael-jordan",
    "gemma-small-big-fr",
)

CLEAN_QPARAMS = {
    "pinnedIds": [],
    "supernodes": [],
    "linkType": "either",
    "clickedId": "",
    "sg_pos": "",
}

# Paper-style subgraph for the DAG acronym demo (pinned + supernodes).
DAG_SUBGRAPH = {
    "pinnedIds": [
        "E_1002_2",
        "E_1003_3",
        "E_1004_4",
        "1_4012_2",
        "1_8821_3",
        "1_2201_4",
        "3_5502_2",
        "4_3300_5",
        "8_9100_5",
        "11_16401_5",
        "11_16402_5",
        "12_17100_5",
        "13_18001_5",
        "13_18002_5",
        "14_19010_5",
        "16_24601_5",
        "16_24602_5",
        "17_25001_5",
        "19_200_5",
        "19_202_5",
    ],
    "supernodes": [
        ["Digital", "1_4012_2", "3_5502_2", "E_1002_2"],
        ["Analytics", "1_8821_3", "E_1003_3"],
        ["Group", "1_2201_4", "E_1004_4"],
        ["say DAG", "16_24601_5", "17_25001_5", "13_18001_5"],
    ],
    "linkType": "either",
    "clickedId": "16_24601_5",
    "sg_pos": "",
}


def _feat(layer: int, feat: int, pos: int, clerp: str, influence: float, act: float = 1.0) -> dict:
    return {
        "node_id": f"{layer}_{feat}_{pos}",
        "feature": feat,
        "layer": str(layer),
        "ctx_idx": pos,
        "feature_type": "cross layer transcoder",
        "jsNodeId": f"{layer}_{feat}-0",
        "clerp": clerp,
        "influence": influence,
        "activation": act,
    }


def _emb(pos: int, vocab: int, influence: float) -> dict:
    return {
        "node_id": f"E_{vocab}_{pos}",
        "feature": pos,
        "layer": "E",
        "ctx_idx": pos,
        "feature_type": "embedding",
        "jsNodeId": f"E_{vocab}-{pos}",
        "clerp": "",
        "influence": influence,
    }


def _err(layer: int, pos: int, influence: float) -> dict:
    return {
        "node_id": f"{layer}_-1_{pos}",
        "feature": -1,
        "layer": str(layer),
        "ctx_idx": pos,
        "feature_type": "mlp reconstruction error",
        "jsNodeId": f"{layer}_-1-{pos}",
        "clerp": "",
        "influence": influence,
    }


def _logit(vocab: int, token: str, prob: float, n_layers: int = 18) -> dict:
    layer = n_layers + 1
    pos = len(TOKENS) - 1
    return {
        "node_id": f"{layer}_{vocab}_{pos}",
        "feature": vocab,
        "layer": str(layer),
        "ctx_idx": pos,
        "feature_type": "logit",
        "token_prob": prob,
        "is_target_logit": token == "N",
        "jsNodeId": f"L_{vocab}-{pos}",
        "clerp": f'Output "{token}" (p={prob:.3f})',
        "influence": prob,
    }


def build_demo_graph() -> dict:
    """Synthetic but structurally faithful graph for the acronym case study."""
    n_layers = 18
    nodes = [
        _emb(0, 1000, 0.12),
        _emb(1, 1001, 0.18),
        _emb(2, 1002, 0.55),
        _emb(3, 1003, 0.52),
        _emb(4, 1004, 0.48),
        _emb(5, 1005, 0.22),
        _feat(1, 4012, 2, "digital", 0.42, 2.1),
        _feat(1, 8821, 3, "analytics", 0.40, 1.9),
        _feat(1, 2201, 4, "group", 0.38, 1.7),
        _feat(2, 1190, 1, "national / organization names", 0.15, 0.9),
        _feat(3, 5502, 2, "Uppercase before an acronym", 0.28, 1.2),
        _feat(4, 3300, 5, "open parenthesis → acronym", 0.35, 1.4),
        _feat(8, 9100, 5, "say first letter of acronym", 0.45, 1.5),
        _feat(11, 16401, 5, 'say "D_"', 0.58, 2.0),
        _feat(11, 16402, 5, 'say "_A"', 0.50, 1.8),
        _feat(12, 17100, 5, "continue acronym", 0.47, 1.6),
        _feat(13, 18001, 5, 'say "DA_"', 0.62, 2.2),
        _feat(13, 18002, 5, 'say "_G"', 0.55, 1.9),
        _feat(14, 19010, 5, "acronym completion", 0.44, 1.5),
        _feat(16, 24601, 5, "[say DAG] say dag", 0.78, 2.8),
        _feat(16, 24602, 5, 'say "N" (acronym start)', 0.70, 2.4),
        _feat(17, 25001, 5, "promote DAG / acronym token", 0.66, 2.1),
        _err(6, 2, 0.08),
        _err(10, 5, 0.11),
        _err(15, 5, 0.09),
        _logit(200, "N", 0.41, n_layers),
        _logit(201, "D", 0.18, n_layers),
        _logit(202, "dag", 0.12, n_layers),
        _logit(203, "A", 0.07, n_layers),
    ]

    def e(src: str, tgt: str, w: float) -> dict:
        return {"source": src, "target": tgt, "weight": w}

    links = [
        e("E_1002_2", "1_4012_2", 0.31),
        e("E_1003_3", "1_8821_3", 0.29),
        e("E_1004_4", "1_2201_4", 0.27),
        e("E_1001_1", "2_1190_1", 0.14),
        e("E_1005_5", "4_3300_5", 0.22),
        e("1_4012_2", "3_5502_2", 0.18),
        e("1_4012_2", "11_16401_5", 0.068),
        e("1_8821_3", "11_16402_5", 0.055),
        e("1_2201_4", "13_18002_5", 0.048),
        e("3_5502_2", "8_9100_5", 0.12),
        e("4_3300_5", "8_9100_5", 0.19),
        e("8_9100_5", "11_16401_5", 0.21),
        e("8_9100_5", "11_16402_5", 0.17),
        e("8_9100_5", "12_17100_5", 0.15),
        e("11_16401_5", "13_18001_5", 0.26),
        e("11_16402_5", "13_18001_5", 0.22),
        e("11_16401_5", "16_24601_5", 0.068),
        e("13_18001_5", "16_24601_5", 0.19),
        e("13_18002_5", "16_24601_5", 0.11),
        e("12_17100_5", "14_19010_5", 0.13),
        e("14_19010_5", "16_24601_5", 0.09),
        e("13_18001_5", "16_24602_5", 0.16),
        e("16_24601_5", "17_25001_5", 0.24),
        e("16_24602_5", "17_25001_5", 0.21),
        e("16_24602_5", "19_200_5", 0.18),
        e("17_25001_5", "19_200_5", 0.22),
        e("16_24601_5", "19_202_5", 0.046),
        e("17_25001_5", "19_202_5", 0.04),
        e("16_24601_5", "19_201_5", -0.03),
        e("11_16401_5", "19_201_5", 0.05),
        e("6_-1_2", "11_16401_5", 0.02),
        e("10_-1_5", "16_24601_5", 0.025),
        e("15_-1_5", "19_200_5", 0.015),
        e("1_4012_2", "12_17100_5", 0.04),
        e("1_8821_3", "13_18001_5", 0.035),
        e("1_2201_4", "14_19010_5", 0.03),
        e("2_1190_1", "8_9100_5", 0.028),
        e("3_5502_2", "11_16401_5", 0.045),
        e("4_3300_5", "13_18002_5", 0.038),
        e("11_16402_5", "16_24602_5", 0.06),
        e("12_17100_5", "16_24602_5", 0.042),
        e("14_19010_5", "17_25001_5", 0.05),
        e("8_9100_5", "16_24601_5", 0.033),
        e("13_18002_5", "17_25001_5", 0.07),
        e("E_1002_2", "11_16401_5", 0.02),
        e("E_1003_3", "13_18001_5", 0.018),
        e("E_1004_4", "13_18002_5", 0.016),
    ]

    colored_links, colored_nodes = annotate_graph(links, nodes, mode="greedy4")
    assignments = assign_edge_2colors(
        [
            {
                "id": f"{x['source']}__{x['target']}",
                "source": x["source"],
                "target": x["target"],
                "weight": x["weight"],
            }
            for x in colored_links
        ],
        mode="greedy4",
    )

    return {
        "metadata": {
            "slug": "gemma-dag-demo",
            "scan": "gemma-2-2b",
            "transcoder_list": ["demo-synthetic-clt"],
            "prompt_tokens": TOKENS,
            "prompt": PROMPT,
            "node_threshold": 0.8,
            "schema_version": 1,
            "title_prefix": "Canonical",
            "edge_coloring": {
                "mode": "greedy4",
                "conflict_rate": round(conflict_rate(assignments, colored_links), 4),
                "n_edges": len(colored_links),
                "n_nodes": len(colored_nodes),
                "hues": {
                    HUE_LABELS[i]: f"#{HUES[i][0]:02X}{HUES[i][1]:02X}{HUES[i][2]:02X}"
                    for i in range(4)
                },
                "intensity": "|weight| / |influence| → saturation",
                "thickness": "sqrt(|weight| / p95) → stroke width (thicker = stronger)",
                "nodes": "weighted-majority of incident edge classes",
                "note": "Synthetic DAG acronym demo; fetch pretrained graphs for real GemmaScope runs.",
            },
        },

        "qParams": dict(DAG_SUBGRAPH),
        "nodes": colored_nodes,
        "links": colored_links,
    }


def _write_metadata(out_dir: Path, graphs: list[dict]) -> None:
    meta = {"graphs": graphs}
    with (out_dir / "graph-metadata.json").open("w") as f:
        json.dump(meta, f, indent=2)

    data_dir = Path(__file__).resolve().parent.parent / "frontend" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "graph-metadata.json").open("w") as f:
        json.dump(meta, f, indent=2)


def reset_canonical_graphs(out_dir: str | Path = "graph_files") -> list[Path]:
    """Rewrite DAG demo, re-color/seed pretrained graphs, sync metadata in canonical order."""
    from interpretability_graph.attribute import annotate_graph_json

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Fresh DAG acronym demo
    dag = build_demo_graph()
    dag_path = out_dir / f"{dag['metadata']['slug']}.json"
    with dag_path.open("w") as f:
        json.dump(dag, f, indent=2)

    paths = [dag_path]

    titles = {
        "gemma-dag-demo": "Canonical · DAG acronym",
        "gemma-addition": "Canonical · 3 + 5",
        "gemma-fact-dallas-austin": "Canonical · capital of Texas",
        "gemma-michael-jordan": "Canonical · Michael Jordan",
        "gemma-small-big-fr": "Canonical · French opposite",
    }

    # 2) Recolor + auto-seed subgraph for every pretrained graph present
    for slug in CANONICAL_SLUGS:
        if slug == "gemma-dag-demo":
            continue
        path = out_dir / f"{slug}.json"
        if path.exists():
            annotate_graph_json(path, mode="greedy4", seed_subgraph=True)
            paths.append(path)

    graph_entries = []
    for slug in CANONICAL_SLUGS:
        path = out_dir / f"{slug}.json"
        if not path.exists():
            continue
        with path.open() as f:
            g = json.load(f)
        md = g.setdefault("metadata", {})
        md["slug"] = slug
        md["title_prefix"] = titles.get(slug, md.get("title_prefix", "Canonical"))
        if slug == "gemma-dag-demo":
            g["qParams"] = dict(DAG_SUBGRAPH)
        with path.open("w") as f:
            if slug == "gemma-dag-demo":
                json.dump(g, f, indent=2)
            else:
                json.dump(g, f)
        graph_entries.append(
            {
                "slug": slug,
                "prompt": md.get("prompt", ""),
                "scan": md.get("scan", "gemma-2-2b"),
                "node_threshold": 1.0 if md.get("node_threshold") is not None else None,
                "title_prefix": titles.get(slug, "Canonical"),
            }
        )

    _write_metadata(out_dir, graph_entries)
    return paths


def write_demo_graph(out_dir: str | Path = "graph_files") -> Path:
    """Write/reset the canonical demo set; return the DAG demo path."""
    paths = reset_canonical_graphs(out_dir)
    return paths[0]
