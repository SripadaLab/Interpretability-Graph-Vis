"""Download featured Gemma-2-2B attribution graphs built with pretrained GemmaScope weights."""

from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path

from interpretability_graph.attribute import annotate_graph_json

logger = logging.getLogger(__name__)

# Precomputed Neuronpedia graphs (no on-device inference required).
# local_slug → remote Neuronpedia slug (or same string).
FEATURED_GRAPHS: dict[str, dict] = {
    "gemma-addition": {
        "remote": "gemma-addition",
        "title_prefix": "Canonical · 3 + 5",
    },
    "gemma-fact-dallas-austin": {
        "remote": "gemma-fact-dallas-austin",
        "title_prefix": "Canonical · capital of Texas",
    },
    "gemma-michael-jordan": {
        "remote": "michaeljordanpla-1773757470825",
        "title_prefix": "Canonical · Michael Jordan",
    },
    "gemma-small-big-fr": {
        "remote": "gemma-small-big-fr",
        "title_prefix": "Canonical · French opposite",
    },
}

DEFAULT_SLUGS = tuple(FEATURED_GRAPHS.keys())


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "interpretability-graph"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.load(resp)


def download_slug(
    local_slug: str,
    *,
    remote_slug: str | None = None,
    title_prefix: str = "GemmaScope",
    model_id: str = "gemma-2-2b",
    out_dir: str | Path = "graph_files",
    color_mode: str = "greedy4",
) -> Path:
    """Fetch a Neuronpedia graph JSON (GemmaScope transcoder attribution) and color it."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    remote = remote_slug or local_slug
    meta = FEATURED_GRAPHS.get(local_slug, {})
    remote = meta.get("remote", remote)
    title_prefix = meta.get("title_prefix", title_prefix)

    info = fetch_json(f"https://www.neuronpedia.org/api/graph/{model_id}/{remote}")
    logger.info("Downloading %s ← %s (%s)", local_slug, remote, info["url"])
    print(f"Downloading {local_slug} ← {info['url']}")
    data = fetch_json(info["url"])

    md = data.setdefault("metadata", {})
    md["slug"] = local_slug
    md.setdefault("scan", info.get("sourceSetName") or model_id)
    md.setdefault("prompt", info.get("prompt") or md.get("prompt", ""))
    if info.get("promptTokens") and not md.get("prompt_tokens"):
        md["prompt_tokens"] = info["promptTokens"]
    md["title_prefix"] = title_prefix
    md["source"] = {
        "neuronpedia": f"https://www.neuronpedia.org/{model_id}/graph?slug={remote}",
        "transcoders": "gemmascope-transcoder-16k (pretrained)",
        "remote_slug": remote,
    }

    path = out_dir / f"{local_slug}.json"
    path.write_text(json.dumps(data))
    annotate_graph_json(path, mode=color_mode)

    g = json.loads(path.read_text())
    coloring = g["metadata"].get("edge_coloring", {})
    print(
        f"  nodes={len(g.get('nodes', []))} "
        f"links={len(g.get('links', []))} "
        f"conflict_rate={coloring.get('conflict_rate')}"
    )
    return path


def fetch_pretrained_graphs(
    slugs: list[str] | None = None,
    *,
    out_dir: str | Path = "graph_files",
    color_mode: str = "greedy4",
) -> list[Path]:
    from interpretability_graph.demo_graph import reset_canonical_graphs

    slugs = slugs or list(DEFAULT_SLUGS)
    paths = []
    for slug in slugs:
        # Allow either local friendly slug or raw remote slug
        if slug in FEATURED_GRAPHS:
            paths.append(download_slug(slug, out_dir=out_dir, color_mode=color_mode))
        else:
            paths.append(
                download_slug(
                    slug.replace("/", "-"),
                    remote_slug=slug,
                    out_dir=out_dir,
                    color_mode=color_mode,
                )
            )
    # Rebuild DAG demo + ordered metadata for all local graphs
    reset_canonical_graphs(out_dir)
    return paths
