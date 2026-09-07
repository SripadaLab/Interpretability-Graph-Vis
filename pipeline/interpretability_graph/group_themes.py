"""Group theme assignment (step 2).

Step 1 (neuronpedia.py) gives every feature node a semantic interpretation.
This module packages each spectral group's interpretations into a request an LLM
can summarize, then folds the returned themes back into the graph metadata.

Groups repeat across k -- a group at k=19 is often identical at k=20 -- so
requests are deduplicated by member set and one theme is reused everywhere that
set appears. Themes land in metadata.spectral_clusters.themes[k][group_index],
aligned with by_k, so the UI can index them directly.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Big leftover bins would swamp a prompt, so sample them and say so.
MAX_MEMBERS_IN_REQUEST = 80

# The terse np_max-act label is the display clerp; these richer descriptions are
# better evidence for a theme, so prefer them when building the request.
_DETAIL_TYPE_PRIORITY = {
    "oai_token-act-pair": 0,
    "np_max-act": 1,
    "np_max-act-logits": 2,
}


def _member_key(members: list[str]) -> str:
    return "|".join(sorted(members))


def group_id(slug: str, members: list[str]) -> str:
    """Short stable handle for a member set, used as the theme lookup key."""
    digest = hashlib.sha1(_member_key(members).encode()).hexdigest()[:8]
    return f"{slug}#{digest}"


def _sample_members(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Even stride through the (layer-sorted) members so all layers show up."""
    if len(rows) <= limit:
        return rows
    ordered = sorted(rows, key=lambda r: (r["layer"], r["node_id"]))
    step = len(ordered) / limit
    return [ordered[int(i * step)] for i in range(limit)]


def _token_label(tokens: list[str], ctx: int) -> str:
    if 0 <= ctx < len(tokens):
        return str(tokens[ctx]).replace("\n", "\\n")
    return "?"


def _detail_text(explanations: list[dict[str, Any]]) -> str | None:
    if not explanations:
        return None
    ranked = sorted(
        explanations,
        key=lambda e: _DETAIL_TYPE_PRIORITY.get(e.get("type", ""), 99),
    )
    return ranked[0].get("text") or None


def _load_cache(graph_dir: Path, model_id: str) -> dict[str, dict[str, Any]]:
    root = graph_dir / "interpretations" / model_id
    out: dict[str, dict[str, Any]] = {}
    if not root.exists():
        return out
    for path in root.glob("*.json"):
        out[path.stem] = json.loads(path.read_text())
    return out


def build_requests(
    graph_dir: str | Path = "graph_files",
    *,
    slugs: list[str] | None = None,
    model_id: str = "gemma-2-2b",
) -> list[dict[str, Any]]:
    """One entry per distinct group member set, with member interpretations."""
    graph_dir = Path(graph_dir)
    paths = sorted(p for p in graph_dir.glob("*.json") if p.name != "graph-metadata.json")
    if slugs:
        paths = [p for p in paths if p.stem in set(slugs)]

    requests: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text())
        meta = data.get("metadata") or {}
        spec = meta.get("spectral_clusters")
        if not spec:
            continue

        cache = _load_cache(graph_dir, meta.get("scan") or model_id)
        tokens = meta.get("prompt_tokens") or []
        by_node = {n.get("node_id"): n for n in data.get("nodes", [])}

        seen: dict[str, dict[str, Any]] = {}
        for k in sorted(spec.get("by_k", {}), key=int):
            for gi, grp in enumerate(spec["by_k"][k]):
                members = list(grp[1:])
                key = _member_key(members)
                entry = seen.get(key)
                if entry is None:
                    rows = []
                    for nid in members:
                        parts = str(nid).split("_")
                        if len(parts) < 3:
                            continue
                        layer, idx, ctx = parts[0], parts[1], parts[2]
                        source = f"{layer}-gemmascope-transcoder-16k"
                        detail = _detail_text((cache.get(source) or {}).get(idx) or [])
                        node = by_node.get(nid) or {}
                        rows.append(
                            {
                                "node_id": nid,
                                # Embedding/logit nodes use a letter here (E_…).
                                "layer": int(layer) if layer.isdigit() else -1,
                                "layer_label": layer,
                                "token": _token_label(
                                    tokens, int(ctx) if ctx.isdigit() else -1
                                ),
                                "label": node.get("clerp") or "",
                                "interpretation": detail or node.get("clerp") or "",
                            }
                        )
                    sample = _sample_members(rows, MAX_MEMBERS_IN_REQUEST)
                    entry = {
                        "id": group_id(path.stem, members),
                        "slug": path.stem,
                        "prompt": meta.get("prompt") or "",
                        "size": len(members),
                        "members_shown": len(sample),
                        "sampled": len(sample) < len(rows),
                        "appears_at": [],
                        "members": sample,
                    }
                    seen[key] = entry
                    requests.append(entry)
                entry["appears_at"].append({"k": int(k), "group_index": gi})

        logger.info("%s: %d distinct groups", path.name, len(seen))

    return requests


def write_requests(
    out_path: str | Path,
    graph_dir: str | Path = "graph_files",
    *,
    slugs: list[str] | None = None,
) -> Path:
    """Dump theme requests as JSONL (one group per line)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    requests = build_requests(graph_dir, slugs=slugs)
    with out_path.open("w") as fh:
        for entry in requests:
            fh.write(json.dumps(entry) + "\n")
    logger.info("Wrote %d theme requests → %s", len(requests), out_path)
    return out_path


def apply_themes(
    themes_path: str | Path,
    graph_dir: str | Path = "graph_files",
    *,
    write: bool = True,
) -> list[Path]:
    """Fold {group_id: {theme, summary}} into metadata.spectral_clusters.themes.

    Themes are keyed by member set, then expanded to every (k, group_index) that
    set occupies so the UI can look them up positionally.
    """
    graph_dir = Path(graph_dir)
    themes: dict[str, Any] = json.loads(Path(themes_path).read_text())

    written: list[Path] = []
    for path in sorted(graph_dir.glob("*.json")):
        if path.name == "graph-metadata.json":
            continue
        data = json.loads(path.read_text())
        spec = (data.get("metadata") or {}).get("spectral_clusters")
        if not spec:
            continue

        out: dict[str, list[Any]] = {}
        hits = 0
        misses = 0
        for k in sorted(spec.get("by_k", {}), key=int):
            row: list[Any] = []
            for grp in spec["by_k"][k]:
                entry = themes.get(group_id(path.stem, list(grp[1:])))
                if entry:
                    row.append(
                        {
                            "theme": entry.get("theme") or "",
                            "summary": entry.get("summary") or "",
                        }
                    )
                    hits += 1
                else:
                    row.append(None)
                    misses += 1
            out[k] = row

        if hits:
            spec["themes"] = out
            logger.info("%s: %d group instances themed, %d unthemed", path.name, hits, misses)
            if write:
                path.write_text(json.dumps(data))
                written.append(path)

    return written
