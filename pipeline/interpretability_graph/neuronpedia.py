"""Semantic interpretation assignment from Neuronpedia.

Step 1 of group theme assignment. A node_id is <layer>_<feature_idx>_<ctx_idx>,
and the middle field indexes the gemmascope transcoder for that layer, so each
feature node maps onto exactly one Neuronpedia feature:

    node 3_7251_6  →  gemma-2-2b / 3-gemmascope-transcoder-16k / 7251

We fetch that feature's explanations, pick one, and write it onto the node as
`clerp` (the field the UI already reads). Every explanation candidate is kept in
the on-disk cache so the theme step can switch preference without refetching.

Explanation preference follows gonogo_task/neuronpedia_api.py: np_max-act, then
oai_token-act-pair, then whatever else the feature has.
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

API_BASE = "https://www.neuronpedia.org/api/feature"
DEFAULT_MODEL_ID = "gemma-2-2b"
DEFAULT_SOURCE_SUFFIX = "gemmascope-transcoder-16k"
EXPLANATION_TYPE_PRIORITY = {
    "np_max-act": 0,
    "oai_token-act-pair": 1,
}
_EXPLANATION_TEXT_KEYS = ("description", "text", "explanation", "value", "title")


def build_source_id(layer: int | str, suffix: str = DEFAULT_SOURCE_SUFFIX) -> str:
    """Convert a layer number into the Neuronpedia source id."""
    return f"{layer}-{suffix}"


def _explanation_text(explanation: dict[str, Any]) -> str | None:
    for key in _EXPLANATION_TEXT_KEYS:
        value = explanation.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _normalize_explanations(feature: dict[str, Any]) -> list[dict[str, Any]]:
    """Reduce the ~100KB feature payload to the explanation candidates."""
    raw = feature.get("explanations") or []
    if not isinstance(raw, list):
        raise TypeError("Expected 'explanations' to be a list in the feature JSON.")
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = _explanation_text(item)
        if not text:
            continue
        out.append(
            {
                "type": item.get("typeName") or "",
                "author": item.get("explanationModelName") or "",
                "text": text,
            }
        )
    return out


def preferred_explanation(explanations: list[dict[str, Any]]) -> str | None:
    """np_max-act, then oai_token-act-pair, then first available."""
    if not explanations:
        return None
    ranked = sorted(
        explanations,
        key=lambda e: EXPLANATION_TYPE_PRIORITY.get(e.get("type", ""), 99),
    )
    return ranked[0].get("text") or None


def fetch_feature(
    layer: int | str,
    index: int | str,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    suffix: str = DEFAULT_SOURCE_SUFFIX,
    attempts: int = 4,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    """Fetch one feature's explanation candidates, retrying on rate limits."""
    url = "/".join(
        [
            API_BASE,
            quote(str(model_id), safe=""),
            quote(build_source_id(layer, suffix), safe=""),
            quote(str(index), safe=""),
        ]
    )
    headers = {"Accept": "application/json"}
    # The public endpoint answers without a key; send one when available.
    api_key = os.getenv("NEURONPEDIA_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urlopen(Request(url, headers=headers), timeout=timeout) as response:
                return _normalize_explanations(
                    json.loads(response.read().decode("utf-8"))
                )
        except HTTPError as err:
            last_error = err
            if err.code == 404:
                return []
            if err.code not in (429, 500, 502, 503, 504):
                raise
        except (URLError, TimeoutError, json.JSONDecodeError) as err:
            last_error = err
        time.sleep(2.0 * (attempt + 1))

    raise RuntimeError(f"Neuronpedia fetch failed for {url}: {last_error}")


class ExplanationCache:
    """Per-source explanation cache: graph_files/interpretations/{model}/{source}.json"""

    def __init__(self, root: str | Path, model_id: str = DEFAULT_MODEL_ID) -> None:
        self.dir = Path(root) / "interpretations" / model_id
        self.model_id = model_id
        self._sources: dict[str, dict[str, Any]] = {}
        self._dirty: set[str] = set()

    def _load(self, source: str) -> dict[str, Any]:
        if source not in self._sources:
            path = self.dir / f"{source}.json"
            if path.exists():
                self._sources[source] = json.loads(path.read_text())
            else:
                self._sources[source] = {}
        return self._sources[source]

    def get(self, layer: int | str, index: int | str) -> list[dict[str, Any]] | None:
        return self._load(build_source_id(layer)).get(str(index))

    def put(
        self, layer: int | str, index: int | str, explanations: list[dict[str, Any]]
    ) -> None:
        source = build_source_id(layer)
        self._load(source)[str(index)] = explanations
        self._dirty.add(source)

    def flush(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        for source in sorted(self._dirty):
            path = self.dir / f"{source}.json"
            payload = dict(sorted(self._sources[source].items(), key=lambda kv: int(kv[0])))
            path.write_text(json.dumps(payload, indent=1, sort_keys=False))
        self._dirty.clear()


def _feature_nodes(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        n
        for n in data.get("nodes", [])
        if n.get("feature_type") == "cross layer transcoder"
    ]


def _node_key(node: dict[str, Any]) -> tuple[int, int] | None:
    """(layer, feature_idx) parsed from node_id, which is authoritative."""
    parts = str(node.get("node_id", "")).split("_")
    if len(parts) < 3:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def assign_interpretations(
    graph_dir: str | Path = "graph_files",
    *,
    slugs: list[str] | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    workers: int = 8,
    refresh: bool = False,
    write: bool = True,
) -> list[Path]:
    """Attach a Neuronpedia explanation to every feature node's `clerp`."""
    graph_dir = Path(graph_dir)
    if slugs:
        paths = []
        for slug in slugs:
            path = graph_dir / f"{slug}.json"
            if not path.exists():
                raise FileNotFoundError(f"No graph matching {slug!r}")
            paths.append(path)
    else:
        paths = sorted(
            p for p in graph_dir.glob("*.json") if p.name != "graph-metadata.json"
        )

    cache = ExplanationCache(graph_dir, model_id)
    written: list[Path] = []

    for path in paths:
        data = json.loads(path.read_text())
        scan = (data.get("metadata") or {}).get("scan")
        if scan and scan != model_id:
            logger.info("Skipping %s (scan=%s, not %s)", path.name, scan, model_id)
            continue

        nodes = _feature_nodes(data)
        keys = {k for k in (_node_key(n) for n in nodes) if k is not None}
        todo = sorted(
            k for k in keys if refresh or cache.get(*k) is None
        )
        logger.info(
            "%s: %d feature nodes, %d unique features, %d to fetch",
            path.name,
            len(nodes),
            len(keys),
            len(todo),
        )

        if todo:
            done = 0
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(fetch_feature, layer, index, model_id=model_id): (
                        layer,
                        index,
                    )
                    for layer, index in todo
                }
                for future in futures:
                    layer, index = futures[future]
                    try:
                        cache.put(layer, index, future.result())
                    except Exception as err:  # keep going; report coverage below
                        logger.warning("  %s/%s failed: %s", layer, index, err)
                    done += 1
                    if done % 100 == 0:
                        logger.info("  fetched %d/%d", done, len(todo))
            cache.flush()

        hits = 0
        for node in nodes:
            key = _node_key(node)
            text = preferred_explanation(cache.get(*key) or []) if key else None
            if text:
                node["clerp"] = text
                hits += 1

        meta = data.setdefault("metadata", {})
        meta["interpretations"] = {
            "source": "neuronpedia",
            "model_id": model_id,
            "sae_suffix": DEFAULT_SOURCE_SUFFIX,
            "n_feature_nodes": len(nodes),
            "n_unique_features": len(keys),
            "n_labeled": hits,
        }
        logger.info("  labeled %d/%d feature nodes", hits, len(nodes))

        if write:
            path.write_text(json.dumps(data))
            written.append(path)

    return written
