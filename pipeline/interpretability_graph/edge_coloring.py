"""Conflict-minimizing k-coloring of attribution-graph edges with intensity.

Attribution graphs are dense DAGs. A proper edge chromatic coloring can need
Δ+1 colors (Vizing). We solve practical greedy k-coloring relaxations:

  1. Sort edges by descending |weight| so strong attributions get priority.
  2. At each vertex, assign the palette color least used among already-colored
     incident edges (Misra–Gries / greedy edge coloring restricted to k colors).
  3. Map |weight| into a lightness/alpha gradient within the chosen hue so
     magnitude stays visible inside the hairball.

Default palette (k=4):
  0 teal, 1 coral, 2 ochre, 3 slate
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence


ColorMode = Literal["greedy4", "greedy2", "layer_parity", "sign"]


# Distinct under dense overlap; avoid purple-on-white defaults.
HUES: tuple[tuple[int, int, int], ...] = (
    (13, 115, 119),   # 0 teal  #0D7377
    (196, 69, 54),    # 1 coral #C44536
    (181, 121, 36),   # 2 ochre #B57924
    (55, 78, 110),    # 3 slate #374E6E
)
HUE_A = HUES[0]
HUE_B = HUES[1]
BG = (245, 244, 238)  # canvas #F5F4EE

HUE_LABELS = ("teal", "coral", "ochre", "slate")


@dataclass(frozen=True)
class EdgeColorAssignment:
    edge_id: str
    color_class: int
    intensity: float  # [0, 1]
    rgba: tuple[int, int, int, float]
    weight: float


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _blend_toward_bg(
    rgb: tuple[int, int, int], intensity: float, bg: tuple[int, int, int] = BG
) -> tuple[int, int, int]:
    """High intensity → saturated hue; low intensity → washed toward canvas."""
    t = max(0.0, min(1.0, intensity))
    # Floor so mid/weak edges stay chromatic on real (skewed) graphs.
    t = 0.32 + 0.68 * t
    return (
        int(_lerp(bg[0], rgb[0], t)),
        int(_lerp(bg[1], rgb[1], t)),
        int(_lerp(bg[2], rgb[2], t)),
    )


def _normalize_weights(weights: Sequence[float]) -> list[float]:
    """Map |weight| → [0,1] with p95 clip + sqrt so real graphs stay vivid.

    Linear /max makes almost every edge wash out when one edge dominates
    (typical of GemmaScope attribution graphs).
    """
    abs_w = [abs(float(w)) for w in weights]
    if not abs_w:
        return []
    sorted_w = sorted(abs_w)
    # 95th percentile clip (or max for tiny graphs)
    idx = min(len(sorted_w) - 1, max(0, int(0.95 * (len(sorted_w) - 1))))
    clip = sorted_w[idx] or max(sorted_w) or 1.0
    out = []
    for w in abs_w:
        t = min(1.0, w / clip)
        # sqrt lifts mid-tier edges into a visible saturation band
        out.append(t**0.5)
    return out


def color_rgba(
    color_class: int,
    intensity: float,
    *,
    alpha_range: tuple[float, float] = (0.28, 0.95),
    n_colors: int = 4,
) -> tuple[int, int, int, float]:
    """Map (class, intensity) → RGBA. Intensity also drives alpha."""
    base = HUES[int(color_class) % max(1, min(n_colors, len(HUES)))]
    t = max(0.0, min(1.0, intensity))
    r, g, b = _blend_toward_bg(base, t)
    a = _lerp(alpha_range[0], alpha_range[1], t)
    return (r, g, b, a)


def _palette_size(mode: ColorMode) -> int:
    if mode == "greedy4":
        return 4
    if mode in ("greedy2", "sign", "layer_parity"):
        return 2
    return 4


def assign_edge_2colors(
    edges: Iterable[dict],
    *,
    mode: ColorMode = "greedy4",
    id_key: str = "id",
    source_key: str = "source",
    target_key: str = "target",
    weight_key: str = "weight",
    layer_key: str = "source_layer",
) -> list[EdgeColorAssignment]:
    """Assign k-colors + intensity to edges (name kept for API compatibility)."""
    edge_list = list(edges)
    if not edge_list:
        return []

    intensities = _normalize_weights([float(e.get(weight_key, 0.0)) for e in edge_list])
    n_colors = _palette_size(mode)

    if mode == "layer_parity":
        classes = [
            int(e.get(layer_key, 0)) % 2 if e.get(layer_key) is not None else i % 2
            for i, e in enumerate(edge_list)
        ]
    elif mode == "sign":
        classes = [0 if float(e.get(weight_key, 0.0)) >= 0 else 1 for e in edge_list]
    elif mode == "greedy2":
        classes = _greedy_kcolor(edge_list, source_key, target_key, weight_key, k=2)
    else:
        classes = _greedy_kcolor(edge_list, source_key, target_key, weight_key, k=4)

    out: list[EdgeColorAssignment] = []
    for i, e in enumerate(edge_list):
        eid = str(e.get(id_key) or e.get("linkId") or f"{e[source_key]}__{e[target_key]}")
        w = float(e.get(weight_key, 0.0))
        c = classes[i]
        intensity = intensities[i]
        out.append(
            EdgeColorAssignment(
                edge_id=eid,
                color_class=c,
                intensity=intensity,
                rgba=color_rgba(c, intensity, n_colors=n_colors),
                weight=w,
            )
        )
    return out


def _greedy_kcolor(
    edges: list[dict],
    source_key: str,
    target_key: str,
    weight_key: str,
    *,
    k: int = 4,
) -> list[int]:
    """Greedy edge k-coloring minimizing same-color conflicts at vertices.

    Processes edges by descending |weight|. For each edge, counts how many
    already-colored incident edges use each palette slot, then picks the
    least-used color (ties → prefer low index for positive weight, high for
    negative — keeps a weak polarity cue).
    """
    k = max(2, min(k, len(HUES)))
    order = sorted(
        range(len(edges)),
        key=lambda i: abs(float(edges[i].get(weight_key, 0.0))),
        reverse=True,
    )
    classes = [-1] * len(edges)
    incident: dict[str, list[tuple[int, int]]] = {}

    for idx in order:
        e = edges[idx]
        u, v = str(e[source_key]), str(e[target_key])
        counts = [0] * k
        for nbr_idx, c in incident.get(u, []) + incident.get(v, []):
            if 0 <= c < k:
                counts[c] += 1

        weight = float(e.get(weight_key, 0.0))
        # Prefer unused / least-used colors; on ties prefer ends of palette by sign.
        min_count = min(counts)
        candidates = [c for c, n in enumerate(counts) if n == min_count]
        if weight >= 0:
            choice = candidates[0]
        else:
            choice = candidates[-1]

        classes[idx] = choice
        incident.setdefault(u, []).append((idx, choice))
        incident.setdefault(v, []).append((idx, choice))

    return classes


def annotate_graph_nodes(
    nodes: list[dict],
    links: list[dict],
    *,
    mode: ColorMode = "greedy4",
) -> list[dict]:
    """Color nodes with the same k-palette as edges.

    Class = |weight|-weighted majority of incident edge color_class values.
    Intensity = normalized |influence| (fallback: strongest incident |weight|).
    """
    if not nodes:
        return []

    if mode == "prgn":  # type: ignore[comparison-overlap]
        out = []
        for n in nodes:
            item = dict(n)
            item["color_class"] = 0
            item["intensity"] = 0.0
            item["nodeColor"] = "#ffffff"
            out.append(item)
        return out

    n_colors = _palette_size(mode)
    scores: dict[str, list[float]] = {}  # id -> [w0..wk-1, max_w]
    for link in links:
        c = link.get("color_class")
        if not isinstance(c, int) or not (0 <= c < n_colors):
            continue
        w = abs(float(link.get("weight", 0.0) or link.get("pctInput", 0.0) or 0.0))
        for end in (link.get("source"), link.get("target")):
            if end is None:
                continue
            key = str(end)
            if key not in scores:
                scores[key] = [0.0] * n_colors + [0.0]
            s = scores[key]
            s[c] += w
            if w > s[-1]:
                s[-1] = w

    max_infl = max((abs(float(n.get("influence") or 0.0)) for n in nodes), default=1.0) or 1.0
    max_edge = max((s[-1] for s in scores.values()), default=1.0) or 1.0

    out = []
    for i, n in enumerate(nodes):
        item = dict(n)
        nid = str(n.get("node_id") or n.get("nodeId") or "")
        s = scores.get(nid)

        if mode == "layer_parity":
            layer = n.get("layer", 0)
            if layer in ("E", "e"):
                color_class = 0
            else:
                try:
                    color_class = int(layer) % 2
                except (TypeError, ValueError):
                    color_class = i % 2
        elif s is not None:
            color_class = max(range(n_colors), key=lambda c: s[c])
        else:
            color_class = i % n_colors

        infl = abs(float(n.get("influence") or 0.0))
        if infl > 0:
            intensity = min(1.0, infl / max_infl)
        elif s is not None:
            intensity = min(1.0, s[-1] / max_edge)
        else:
            intensity = 0.35

        r, g, b, _ = color_rgba(color_class, intensity, alpha_range=(0.55, 1.0), n_colors=n_colors)
        item["color_class"] = color_class
        item["intensity"] = intensity
        item["nodeColor"] = f"rgb({r},{g},{b})"
        item["rgba"] = [r, g, b, 1.0]
        out.append(item)
    return out


def annotate_graph_links(
    links: list[dict],
    nodes: list[dict] | None = None,
    *,
    mode: ColorMode = "greedy4",
) -> list[dict]:
    """Return links with color_class, intensity, and rgba fields attached."""
    node_layer: dict[str, int] = {}
    if nodes:
        for n in nodes:
            nid = n.get("node_id") or n.get("nodeId")
            if nid is None:
                continue
            layer = n.get("layer", 0)
            try:
                node_layer[str(nid)] = int(layer) if layer not in ("E", "e") else -1
            except (TypeError, ValueError):
                node_layer[str(nid)] = 0

    enriched = []
    for link in links:
        item = dict(link)
        src = item.get("source")
        if "source_layer" not in item and src is not None:
            item["source_layer"] = node_layer.get(str(src), 0)
        if "id" not in item and "linkId" not in item:
            item["id"] = f"{item.get('source')}__{item.get('target')}"
        enriched.append(item)

    assignments = assign_edge_2colors(enriched, mode=mode)
    by_id = {a.edge_id: a for a in assignments}
    n_colors = _palette_size(mode)

    result = []
    for item in enriched:
        eid = str(item.get("id") or item.get("linkId"))
        a = by_id[eid]
        out = dict(item)
        out["color_class"] = a.color_class
        out["intensity"] = a.intensity
        out["rgba"] = list(a.rgba)
        r, g, b, alpha = a.rgba
        out["color"] = f"rgba({r},{g},{b},{alpha:.3f})"
        out["n_colors"] = n_colors
        result.append(out)
    return result


def annotate_graph(
    links: list[dict],
    nodes: list[dict],
    *,
    mode: ColorMode = "greedy4",
) -> tuple[list[dict], list[dict]]:
    """Annotate links then nodes so both share the k-color palette."""
    colored_links = annotate_graph_links(links, nodes, mode=mode)
    colored_nodes = annotate_graph_nodes(nodes, colored_links, mode=mode)
    return colored_links, colored_nodes


def seed_subgraph_from_logits(
    nodes: list[dict],
    links: list[dict],
    *,
    max_nodes: int = 28,
    max_depth: int = 5,
) -> tuple[list[str], list[list[str]]]:
    """Grow a readable subgraph by walking strongest edges upstream from logits.

    Returns (pinned_ids, supernodes) where supernodes group by shared clerp prefix
    when multiple features share a label family.
    """
    id_key = lambda n: str(n.get("node_id") or n.get("nodeId") or "")
    by_id = {id_key(n): n for n in nodes if id_key(n)}

    # Incoming adjacency: target -> [(source, |w|, w)]
    incoming: dict[str, list[tuple[str, float, float]]] = {}
    for link in links:
        src, tgt = str(link.get("source")), str(link.get("target"))
        w = float(link.get("weight", 0.0) or 0.0)
        incoming.setdefault(tgt, []).append((src, abs(w), w))
    for tgt in incoming:
        incoming[tgt].sort(key=lambda t: t[1], reverse=True)

    logits = [
        n
        for n in nodes
        if (
            n.get("feature_type") == "logit"
            or n.get("is_target_logit")
            or (
                str(n.get("layer", "")).isdigit()
                and int(n["layer"]) > 20
                and "logit" in str(n.get("feature_type", "")).lower()
            )
        )
    ]
    if not logits:
        logits = sorted(nodes, key=lambda n: -abs(float(n.get("influence") or 0)))[:3]

    seeds = sorted(
        logits,
        key=lambda n: (
            0 if n.get("is_target_logit") else 1,
            -float(n.get("token_prob") or n.get("influence") or 0),
        ),
    )[:3]

    pinned: list[str] = []
    seen: set[str] = set()
    frontier: list[tuple[str, int]] = [(id_key(n), 0) for n in seeds if id_key(n)]

    while frontier and len(pinned) < max_nodes:
        nid, depth = frontier.pop(0)
        if nid in seen or nid not in by_id:
            continue
        seen.add(nid)
        pinned.append(nid)
        if depth >= max_depth:
            continue
        for src, _aw, _w in incoming.get(nid, [])[:6]:
            if src not in seen:
                frontier.append((src, depth + 1))

    # Group feature nodes that share a non-empty clerp into supernodes (cap 6 groups)
    groups: dict[str, list[str]] = {}
    for nid in pinned:
        n = by_id[nid]
        clerp = (n.get("clerp") or n.get("ppClerp") or "").strip()
        if not clerp or n.get("feature_type") in ("embedding", "logit", "mlp reconstruction error"):
            continue
        # Use first 2 tokens / bracket label as group key
        label = clerp.split("]")[0].strip("[") if clerp.startswith("[") else clerp.split()[0]
        if len(label) < 2:
            continue
        groups.setdefault(label[:32], []).append(nid)

    supernodes: list[list[str]] = []
    for label, ids in list(groups.items())[:6]:
        if len(ids) >= 2:
            supernodes.append([label, *ids])

    return pinned, supernodes


def conflict_rate(assignments: Sequence[EdgeColorAssignment], edges: Sequence[dict]) -> float:
    """Fraction of vertex-incident edge pairs that share a color (debug metric)."""
    by_id = {a.edge_id: a for a in assignments}
    incident: dict[str, list[int]] = {}
    for e in edges:
        eid = str(e.get("id") or e.get("linkId") or f"{e['source']}__{e['target']}")
        a = by_id.get(eid)
        if a is None:
            continue
        for node in (str(e["source"]), str(e["target"])):
            incident.setdefault(node, []).append(a.color_class)

    conflicts = pairs = 0
    for colors in incident.values():
        n = len(colors)
        for i in range(n):
            for j in range(i + 1, n):
                pairs += 1
                if colors[i] == colors[j]:
                    conflicts += 1
    return conflicts / pairs if pairs else 0.0
