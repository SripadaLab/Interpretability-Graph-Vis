/**
 * Conflict-minimizing k-color edge + node encoding with intensity gradients.
 *
 * Default: greedy 4-color heuristic (teal / coral / ochre / slate).
 * |pctInput| or |weight| maps to saturation + alpha within the chosen hue.
 * Nodes inherit a weighted-majority class from incident edges.
 */
window.edgeColoring = (function () {
  const HUES = [
    [13, 115, 119], // 0 teal
    [196, 69, 54], // 1 coral
    [181, 121, 36], // 2 ochre
    [55, 78, 110], // 3 slate
  ];
  const HUE_LABELS = ['teal', 'coral', 'ochre', 'slate'];
  const HUE_A = HUES[0];
  const HUE_B = HUES[1];
  const BG = [245, 244, 238];

  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  function blend(rgb, intensity) {
    // Floor saturation so mid/weak edges stay chromatic, not canvas-beige.
    const t = 0.32 + 0.68 * Math.max(0, Math.min(1, intensity));
    return [
      Math.round(lerp(BG[0], rgb[0], t)),
      Math.round(lerp(BG[1], rgb[1], t)),
      Math.round(lerp(BG[2], rgb[2], t)),
    ];
  }

  function paletteSize(mode) {
    if (mode === 'greedy4') return 4;
    if (mode === 'greedy2' || mode === 'sign' || mode === 'layer_parity') return 2;
    return 4;
  }

  function rgba(colorClass, intensity, alphaRange, nColors) {
    const k = nColors || 4;
    const base = HUES[colorClass % k];
    const t = Math.max(0, Math.min(1, intensity));
    const [r, g, b] = blend(base, t);
    const ar = alphaRange || [0.28, 0.95];
    const a = lerp(ar[0], ar[1], t);
    return { r, g, b, a, css: `rgba(${r},${g},${b},${a.toFixed(3)})` };
  }

  function nodeFill(colorClass, intensity, nColors) {
    const col = rgba(colorClass, intensity, [0.65, 1], nColors);
    return `rgb(${col.r},${col.g},${col.b})`;
  }

  /** p95-clip + sqrt so GemmaScope weight skew doesn't wash out colors. */
  function intensityScale(values) {
    const abs = values.map((v) => Math.abs(v || 0)).filter((v) => v > 0);
    if (!abs.length) return () => 0.35;
    abs.sort((a, b) => a - b);
    const clip = abs[Math.min(abs.length - 1, Math.floor(0.95 * (abs.length - 1)))] || abs[abs.length - 1];
    return (v) => {
      const t = Math.min(1, Math.abs(v || 0) / (clip || 1));
      return Math.sqrt(t);
    };
  }

  /** Greedy edge k-coloring: least-used incident color, strong edges first. */
  function greedyKColor(links, k) {
    const order = links
      .map((d, i) => ({ i, w: Math.abs(d.pctInput ?? d.weight ?? 0) }))
      .sort((a, b) => b.w - a.w);

    const classes = new Array(links.length).fill(-1);
    const incident = new Map();

    function touch(nodeKey, edgeIdx, color) {
      if (!incident.has(nodeKey)) incident.set(nodeKey, []);
      incident.get(nodeKey).push([edgeIdx, color]);
    }

    for (const { i } of order) {
      const link = links[i];
      const u = link.sourceNode?.nodeId || link.source;
      const v = link.targetNode?.nodeId || link.target;
      const counts = new Array(k).fill(0);
      for (const list of [incident.get(u) || [], incident.get(v) || []]) {
        for (const [, c] of list) {
          if (c >= 0 && c < k) counts[c]++;
        }
      }
      const w = link.pctInput ?? link.weight ?? 0;
      let minCount = Infinity;
      for (let c = 0; c < k; c++) if (counts[c] < minCount) minCount = counts[c];
      const candidates = [];
      for (let c = 0; c < k; c++) if (counts[c] === minCount) candidates.push(c);
      const choice = w >= 0 ? candidates[0] : candidates[candidates.length - 1];

      classes[i] = choice;
      touch(u, i, choice);
      touch(v, i, choice);
    }
    return classes;
  }

  function layerParity(links) {
    return links.map((d, i) => {
      const layer = d.sourceNode?.layer;
      if (layer === 'E' || layer === 'e') return 0;
      const n = parseInt(layer, 10);
      return Number.isFinite(n) ? n % 2 : i % 2;
    });
  }

  function signColor(links) {
    return links.map((d) => ((d.pctInput ?? d.weight ?? 0) >= 0 ? 0 : 1));
  }

  function intensityOf(link, maxAbs) {
    const v = Math.abs(link.pctInput ?? link.weight ?? 0);
    if (!maxAbs) return 0;
    return Math.min(1, v / maxAbs);
  }

  function linkWeight(d) {
    return Math.abs(d.pctInput ?? d.weight ?? 0);
  }

  function applyNodes(nodes, links, mode) {
    if (!nodes?.length) return;

    if (mode === 'prgn') {
      nodes.forEach((n) => {
        n.colorClass = 0;
        n.intensity = 0;
        n.nodeColor = '#fff';
      });
      return;
    }

    const k = paletteSize(mode);
    const scores = new Map(); // nodeId -> Float64Array(k+1) last = maxW
    for (const link of links || []) {
      const c = link.colorClass;
      if (!(c >= 0 && c < k)) continue;
      const w = linkWeight(link);
      const ends = [
        link.sourceNode?.nodeId || link.source,
        link.targetNode?.nodeId || link.target,
      ];
      for (const id of ends) {
        if (id == null) continue;
        const key = String(id);
        if (!scores.has(key)) scores.set(key, new Float64Array(k + 1));
        const s = scores.get(key);
        s[c] += w;
        if (w > s[k]) s[k] = w;
      }
    }

    const toInfl = intensityScale(nodes.map((n) => n.influence ?? n.activation ?? 0));
    const toEdge = intensityScale([...scores.values()].map((s) => s[k]));

    nodes.forEach((n, i) => {
      const key = String(n.nodeId ?? n.node_id ?? '');
      const s = scores.get(key);
      let colorClass;
      if (mode === 'layer_parity') {
        const layer = n.layer;
        if (layer === 'E' || layer === 'e') colorClass = 0;
        else {
          const ln = parseInt(layer, 10);
          colorClass = Number.isFinite(ln) ? ln % 2 : i % 2;
        }
      } else if (s) {
        let best = 0;
        for (let c = 1; c < k; c++) if (s[c] > s[best]) best = c;
        colorClass = best;
      } else {
        colorClass = i % k;
      }

      const infl = Math.abs(n.influence ?? 0);
      const intensity = infl > 0 ? toInfl(infl) : s ? toEdge(s[k]) : 0.45;

      n.colorClass = colorClass;
      n.intensity = intensity;
      n.nodeColor = nodeFill(colorClass, intensity, k);
    });
  }

  /**
   * Color a list of subgraph aggregate links using the same heuristic.
   * Mutates each link's color / colorClass / intensity.
   */
  function applySubgraphLinks(sgLinks, mode) {
    if (!sgLinks?.length) return;
    if (mode === 'prgn') {
      sgLinks.forEach((d) => {
        d.color = utilCg.pctInputColorFn(d.weight ?? d.pctInput ?? 0);
        d.colorClass = (d.weight ?? 0) >= 0 ? 0 : 1;
      });
      return;
    }
    const toI = intensityScale(sgLinks.map((d) => d.weight ?? d.pctInput ?? 0));
    const k = paletteSize(mode);
    const classes = greedyKColor(sgLinks, k);
    sgLinks.forEach((d, i) => {
      const intensity = toI(d.weight ?? d.pctInput ?? 0);
      const c = classes[i];
      const col = rgba(c, intensity, [0.4, 0.98], k);
      d.colorClass = c;
      d.intensity = intensity;
      d.color = col.css;
      d.pctInputColor = d.color;
    });
  }

  /**
   * Stroke-width intensity.
   *   - 'relative' (default): √-compressed |pctInput|, normalized against the
   *     currently-visible edges (p95). Widths rescale as you prune.
   *   - 'stable': √-compressed |weight|, normalized against a fixed full-graph
   *     p95 (globalWeightP95). An edge keeps the same width regardless of how
   *     many nodes are pruned.
   */
  function widthIntensityFn(links, opts) {
    const mode = opts?.thicknessMode || 'relative';
    if (mode === 'stable') {
      const clip = opts?.globalWeightP95 || 1;
      return (d) => Math.sqrt(Math.min(1, Math.abs(d.weight ?? d.pctInput ?? 0) / (clip || 1)));
    }
    const toI = intensityScale(links.map((d) => d.pctInput ?? d.weight ?? 0));
    return (d) => toI(d.pctInput ?? d.weight ?? 0);
  }

  /**
   * @param {object[]} links
   * @param {'greedy4'|'greedy2'|'layer_parity'|'sign'|'prgn'} mode
   * @param {object[]} [nodes]
   * @param {{thicknessMode?: 'relative'|'stable', globalWeightP95?: number}} [opts]
   */
  function apply(links, mode, nodes, opts) {
    if (!links?.length) {
      if (nodes) applyNodes(nodes, links, mode);
      return;
    }

    const widthScale = d3.scaleSqrt().domain([0, 1]).range([0.5, 3.6]);
    const widthIntensity = widthIntensityFn(links, opts);

    if (mode === 'prgn') {
      const pctScale = d3.scaleLinear().domain([-0.4, 0.4]);
      const tScale = d3.scaleLinear().domain([0, 0.5, 0.5, 1]).range([0, 0.499, 0.501, 1]);
      links.forEach((d) => {
        const pct = d.pctInput ?? 0;
        d.colorClass = pct >= 0 ? 0 : 1;
        d.intensity = Math.min(1, Math.abs(pct) / 0.4);
        d.strokeWidth = widthScale(Math.max(0.02, widthIntensity(d)));
        d.pctInputColor = d3.interpolatePRGn(tScale(pctScale(pct)));
        d.color = d.pctInputColor;
        d.twoColorCss = d.color;
      });
      applyNodes(nodes, links, mode);
      return;
    }

    const k = paletteSize(mode);
    const classes =
      mode === 'layer_parity'
        ? layerParity(links)
        : mode === 'sign'
          ? signColor(links)
          : mode === 'greedy2'
            ? greedyKColor(links, 2)
            : greedyKColor(links, 4);

    // Color intensity stays relative to visible edges; only width can be stable.
    const toI = intensityScale(links.map((d) => d.pctInput ?? d.weight ?? 0));

    links.forEach((d, i) => {
      const intensity = toI(d.pctInput ?? d.weight ?? 0);
      const c = classes[i];
      const col = rgba(c, intensity, null, k);
      d.colorClass = c;
      d.intensity = intensity;
      d.strokeWidth = widthScale(Math.max(0.15, widthIntensity(d)));
      d.twoColorCss = col.css;
      d.color = col.css;
      d.pctInputColor = d3.interpolatePRGn(
        (d.pctInput ?? 0) >= 0 ? 0.75 + 0.25 * intensity : 0.25 - 0.25 * intensity
      );
    });

    applyNodes(nodes, links, mode);
  }

  /** Pin a readable subgraph: walk strongest edges upstream from logits / click. */
  function fillSubgraph(visState, renderAll, data, opts) {
    const maxNodes = opts?.maxNodes || 28;
    const maxDepth = opts?.maxDepth || 5;
    const nodes = data.nodes;
    const links = data.links;

    const incoming = new Map();
    links.forEach((l) => {
      const tgt = l.targetNode?.nodeId || l.target;
      const src = l.sourceNode?.nodeId || l.source;
      const w = Math.abs(l.pctInput ?? l.weight ?? 0);
      if (!incoming.has(tgt)) incoming.set(tgt, []);
      incoming.get(tgt).push({ src, w });
    });
    incoming.forEach((arr) => arr.sort((a, b) => b.w - a.w));

    let seeds = [];
    if (visState.clickedId && data.nodes.idToNode?.[visState.clickedId]) {
      seeds = [visState.clickedId];
    } else {
      const logits = nodes.filter(
        (n) => n.feature_type === 'logit' || n.is_target_logit
      );
      seeds = d3
        .sort(logits, (n) => [
          n.is_target_logit ? 0 : 1,
          -(n.token_prob ?? n.influence ?? 0),
        ])
        .slice(0, 3)
        .map((n) => n.nodeId);
      if (!seeds.length) {
        seeds = d3
          .sort(nodes, (n) => -(Math.abs(n.influence ?? 0)))
          .slice(0, 3)
          .map((n) => n.nodeId);
      }
    }

    const pinned = [];
    const seen = new Set();
    const frontier = seeds.map((id) => ({ id, depth: 0 }));
    while (frontier.length && pinned.length < maxNodes) {
      const { id, depth } = frontier.shift();
      if (!id || seen.has(id)) continue;
      seen.add(id);
      pinned.push(id);
      if (depth >= maxDepth) continue;
      for (const { src } of (incoming.get(id) || []).slice(0, 6)) {
        if (!seen.has(src)) frontier.push({ id: src, depth: depth + 1 });
      }
    }

    visState.pinnedIds = pinned;
    util.params.set('pinnedIds', pinned.join(','));
    renderAll.pinnedIds();
    return pinned;
  }

  return {
    apply,
    applyNodes,
    applySubgraphLinks,
    fillSubgraph,
    rgba,
    nodeFill,
    paletteSize,
    HUES,
    HUE_LABELS,
    HUE_A,
    HUE_B,
  };
})();
