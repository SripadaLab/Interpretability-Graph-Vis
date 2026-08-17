/**
 * Left sidebar: graph stats + visual legend (color classes + line thickness).
 *
 * Line thickness = sqrt-scaled |attribution| (pctInput / weight).
 * Thicker / more opaque strokes = stronger contribution.
 */
window.initCgGraphStats = function ({visState, renderAll, data, cgSel}) {
  var sel = cgSel.select('.graph-stats')
  if (sel.empty()) return

  function sectionTitle(text) {
    return sel.append('div')
      .text(text)
      .st({
        fontFamily: 'system-ui, sans-serif',
        fontWeight: 600,
        fontSize: '12px',
        margin: '14px 0 8px',
        letterSpacing: '0.02em',
      })
  }

  function render() {
    var nodes = data.nodes || []
    var links = data.links || []
    var md = data.metadata || {}
    var thr = parseFloat(visState.pruningThreshold)
    var byType = d3.rollup(
      nodes,
      (v) => v.length,
      (d) => d.feature_type || 'other'
    )
    var nFeat = nodes.filter((d) =>
      (d.feature_type || '').includes('transcoder') || d.feature_type === 'cross layer transcoder'
    ).length
    var nEmb = nodes.filter((d) => d.feature_type === 'embedding').length
    var nErr = nodes.filter((d) => (d.feature_type || '').includes('error')).length
    var nLogit = nodes.filter((d) => d.feature_type === 'logit').length
    var coloring = md.edge_coloring || {}
    var rawNodes = data._rawNodeCount || nodes.length
    var rawLinks = data._rawLinkCount || links.length
    var supernodes = (visState.subgraph && visState.subgraph.supernodes) ||
      visState.supernodes ||
      (data.qParams && data.qParams.supernodes) ||
      []

    var rows = [
      ['Prompt', (md.prompt || '').slice(0, 72) + ((md.prompt || '').length > 72 ? '…' : '')],
      ['Scan', md.scan || '—'],
      ['', ''],
      ['Nodes (visible)', String(nodes.length)],
      ['Nodes (loaded)', String(rawNodes)],
      ['Links (visible)', String(links.length)],
      ['Links (loaded)', String(rawLinks)],
      ['', ''],
      ['Features (CLT)', String(nFeat)],
      ['Embeddings', String(nEmb)],
      ['Errors', String(nErr)],
      ['Logits', String(nLogit)],
      ['Clusters', String(supernodes.length)],
      ['', ''],
      ['Pruning', Number.isFinite(thr) ? (thr >= 0.995 ? '1.00 (full)' : thr.toFixed(2)) : '—'],
      ['Edge color', coloring.mode || visState.edgeColorMode || '—'],
      ['Thickness', (visState.thicknessMode === 'stable' ? 'stable (fixed)' : 'relative (pruned)')],
      ['Conflict rate', coloring.conflict_rate != null ? String(coloring.conflict_rate) : '—'],
    ]

    sel.html('').st({
      fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
      fontSize: '11px',
      lineHeight: '1.45',
      color: '#333',
      padding: '12px 14px',
      boxSizing: 'border-box',
      overflowY: 'auto',
      height: '100%',
      background: '#F7F6F1',
      borderRight: '1px solid #E4E2D8',
    })

    sel.append('div')
      .text('Legend & stats')
      .st({
        fontFamily: 'system-ui, sans-serif',
        fontWeight: 600,
        fontSize: '13px',
        marginBottom: '4px',
        letterSpacing: '0.02em',
      })

    function latexBlock(tex) {
      var box = sel.append('div').st({
        background: '#fff',
        border: '1px solid #E4E2D8',
        borderRadius: '4px',
        padding: '10px 8px',
        marginBottom: '8px',
        overflowX: 'auto',
        textAlign: 'center',
        color: '#2a2a2a',
      })
      if (window.katex) {
        try {
          box.html(katex.renderToString(tex, {
            displayMode: true,
            throwOnError: false,
            strict: 'ignore',
          }))
        } catch (e) {
          box.text(tex)
        }
      } else {
        box.st({
          fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
          fontSize: '10px',
          whiteSpace: 'pre-wrap',
          textAlign: 'left',
        }).text(tex)
      }
      // Keep KaTeX from overflowing the narrow sidebar
      box.selectAll('.katex-display').st({margin: '0.35em 0', fontSize: '0.92em'})
      return box
    }

    // ——— Attribution ———
    sectionTitle('Attribution')
    sel.append('div')
      .text('Each edge weight w is the attribution from source → target. For a node, pctInput is that edge’s share of all incoming |w|:')
      .st({
        fontFamily: 'system-ui, sans-serif',
        fontSize: '10px',
        color: '#666',
        marginBottom: '6px',
        lineHeight: '1.35',
      })
    latexBlock(String.raw`
      \mathrm{pctInput}(e)
        = \frac{w(e)}{\displaystyle\sum_{\substack{e'\\e'\to\mathrm{target}(e)}} |w(e')|}
    `)

    // ——— Line thickness legend ———
    sectionTitle('Line thickness')
    sel.append('div')
      .text('Stroke width follows intensity (√-compressed |pctInput|). Thicker = stronger contribution to the target.')
      .st({
        fontFamily: 'system-ui, sans-serif',
        fontSize: '10px',
        color: '#666',
        marginBottom: '6px',
        lineHeight: '1.35',
      })
    latexBlock(String.raw`
      \begin{aligned}
        v &= |\mathrm{pctInput}|\ \text{(else }|w|\text{)} \\[0.35em]
        \mathrm{intensity}
          &= \sqrt{\min\!\left(1,\ \frac{v}{p_{95}}\right)} \\[0.35em]
        \mathrm{strokeWidth}
          &\in [0.5,\ 3.6]\ \mathrm{px}
            \ \text{via }\sqrt{\text{-scale}}(\mathrm{intensity})
      \end{aligned}
    `)
    sel.append('div')
      .html(visState.thicknessMode === 'stable'
        ? '<b>Stable</b> mode: v = |w| and p₉₅ is the fixed full-graph value, so width does not change when you prune.'
        : '<b>Relative</b> mode: p₉₅ is taken over the currently-visible edges, so widths rescale as you prune. Toggle “Thickness” in the toolbar for a fixed scale.')
      .st({fontFamily: 'system-ui, sans-serif', fontSize: '10px', color: '#888', margin: '2px 0 8px', lineHeight: '1.35'})

    var thickSvg = sel.append('svg')
      .at({width: '100%', height: 72})
      .st({display: 'block', marginBottom: '4px'})

    var samples = [
      {label: 'weak', w: 0.6},
      {label: 'medium', w: 1.6},
      {label: 'strong', w: 3.2},
    ]
    samples.forEach((s, i) => {
      var y = 14 + i * 20
      thickSvg.append('line')
        .at({
          x1: 4, x2: 88, y1: y, y2: y,
          stroke: '#0D7377',
          'stroke-width': s.w,
          'stroke-linecap': 'round',
        })
      thickSvg.append('text')
        .text(s.label)
        .at({x: 96, y: y + 3, fill: '#555', 'font-size': 10, 'font-family': 'system-ui'})
    })

    // ——— Color class legend ———
    sectionTitle('Edge / node colors')
    sel.append('div')
      .text('4-color greedy: nearby edges prefer different hues. Saturation / alpha also track intensity (same formula as thickness).')
      .st({
        fontFamily: 'system-ui, sans-serif',
        fontSize: '10px',
        color: '#666',
        marginBottom: '8px',
        lineHeight: '1.35',
      })

    var hues = coloring.hues || {
      teal: '#0D7377', coral: '#C44536', ochre: '#B57924', slate: '#374E6E',
    }
    var labels = window.edgeColoring?.HUE_LABELS || Object.keys(hues)
    var hexes = window.edgeColoring?.HUES
      ? window.edgeColoring.HUES.map((rgb) =>
          '#' + rgb.map((x) => x.toString(16).padStart(2, '0')).join('')
        )
      : Object.values(hues)

    var mode = visState.edgeColorMode || coloring.mode || 'greedy4'
    var k = window.edgeColoring?.paletteSize?.(mode) || Math.min(4, hexes.length)
    if (mode === 'prgn') {
      sel.append('div').text('PRGn: green = positive, purple = negative')
        .st({fontFamily: 'system-ui', fontSize: '10px', color: '#555'})
    } else {
      var pal = sel.append('div').st({display: 'flex', flexDirection: 'column', gap: '6px'})
      for (var i = 0; i < k; i++) {
        var hex = hexes[i] || Object.values(hues)[i]
        var name = labels[i] || ('C' + i)
        var row = pal.append('div').st({display: 'flex', alignItems: 'center', gap: '8px'})
        row.append('span').st({
          width: '12px', height: '12px', borderRadius: '50%',
          background: hex, border: '1px solid #000', flexShrink: 0,
        })
        row.append('span').st({
          width: '28px', height: '3px', background: hex, flexShrink: 0,
        })
        row.append('span').text(name).st({fontSize: '10px', color: '#555'})
      }
    }

    // ——— Link filters ———
    sectionTitle('Links filter')
    var linkHelp = [
      ['Input', 'edges into clicked / pinned'],
      ['Output', 'edges out of clicked / pinned'],
      ['Either', 'into or out of focus nodes'],
      ['Both', 'only edges between pinned nodes'],
    ]
    linkHelp.forEach(([name, desc]) => {
      var row = sel.append('div').st({
        display: 'flex', gap: '8px', fontFamily: 'system-ui', fontSize: '10px',
        color: '#555', padding: '2px 0',
      })
      row.append('span').text(name).st({fontWeight: 600, width: '48px', flexShrink: 0})
      row.append('span').text(desc)
    })
    sel.append('div')
      .text('Picking Input / Output / Both turns off “Show all links” so the filter is visible.')
      .st({fontFamily: 'system-ui', fontSize: '10px', color: '#888', margin: '6px 0 4px', lineHeight: '1.35'})

    // ——— Node shapes ———
    sectionTitle('Node shapes')
    var shapes = [
      ['●', 'CLT / transcoder feature'],
      ['■', 'Embedding or logit'],
      ['◆', 'MLP reconstruction error'],
    ]
    shapes.forEach(([glyph, label]) => {
      var row = sel.append('div').st({
        display: 'flex', gap: '8px', alignItems: 'center',
        fontFamily: 'system-ui', fontSize: '10px', color: '#555', padding: '2px 0',
      })
      row.append('span').text(glyph).st({width: '16px', textAlign: 'center'})
      row.append('span').text(label)
    })

    // ——— Counts ———
    sectionTitle('Counts')
    var table = sel.append('div.stats-table')
    rows.forEach(([k, v]) => {
      if (!k && !v) {
        table.append('div').st({height: '8px'})
        return
      }
      var row = table.append('div').st({
        display: 'flex',
        justifyContent: 'space-between',
        gap: '10px',
        padding: '3px 0',
        borderBottom: '1px solid #ECEAE3',
      })
      row.append('span').text(k).st({color: '#666', flex: '1'})
      row.append('span').text(v).st({
        fontWeight: 600,
        textAlign: 'right',
        maxWidth: '55%',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap',
      })
    })

    sectionTitle('By type')
    var typeList = sel.append('div')
    ;[...byType.entries()]
      .sort((a, b) => b[1] - a[1])
      .forEach(([t, n]) => {
        var row = typeList.append('div').st({
          display: 'flex',
          justifyContent: 'space-between',
          gap: '8px',
          padding: '2px 0',
          fontSize: '10px',
          color: '#555',
        })
        row.append('span').text(t.replace('cross layer transcoder', 'CLT feature'))
        row.append('span').text(String(n)).st({fontWeight: 600})
      })
  }

  render()
  renderAll.pinnedIds.fns['graphStats'] = render
  renderAll.expandedIds.fns['graphStats'] = render
  renderAll.edgeColorMode.fns['graphStats'] = render
  if (renderAll.thicknessMode) renderAll.thicknessMode.fns['graphStats'] = render
  renderAll.features.fns['graphStats'] = render
}

window.init?.()
