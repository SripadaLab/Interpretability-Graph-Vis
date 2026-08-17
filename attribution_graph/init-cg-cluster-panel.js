/**
 * Clustering sidebar — investigate manual supernodes (Neuronpedia / paper groupings).
 * Click a cluster to pin its members into the subgraph and focus the first member.
 */
window.initCgClusterPanel = function ({visState, renderAll, data, cgSel}) {
  var sel = cgSel.select('.cluster-panel')
  if (sel.empty()) return

  function getSupernodes() {
    if (visState.subgraph?.supernodes?.length) return visState.subgraph.supernodes
    if (visState.supernodes?.length) return visState.supernodes
    return data.qParams?.supernodes || []
  }

  function nodeById(id) {
    return data.nodes.idToNode?.[id] || data.nodes.find((n) => n.nodeId === id || n.node_id === id)
  }

  function render() {
    var supernodes = getSupernodes()
    var byId = data.nodes.idToNode || {}

    sel.html('').st({
      fontFamily: 'system-ui, sans-serif',
      fontSize: '12px',
      color: '#333',
      padding: '12px 14px',
      boxSizing: 'border-box',
      overflowY: 'auto',
      height: '100%',
      background: '#F7F6F1',
      borderRight: '1px solid #E4E2D8',
    })

    sel.append('div')
      .text('Clusters')
      .st({fontWeight: 600, fontSize: '13px', marginBottom: '4px', letterSpacing: '0.02em'})

    sel.append('div')
      .text(
        supernodes.length
          ? 'Manual groupings from the published graph. Click a cluster to load it in the subgraph.'
          : 'No manual clusters on this graph. Use Fill subgraph, or hold ⌘/Ctrl and click nodes to pin, then press G to group.'
      )
      .st({fontSize: '10px', color: '#666', marginBottom: '10px', lineHeight: '1.35'})

    if (!supernodes.length) {
      sel.append('div')
        .text('—')
        .st({color: '#aaa', fontSize: '11px', fontFamily: 'ui-monospace, monospace'})
      return
    }

    // Pin every member of every cluster so all supernodes render at once.
    sel.append('div')
      .text('Show all clusters')
      .st({
        display: 'inline-block', cursor: 'pointer', fontSize: '11px', fontWeight: 600,
        color: '#0D7377', border: '1px solid #0D7377', borderRadius: '5px',
        padding: '5px 9px', marginBottom: '10px', background: '#fff',
      })
      .on('click', () => {
        var pinned = new Set(visState.pinnedIds || [])
        supernodes.forEach((sn) => sn.slice(1).forEach((id) => pinned.add(id)))
        visState.pinnedIds = [...pinned]
        util.params.set('pinnedIds', visState.pinnedIds.join(','))
        if (!visState.subgraph) visState.subgraph = {supernodes: []}
        visState.subgraph.supernodes = getSupernodes()
        util.params.set('supernodes', JSON.stringify(visState.subgraph.supernodes))
        renderAll.pinnedIds()
      })

    // Cosine-similarity verification. The methodology is always explained; the
    // pass/fail numbers only appear once verify_supernodes.py has written
    // metadata.supernode_similarity = {in_group_mean, out_group_mean, per_cluster}.
    var sim = (data.metadata || {}).supernode_similarity
    var ok = sim ? sim.in_group_mean > sim.out_group_mean : null
    var box = sel.append('div').st({
      background: sim ? (ok ? '#EEF7EF' : '#FBEEEE') : '#FBFAF5',
      border: '1px solid ' + (sim ? (ok ? '#BBD9BE' : '#E4B7B7') : '#E4E2D8'),
      borderRadius: '5px', padding: '8px 10px', marginBottom: '12px',
      fontSize: '10px', color: '#333', lineHeight: '1.45',
    })
    box.append('div').text((sim ? (ok ? '✓ ' : '✗ ') : '') + 'Inflow + outflow cosine check')
      .st({fontWeight: 600, marginBottom: '4px', fontSize: '11px'})

    // Plain-language explanation of what the check measures.
    box.append('div')
      .html(
        'For each feature <b>f</b> we build a direction by concatenating, left→right, ' +
        'its <b>inflow</b> (GemmaScope encoder column, how it <i>reads</i> the residual stream) ' +
        'and its <b>outflow</b> (decoder row, how it <i>writes</i>). Each half is unit-normalized ' +
        'so both contribute equally, then we take cosine similarity between features. ' +
        'A good cluster has higher cosine <i>within</i> the group than <i>across</i> groups.'
      )
      .st({fontSize: '9.5px', color: '#555', marginBottom: '6px', lineHeight: '1.4'})

    // Rendered formula (falls back to monospace text if KaTeX is unavailable).
    var fBox = box.append('div').st({
      background: '#fff', border: '1px solid #E4E2D8', borderRadius: '4px',
      padding: '6px 6px', marginBottom: '6px', overflowX: 'auto', textAlign: 'center',
    })
    var tex =
      '\\mathbf{v}_f = \\big[\\,\\widehat{W_{\\text{enc}}}[:,f]\\; \\big\\|\\; \\widehat{W_{\\text{dec}}}[f,:]\\,\\big],' +
      '\\quad \\cos(\\mathbf{v}_a,\\mathbf{v}_b) = \\tfrac{1}{2}\\big(\\cos_{\\text{in}} + \\cos_{\\text{out}}\\big)'
    var texFallback = 'v_f = [ unit(W_enc[:,f]) || unit(W_dec[f,:]) ] ; cos = ½(cos_in + cos_out)'
    if (window.katex) {
      try {
        fBox.html(katex.renderToString(tex, {displayMode: true, throwOnError: false, strict: 'ignore'}))
        fBox.selectAll('.katex-display').st({margin: '0.2em 0', fontSize: '0.8em'})
      } catch (e) {
        fBox.st({fontFamily: 'ui-monospace, monospace', fontSize: '9px', textAlign: 'left'}).text(texFallback)
      }
    } else {
      fBox.st({fontFamily: 'ui-monospace, monospace', fontSize: '9px', textAlign: 'left'}).text(texFallback)
    }

    if (sim) {
      box.append('div').text('in-group mean  ' + sim.in_group_mean.toFixed(3))
        .st({fontFamily: 'ui-monospace, monospace'})
      box.append('div').text('out-group mean ' + sim.out_group_mean.toFixed(3))
        .st({fontFamily: 'ui-monospace, monospace'})
      box.append('div').text(ok ? 'in-group > out-group ✓' : 'in-group ≤ out-group — review clusters')
        .st({color: ok ? '#2f6f36' : '#a23', marginTop: '3px', fontWeight: 600})
    } else {
      box.append('div')
        .text('Run `python -m interpretability_graph verify` to compute in-group vs out-group cosine.')
        .st({fontFamily: 'ui-monospace, monospace', fontSize: '9px', color: '#999', marginTop: '2px'})
    }

    if (sim && sim.matrix && sim.matrix.labels && sim.matrix.labels.length) {
      renderMatrix(sim.matrix)
    }

    var list = sel.append('div').st({display: 'flex', flexDirection: 'column', gap: '8px'})

    supernodes.forEach((sn, idx) => {
      var label = sn[0]
      var memberIds = sn.slice(1)
      var members = memberIds.map(nodeById).filter(Boolean)
      var layers = members
        .map((m) => m.layer)
        .filter((l) => l != null && l !== 'E')
        .map(Number)
        .filter(Number.isFinite)
      var layerRange = layers.length
        ? (d3.min(layers) === d3.max(layers) ? 'L' + d3.min(layers) : 'L' + d3.min(layers) + '–' + d3.max(layers))
        : '—'
      var avgInfl = members.length
        ? d3.mean(members, (m) => Math.abs(m.influence || 0))
        : 0

      var card = list.append('div.cluster-card')
        .st({
          background: '#fff',
          border: '1px solid #E4E2D8',
          borderRadius: '6px',
          padding: '8px 10px',
          cursor: 'pointer',
        })
        .on('mouseenter', function () {
          d3.select(this).st({borderColor: '#0D7377'})
        })
        .on('mouseleave', function () {
          d3.select(this).st({borderColor: '#E4E2D8'})
        })
        .on('click', () => {
          // Pin all members and open subgraph focus on first
          var pinned = new Set(visState.pinnedIds || [])
          memberIds.forEach((id) => pinned.add(id))
          visState.pinnedIds = [...pinned]
          util.params.set('pinnedIds', visState.pinnedIds.join(','))

          // Keep / sync supernodes into subgraph state
          if (!visState.subgraph) visState.subgraph = {supernodes: []}
          visState.subgraph.supernodes = getSupernodes()
          util.params.set('supernodes', JSON.stringify(visState.subgraph.supernodes))

          var focus = members[0]
          if (focus) {
            visState.clickedId = focus.nodeId
            visState.clickedCtxIdx = focus.ctx_idx
            util.params.set('clickedId', focus.nodeId)
          }
          renderAll.pinnedIds()
          renderAll.clickedId()
        })

      card.append('div')
        .text(label)
        .st({fontWeight: 600, fontSize: '12px', marginBottom: '4px'})

      card.append('div')
        .text(members.length + ' features · ' + layerRange +
          (avgInfl ? ' · avg infl ' + avgInfl.toFixed(3) : ''))
        .st({fontSize: '10px', color: '#777', marginBottom: '6px', fontFamily: 'ui-monospace, monospace'})

      var perCluster = sim && sim.per_cluster && sim.per_cluster[label]
      if (perCluster) {
        var cok = perCluster.in >= perCluster.out
        card.append('div')
          .text('cos in ' + perCluster.in.toFixed(3) + ' vs out ' + perCluster.out.toFixed(3))
          .st({
            fontSize: '10px', fontFamily: 'ui-monospace, monospace',
            color: cok ? '#2f6f36' : '#a23', marginBottom: '6px',
          })
      }

      var memberList = card.append('div').st({display: 'flex', flexDirection: 'column', gap: '2px'})
      members.slice(0, 8).forEach((m) => {
        var row = memberList.append('div')
          .st({
            display: 'flex',
            justifyContent: 'space-between',
            gap: '6px',
            fontSize: '10px',
            color: '#555',
            padding: '2px 0',
            borderTop: '1px solid #F0EEE7',
          })
          .on('click', (ev) => {
            ev.stopPropagation()
            visState.clickedId = m.nodeId
            visState.clickedCtxIdx = m.ctx_idx
            util.params.set('clickedId', m.nodeId)
            if (!(visState.pinnedIds || []).includes(m.nodeId)) {
              visState.pinnedIds = [...(visState.pinnedIds || []), m.nodeId]
              util.params.set('pinnedIds', visState.pinnedIds.join(','))
              renderAll.pinnedIds()
            }
            renderAll.clickedId()
          })
        row.append('span')
          .text((m.ppClerp || m.clerp || m.nodeId || '').slice(0, 36))
          .st({overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1})
        row.append('span')
          .text(m.layer === 'E' ? 'Emb' : ('L' + m.layer))
          .st({color: '#888', flexShrink: 0})
      })
      if (members.length > 8) {
        card.append('div')
          .text('+' + (members.length - 8) + ' more')
          .st({fontSize: '10px', color: '#999', marginTop: '4px'})
      }
    })
  }

  // Cluster × cluster cosine heatmap. Diagonal = within-cluster similarity,
  // off-diagonal = between-cluster. A healthy grouping shows a bright diagonal.
  function renderMatrix(matrix) {
    var labels = matrix.labels
    var values = matrix.values
    var counts = matrix.counts || labels.map(() => null)
    var n = labels.length

    var wrap = sel.append('div').st({marginBottom: '12px'})
    wrap.append('div').text('Cluster similarity matrix')
      .st({fontWeight: 600, fontSize: '11px', marginBottom: '2px'})
    wrap.append('div')
      .text('diagonal = within-cluster · off-diagonal = between clusters · brighter = more similar')
      .st({fontSize: '9px', color: '#777', marginBottom: '6px', lineHeight: '1.35'})

    // Color scale over the observed value range (light → teal).
    var flat = values.flat().filter(v => v != null && Number.isFinite(v))
    var vmin = flat.length ? Math.min.apply(null, flat) : 0
    var vmax = flat.length ? Math.max.apply(null, flat) : 1
    if (vmax - vmin < 1e-6) vmax = vmin + 1e-6
    function lerp(a, b, t) { return Math.round(a + (b - a) * t) }
    function tOf(v) { return Math.max(0, Math.min(1, (v - vmin) / (vmax - vmin))) }
    function colorFor(v) {
      if (v == null || !Number.isFinite(v)) return '#EEECE4'
      var t = tOf(v)
      return 'rgb(' + lerp(247, 13, t) + ',' + lerp(246, 115, t) + ',' + lerp(241, 119, t) + ')'
    }
    function textColor(v) {
      if (v == null || !Number.isFinite(v)) return '#bbb'
      return tOf(v) > 0.55 ? '#fff' : '#333'
    }

    // Size cells to the available panel width.
    var avail = (sel.node().clientWidth || 260) - 28
    var gutter = 16
    var cell = Math.max(15, Math.min(34, Math.floor((avail - gutter) / n)))
    var top = 16
    var svgW = gutter + n * cell
    var svgH = top + n * cell

    var svg = wrap.append('svg')
      .at({width: svgW, height: svgH})
      .st({display: 'block', overflow: 'visible', fontFamily: 'ui-monospace, monospace'})

    // Column index labels (top) and row index labels (left).
    d3.range(n).forEach(j => {
      svg.append('text').text(j + 1)
        .at({x: gutter + j * cell + cell / 2, y: top - 5, textAnchor: 'middle'})
        .st({fontSize: '9px', fill: '#888'})
    })
    d3.range(n).forEach(i => {
      svg.append('text').text(i + 1)
        .at({x: gutter - 4, y: top + i * cell + cell / 2 + 3, textAnchor: 'end'})
        .st({fontSize: '9px', fill: '#888'})
    })

    // Cells.
    for (var i = 0; i < n; i++) {
      for (var j = 0; j < n; j++) {
        var v = values[i][j]
        var isDiag = i === j
        var g = svg.append('g')
        g.append('rect')
          .at({
            x: gutter + j * cell, y: top + i * cell,
            width: cell - 1, height: cell - 1,
            fill: colorFor(v),
            stroke: isDiag ? '#0D7377' : '#E4E2D8',
            strokeWidth: isDiag ? 1.5 : 0.5,
          })
          .append('title')
          .text(labels[i] + '  ×  ' + labels[j] + '\n' + (v == null ? 'n/a (single feature)' : v.toFixed(3)))
        if (cell >= 22 && v != null && Number.isFinite(v)) {
          g.append('text')
            .text(v.toFixed(2).replace(/^0\./, '.').replace(/^-0\./, '-.'))
            .at({
              x: gutter + j * cell + (cell - 1) / 2,
              y: top + i * cell + (cell - 1) / 2 + 3,
              textAnchor: 'middle',
            })
            .st({fontSize: '8px', fill: textColor(v), pointerEvents: 'none'})
        }
      }
    }

    // Index → label legend.
    var legend = wrap.append('div').st({marginTop: '6px', display: 'flex', flexDirection: 'column', gap: '2px'})
    labels.forEach((label, i) => {
      var diag = values[i][i]
      var row = legend.append('div')
        .st({display: 'flex', alignItems: 'center', gap: '5px', fontSize: '9.5px', color: '#555'})
      row.append('span')
        .st({
          display: 'inline-block', width: '10px', height: '10px', borderRadius: '2px',
          background: colorFor(diag), border: '1px solid #ccc', flexShrink: 0,
        })
      row.append('span').text((i + 1) + '.')
        .st({color: '#888', flexShrink: 0, minWidth: '14px'})
      row.append('span').text(label + (counts[i] != null ? ' (' + counts[i] + ')' : ''))
        .st({overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap'})
    })
  }

  render()
  renderAll.pinnedIds.fns['clusterPanel'] = render
  renderAll.clickedId.fns['clusterPanel'] = render
  renderAll.features.fns['clusterPanel'] = render
}

window.init?.()
