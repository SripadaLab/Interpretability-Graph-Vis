/**
 * Full-width SVD spectrum panel: cosine similarity + three attribution matrices,
 * side by side. Node set = transcoder features surviving the current prune
 * (capped), same order for every matrix. Rank-k slider shows % Frobenius energy.
 */
window.initCgSvdPanel = function ({visState, renderAll, data, cgSel}) {
  var sel = cgSel.select('.svd-panel')
  if (sel.empty()) return
  sel.datum().resizeFn = render

  var maxN = 250 // fallback edge-expanded sample if no backend bundle
  var spectrumKMax = 32 // how many σ bars / modes we compute & show
  var rankK = Math.min(5, spectrumKMax)
  var clusterMatrixIdx = 0 // which spectrum matrix drives the cluster graph
  var clusterTopN = 8 // top-|u| features per σ in the cluster graph
  var componentIdx = 0 // which singular component for token-position view
  var focusCtxIdx = null // null = all token positions; else filter graphs to that ctx
  var heatShowAll = true // component UUᵀ heatmap: all loadings vs top-28 preview
  // Order inside a block: 'none' leaves features in the order they arrive
  // (|influence| rank), which is arbitrary with respect to this component, so a
  // bright block is the grouping's doing and not the sort's. 'abs' / 'signed'
  // sort by loading.
  var heatSortMode = 'none' // 'none' | 'abs' | 'signed'
  // Which axis blocks the component matrix: 'token' (ctx_idx), 'layer', or one
  // of the two nestings, 'tokenLayer' / 'layerToken'. A mode can be local to a
  // prompt position, local to a depth, or neither, and only the matching
  // grouping makes that visible as a diagonal block. The two nestings show the
  // same partition with the roles of heavy and light rules swapped.
  var heatGroupBy = 'token'
  var spectralK = null // k for spectral cluster viz (metadata.spectral_clusters)
  var spectralFocusGroup = null // null = all groups; else group index to emphasize
  var spectralHeatShowAll = true // affinity W heatmap: all nodes vs top-by-influence preview
  var affinityNpy = null // cached npy parse for W
  var affinityNpyPath = null
  var affinityNpyStatus = 'idle' // idle | loading | ready | missing | error
  var affinityLoadError = null
  var affinityWaiters = [] // callbacks waiting on in-flight affinity load
  var modeColors = ['#0D7377', '#C45C26', '#2F6F36', '#A67C00', '#4A6FA5', '#8B4513', '#5F7A6A', '#9C4A3A']
  var clusterSim = null // active force simulation; stop on re-render
  var svdBundle = null // backend full-graph SVD (from export-svd)
  var svdBundleStatus = 'idle' // idle | loading | ready | missing | error
  // Shared matrix-picker state: both SV-cluster + token-position views register
  // here so changing Matrix on either control updates both (avoids getting stuck
  // on an empty Unsigned adjacency while Cosine would still work).
  var matrixPickerStyles = []
  var matrixViewRedraws = []
  var positionPickerStyles = []

  function bundlePath() {
    var meta = (data.metadata || {}).svd_bundle
    return meta && meta.path ? String(meta.path).replace(/^\/+/, '') : null
  }

  var svdBundleWaiters = []

  function ensureSvdBundle(done) {
    if (svdBundleStatus === 'ready' || svdBundleStatus === 'missing' || svdBundleStatus === 'error') {
      done && done()
      return
    }
    var path = bundlePath()
    if (!path) {
      svdBundleStatus = 'missing'
      done && done()
      return
    }
    if (done) svdBundleWaiters.push(done)
    if (svdBundleStatus === 'loading') return
    svdBundleStatus = 'loading'
    // no-store: avoid stale empty bundles after re-export
    fetch(util.graphDataUrl(path) + '?t=' + Date.now(), {cache: 'no-store'})
      .then(r => {
        if (!r.ok) throw new Error('HTTP ' + r.status)
        return r.json()
      })
      .then(bundle => {
        svdBundle = bundle
        svdBundleStatus = 'ready'
        spectrumKMax = Math.min(32, bundle.k || 32)
        var waiters = svdBundleWaiters.splice(0)
        waiters.forEach(fn => { try { fn() } catch (e) {} })
      })
      .catch(err => {
        console.warn('SVD bundle load failed:', err)
        svdBundle = null
        svdBundleStatus = 'error'
        var waiters = svdBundleWaiters.splice(0)
        waiters.forEach(fn => { try { fn() } catch (e) {} })
      })
  }

  function nodesFromBundle() {
    if (!svdBundle?.node_ids?.length) return null
    var byId = {}
    ;(data.allNodes || data.nodes || []).forEach(n => {
      var id = n.nodeId || n.node_id
      if (id != null) byId[id] = n
      if (n.node_id != null) byId[n.node_id] = n
    })
    return svdBundle.node_ids.map(id => byId[id]).filter(Boolean)
  }

  function chartsFromBundle() {
    if (!svdBundle?.charts) return null
    var order = ['cosine', 'unsigned', 'signed', 'symmetric']
    return order.map(key => {
      var c = svdBundle.charts[key]
      if (!c) return null
      return {
        key: key,
        title: c.title || key,
        equation: c.equation || '',
        note: (c.note || '') + ' · backend full graph n=' + svdBundle.n,
        matrix: null,
        frob2: c.frob2,
        svdResult: {
          sigmas: c.sigmas || [],
          vectors: c.vectors || [],
          truncated: !!c.truncated,
          k: c.k || (c.sigmas || []).length,
        },
      }
    }).filter(Boolean)
  }

  function tokenLabel(ctx, tokens) {
    tokens = tokens || (data.metadata && data.metadata.prompt_tokens) || []
    var t = tokens[ctx] != null ? tokens[ctx] : ('#' + ctx)
    t = String(util.ppToken ? util.ppToken(t) : t)
    return t.length > 10 ? t.slice(0, 9) + '…' : t
  }

  /** Restrict a node×node matrix to the rows/cols at focusCtxIdx (or all). */
  function sliceByCtx(matrix, orderedNodes, ctx) {
    if (ctx == null || ctx === undefined) {
      return {matrix: matrix, nodes: orderedNodes, indices: orderedNodes.map((_, i) => i)}
    }
    var indices = []
    var nodes = []
    orderedNodes.forEach((n, i) => {
      if (n && n.ctx_idx === ctx) {
        indices.push(i)
        nodes.push(n)
      }
    })
    var sub = indices.map(i => indices.map(j => (matrix[i] && matrix[i][j]) || 0))
    return {matrix: sub, nodes: nodes, indices: indices}
  }

  function withScrollPreserved(fn) {
    // Position / matrix redraws wipe large SVG blocks; without this the SVD
    // panel (and sometimes the window) jumps back to the top.
    var panel = sel.node()
    var panelTop = panel ? panel.scrollTop : 0
    var winX = window.scrollX || 0
    var winY = window.scrollY || 0
    try { fn() } finally {
      var restore = () => {
        if (panel) panel.scrollTop = panelTop
        window.scrollTo(winX, winY)
      }
      restore()
      requestAnimationFrame(restore)
      setTimeout(restore, 0)
    }
  }

  function setFocusCtxIdx(ctx, opts) {
    opts = opts || {}
    var toggle = opts.toggle !== false
    // Toggle: clicking the active position again clears the filter (chips/bars).
    if (toggle && ctx != null && ctx === focusCtxIdx) focusCtxIdx = null
    else focusCtxIdx = ctx
    withScrollPreserved(() => {
      positionPickerStyles.forEach(fn => { try { fn() } catch (e) {} })
      // Both SV-cluster + token-position bodies register on matrixViewRedraws.
      matrixViewRedraws.forEach(fn => { try { fn() } catch (e) {} })
    })
  }

  function renderPositionPicker(parent, rows, tokens) {
    var row = parent.append('div').st({
      display: 'flex', alignItems: 'center', gap: '4px', flexWrap: 'wrap',
      maxWidth: '520px', justifyContent: 'flex-end',
    })
    row.append('span').text('Position')
      .st({fontSize: '10px', color: '#888', textTransform: 'uppercase'})
    var opts = [{ctx: null, label: 'All', frac: 1}]
    ;(rows || []).forEach(r => {
      opts.push({
        ctx: r.ctx_idx,
        label: tokenLabel(r.ctx_idx, tokens),
        frac: r.frac || 0,
        n: (r.nodes && r.nodes.length) || 0,
      })
    })
    var selBtns = row.append('div.link-type-buttons').st({display: 'flex', flexWrap: 'wrap'})
      .appendMany('div', opts)
      .text(d => d.ctx == null
        ? 'All'
        : (d.label + (d.frac > 0 ? ' ' + Math.round(100 * d.frac) + '%' : '')))
      .st({
        fontSize: '10px', padding: '4px 7px', border: '1px solid #ccc',
        background: '#fff', cursor: 'pointer', marginLeft: '-1px', userSelect: 'none',
        fontFamily: 'ui-monospace, monospace',
      })
      .at({title: d => d.ctx == null
        ? 'Show graphs over all token positions'
        : ('Focus ctx ' + d.ctx + ' · "' + d.label + '"'
          + (d.n != null ? ' · ' + d.n + ' features' : '')
          + ' — click again to clear')})
      .on('click', (ev, d) => setFocusCtxIdx(d.ctx))
    function stylePos() {
      selBtns.st({
        background: d => (d.ctx === focusCtxIdx || (d.ctx == null && focusCtxIdx == null))
          ? '#0D7377' : '#fff',
        color: d => (d.ctx === focusCtxIdx || (d.ctx == null && focusCtxIdx == null))
          ? '#fff' : '#333',
        borderColor: d => (d.ctx === focusCtxIdx || (d.ctx == null && focusCtxIdx == null))
          ? '#0D7377' : '#ccc',
      })
    }
    stylePos()
    positionPickerStyles.push(stylePos)
    return row
  }

  function matrixIsEmpty(matrix) {
    if (!matrix?.length || !matrix[0]?.length) return true
    for (var i = 0; i < matrix.length; i++) {
      var row = matrix[i]
      for (var j = 0; j < row.length; j++) {
        if (Math.abs(row[j]) > 1e-12) return false
      }
    }
    return true
  }

  function chartIsEmpty(ch) {
    if (!ch) return true
    // Prefer SVD signal: a non-empty matrix with σ₁≈0 is still unusable
    // (e.g. diagonal-only cosine bug, or noise-floor adjacency).
    if (ch.svdResult?.sigmas?.length) {
      return !(ch.svdResult.sigmas[0] >= 1e-12)
    }
    return matrixIsEmpty(ch.matrix)
  }

  function firstUsableMatrixIdx(charts) {
    for (var i = 0; i < charts.length; i++) {
      if (!chartIsEmpty(charts[i])) return i
    }
    return 0
  }

  function setClusterMatrixIdx(i, charts) {
    if (!charts?.length) return
    var next = Math.max(0, Math.min(i, charts.length - 1))
    // Refuse empty matrices — keep the last usable selection.
    if (chartIsEmpty(charts[next])) {
      var usable = firstUsableMatrixIdx(charts)
      if (chartIsEmpty(charts[usable])) return
      next = usable
    }
    clusterMatrixIdx = next
    withScrollPreserved(() => {
      matrixPickerStyles.forEach(fn => { try { fn() } catch (e) {} })
      matrixViewRedraws.forEach(fn => { try { fn() } catch (e) {} })
    })
  }

  function selectedNodes() {
    var fromBundle = nodesFromBundle()
    if (fromBundle && fromBundle.length >= 2) return fromBundle

    // Fallback: edge-expanded sample from the full transcoder graph.
    var pool = (data.allNodes || data.nodes || []).filter(d => {
      var ft = (d.feature_type || '').toLowerCase()
      return ft.includes('transcoder') && !ft.includes('error')
    })
    var links = data.allLinks || data.links || []
    var byId = {}
    pool.forEach(n => { byId[n.nodeId] = n })

    // Adjacency among transcoder nodes (undirected for expansion).
    var nbr = {}
    pool.forEach(n => { nbr[n.nodeId] = [] })
    links.forEach(l => {
      var s = l.sourceNode?.nodeId || l.source
      var t = l.targetNode?.nodeId || l.target
      if (!byId[s] || !byId[t] || s === t) return
      var w = Math.abs(+(l.weight ?? l.pctInput ?? 0))
      if (!(w > 0)) return
      nbr[s].push({id: t, w})
      nbr[t].push({id: s, w})
    })

    var geomIds = ((data.metadata || {}).svd_geom || {}).node_ids || []
    var seed = []
    var seen = new Set()
    // Seed with geom nodes (stable cosine), then top influence.
    geomIds.forEach(id => {
      if (byId[id] && !seen.has(id)) { seen.add(id); seed.push(id) }
    })
    d3.sort(pool, d => -(Math.abs(d.influence || 0))).forEach(n => {
      if (!seen.has(n.nodeId)) { seen.add(n.nodeId); seed.push(n.nodeId) }
    })

    var chosen = []
    var chosenSet = new Set()
    var frontier = []
    function addNode(id) {
      if (chosenSet.has(id) || !byId[id]) return
      if (chosen.length >= maxN) return
      chosenSet.add(id)
      chosen.push(id)
      frontier.push(id)
    }
    // Take initial seeds, then grow by strongest edges into the set.
    for (var i = 0; i < seed.length && chosen.length < Math.min(40, maxN); i++) {
      addNode(seed[i])
    }
    while (chosen.length < maxN && frontier.length) {
      var best = null
      var bestW = -1
      for (var fi = 0; fi < frontier.length; fi++) {
        var u = frontier[fi]
        var list = nbr[u] || []
        for (var j = 0; j < list.length; j++) {
          var v = list[j]
          if (chosenSet.has(v.id)) continue
          if (v.w > bestW) { bestW = v.w; best = v.id }
        }
      }
      if (best == null) {
        var added = false
        for (var si = 0; si < seed.length; si++) {
          if (!chosenSet.has(seed[si])) { addNode(seed[si]); added = true; break }
        }
        if (!added) break
        continue
      }
      addNode(best)
    }

    var gset = new Set(geomIds)
    var nodes = chosen.map(id => byId[id]).filter(Boolean)
    var preferred = d3.sort(nodes.filter(d => gset.has(d.nodeId)), d => -(Math.abs(d.influence || 0)))
    var rest = d3.sort(nodes.filter(d => !gset.has(d.nodeId)), d => -(Math.abs(d.influence || 0)))
    return preferred.concat(rest)
  }

  function buildAttributionMatrices(nodeIds, links) {
    var idx = {}
    nodeIds.forEach((id, i) => { idx[id] = i })
    var n = nodeIds.length
    var unsigned = Array.from({length: n}, () => new Array(n).fill(0))
    var signed = Array.from({length: n}, () => new Array(n).fill(0))
    var symmetric = Array.from({length: n}, () => new Array(n).fill(0))
    ;(links || []).forEach(l => {
      var s = l.sourceNode?.nodeId || l.source
      var t = l.targetNode?.nodeId || l.target
      if (idx[s] == null || idx[t] == null) return
      var w = +(l.weight ?? l.pctInput ?? 0)
      if (!w) return
      unsigned[idx[s]][idx[t]] += Math.abs(w)
      signed[idx[s]][idx[t]] += w
      symmetric[idx[s]][idx[t]] += Math.abs(w)
      symmetric[idx[t]][idx[s]] += Math.abs(w)
    })
    return {unsigned, signed, symmetric}
  }

  function renderLatex(parent, tex, displayMode) {
    var box = parent.append('div').classed('svd-eq', 1).st({
      overflowX: 'auto',
      overflowY: 'visible',
      textAlign: displayMode ? 'center' : 'left',
      color: '#1a1a1a',
      marginBottom: displayMode ? '8px' : '4px',
      minHeight: displayMode ? '32px' : '18px',
      lineHeight: '1.45',
    })
    var node = box.node()
    var katexApi = window.katex
    if (katexApi && typeof katexApi.render === 'function') {
      try {
        katexApi.render(tex, node, {
          displayMode: !!displayMode,
          throwOnError: false,
          strict: 'ignore',
          trust: false,
          output: 'html',
        })
      } catch (e) {
        console.warn('KaTeX render failed:', tex, e)
        node.textContent = tex
      }
    } else if (katexApi && typeof katexApi.renderToString === 'function') {
      try {
        node.innerHTML = katexApi.renderToString(tex, {
          displayMode: !!displayMode,
          throwOnError: false,
          strict: 'ignore',
        })
      } catch (e) {
        console.warn('KaTeX renderToString failed:', tex, e)
        node.textContent = tex
      }
    } else {
      node.textContent = tex
      box.st({fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace', fontSize: '11px'})
    }
    // Never shrink KaTeX to illegible / invisible sizes.
    box.selectAll('.katex-display').st({margin: '0.4em 0'})
    box.selectAll('.katex').st({
      fontSize: displayMode ? '1.15em' : '1.05em',
      color: '#1a1a1a',
      visibility: 'visible',
      opacity: 1,
    })
    return box
  }

  function featureIdOf(node, id) {
    if (node && node.feature != null && node.feature !== '') return String(node.feature)
    if (node && node.active_feature_idx != null && node.active_feature_idx !== '') {
      return String(node.active_feature_idx)
    }
    var raw = (node && (node.nodeId || node.featureId || node.node_id)) || id || ''
    var parts = String(raw).split('_')
    if (parts.length >= 2 && parts[1] !== '') return parts[1]
    return raw ? String(raw) : ''
  }

  function featureLabel(node, id) {
    var feat = featureIdOf(node, id)
    if (!feat) return id || ''
    var clerp = node && (node.localClerp || node.clerp)
    if (clerp) {
      clerp = String(clerp).trim()
      if (clerp && !/^\[group\s+\d+\]/i.test(clerp) && clerp !== feat) {
        var t = '[' + feat + '] ' + clerp
        return t.length > 48 ? t.slice(0, 46) + '…' : t
      }
    }
    return '[' + feat + ']'
  }

  function downloadBlobCsv(matrix, nodes, slug) {
    if (!matrix?.length || !nodes?.length) return
    var ids = nodes.map(n => n.nodeId || n.node_id || '')
    var esc = s => {
      s = String(s == null ? '' : s)
      return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s
    }
    var lines = ['node_id,' + ids.map(esc).join(',')]
    for (var i = 0; i < matrix.length; i++) {
      lines.push(esc(ids[i]) + ',' + matrix[i].map(v =>
        (v == null || !Number.isFinite(v)) ? '' : Number(v).toFixed(8)
      ).join(','))
    }
    var blob = new Blob([lines.join('\n') + '\n'], {type: 'text/csv;charset=utf-8'})
    var a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = (slug || 'graph') + '-cosine.csv'
    document.body.appendChild(a)
    a.click()
    setTimeout(() => {
      URL.revokeObjectURL(a.href)
      a.remove()
    }, 500)
  }

  /** Prefer the unthresholded server export (all transcoder nodes); else in-panel matrix. */
  function downloadCosineCsv(matrix, nodes, slug) {
    var full = (data.metadata || {}).svd_geom_full
    var rel = full && full.csv
    if (rel) {
      var url = util.graphDataUrl(rel)
      fetch(url).then(r => {
        if (!r.ok) throw new Error('missing full export')
        return r.blob()
      }).then(blob => {
        var a = document.createElement('a')
        a.href = URL.createObjectURL(blob)
        a.download = (slug || 'graph') + '-cosine-full.csv'
        document.body.appendChild(a)
        a.click()
        setTimeout(() => {
          URL.revokeObjectURL(a.href)
          a.remove()
        }, 500)
      }).catch(() => downloadBlobCsv(matrix, nodes, slug))
      return
    }
    downloadBlobCsv(matrix, nodes, slug)
  }

  /**
   * Full node×node cosine heatmap (same matrix as the Cosine SVD spectrum)
   * with axis labels + CSV export.
   */
  function renderCosineMatrixHeatmap(container, matrix, nodes, note) {
    if (!matrix?.length || !nodes?.length) return
    var n = Math.min(matrix.length, nodes.length)
    var wrap = container.append('div.svd-cosine-matrix').st({
      marginTop: '12px', background: '#fff', border: '1px solid #E4E2D8',
      borderRadius: '6px', padding: '10px 12px', boxSizing: 'border-box',
    })
    var head = wrap.append('div').st({
      display: 'flex', flexWrap: 'wrap', alignItems: 'flex-start',
      justifyContent: 'space-between', gap: '8px', marginBottom: '6px',
    })
    var titleCol = head.append('div')
    var fullMeta = (data.metadata || {}).svd_geom_full
    var fullN = fullMeta && fullMeta.n
    titleCol.append('div').text('Node × node cosine similarity')
      .st({fontWeight: 600, fontSize: '12px', color: '#222'})
    titleCol.append('div')
      .text(
        (note || 'Cᵢⱼ = cos(vᵢ, vⱼ)')
          + ' · panel shows ' + n + '×' + n
          + (fullN ? (' · Export CSV = full unthresholded ' + fullN + '×' + fullN) : '')
      )
      .st({fontSize: '10px', color: '#666', marginTop: '2px', maxWidth: '640px', lineHeight: '1.35'})
    var slug = (data.metadata && data.metadata.slug) || 'graph'
    head.append('button')
      .text(fullN ? ('Export full ' + fullN + '×' + fullN + ' CSV') : 'Export CSV')
      .st({
        fontSize: '11px', fontWeight: 600, cursor: 'pointer',
        border: '1px solid #0D7377', background: '#F3FAFA', color: '#0D7377',
        borderRadius: '5px', padding: '5px 10px',
      })
      .on('click', () => downloadCosineCsv(matrix, nodes.slice(0, n), slug))

    var eq = wrap.append('div').st({marginBottom: '8px'})
    renderLatex(
      eq,
      'C_{ij} = \\cos(\\mathbf{v}_i, \\mathbf{v}_j)'
        + '\\quad\\text{(teal }=+1\\text{, cream }=0\\text{, orange }=-1\\text{)}',
      true
    )
    eq.selectAll('.svd-eq').st({textAlign: 'left', minHeight: '0'})
    eq.selectAll('.katex').st({fontSize: '1.05em'})

    var showN = Math.min(n, 36)
    var cell = Math.max(5, Math.min(14, Math.floor(420 / showN)))
    var pad = {top: 10, right: 10, bottom: 52, left: 56}
    var plotW = cell * showN
    var hw = plotW + pad.left + pad.right
    var hh = plotW + pad.top + pad.bottom
    var svg = wrap.append('svg').at({
      width: Math.min(hw, 520), height: Math.min(hh, 520),
      viewBox: `0 0 ${hw} ${hh}`,
    })
    var g = svg.append('g').at({transform: `translate(${pad.left},${pad.top})`})

    var cells = []
    for (var i = 0; i < showN; i++) {
      for (var j = 0; j < showN; j++) {
        cells.push({i, j, v: matrix[i][j]})
      }
    }
    g.appendMany('rect', cells).at({
      x: d => d.j * cell,
      y: d => d.i * cell,
      width: cell - 0.4,
      height: cell - 0.4,
      fill: d => {
        var v = d.v
        if (v == null || !Number.isFinite(v)) return '#EEECE4'
        if (v >= 0) return d3.interpolateRgb('#FBFAF5', '#0D7377')(Math.min(1, v))
        return d3.interpolateRgb('#FBFAF5', '#C45C26')(Math.min(1, -v))
      },
      stroke: d => d.i === d.j ? '#333' : 'none',
      strokeWidth: d => d.i === d.j ? 0.6 : 0,
    }).append('title').text(d => {
      var a = featureLabel(nodes[d.i])
      var b = featureLabel(nodes[d.j])
      var v = d.v
      return a + '  ×  ' + b + '\n'
        + (v == null || !Number.isFinite(v) ? 'n/a' : 'cos = ' + v.toFixed(4))
        + '\n' + (nodes[d.i].nodeId || '') + ' × ' + (nodes[d.j].nodeId || '')
    })

    // Sparse tick labels (node index + short clerp) on both axes
    var tickStep = showN <= 12 ? 1 : (showN <= 24 ? 2 : 3)
    for (var t = 0; t < showN; t += tickStep) {
      var lab = String(t + 1)
      g.append('text').text(lab).at({
        x: t * cell + cell / 2, y: plotW + 12,
        textAnchor: 'middle', fill: '#666', fontSize: 8,
      })
      g.append('text').text(lab).at({
        x: -6, y: t * cell + cell / 2 + 3,
        textAnchor: 'end', fill: '#666', fontSize: 8,
      })
    }
    g.append('text').text('node j').at({
      x: plotW / 2, y: plotW + 28,
      textAnchor: 'middle', fill: '#888', fontSize: 9,
    })
    g.append('text').text('node i').at({
      transform: 'rotate(-90)',
      x: -plotW / 2, y: -40,
      textAnchor: 'middle', fill: '#888', fontSize: 9,
    })

    if (showN < n) {
      wrap.append('div')
        .text('Showing top ' + showN + ' of ' + n + ' nodes in the heatmap; Export CSV includes the full ' + n + '×' + n + ' matrix.')
        .st({fontSize: '10px', color: '#888', marginTop: '6px'})
    }

    // Compact index → feature legend
    var legend = wrap.append('div').st({
      marginTop: '8px', display: 'flex', flexWrap: 'wrap', gap: '4px 12px',
      maxHeight: '120px', overflowY: 'auto',
    })
    nodes.slice(0, showN).forEach((node, i) => {
      legend.append('div')
        .text((i + 1) + '. ' + featureLabel(node))
        .st({fontSize: '9px', color: '#555', maxWidth: '220px',
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap'})
        .at({title: (node.nodeId || '') + '\n' + (node.ppClerp || node.clerp || '')})
    })
  }

  function focusSvdFeature(node) {
    if (!node) return
    // Sticky selection: click focus survives mouseleave / scrolling away.
    visState.clickedId = node.nodeId
    visState.clickedCtxIdx = node.ctx_idx
    visState.hoveredId = null
    visState.hoveredCtxIdx = null
    util.params.set('clickedId', node.nodeId)
    if (!(visState.pinnedIds || []).includes(node.nodeId)) {
      visState.pinnedIds = [...(visState.pinnedIds || []), node.nodeId]
      util.params.set('pinnedIds', visState.pinnedIds.join(','))
      renderAll.pinnedIds()
    }
    if (visState.expandEdges && !(visState.expandedIds || []).includes(node.nodeId)) {
      visState.expandedIds = [node.nodeId, ...(visState.expandedIds || [])]
      util.params.set('expandedIds', visState.expandedIds.join(','))
      renderAll.expandedIds()
    }
    renderAll.clickedId()
    var graphEl = cgSel.select('.link-graph').node()
    if (graphEl?.scrollIntoView) {
      graphEl.scrollIntoView({behavior: 'smooth', block: 'center'})
    }
  }

  /**
   * Graph of σ₁…σₖ hubs with top-|u| features clustered around each mode.
   * Features shared across modes sit once and link to every relevant σ.
   */
  function renderModeClusterGraph(container, charts, orderedNodes, k) {
    var wrap = container.append('div.svd-mode-graph').st({
      marginTop: '12px', background: '#fff', border: '1px solid #E4E2D8',
      borderRadius: '6px', padding: '10px 12px', boxSizing: 'border-box',
    })

    var head = wrap.append('div').st({
      display: 'flex', flexWrap: 'wrap', alignItems: 'center',
      gap: '10px', justifyContent: 'space-between', marginBottom: '8px',
    })
    var titleCol = head.append('div')
    titleCol.append('div').text('SV → feature clusters')
      .st({fontWeight: 600, fontSize: '12px', color: '#222'})
    titleCol.append('div')
      .text('Each σᵣ hub links to its top-|uᵣ| features. Shared features keep one node with edges to every mode they load on. Click a feature to select (sticky).')
      .st({fontSize: '10px', color: '#666', marginTop: '2px', maxWidth: '640px', lineHeight: '1.35'})

    clusterMatrixIdx = Math.max(0, Math.min(clusterMatrixIdx, charts.length - 1))
    if (chartIsEmpty(charts[clusterMatrixIdx])) {
      clusterMatrixIdx = firstUsableMatrixIdx(charts)
    }
    var topNMax = Math.max(3, Math.min(orderedNodes.length, 16))
    clusterTopN = Math.max(3, Math.min(clusterTopN, topNMax))

    var controls = head.append('div').st({
      display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '6px',
    })

    var picker = controls.append('div').st({display: 'flex', alignItems: 'center', gap: '4px', flexWrap: 'wrap'})
    picker.append('span').text('Matrix').st({fontSize: '10px', color: '#888', textTransform: 'uppercase'})
    var pickSel = picker.append('div.link-type-buttons').st({display: 'flex'})
      .appendMany('div', charts.map((c, i) => ({i, title: c.title, empty: chartIsEmpty(c)})))
      .text(d => d.title.replace(' adjacency', '').replace(' similarity', ''))
      .st({
        fontSize: '11px', padding: '5px 8px', border: '1px solid #ccc',
        background: '#fff', cursor: 'pointer', marginLeft: '-1px', userSelect: 'none',
      })
      .at({title: d => d.empty
        ? (d.title + ' — no usable singular values (σ₁≈0); Cosine is used instead')
        : d.title})
      .on('click', (ev, d) => {
        if (d.empty) {
          var usable = firstUsableMatrixIdx(charts)
          setClusterMatrixIdx(usable, charts)
          return
        }
        setClusterMatrixIdx(d.i, charts)
      })
    function styleModeMat() {
      pickSel.st({
        background: d => d.i === clusterMatrixIdx ? '#000' : (d.empty ? '#F3F1EA' : '#fff'),
        color: d => d.i === clusterMatrixIdx ? '#fff' : (d.empty ? '#aaa' : '#333'),
        borderColor: d => d.i === clusterMatrixIdx ? '#000' : '#ccc',
        opacity: d => (d.empty && d.i !== clusterMatrixIdx) ? 0.65 : 1,
      })
    }
    styleModeMat()
    matrixPickerStyles.push(styleModeMat)

    var topCtrl = controls.append('div').st({
      display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px',
      background: '#FBFAF5', border: '1px solid #E4E2D8', borderRadius: '5px',
      padding: '4px 8px',
    })
    topCtrl.append('span').text('top per σ')
      .st({color: '#666', fontSize: '10px'})
    topCtrl.append('span').text('3').st({color: '#999', fontSize: '10px'})
    var topSlider = topCtrl.append('input')
      .at({type: 'range', min: 3, max: topNMax, step: 1, value: clusterTopN})
      .st({width: '110px', cursor: 'pointer'})
    topCtrl.append('span').text(String(topNMax))
      .st({color: '#999', fontSize: '10px', fontWeight: 600})
    var topLabel = topCtrl.append('span').text(String(clusterTopN))
      .st({fontWeight: 700, color: '#0D7377', minWidth: '18px'})
    topSlider.on('input', function () {
      clusterTopN = +this.value
      topLabel.text(String(clusterTopN))
      drawGraphBody()
    })

    // Position chips from node ctx counts (mass filled later from active component).
    var tokensForPos = (data.metadata && data.metadata.prompt_tokens) || []
    var ctxRows = []
    var byCtxTmp = {}
    orderedNodes.forEach(n => {
      var c = n.ctx_idx
      if (c == null) return
      if (!byCtxTmp[c]) {
        byCtxTmp[c] = {
          ctx_idx: c,
          token: tokensForPos[c],
          nodes: [],
          frac: 0,
        }
        ctxRows.push(byCtxTmp[c])
      }
      byCtxTmp[c].nodes.push(n)
    })
    ctxRows.sort((a, b) => a.ctx_idx - b.ctx_idx)
    var nAll = orderedNodes.length || 1
    ctxRows.forEach(r => { r.frac = r.nodes.length / nAll })
    renderPositionPicker(controls, ctxRows, tokensForPos)

    var body = wrap.append('div.svd-mode-graph-body')

    function drawGraphBody() {
      if (clusterSim) { clusterSim.stop(); clusterSim = null }
      body.html('')
      var ch = charts[clusterMatrixIdx]
      if (!ch) return

      var svdResult
      var viewNodes
      var focusNodes = focusCtxIdx == null
        ? orderedNodes
        : orderedNodes.filter(n => n && n.ctx_idx === focusCtxIdx)

      if (ch.svdResult?.sigmas?.length && ch.svdResult.sigmas[0] >= 1e-12) {
        // Prefer precomputed (backend) SVD. Position focus only filters which
        // features are drawn — never re-SVD a sparse among-token adjacency.
        viewNodes = focusNodes
        if (focusCtxIdx == null) {
          svdResult = ch.svdResult
        } else {
          svdResult = window.svdSpectrum.projectSvdToNodes(
            ch.svdResult, orderedNodes, viewNodes
          ) || ch.svdResult
        }
        if (focusCtxIdx != null) {
          var focusNote = body.append('div').st({
            fontSize: '10px', color: '#0D7377', marginBottom: '6px',
            background: '#F3FAFA', border: '1px solid #C5E3E4', borderRadius: '4px',
            padding: '5px 8px',
          })
          focusNote.html(
            'Position filter · ctx <b>' + focusCtxIdx + '</b> "'
            + tokenLabel(focusCtxIdx, tokensForPos) + '" · '
            + viewNodes.length + ' features · full-graph singular vectors (filtered)'
            + ' · <span style="cursor:pointer;text-decoration:underline" data-clear-pos="1">clear</span>'
          )
          focusNote.select('[data-clear-pos]').on('click', () => setFocusCtxIdx(null))
        }
      } else {
        var matrix = ch.matrix
        if (!matrix && focusCtxIdx != null) {
          // Last resort: build a small client matrix for the focused token subset.
          var focusIds = focusNodes.map(n => n.nodeId || n.node_id)
          if (ch.key === 'cosine' || (ch.title || '').toLowerCase().includes('cosine')) {
            var geom = (data.metadata || {}).svd_geom
            matrix = window.svdSpectrum.cosineFromGeom(focusIds, geom)
              || window.svdSpectrum.linkNeighborhoodCosine(focusIds, data.allLinks || data.links || [])
          } else {
            var attr = buildAttributionMatrices(focusIds, data.allLinks || data.links || [])
            matrix = ch.key === 'signed' ? attr.signed
              : ch.key === 'symmetric' ? attr.symmetric
              : attr.unsigned
          }
          viewNodes = focusNodes
        } else {
          var sliced = sliceByCtx(matrix, orderedNodes, focusCtxIdx)
          viewNodes = sliced.nodes
          matrix = sliced.matrix
        }
        if (focusCtxIdx != null) {
          var focusNote2 = body.append('div').st({
            fontSize: '10px', color: '#0D7377', marginBottom: '6px',
            background: '#F3FAFA', border: '1px solid #C5E3E4', borderRadius: '4px',
            padding: '5px 8px',
          })
          focusNote2.html(
            'Position filter · ctx <b>' + focusCtxIdx + '</b> "'
            + tokenLabel(focusCtxIdx, tokensForPos) + '" · '
            + viewNodes.length + ' features · SVD recomputed on this subset'
            + ' · <span style="cursor:pointer;text-decoration:underline" data-clear-pos="1">clear</span>'
          )
          focusNote2.select('[data-clear-pos]').on('click', () => setFocusCtxIdx(null))
        }
        if (viewNodes.length < 2) {
          body.append('div')
            .text('Need at least 2 features at this token position to build a graph.')
            .st({fontSize: '11px', color: '#999', padding: '12px 0'})
          return
        }
        svdResult = window.svdSpectrum.svd(matrix, {
          k: Math.min(spectrumKMax, Math.max(k, 8), viewNodes.length),
        })
      }

      if (viewNodes.length < 2) {
        body.append('div')
          .text('Need at least 2 features at this token position to build a graph.')
          .st({fontSize: '11px', color: '#999', padding: '12px 0'})
        return
      }
      if (!svdResult?.sigmas?.length || svdResult.sigmas[0] < 1e-12) {
        var cosIdx = firstUsableMatrixIdx(charts)
        if (cosIdx !== clusterMatrixIdx && !chartIsEmpty(charts[cosIdx])) {
          setClusterMatrixIdx(cosIdx, charts)
          return
        }
        var emptyMsg = body.append('div').st({
          fontSize: '11px', color: '#a23', padding: '12px 0',
          display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '8px',
        })
        emptyMsg.append('span')
          .text(focusCtxIdx != null
            ? 'All σ ≈ 0 among features at this token position.'
            : 'All σ ≈ 0 for this matrix — empty adjacency among selected nodes.')
        if (!chartIsEmpty(charts[0]) && clusterMatrixIdx !== 0) {
          emptyMsg.append('button')
            .text('Switch to Cosine')
            .st({
              fontSize: '11px', fontWeight: 600, cursor: 'pointer',
              border: '1px solid #0D7377', background: '#F3FAFA', color: '#0D7377',
              borderRadius: '5px', padding: '4px 8px',
            })
            .on('click', () => setClusterMatrixIdx(0, charts))
        } else if (svdBundleStatus !== 'ready') {
          emptyMsg.append('span')
            .text('(backend SVD not loaded — run: interpretability-graph export-svd)')
            .st({color: '#888'})
        }
        return
      }
      var kView = Math.min(k, (svdResult.sigmas || []).length, viewNodes.length)
      var graph = window.svdSpectrum.buildModeGraph(svdResult, viewNodes, kView, clusterTopN)
      if (!graph.modes.length) {
        body.append('div').text('No modes to plot for this matrix.')
          .st({fontSize: '11px', color: '#999', padding: '16px 0'})
        return
      }

      var legend = body.append('div').st({
        display: 'flex', flexWrap: 'wrap', gap: '8px', marginBottom: '6px',
      })
      graph.modes.forEach(m => {
        var item = legend.append('div').st({
          display: 'flex', alignItems: 'center', gap: '4px', fontSize: '10px',
          fontFamily: 'ui-monospace, monospace', color: '#444',
        })
        item.append('span').st({
          width: '10px', height: '10px', borderRadius: '50%',
          background: modeColors[m.r % modeColors.length], display: 'inline-block',
        })
        item.append('span').text(m.label + '=' + m.sigma.toFixed(3))
      })

      var width = Math.max(640, (body.node()?.clientWidth || 800) - 4)
      // Grow the canvas when many features are shown so clusters have room.
      var height = Math.min(560, 280 + Math.ceil(graph.features.length / Math.max(1, graph.modes.length)) * 36)
      var margin = {top: 28, right: 16, bottom: 16, left: 16}
      var innerW = width - margin.left - margin.right
      var innerH = height - margin.top - margin.bottom

      var svg = body.append('svg')
        .at({width: '100%', viewBox: `0 0 ${width} ${height}`})
        .st({display: 'block', background: '#FBFAF5', borderRadius: '4px', border: '1px solid #EFEDE4'})
      var g = svg.append('g').at({transform: `translate(${margin.left},${margin.top})`})

      // Seed positions: mode hubs along the top; features under their primary mode.
      var colW = innerW / Math.max(1, graph.modes.length)
      graph.modes.forEach((m, i) => {
        m.x = (i + 0.5) * colW
        m.y = 18
        m.fx = m.x
        m.fy = m.y
      })
      var byMode = {}
      graph.modes.forEach(m => { byMode[m.r] = [] })
      graph.features.forEach(f => {
        ;(byMode[f.primaryMode] || (byMode[f.primaryMode] = [])).push(f)
      })
      Object.keys(byMode).forEach(rk => {
        var list = byMode[rk]
        var mode = graph.modes[+rk]
        if (!mode) return
        list.forEach((f, j) => {
          var n = list.length
          var spread = Math.min(colW * 0.72, 28 * n)
          f.x = mode.x + (n === 1 ? 0 : ((j / (n - 1)) - 0.5) * spread)
          f.y = 110 + (j % 3) * 42 + ((j * 17) % 23)
        })
      })

      var nodes = graph.modes.concat(graph.features)
      var idToNode = {}
      nodes.forEach(n => { idToNode[n.id] = n })
      var links = graph.links.map(l => ({
        ...l,
        source: idToNode[l.source],
        target: idToNode[l.target],
      })).filter(l => l.source && l.target)

      var maxAbs = d3.max(links, d => d.abs) || 1
      var linkSel = g.append('g').selectAll('line').data(links).join('line')
        .at({
          stroke: d => modeColors[d.mode % modeColors.length],
          strokeOpacity: 0.45,
          strokeWidth: d => 1 + 3.5 * (d.abs / maxAbs),
        })

      var modeSel = g.append('g').selectAll('g.mode-hub').data(graph.modes).join('g')
        .attr('class', 'mode-hub')
        .st({cursor: 'default'})
      modeSel.append('circle')
        .at({
          r: d => 14 + 10 * (d.sigma / (graph.modes[0].sigma || 1)),
          fill: d => modeColors[d.r % modeColors.length],
          stroke: '#fff',
          strokeWidth: 2,
        })
      modeSel.append('text')
        .text(d => d.label)
        .at({textAnchor: 'middle', dy: '0.35em', fill: '#fff', fontSize: 11, fontWeight: 700})
        .st({pointerEvents: 'none', fontFamily: 'ui-monospace, monospace'})

      var featSel = g.append('g').selectAll('g.feat-node').data(graph.features).join('g')
        .attr('class', 'feat-node')
        .st({cursor: 'pointer'})
        .classed('is-selected', d => d.nodeId === visState.clickedId)
        .on('click', (ev, d) => {
          ev.preventDefault()
          ev.stopPropagation()
          focusSvdFeature(d.node)
          featSel.classed('is-selected', x => x.nodeId === visState.clickedId)
          styleFeatNodes()
        })
        .on('mouseenter', (ev, d) => {
          if (d.node) utilCg.hoverFeature(visState, renderAll, d.node)
          highlightNeighborhood(d)
        })
        .on('mouseleave', () => {
          utilCg.unHoverFeature(visState, renderAll)
          clearHighlight()
        })

      featSel.append('circle')
        .at({
          r: 9,
          fill: d => modeColors[d.primaryMode % modeColors.length],
          stroke: '#fff',
          strokeWidth: 1.5,
          fillOpacity: 0.9,
        })
      featSel.append('title')
        .text(d => {
          var parts = Object.keys(d.loadings).map(r =>
            'σ' + (+r + 1) + '=' + (+d.loadings[r] >= 0 ? '+' : '') + (+d.loadings[r]).toFixed(2)
          )
          return featureLabel(d.node) + '\n' + parts.join(' · ')
        })
      featSel.append('text')
        .text(d => {
          var t = featureLabel(d.node)
          return t.length > 18 ? t.slice(0, 16) + '…' : t
        })
        .at({
          textAnchor: 'middle', dy: 22, fill: '#333', fontSize: 9,
        })
        .st({pointerEvents: 'none'})

      function styleFeatNodes() {
        featSel.select('circle')
          .at({
            stroke: d => d.nodeId === visState.clickedId ? '#111' : '#fff',
            strokeWidth: d => d.nodeId === visState.clickedId ? 2.5 : 1.5,
            r: d => d.nodeId === visState.clickedId ? 11 : 9,
          })
      }
      styleFeatNodes()

      function highlightNeighborhood(feat) {
        var touch = new Set([feat.id])
        links.forEach(l => {
          if (l.target.id === feat.id) touch.add(l.source.id)
        })
        linkSel.at({
          strokeOpacity: d => (d.target.id === feat.id ? 0.9 : 0.08),
          strokeWidth: d => d.target.id === feat.id
            ? 1.5 + 4 * (d.abs / maxAbs) : 0.6,
        })
        modeSel.st({opacity: d => touch.has(d.id) ? 1 : 0.25})
        featSel.st({opacity: d => d.id === feat.id || touch.has(d.id) ? 1 : 0.2})
      }
      function clearHighlight() {
        linkSel.at({
          strokeOpacity: 0.45,
          strokeWidth: d => 1 + 3.5 * (d.abs / maxAbs),
        })
        modeSel.st({opacity: 1})
        featSel.st({opacity: 1})
      }

      clusterSim = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(d => d.id)
          .distance(d => 55 + 30 * (1 - d.abs / maxAbs))
          .strength(0.55))
        .force('charge', d3.forceManyBody().strength(d => d.type === 'mode' ? -40 : -120))
        .force('collide', d3.forceCollide().radius(d => d.type === 'mode' ? 28 : 22))
        .force('y', d3.forceY(d => {
          if (d.type === 'mode') return 18
          return 130 + (d.primaryMode % 2) * 20
        }).strength(d => d.type === 'mode' ? 1 : 0.18))
        .force('x', d3.forceX(d => {
          if (d.type === 'mode') return d.fx
          var home = graph.modes[d.primaryMode]
          return home ? home.x : innerW / 2
        }).strength(d => d.type === 'mode' ? 1 : 0.22))
        .on('tick', () => {
          linkSel
            .at({
              x1: d => d.source.x, y1: d => d.source.y,
              x2: d => d.target.x, y2: d => d.target.y,
            })
          modeSel.at({transform: d => `translate(${d.x},${d.y})`})
          featSel.at({transform: d => `translate(${d.x},${d.y})`})
        })

      // Soft drag on features only
      featSel.call(d3.drag()
        .on('start', (ev, d) => {
          if (!ev.active) clusterSim.alphaTarget(0.25).restart()
          d.fx = d.x; d.fy = d.y
        })
        .on('drag', (ev, d) => { d.fx = ev.x; d.fy = ev.y })
        .on('end', (ev, d) => {
          if (!ev.active) clusterSim.alphaTarget(0)
          d.fx = null; d.fy = null
        }))
    }

    matrixViewRedraws.push(drawGraphBody)
    drawGraphBody()
  }

  /**
   * Token-position view of one singular component.
   * Hypothesis: high-|u| features for a mode cluster on one token position
   * rather than spanning positions. Shows:
   *  - |u|² mass by token (with decision-boundary columns)
   *  - node×node outer product u uᵀ reordered by ctx_idx (block structure)
   *  - scatter of features in token columns
   */
  function renderComponentTokenView(container, charts, orderedNodes, k) {
    var tokens = (data.metadata && data.metadata.prompt_tokens) || []
    var wrap = container.append('div.svd-token-component').st({
      marginTop: '12px', background: '#fff', border: '1px solid #E4E2D8',
      borderRadius: '6px', padding: '10px 12px', boxSizing: 'border-box',
    })

    var head = wrap.append('div').st({
      display: 'flex', flexWrap: 'wrap', alignItems: 'flex-start',
      gap: '10px', justifyContent: 'space-between', marginBottom: '8px',
    })
    var titleCol = head.append('div')
    titleCol.append('div').text('Component × token position')
      .st({fontWeight: 600, fontSize: '12px', color: '#222'})
    titleCol.append('div')
      .text('Hypothesis: high-loading features of one singular component tend to share a single prompt token position (ctx), rather than forming a position-spanning cluster.')
      .st({fontSize: '10px', color: '#666', marginTop: '2px', maxWidth: '720px', lineHeight: '1.35'})
    var mathIntro = titleCol.append('div').st({
      marginTop: '8px', background: '#FBFAF5', border: '1px solid #E4E2D8',
      borderRadius: '5px', padding: '10px 12px', maxWidth: '820px',
    })
    mathIntro.append('div').text('Index · how to read this')
      .st({
        fontSize: '10px', fontWeight: 600, color: '#666',
        textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: '8px',
      })

    // Equation + one-line index note (complements the math, not a restatement).
    function eqNote(parent, tex, note) {
      var row = parent.append('div').st({
        display: 'flex', flexWrap: 'wrap', alignItems: 'baseline',
        gap: '6px 14px', marginBottom: '6px',
      })
      var eq = row.append('div').st({flex: '1 1 280px', minWidth: '200px'})
      renderLatex(eq, tex, true)
      eq.selectAll('.svd-eq').st({textAlign: 'left', minHeight: '0', marginBottom: '0'})
      eq.selectAll('.katex').st({fontSize: '1.05em'})
      if (note) {
        row.append('div').text(note)
          .st({
            flex: '1 1 220px', fontSize: '10px', color: '#666', lineHeight: '1.35',
            borderLeft: '2px solid #E4E2D8', paddingLeft: '10px',
          })
      }
    }

    eqNote(mathIntro, 'M = U\\Sigma V^{\\top}',
      'SVD of the selected matrix (cosine / adjacency). Modes = columns of U.')
    eqNote(
      mathIntro,
      '\\mathbf{u}_r \\in \\mathbb{R}^{n},\\quad '
        + '\\lVert\\mathbf{u}_r\\rVert_2 = 1,\\quad '
        + '\\sigma_r \\ge 0',
      'uᵣ = soft weights over feature-nodes. σᵣ = how strong that mode is.'
    )
    eqNote(
      mathIntro,
      'u_{r,i}'
        + '\\text{ = loading of node } i'
        + '\\text{ at token position }'
        + '\\mathrm{ctx}_i',
      'Each node lives on one prompt token. |u| large ⇒ that feature drives the mode.'
    )
    eqNote(
      mathIntro,
      'p_t = \\sum_{i=1}^{n} u_{r,i}^{2}\\,\\mathbf{1}_{\\{\\mathrm{ctx}_i=t\\}}',
      'Share of mode energy on token t. Bars = these heights. Sum_t p_t = 1.'
    )
    eqNote(
      mathIntro,
      't^{\\star} = \\operatorname*{arg\\,max}_{t} p_t',
      'The token the mode “points at.” Teal in bars/scatter.'
    )
    eqNote(
      mathIntro,
      'H = -\\sum_{t} p_t \\log p_t'
        + '\\quad\\text{ and }\\quad'
        + ' e^{H}\\text{ = effective number of positions}',
      'eᴴ≈1 → one token. eᴴ≈k → roughly k tokens share the mode.'
    )
    eqNote(
      mathIntro,
      '\\text{Concentrated if } p_{t^{\\star}} \\ge 0.6',
      'Rule of thumb: ≥60% on one token ⇒ not position-spanning.'
    )
    eqNote(
      mathIntro,
      '(UU^{\\top})_{ij} = u_{r,i} u_{r,j}'
        + '\\quad\\text{ (heatmap; nodes blocked by }\\mathrm{ctx}\\text{ or layer)}',
      'A bright diagonal block ⇒ those features co-load; which block depends on the Group by toggle.'
    )
    eqNote(
      mathIntro,
      '\\text{For symmetric }M\\text{: rank-1 piece }'
        + '\\approx \\sigma_r\\,\\mathbf{u}_r\\mathbf{u}_r^{\\top}',
      'Heatmap is u uᵀ (structure). Full rank-1 piece also scales by σᵣ.'
    )

    clusterMatrixIdx = Math.max(0, Math.min(clusterMatrixIdx, charts.length - 1))
    if (chartIsEmpty(charts[clusterMatrixIdx])) {
      clusterMatrixIdx = firstUsableMatrixIdx(charts)
    }
    componentIdx = Math.max(0, Math.min(componentIdx, Math.max(0, k - 1)))

    var controls = head.append('div').st({
      display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '6px',
    })

    var matRow = controls.append('div').st({display: 'flex', alignItems: 'center', gap: '4px', flexWrap: 'wrap'})
    matRow.append('span').text('Matrix').st({fontSize: '10px', color: '#888', textTransform: 'uppercase'})
    var matSel = matRow.append('div.link-type-buttons').st({display: 'flex'})
      .appendMany('div', charts.map((c, i) => ({i, title: c.title, empty: chartIsEmpty(c)})))
      .text(d => d.title.replace(' adjacency', '').replace(' similarity', ''))
      .st({
        fontSize: '11px', padding: '5px 8px', border: '1px solid #ccc',
        background: '#fff', cursor: 'pointer', marginLeft: '-1px', userSelect: 'none',
      })
      .at({title: d => d.empty
        ? (d.title + ' — no usable singular values (σ₁≈0); Cosine is used instead')
        : d.title})
      .on('click', (ev, d) => {
        if (d.empty) {
          var usable = firstUsableMatrixIdx(charts)
          setClusterMatrixIdx(usable, charts)
          return
        }
        setClusterMatrixIdx(d.i, charts)
      })
    function styleMat() {
      matSel.st({
        background: d => d.i === clusterMatrixIdx ? '#000' : (d.empty ? '#F3F1EA' : '#fff'),
        color: d => d.i === clusterMatrixIdx ? '#fff' : (d.empty ? '#aaa' : '#333'),
        borderColor: d => d.i === clusterMatrixIdx ? '#000' : '#ccc',
        opacity: d => (d.empty && d.i !== clusterMatrixIdx) ? 0.65 : 1,
      })
    }
    styleMat()
    matrixPickerStyles.push(styleMat)

    var compRow = controls.append('div').st({display: 'flex', alignItems: 'center', gap: '4px', flexWrap: 'wrap'})
    compRow.append('span').text('Component').st({fontSize: '10px', color: '#888', textTransform: 'uppercase'})
    var compOpts = d3.range(Math.max(1, k)).map(i => ({i, label: 'σ' + (i + 1)}))
    var compSel = compRow.append('div.link-type-buttons').st({display: 'flex'})
      .appendMany('div', compOpts)
      .text(d => d.label)
      .st({
        fontSize: '11px', padding: '5px 8px', border: '1px solid #ccc',
        background: '#fff', cursor: 'pointer', marginLeft: '-1px', userSelect: 'none',
        fontFamily: 'ui-monospace, monospace',
      })
      .on('click', (ev, d) => {
        componentIdx = d.i
        styleComp()
        drawBody()
      })
    function styleComp() {
      compSel.st({
        background: d => d.i === componentIdx ? '#0D7377' : '#fff',
        color: d => d.i === componentIdx ? '#fff' : '#333',
        borderColor: d => d.i === componentIdx ? '#0D7377' : '#ccc',
      })
    }
    styleComp()
    // Position chips use mass fractions from the active component (filled in drawBody
    // via a placeholder host that we rebuild — chips live in controls and are
    // created once from node ctx counts, then restyled when focus changes).
    var ctxRowsTok = []
    var byCtxTok = {}
    orderedNodes.forEach(n => {
      var c = n.ctx_idx
      if (c == null) return
      if (!byCtxTok[c]) {
        byCtxTok[c] = {ctx_idx: c, token: tokens[c], nodes: [], frac: 0}
        ctxRowsTok.push(byCtxTok[c])
      }
      byCtxTok[c].nodes.push(n)
    })
    ctxRowsTok.sort((a, b) => a.ctx_idx - b.ctx_idx)
    var nTokAll = orderedNodes.length || 1
    ctxRowsTok.forEach(r => { r.frac = r.nodes.length / nTokAll })
    renderPositionPicker(controls, ctxRowsTok, tokens)

    var body = wrap.append('div.svd-token-component-body')

    function drawBody() {
      body.html('')
      var ch = charts[clusterMatrixIdx]
      if (!ch || !orderedNodes.length) return
      // Always prefer precomputed SVD (backend bundle). Never require a dense
      // client matrix for the token-position panel.
      var svdResult = ch.svdResult
      if ((!svdResult?.sigmas?.length || svdResult.sigmas[0] < 1e-12) && ch.matrix) {
        svdResult = window.svdSpectrum.svd(ch.matrix, {
          k: Math.min(spectrumKMax, Math.max(k, 8), orderedNodes.length),
        })
        ch.svdResult = svdResult
      }
      if (!svdResult) {
        body.append('div').text('No SVD available for this matrix.')
          .st({fontSize: '11px', color: '#999', padding: '12px 0'})
        return
      }
      var r = Math.min(componentIdx, Math.max(0, (svdResult.sigmas || []).length - 1))
      if (!svdResult.sigmas?.length || svdResult.sigmas[0] < 1e-12) {
        var cosIdx = firstUsableMatrixIdx(charts)
        if (cosIdx !== clusterMatrixIdx && !chartIsEmpty(charts[cosIdx])) {
          setClusterMatrixIdx(cosIdx, charts)
          return
        }
        var emptyMsg = body.append('div').st({
          fontSize: '11px', color: '#a23', padding: '12px 0',
          display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '8px',
        })
        emptyMsg.append('span')
          .text('All σ ≈ 0 for this matrix — nothing to position-cluster.')
        if (!chartIsEmpty(charts[0]) && clusterMatrixIdx !== 0) {
          emptyMsg.append('button')
            .text('Switch to Cosine')
            .st({
              fontSize: '11px', fontWeight: 600, cursor: 'pointer',
              border: '1px solid #0D7377', background: '#F3FAFA', color: '#0D7377',
              borderRadius: '5px', padding: '4px 8px',
            })
            .on('click', () => setClusterMatrixIdx(0, charts))
        } else if (svdBundleStatus !== 'ready') {
          emptyMsg.append('span')
            .text('(backend SVD bundle not loaded — run interpretability-graph export-svd)')
            .st({color: '#888'})
        }
        return
      }
      var vector = svdResult.vectors[r] || []
      var sigma = svdResult.sigmas[r] || 0
      var statsNodes = orderedNodes
      var statsVector = vector
      if (focusCtxIdx != null) {
        // Keep full-graph σ for the verdict, but mass/scatter can emphasize focus.
        // Stats still use the full vector so p_t / t★ stay meaningful.
        statsNodes = orderedNodes
        statsVector = vector
      }
      var stats = window.svdSpectrum.componentPositionStats(statsVector, statsNodes, tokens)
      // Refresh chip labels with p_t mass when available
      stats.rows.forEach(row => {
        var hit = ctxRowsTok.find(x => x.ctx_idx === row.ctx_idx)
        if (hit) hit.frac = row.frac
      })
      positionPickerStyles.forEach(fn => { try { fn() } catch (e) {} })

      // —— Verdict strip ——
      var verdict = body.append('div').st({
        display: 'flex', flexWrap: 'wrap', gap: '10px 18px', alignItems: 'baseline',
        padding: '8px 10px', marginBottom: '10px', borderRadius: '5px',
        background: stats.concentrated ? '#E8F5EE' : '#F8F0E8',
        border: '1px solid ' + (stats.concentrated ? '#B7DCC4' : '#E8D5C0'),
      })
      verdict.append('div')
        .text(stats.concentrated
          ? 'Supports concentration — not token-spanning'
          : 'Spans positions — not a single-token cluster')
        .st({
          fontWeight: 700, fontSize: '12px',
          color: stats.concentrated ? '#2F6F36' : '#8B4513',
        })
      var tokLabel = stats.dominant
        ? ('"' + String(util.ppToken ? util.ppToken(stats.dominant.token) : stats.dominant.token) + '"')
        : '—'
      verdict.append('div')
        .html(
          '<span style="font-family:ui-monospace,monospace;font-weight:700;color:#0D7377">σ'
          + (r + 1) + '=' + sigma.toFixed(3) + '</span>'
          + ' · t* = <b>' + (stats.dominant ? stats.dominant.ctx_idx : '—') + '</b> '
          + tokLabel
          + ' · p<sub>t*</sub> = <b>' + (100 * stats.dominantFrac).toFixed(0) + '%</b>'
          + ' · #\{t : p<sub>t</sub>&gt;0\} = <b>' + stats.nPositions + '</b>'
          + ' · e<sup>H</sup> = <b>' + stats.effectivePositions.toFixed(2) + '</b>'
          + (focusCtxIdx != null
            ? (' · <span style="color:#0D7377">focusing ctx ' + focusCtxIdx
              + ' "' + tokenLabel(focusCtxIdx, tokens) + '"</span>')
            : ' · <span style="color:#888">click a position chip or bar to focus</span>')
        )
        .st({fontSize: '11px', color: '#444'})
      verdict.append('div')
        .text(
          stats.concentrated
            ? 'Index: one token owns the mode (≥60%). Spillover = eᴴ − 1 roughly.'
            : 'Index: mass split across tokens — mode is position-spanning, not a single-token cluster.'
        )
        .st({fontSize: '10px', color: '#666', flex: '1 1 100%', marginTop: '2px', lineHeight: '1.35'})

      // —— Color legend + math reminders ——
      var legend = body.append('div').st({
        display: 'flex', flexWrap: 'wrap', gap: '14px 22px', alignItems: 'flex-start',
        padding: '8px 10px', marginBottom: '10px', borderRadius: '5px',
        background: '#FBFAF5', border: '1px solid #E4E2D8', fontSize: '11px', color: '#444',
      })
      var posLeg = legend.append('div').st({display: 'flex', flexDirection: 'column', gap: '4px', flex: '1 1 280px'})
      posLeg.append('div').text('Token position encoding (bars & scatter)')
        .st({fontSize: '10px', fontWeight: 600, color: '#666', textTransform: 'uppercase', letterSpacing: '0.03em'})
      var posRow = posLeg.append('div').st({display: 'flex', flexDirection: 'column', gap: '6px'})
      var tealPos = posRow.append('div').st({display: 'flex', alignItems: 'center', gap: '6px'})
      tealPos.append('span').st({
        width: '12px', height: '12px', borderRadius: '3px', background: '#0D7377',
        border: '1px solid rgba(0,0,0,0.12)', flexShrink: 0,
      })
      renderLatex(tealPos, '\\text{Teal: }\\mathrm{ctx}_i = t^{\\star}', false)
      var oraPos = posRow.append('div').st({display: 'flex', alignItems: 'center', gap: '6px'})
      oraPos.append('span').st({
        width: '12px', height: '12px', borderRadius: '3px', background: '#C45C26',
        border: '1px solid rgba(0,0,0,0.12)', flexShrink: 0,
      })
      renderLatex(oraPos, '\\text{Orange: }\\mathrm{ctx}_i \\neq t^{\\star}', false)
      posLeg.append('div')
        .text('Index: teal = where the mode lives; orange = everything else.')
        .st({fontSize: '10px', color: '#888', marginTop: '4px', lineHeight: '1.3'})
      posRow.selectAll('.svd-eq').st({textAlign: 'left', minHeight: '0', marginBottom: '0'})
      posRow.selectAll('.katex').st({fontSize: '1em'})

      var heatLeg = legend.append('div').st({display: 'flex', flexDirection: 'column', gap: '4px', flex: '1 1 300px'})
      heatLeg.append('div').text('Heatmap cell color')
        .st({fontSize: '10px', fontWeight: 600, color: '#666', textTransform: 'uppercase', letterSpacing: '0.03em'})
      var heatRow = heatLeg.append('div').st({display: 'flex', flexDirection: 'column', gap: '6px'})
      var tealH = heatRow.append('div').st({display: 'flex', alignItems: 'center', gap: '6px'})
      tealH.append('span').st({
        width: '12px', height: '12px', borderRadius: '3px', background: '#0D7377',
        border: '1px solid rgba(0,0,0,0.12)', flexShrink: 0,
      })
      renderLatex(tealH, '\\text{Teal: }u_{r,i}u_{r,j} > 0', false)
      var oraH = heatRow.append('div').st({display: 'flex', alignItems: 'center', gap: '6px'})
      oraH.append('span').st({
        width: '12px', height: '12px', borderRadius: '3px', background: '#C45C26',
        border: '1px solid rgba(0,0,0,0.12)', flexShrink: 0,
      })
      renderLatex(oraH, '\\text{Orange: }u_{r,i}u_{r,j} < 0', false)
      renderLatex(
        heatRow,
        '\\text{Intensity }\\propto\\lvert u_{r,i}u_{r,j}\\rvert'
          + '\\,/\\,\\max_{a,b}\\lvert u_{r,a}u_{r,b}\\rvert',
        false
      )
      heatLeg.append('div')
        .text('Index: look for a diagonal block at t★ — co-loading within one token. Orange = opposite-sign loadings (anti-aligned on this mode).')
        .st({fontSize: '10px', color: '#888', marginTop: '4px', lineHeight: '1.3'})
      var nPosU = 0, nNegU = 0, massNegU = 0
      for (var ui = 0; ui < vector.length; ui++) {
        var uv = vector[ui] || 0
        if (uv > 1e-12) nPosU++
        else if (uv < -1e-12) { nNegU++; massNegU += uv * uv }
      }
      var signNote = heatLeg.append('div').st({
        fontSize: '10px', marginTop: '6px', lineHeight: '1.35',
        padding: '5px 7px', borderRadius: '4px',
        background: nNegU === 0 ? '#F3FAFA' : '#FBF6F1',
        border: '1px solid ' + (nNegU === 0 ? '#C5E3E4' : '#E8D5C0'),
        color: '#444',
      })
      if (nNegU === 0) {
        signNote.html(
          'This σ<sub>r</sub> has <b>no negative loadings</b> (all u<sub>i</sub>≥0) → '
          + '<span style="color:#C45C26">orange cells cannot appear</span>. '
          + 'Expected for <b>Unsigned</b> / <b>Symmetric</b> (nonnegative M; Perron-like leading vector). '
          + 'Switch Matrix to <b>Cosine</b> or <b>Signed</b> to see opposite-sign structure.'
        )
      } else {
        signNote.html(
          'Loadings: <b>' + nPosU + '</b> positive, <b>' + nNegU + '</b> negative'
          + ' (negative mass ' + (100 * massNegU).toFixed(0) + '%). '
          + 'Orange cells = products u<sub>i</sub>u<sub>j</sub>&lt;0 — coloring is valid.'
        )
      }
      heatRow.selectAll('.svd-eq').st({textAlign: 'left', minHeight: '0', marginBottom: '0'})
      heatRow.selectAll('.katex').st({fontSize: '1em'})

      var defBox = legend.append('div').st({flex: '1 1 100%', marginTop: '4px'})
      defBox.append('div').text('Quantities in this panel')
        .st({fontSize: '10px', fontWeight: 600, color: '#666', textTransform: 'uppercase', letterSpacing: '0.03em', marginBottom: '4px'})
      renderLatex(
        defBox,
        '\\mathbf{u}_r'
          + '\\text{ = left singular vector for }'
          + '\\sigma_r'
          + '\\text{ of selected }'
          + 'M',
        true
      )
      renderLatex(
        defBox,
        'p_t = \\sum_{i=1}^{n} u_{r,i}^{2}\\,\\mathbf{1}_{\\{\\mathrm{ctx}_i=t\\}}'
          + '\\quad\\text{ (bar height)}',
        true
      )
      renderLatex(
        defBox,
        't^{\\star} = \\operatorname*{arg\\,max}_{t} p_t'
          + '\\quad\\text{ (teal)}',
        true
      )
      renderLatex(
        defBox,
        '\\text{Scatter: }'
          + 'x=\\mathrm{ctx}_i,\\;'
          + 'y=\\mathrm{layer}_i,\\;'
          + '\\text{radius}\\propto\\lvert u_{r,i}\\rvert',
        true
      )
      renderLatex(
        defBox,
        '\\text{Heatmap: }'
          + '(UU^{\\top})_{ij}=u_{r,i}u_{r,j}',
        true
      )
      defBox.selectAll('.svd-eq').st({textAlign: 'left'})

      var panels = body.append('div').st({
        display: 'flex', flexWrap: 'wrap', gap: '12px', alignItems: 'stretch',
      })

      // —— Mass by token position ——
      var massBox = panels.append('div').st({
        flex: '1 1 220px', minWidth: '200px', maxWidth: '320px',
        border: '1px solid #E4E2D8', borderRadius: '5px', padding: '8px',
        background: '#FBFAF5',
      })
      massBox.append('div').text('Mass by token position · click a bar to focus')
        .st({fontSize: '11px', fontWeight: 600, marginBottom: '2px'})
      massBox.append('div')
        .text('Index: tall teal bar = mode parks on that word; flat orange spread = spans the prompt.')
        .st({fontSize: '10px', color: '#888', marginBottom: '4px', lineHeight: '1.3'})
      var massCap = massBox.append('div').st({marginBottom: '6px'})
      renderLatex(
        massCap,
        'p_t = \\sum_{i=1}^{n} u_{r,i}^{2}\\,\\mathbf{1}_{\\{\\mathrm{ctx}_i=t\\}}'
          + '\\quad\\text{ (teal }=t^{\\star}\\text{)}',
        true
      )
      massCap.selectAll('.svd-eq').st({textAlign: 'left', minHeight: '0'})
      massCap.selectAll('.katex').st({fontSize: '1.05em'})
      var mw = 280, mh = 140
      var mm = {top: 8, right: 8, bottom: 48, left: 36}
      var msvg = massBox.append('svg').at({width: '100%', viewBox: `0 0 ${mw} ${mh}`})
      var mg = msvg.append('g').at({transform: `translate(${mm.left},${mm.top})`})
      var innerWm = mw - mm.left - mm.right
      var innerHm = mh - mm.top - mm.bottom
      var ctxDomain = stats.rows.map(d => d.ctx_idx)
      var xMass = d3.scaleBand().domain(ctxDomain).range([0, innerWm]).padding(0.2)
      var yMass = d3.scaleLinear().domain([0, d3.max(stats.rows, d => d.frac) || 1]).range([innerHm, 0]).nice()
      mg.appendMany('rect', stats.rows).at({
        x: d => xMass(d.ctx_idx),
        y: d => yMass(d.frac),
        width: xMass.bandwidth(),
        height: d => Math.max(0, innerHm - yMass(d.frac)),
        fill: d => d === stats.dominant ? '#0D7377' : '#C45C26',
        fillOpacity: d => {
          if (focusCtxIdx == null) return d === stats.dominant ? 1 : 0.55
          return d.ctx_idx === focusCtxIdx ? 1 : 0.18
        },
        stroke: d => d.ctx_idx === focusCtxIdx ? '#111' : (d === stats.dominant ? '#085456' : 'none'),
        strokeWidth: d => d.ctx_idx === focusCtxIdx ? 2 : 1,
        cursor: 'pointer',
      })
        .on('click', (ev, d) => {
          ev.stopPropagation()
          setFocusCtxIdx(d.ctx_idx)
        })
        .append('title')
        .text(d => 'Click to focus ctx ' + d.ctx_idx + ' · p=' + (100 * d.frac).toFixed(0) + '%')
      // decision-boundary ticks between positions
      ctxDomain.slice(0, -1).forEach((ctx, i) => {
        var x = xMass(ctx) + xMass.bandwidth() + xMass.padding() * xMass.step() / 2
        if (!isFinite(x)) return
        mg.append('line').at({
          x1: xMass(ctx) + xMass.bandwidth(), x2: xMass(ctx) + xMass.bandwidth(),
          y1: 0, y2: innerHm, stroke: '#D0CEC4', strokeDasharray: '2,2',
        })
      })
      mg.append('g').at({transform: `translate(0,${innerHm})`})
        .call(d3.axisBottom(xMass).tickFormat(ctx => {
          var row = stats.rows.find(d => d.ctx_idx === ctx)
          var t = row ? String(util.ppToken ? util.ppToken(row.token) : row.token) : String(ctx)
          return t.length > 6 ? t.slice(0, 5) + '…' : t
        }))
        .selectAll('text')
        .st({fontSize: '8px', fill: '#666'})
        .attr('transform', 'rotate(-35)')
        .attr('text-anchor', 'end')
      mg.append('g').call(d3.axisLeft(yMass).ticks(3).tickFormat(d3.format('.0%')))
        .selectAll('text').st({fontSize: '8px', fill: '#888'})
      mg.selectAll('.domain').st({stroke: '#ccc'})

      // —— Scatter: features in token columns ——
      var scatBox = panels.append('div').st({
        flex: '1 1 280px', minWidth: '240px',
        border: '1px solid #E4E2D8', borderRadius: '5px', padding: '8px',
        background: '#FBFAF5',
      })
      scatBox.append('div').text('Features in token columns')
        .st({fontSize: '11px', fontWeight: 600, marginBottom: '2px'})
      scatBox.append('div')
        .text('Index: columns = tokens; big dots = high |u|. A pile in one column ⇒ concentrated.')
        .st({fontSize: '10px', color: '#888', marginBottom: '4px', lineHeight: '1.3'})
      var scatCap = scatBox.append('div').st({marginBottom: '6px'})
      renderLatex(
        scatCap,
        'x=\\mathrm{ctx}_i,\\;'
          + 'y=\\mathrm{layer}_i,\\;'
          + '\\text{radius}\\propto\\lvert u_{r,i}\\rvert',
        true
      )
      scatCap.selectAll('.svd-eq').st({textAlign: 'left', minHeight: '0'})
      scatCap.selectAll('.katex').st({fontSize: '1.05em'})
      var sw = 360, sh = 180
      var sm = {top: 12, right: 12, bottom: 48, left: 40}
      var ssvg = scatBox.append('svg').at({width: '100%', viewBox: `0 0 ${sw} ${sh}`})
      var sg = ssvg.append('g').at({transform: `translate(${sm.left},${sm.top})`})
      var innerWs = sw - sm.left - sm.right
      var innerHs = sh - sm.top - sm.bottom
      var pts = orderedNodes.map((node, i) => ({
        node, i, loading: vector[i] || 0, abs: Math.abs(vector[i] || 0),
        ctx: node.ctx_idx, layer: +node.layer || 0,
      })).filter(d => d.abs > 1e-8)
      var xScat = d3.scalePoint().domain(ctxDomain).range([0, innerWs]).padding(0.5)
      var yScat = d3.scaleLinear()
        .domain(d3.extent(pts, d => d.layer)).nice()
        .range([innerHs, 0])
      if (!pts.length || !isFinite(yScat.domain()[0])) {
        yScat.domain([0, 1]).range([innerHs, 0])
      }
      // column bands + decision boundaries
      ctxDomain.forEach((ctx, i) => {
        var x = xScat(ctx)
        if (x == null) return
        if (i > 0) {
          var prev = xScat(ctxDomain[i - 1])
          var mid = (prev + x) / 2
          sg.append('line').at({
            x1: mid, x2: mid, y1: 0, y2: innerHs,
            stroke: '#D8D5C8', strokeWidth: 1, strokeDasharray: '3,2',
          })
        }
      })
      var rScat = d3.scaleSqrt().domain([0, d3.max(pts, d => d.abs) || 1]).range([2.5, 11])
      var dotSel = sg.appendMany('circle', pts).at({
        cx: d => xScat(d.ctx) + ((d.i * 17) % 11 - 5),
        cy: d => yScat(d.layer),
        r: d => rScat(d.abs),
        fill: d => d.ctx === stats.dominant?.ctx_idx ? '#0D7377' : '#C45C26',
        fillOpacity: d => {
          if (focusCtxIdx == null) return 0.75
          return d.ctx === focusCtxIdx ? 0.9 : 0.12
        },
        stroke: d => {
          if (d.node.nodeId === visState.clickedId) return '#111'
          if (focusCtxIdx != null && d.ctx === focusCtxIdx) return '#0D7377'
          return '#fff'
        },
        strokeWidth: d => d.node.nodeId === visState.clickedId ? 2 : 1,
        cursor: 'pointer',
      })
        .on('click', (ev, d) => {
          ev.stopPropagation()
          focusSvdFeature(d.node)
          // Focus this token column (no toggle) so the cluster graph updates to this position.
          if (focusCtxIdx !== d.ctx) setFocusCtxIdx(d.ctx, {toggle: false})
        })
        .on('mouseenter', (ev, d) => utilCg.hoverFeature(visState, renderAll, d.node))
        .on('mouseleave', () => utilCg.unHoverFeature(visState, renderAll))
      dotSel.append('title').text(d =>
        featureLabel(d.node) + '\nctx ' + d.ctx + ' · u=' + (d.loading >= 0 ? '+' : '') + d.loading.toFixed(3)
          + '\nClick to focus this token position'
      )
      sg.append('g').at({transform: `translate(0,${innerHs})`})
        .call(d3.axisBottom(xScat).tickFormat(ctx => {
          var row = stats.rows.find(d => d.ctx_idx === ctx)
          var t = row ? String(util.ppToken ? util.ppToken(row.token) : row.token) : String(ctx)
          return t.length > 6 ? t.slice(0, 5) + '…' : t
        }))
        .selectAll('text').st({fontSize: '8px', fill: '#666'})
        .attr('transform', 'rotate(-35)').attr('text-anchor', 'end')
      sg.append('g').call(d3.axisLeft(yScat).ticks(4))
        .selectAll('text').st({fontSize: '8px', fill: '#888'})
      sg.append('text').text('layer').at({
        transform: 'rotate(-90)', x: -innerHs / 2, y: -28,
        textAnchor: 'middle', fill: '#888', fontSize: 9,
      })
      sg.selectAll('.domain').st({stroke: '#ccc'})

      // —— Outer-product heatmap u uᵀ, nodes blocked by token / layer, then |u| ——
      // Its own row: the mass bars and scatter are small and summary, this is the
      // thing you actually read cell by cell, so it gets the panel's full width.
      var heatBox = panels.append('div').st({
        flex: '1 1 100%', minWidth: '280px', maxWidth: '1040px',
        border: '1px solid #E4E2D8', borderRadius: '5px', padding: '8px',
        background: '#FBFAF5',
      })
      var heatHead = heatBox.append('div').st({
        display: 'flex', flexWrap: 'wrap', alignItems: 'center',
        justifyContent: 'space-between', gap: '6px', marginBottom: '4px',
      })
      heatHead.append('div').text('Component matrix')
        .st({fontSize: '11px', fontWeight: 600})
      var heatControls = heatHead.append('div').st({
        display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '8px',
      })
      var heatToggle = heatControls.append('div.link-type-buttons').st({display: 'flex'})
      ;[
        {all: false, label: 'Top 28'},
        {all: true, label: 'All'},
      ].forEach(opt => {
        heatToggle.append('div')
          .text(opt.label)
          .st({
            fontSize: '10px', padding: '3px 7px', border: '1px solid #ccc',
            marginLeft: '-1px', cursor: 'pointer', userSelect: 'none',
            background: heatShowAll === opt.all ? '#0D7377' : '#fff',
            color: heatShowAll === opt.all ? '#fff' : '#333',
            borderColor: heatShowAll === opt.all ? '#0D7377' : '#ccc',
          })
          .at({title: opt.all
            ? 'Show UUᵀ over every feature in scope (full component matrix)'
            : 'Preview: strongest 28 loadings only'})
          .on('click', () => {
            if (heatShowAll === opt.all) return
            heatShowAll = opt.all
            withScrollPreserved(() => {
              matrixViewRedraws.forEach(fn => { try { fn() } catch (e) {} })
            })
          })
      })
      var groupRow = heatControls.append('div').st({display: 'flex', alignItems: 'center', gap: '4px'})
      groupRow.append('span').text('Group by')
        .st({fontSize: '9px', color: '#888', textTransform: 'uppercase'})
      var groupToggle = groupRow.append('div.link-type-buttons').st({display: 'flex'})
      ;[
        {mode: 'token', label: 'token', title: 'Blocks are prompt positions (ctx_idx asc) — the original view'},
        {mode: 'layer', label: 'layer', title: 'Blocks are layers (asc), token position ignored: shows whether a mode lives at one depth'},
        {mode: 'tokenLayer', label: 'token → layer', title: 'Token blocks with layer sub-blocks inside them (heavy lines = tokens, light = layers)'},
        {mode: 'layerToken', label: 'layer → token', title: 'Layer blocks with token sub-blocks inside them (heavy lines = layers, light = tokens)'},
      ].forEach(opt => {
        groupToggle.append('div')
          .text(opt.label)
          .st({
            fontSize: '10px', padding: '3px 7px', border: '1px solid #ccc',
            marginLeft: '-1px', cursor: 'pointer', userSelect: 'none',
            background: heatGroupBy === opt.mode ? '#0D7377' : '#fff',
            color: heatGroupBy === opt.mode ? '#fff' : '#333',
            borderColor: heatGroupBy === opt.mode ? '#0D7377' : '#ccc',
          })
          .at({title: opt.title})
          .on('click', () => {
            if (heatGroupBy === opt.mode) return
            heatGroupBy = opt.mode
            withScrollPreserved(() => {
              matrixViewRedraws.forEach(fn => { try { fn() } catch (e) {} })
            })
          })
      })
      var sortRow = heatControls.append('div').st({display: 'flex', alignItems: 'center', gap: '4px'})
      sortRow.append('span').text('Within block')
        .st({fontSize: '9px', color: '#888', textTransform: 'uppercase'})
      var sortToggle = sortRow.append('div.link-type-buttons').st({display: 'flex'})
      ;[
        {mode: 'none', label: 'unsorted', title: 'Inside each block: leave features in the order they arrive (|influence| rank) — no ordering by this component'},
        {mode: 'abs', label: 'by |u|', title: 'Inside each block: strongest |loading| first'},
        {mode: 'signed', label: 'by signed u', title: 'Inside each block: +pole → −pole (contiguous teal/orange regions)'},
      ].forEach(opt => {
        sortToggle.append('div')
          .text(opt.label)
          .st({
            fontSize: '10px', padding: '3px 7px', border: '1px solid #ccc',
            marginLeft: '-1px', cursor: 'pointer', userSelect: 'none',
            background: heatSortMode === opt.mode ? '#0D7377' : '#fff',
            color: heatSortMode === opt.mode ? '#fff' : '#333',
            borderColor: heatSortMode === opt.mode ? '#0D7377' : '#ccc',
            fontFamily: opt.mode === 'abs' ? 'ui-monospace, monospace' : 'inherit',
          })
          .at({title: opt.title})
          .on('click', () => {
            if (heatSortMode === opt.mode) return
            heatSortMode = opt.mode
            withScrollPreserved(() => {
              matrixViewRedraws.forEach(fn => { try { fn() } catch (e) {} })
            })
          })
      })
      // Filled in once the matrix exists — the button needs its geometry, but it
      // belongs up here in the header with the other controls.
      var exportSlot = heatControls.append('div').st({display: 'flex'})
      var groupWord = {
        token: 'token',
        layer: 'layer',
        tokenLayer: 'token, then layer',
        layerToken: 'layer, then token',
      }[heatGroupBy]
      var heatIndexNote = {
        token: 'Bright diagonal block at t★ = co-loading at that position.',
        layer: 'One bright diagonal block = the mode lives at one depth; bright off-diagonal blocks = two layers co-loading.',
        tokenLayer: 'Heavy lines separate tokens, light lines the layers inside them.',
        layerToken: 'Heavy lines separate layers, light lines the tokens inside them.',
      }[heatGroupBy]
      var sortWord = heatSortMode === 'none'
        ? 'arbitrary within a block'
        : heatSortMode === 'signed' ? 'signed u (+pole → −pole)' : '|u| (strongest first)'
      heatBox.append('div')
        .text('Index: nodes by ' + groupWord + ', '
          + (heatSortMode === 'none' ? sortWord : 'then ' + sortWord)
          + '. ' + heatIndexNote)
        .st({fontSize: '10px', color: '#888', marginBottom: '4px', lineHeight: '1.3'})
      var heatEq = heatBox.append('div').st({marginBottom: '6px'})
      renderLatex(
        heatEq,
        '(UU^{\\top})_{ij}=u_{r,i}u_{r,j}'
          + '\\quad\\text{(teal if }>0\\text{, orange if }<0\\text{)}',
        true
      )
      heatEq.selectAll('.svd-eq').st({textAlign: 'left', minHeight: '0'})
      heatEq.selectAll('.katex').st({fontSize: '1.05em'})

      // Row/col order: block by token, by layer, or by layer then token; inside
      // the innermost block leave the incoming order alone (default), or sort by
      // |u| / signed u. Top 28 is a filter on |u|, not an order, so it composes
      // with all three.
      var heatCtxKey = d => d.node.ctx_idx
      var heatLayerKey = d => +d.node.layer || 0
      var groupKeys = {
        token: [heatCtxKey],
        layer: [heatLayerKey],
        tokenLayer: [heatCtxKey, heatLayerKey],
        layerToken: [heatLayerKey, heatCtxKey],
      }[heatGroupBy]
      var order = orderedNodes.map((node, i) => ({
        node, i, loading: vector[i] || 0, abs: Math.abs(vector[i] || 0),
      }))
        .filter(d => focusCtxIdx == null || d.node.ctx_idx === focusCtxIdx)
        .filter(d => heatShowAll || d.abs > 1e-8)
      if (!heatShowAll) {
        order = order.sort((a, b) => b.abs - a.abs).slice(0, 28)
      }
      order.sort((a, b) => {
        for (var gk = 0; gk < groupKeys.length; gk++) {
          var keyDiff = groupKeys[gk](a) - groupKeys[gk](b)
          if (keyDiff) return keyDiff
        }
        if (heatSortMode === 'none') return a.i - b.i
        return heatSortMode === 'signed' ? (b.loading - a.loading) : (b.abs - a.abs)
      })
      var nH = order.length
      var nScope = orderedNodes.filter(n =>
        focusCtxIdx == null || n.ctx_idx === focusCtxIdx
      ).length
      heatBox.append('div')
        .text(
          (heatShowAll
            ? ('Showing all ' + nH + ' / ' + nScope + ' features in scope'
              + (focusCtxIdx != null ? ' (focused position)' : ' (full component)'))
            : ('Preview ' + nH + ' strongest of ' + nScope + ' in scope — switch to All for the full matrix'))
          + ' · blocked by ' + groupWord
          + ' · within-block order: '
          + (heatSortMode === 'none' ? 'unsorted' : heatSortMode === 'signed' ? 'signed u' : '|u|')
        )
        .st({fontSize: '10px', color: '#666', marginBottom: '6px'})
      if (!nH) {
        heatBox.append('div').text('No features in scope').st({fontSize: '10px', color: '#aaa'})
      } else {
        var absU = order.map(d => d.abs)
        // Scale by a mid-high quantile of |u| so more cells hit full color
        // (too high a quantile leaves the full matrix washed out).
        var sortedAbs = absU.slice().sort((a, b) => a - b)
        var q = sortedAbs[Math.max(0, Math.floor(0.90 * (sortedAbs.length - 1)))] || 0
        var uScale = Math.max(q, sortedAbs[sortedAbs.length - 1] * 0.15, 1e-8)

        function heatTokenLabel(ctx) {
          var row = stats.rows.find(d => d.ctx_idx === ctx)
          var t = row ? String(util.ppToken ? util.ppToken(row.token) : row.token) : String(ctx)
          return t.length > 5 ? t.slice(0, 4) + '…' : t
        }
        // Contiguous runs of equal key, in the order already sorted above.
        function heatRuns(keyFn) {
          var out = []
          order.forEach((d, i) => {
            var key = keyFn(d)
            var last = out[out.length - 1]
            if (!last || last.key !== key) out.push({key, start: i, end: i + 1, d})
            else last.end = i + 1
          })
          return out
        }
        var nested = groupKeys.length > 1
        var majorRuns = heatRuns(d => groupKeys[0](d))
        var minorRuns = nested ? heatRuns(d => groupKeys.map(k => k(d)).join('|')) : []
        var tokenText = run => heatTokenLabel(run.d.node.ctx_idx)
        var layerText = run => 'L' + (+run.d.node.layer || 0)
        var primaryIsToken = heatGroupBy === 'token' || heatGroupBy === 'tokenLayer'
        var majorText = primaryIsToken ? tokenText : layerText
        var minorText = primaryIsToken ? layerText : tokenText

        // Nested mode needs a second ring of labels: sub-blocks next to the
        // matrix, outer blocks outside them.
        var heatPad = nested
          ? {top: 8, right: 8, bottom: 50, left: 66}
          : {top: 8, right: 8, bottom: 36, left: 48}
        // Take the width the row actually gives us (the matrix owns a full row),
        // capped so a 28-cell preview does not stretch into a poster.
        var heatOuter = heatBox.node().getBoundingClientRect().width || 520
        var heatAvail = Math.floor(heatOuter - 18 - heatPad.left - heatPad.right)
        var disp = Math.max(220, Math.min(heatAvail, 960, Math.max(220, nH * 14)))
        var frame = heatBox.append('div').st({
          position: 'relative',
          width: (disp + heatPad.left + heatPad.right) + 'px',
          height: (disp + heatPad.top + heatPad.bottom) + 'px',
          maxWidth: '100%',
        })
        var cnv = frame.append('canvas').node()
        // Draw at matrix resolution (or capped), display CSS-scaled. The cap sits
        // above the largest display size so a full component is downsampled once,
        // by the sampler below, and not blurred again on the way to the screen.
        var pixN = Math.min(nH, 1200)
        cnv.style.position = 'absolute'
        cnv.style.left = heatPad.left + 'px'
        cnv.style.top = heatPad.top + 'px'
        cnv.style.width = disp + 'px'
        cnv.style.height = disp + 'px'
        cnv.style.imageRendering = nH > 200 ? 'auto' : 'pixelated'
        cnv.style.border = '1px solid #DDD9CC'
        cnv.style.background = '#FBFAF5'

        function lerpRgb(t, rgb) {
          // Near-white background → fully saturated endpoint. Aggressive gamma
          // so mid-range products still read as strong teal / orange.
          var r0 = 0xF7, g0 = 0xF5, b0 = 0xEE
          t = Math.max(0, Math.min(1, t))
          t = Math.pow(t, 0.35)
          return [
            (r0 + (rgb[0] - r0) * t) | 0,
            (g0 + (rgb[1] - g0) * t) | 0,
            (b0 + (rgb[2] - b0) * t) | 0,
          ]
        }
        // Punchier endpoints than the panel accents.
        var teal = [0x00, 0x6B, 0x70]
        var ora = [0xD4, 0x3B, 0x08]
        // Paint u uᵀ at whatever resolution the caller wants: the on-screen
        // canvas, or a larger one for export. Pixel → feature index is nearest
        // neighbour, so a block stays a block instead of being averaged away.
        function paintComponent(canvasEl, size) {
          canvasEl.width = size
          canvasEl.height = size
          var cx = canvasEl.getContext('2d')
          var im = cx.createImageData(size, size)
          var d4 = im.data
          for (var py = 0; py < size; py++) {
            var ai = Math.min(nH - 1, Math.floor(py * nH / size))
            var ua = order[ai].loading / uScale
            for (var px = 0; px < size; px++) {
              var bi = Math.min(nH - 1, Math.floor(px * nH / size))
              var ub = order[bi].loading / uScale
              // Product in scaled units; clip to [-1,1] for color.
              var t = Math.max(-1, Math.min(1, ua * ub))
              var rgb = t >= 0 ? lerpRgb(t, teal) : lerpRgb(-t, ora)
              var pix = (py * size + px) * 4
              d4[pix] = rgb[0]
              d4[pix + 1] = rgb[1]
              d4[pix + 2] = rgb[2]
              d4[pix + 3] = 255
            }
          }
          cx.putImageData(im, 0, 0)
        }
        paintComponent(cnv, pixN)

        // Overlay: token-block boundaries + axis labels (SVG on top of canvas).
        var overlay = frame.append('svg').at({
          width: disp + heatPad.left + heatPad.right,
          height: disp + heatPad.top + heatPad.bottom,
        }).st({
          position: 'absolute', left: 0, top: 0,
          width: (disp + heatPad.left + heatPad.right) + 'px',
          height: (disp + heatPad.top + heatPad.bottom) + 'px',
          pointerEvents: 'none',
        })
        var hg = overlay.append('g').at({
          transform: `translate(${heatPad.left},${heatPad.top})`,
        })
        var scale = disp / nH
        function heatRule(at, opacity, width, dash) {
          hg.append('line').at({
            x1: at, x2: at, y1: 0, y2: disp,
            stroke: '#111', strokeOpacity: opacity, strokeWidth: width,
            strokeDasharray: dash || null,
          })
          hg.append('line').at({
            x1: 0, x2: disp, y1: at, y2: at,
            stroke: '#111', strokeOpacity: opacity, strokeWidth: width,
            strokeDasharray: dash || null,
          })
        }
        // Sub-blocks first, so the layer rules draw over them.
        minorRuns.slice(1).forEach(run => heatRule(run.start * scale, 0.16, 0.6, '2,2'))
        majorRuns.slice(1).forEach(run => heatRule(run.start * scale, 0.45, 1))
        // Label as many blocks as fit: thin blocks are common (one feature at a
        // late layer) and drawing every label overprints them into a smear. Keep
        // a running extent per axis and skip anything that would collide.
        function heatLabels(runs, text, xOff, yOff) {
          var lastX = -Infinity
          var lastY = -Infinity
          runs.forEach(run => {
            var mid = ((run.start + run.end) / 2) * scale
            var label = text(run)
            var halfW = label.length * 2.4 + 2
            if (mid - halfW > lastX) {
              lastX = mid + halfW
              hg.append('text').text(label).at({
                x: mid, y: disp + yOff,
                textAnchor: 'middle', fill: '#666', fontSize: 8,
              })
            }
            if (mid - 4.5 > lastY) {
              lastY = mid + 4.5
              hg.append('text').text(label).at({
                x: xOff, y: mid + 3,
                textAnchor: 'end', fill: '#666', fontSize: 8,
              })
            }
          })
        }
        if (nested) {
          heatLabels(minorRuns, minorText, -6, 11)
          heatLabels(majorRuns, majorText, -30, 24)
        } else {
          heatLabels(majorRuns, majorText, -6, 12)
        }
        var heatAxisWord = 'by ' + groupWord
        hg.append('text').text('node j (' + heatAxisWord + ')').at({
          x: disp / 2, y: disp + (nested ? 40 : 28),
          textAnchor: 'middle', fill: '#888', fontSize: 9,
        })
        hg.append('text').text('node i (' + heatAxisWord + ')').at({
          transform: 'rotate(-90)',
          x: -disp / 2, y: nested ? -52 : -34,
          textAnchor: 'middle', fill: '#888', fontSize: 9,
        })

        // —— PNG export: matrix + rules + labels + what it is ——
        // Repaints the pixels at export resolution and rasterizes the same
        // overlay SVG on top, so the file is the view, not a re-derivation of it.
        var meta = data.metadata || {}
        var exSlug = meta.slug || 'graph'
        var exFont = '-apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif'
        var exLines = [
          {
            text: 'Component matrix · σ' + (r + 1) + ' · ' + ch.title
              + ' · ' + (heatShowAll ? 'all' : 'top') + ' ' + nH + ' features',
            size: 14, weight: '600', fill: '#1d1d1b',
          },
          {
            text: exSlug + (meta.prompt ? ' — "' + meta.prompt + '"' : ''),
            size: 11, weight: '400', fill: '#666',
          },
          {
            text: '(UUᵀ)ij = u' + (r + 1) + ',i · u' + (r + 1) + ',j'
              + '   ·   teal = positive product, orange = negative'
              + '   ·   σ' + (r + 1) + ' = ' + sigma.toPrecision(4),
            size: 11, weight: '400', fill: '#444',
          },
          {
            text: 'blocked by ' + groupWord + ' · within-block order: '
              + (heatSortMode === 'none' ? 'unsorted (arbitrary)'
                : heatSortMode === 'signed' ? 'signed u' : '|u| descending')
              + ' · ' + nH + ' of ' + nScope + ' features in scope'
              + (focusCtxIdx != null ? ' · position ' + focusCtxIdx + ' only' : ''),
            size: 11, weight: '400', fill: '#444',
          },
          {
            text: nested
              ? 'Heavy rules separate ' + (primaryIsToken ? 'tokens' : 'layers')
                + ', light rules the ' + (primaryIsToken ? 'layers' : 'tokens') + ' inside them.'
              : 'Rules separate ' + groupWord + ' blocks.',
            size: 10, weight: '400', fill: '#888',
          },
        ]
        var exPadX = 18
        var exLineH = 17
        var exHeadH = 14 + exLines.length * exLineH + 10
        var frameW = disp + heatPad.left + heatPad.right
        var frameH = disp + heatPad.top + heatPad.bottom
        function exportComponentPng() {
          var s = 2 // retina-ish; text stays sharp when the file is zoomed
          var W = frameW + exPadX * 2
          var H = exHeadH + frameH + 14
          var out = document.createElement('canvas')
          out.width = Math.round(W * s)
          out.height = Math.round(H * s)
          var g = out.getContext('2d')
          g.scale(s, s)
          g.fillStyle = '#fff'
          g.fillRect(0, 0, W, H)
          g.textBaseline = 'alphabetic'
          exLines.forEach((ln, i) => {
            g.font = ln.weight + ' ' + ln.size + 'px ' + exFont
            g.fillStyle = ln.fill
            g.fillText(ln.text, exPadX, 14 + i * exLineH + ln.size)
          })
          // Pixels: at least the matrix's own resolution, and enough that the
          // exported cells are not upscaled from the on-screen canvas.
          var exPix = Math.min(2000, Math.max(nH, Math.round(disp * s)))
          var tmp = document.createElement('canvas')
          paintComponent(tmp, exPix)
          g.imageSmoothingEnabled = false
          g.drawImage(
            tmp, exPadX + heatPad.left, exHeadH + heatPad.top, disp, disp
          )
          g.strokeStyle = '#DDD9CC'
          g.lineWidth = 1
          g.strokeRect(
            exPadX + heatPad.left - 0.5, exHeadH + heatPad.top - 0.5, disp + 1, disp + 1
          )
          // Rules and labels: reuse the live overlay rather than redrawing it.
          var svgNode = overlay.node().cloneNode(true)
          svgNode.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
          svgNode.setAttribute('width', frameW)
          svgNode.setAttribute('height', frameH)
          svgNode.setAttribute('font-family', exFont)
          var svgUrl = 'data:image/svg+xml;charset=utf-8,'
            + encodeURIComponent(new XMLSerializer().serializeToString(svgNode))
          var svgImg = new Image()
          svgImg.onload = () => {
            g.drawImage(svgImg, exPadX, exHeadH, frameW, frameH)
            var name = [
              exSlug, 'sigma' + (r + 1), String(ch.title).toLowerCase(),
              heatGroupBy.toLowerCase(), heatSortMode,
              focusCtxIdx != null ? 'ctx' + focusCtxIdx : null,
            ].filter(Boolean).join('-').replace(/[^a-z0-9._-]+/gi, '-') + '.png'
            out.toBlob(blob => {
              if (!blob) return
              var url = URL.createObjectURL(blob)
              var a = document.createElement('a')
              a.href = url
              a.download = name
              document.body.appendChild(a)
              a.click()
              a.remove()
              setTimeout(() => URL.revokeObjectURL(url), 5000)
            }, 'image/png')
          }
          svgImg.onerror = () => {
            exportBtn.text('export failed')
            setTimeout(() => exportBtn.text('PNG'), 1800)
          }
          svgImg.src = svgUrl
        }
        var exportBtn = exportSlot.append('div.link-type-buttons').append('div')
          .text('PNG')
          .st({
            fontSize: '10px', padding: '3px 7px', border: '1px solid #0D7377',
            cursor: 'pointer', userSelect: 'none',
            background: '#F3FAFA', color: '#0D7377',
          })
          .at({title: 'Download this matrix as a PNG: cells, block rules, axis labels, and the settings that produced it'})
          .on('click', () => {
            exportBtn.text('saving…')
            exportComponentPng()
            setTimeout(() => exportBtn.text('PNG'), 900)
          })
      }

      // Top features list for this component, grouped by position
      var list = body.append('div').st({marginTop: '10px'})
      list.append('div').text('Top |u| features on σ' + (r + 1) + ', grouped by token position · click a group to focus')
        .st({fontSize: '11px', fontWeight: 600, marginBottom: '6px'})
      var groups = list.append('div').st({
        display: 'flex', flexWrap: 'wrap', gap: '8px',
      })
      var listRows = stats.rows.filter(row => row.mass > 1e-12)
      if (focusCtxIdx != null) {
        listRows = listRows.slice().sort((a, b) =>
          (a.ctx_idx === focusCtxIdx ? 0 : 1) - (b.ctx_idx === focusCtxIdx ? 0 : 1)
        )
      }
      listRows.forEach(row => {
        var focused = focusCtxIdx != null && row.ctx_idx === focusCtxIdx
        var dimmed = focusCtxIdx != null && !focused
        var card = groups.append('div').st({
          flex: '1 1 160px', maxWidth: '220px',
          border: focused ? '2px solid #0D7377' : '1px solid #E4E2D8',
          borderRadius: '5px', padding: '6px 8px',
          background: focused ? '#E8F5F5' : (row === stats.dominant ? '#F3FAFA' : '#FAFAF8'),
          opacity: dimmed ? 0.4 : 1,
          cursor: 'pointer',
        })
          .at({title: 'Click to focus this token position'})
          .on('click', (ev) => {
            // Don't steal feature-row clicks
            if (ev.target && ev.target.closest && ev.target.closest('[data-feat-row]')) return
            setFocusCtxIdx(row.ctx_idx)
          })
        card.append('div')
          .text('pos ' + row.ctx_idx + ' · "'
            + String(util.ppToken ? util.ppToken(row.token) : row.token) + '"'
            + ' · ' + (100 * row.frac).toFixed(0) + '%'
            + (focused ? ' · focused' : ''))
          .st({
            fontSize: '10px', fontWeight: 700, marginBottom: '4px',
            color: (focused || row === stats.dominant) ? '#0D7377' : '#555',
            fontFamily: 'ui-monospace, monospace',
          })
        row.nodes.slice(0, 6).forEach((t, ti) => {
          var rowEl = card.append('div')
            .at({'data-feat-row': 1})
            .st({
              display: 'flex', justifyContent: 'space-between', gap: '6px',
              fontSize: '10px', padding: '1px 0', cursor: 'pointer',
              color: t.node.nodeId === visState.clickedId ? '#0D7377' : '#333',
              fontWeight: t.node.nodeId === visState.clickedId ? 700 : 400,
            })
            .on('click', (ev) => {
              ev.stopPropagation()
              focusSvdFeature(t.node)
              drawBody() // refresh selection styles
            })
            .on('mouseenter', () => utilCg.hoverFeature(visState, renderAll, t.node))
            .on('mouseleave', () => utilCg.unHoverFeature(visState, renderAll))
          rowEl.append('span')
            .text((ti + 1) + '. ' + featureLabel(t.node))
            .st({overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1})
          rowEl.append('span')
            .text((t.loading >= 0 ? '+' : '') + t.loading.toFixed(2))
            .st({fontFamily: 'ui-monospace, monospace', color: '#666', flexShrink: 0})
        })
      })
    }

    matrixViewRedraws.push(drawBody)
    drawBody()
  }

  function renderSpectrum(container, title, equation, note, svdResult, nodes, k, frob2) {
    var sigmas = svdResult.sigmas || []
    var vectors = svdResult.vectors || []
    var wrap = container.append('div').st({
      flex: '1 1 240px', minWidth: '220px', maxWidth: '340px',
      background: '#fff', border: '1px solid #E4E2D8', borderRadius: '6px',
      padding: '10px 10px 8px', boxSizing: 'border-box',
    })
    wrap.append('div').text(title)
      .st({fontWeight: 600, fontSize: '12px', marginBottom: '6px', color: '#222'})

    var eqBox = wrap.append('div').st({
      background: '#FBFAF5', border: '1px solid #E4E2D8', borderRadius: '4px',
      padding: '8px 6px', marginBottom: '6px',
    })
    renderLatex(eqBox, equation, true)

    if (note) {
      wrap.append('div').text(note)
        .st({fontSize: '10px', color: '#666', marginBottom: '6px', lineHeight: '1.35'})
    }

    if (!sigmas.length) {
      wrap.append('div').text('Not enough nodes')
        .st({fontSize: '10px', color: '#aaa', padding: '20px 0'})
      return
    }

    var energy = window.svdSpectrum.energyFraction(sigmas, k, frob2)
    var energyRow = wrap.append('div').st({
      marginBottom: '6px', color: '#0D7377', fontWeight: 600,
    })
    renderLatex(
      energyRow,
      '\\dfrac{\\sum_{i=1}^{k}\\sigma_i^{2}}{\\lVert M\\rVert_F^{2}}='
        + (100 * energy).toFixed(1) + '\\%',
      true
    )
    if (svdResult.truncated) {
      wrap.append('div')
        .text('Truncated top-' + (svdResult.k || sigmas.length) + ' SVD (full graph).')
        .st({fontSize: '9px', color: '#999', marginBottom: '4px'})
    }

    var w = 220, h = 120
    var margin = {top: 6, right: 6, bottom: 22, left: 28}
    var svg = wrap.append('svg').at({width: '100%', viewBox: `0 0 ${w} ${h}`})
      .st({display: 'block', overflow: 'visible'})

    var innerW = w - margin.left - margin.right
    var innerH = h - margin.top - margin.bottom
    var g = svg.append('g').at({transform: `translate(${margin.left},${margin.top})`})

    var x = d3.scaleBand().domain(d3.range(sigmas.length)).range([0, innerW]).padding(0.15)
    var yMax = sigmas[0] || 1
    var y = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]).nice()

    g.appendMany('rect', sigmas).at({
      x: (d, i) => x(i),
      y: d => y(d),
      width: x.bandwidth(),
      height: d => Math.max(0, innerH - y(d)),
      fill: (d, i) => i < k ? '#0D7377' : '#C8C5B8',
    })

    g.append('g').at({transform: `translate(0,${innerH})`})
      .call(d3.axisBottom(x).tickValues(
        d3.range(sigmas.length).filter(i => i === 0 || i === sigmas.length - 1 || (i + 1) % 5 === 0)
      ).tickFormat(i => i + 1).tickSize(3))
      .selectAll('text').st({fontSize: '8px', fill: '#888'})
    g.append('g')
      .call(d3.axisLeft(y).ticks(3).tickSize(3))
      .selectAll('text').st({fontSize: '8px', fill: '#888'})
    g.selectAll('.domain').st({stroke: '#ccc'})
    g.selectAll('.tick line').st({stroke: '#ddd'})

    renderLatex(wrap, '\\sigma_1 \\ge \\sigma_2 \\ge \\cdots \\ge \\sigma_n \\ge 0'
      + '\\;\\;(n=' + sigmas.length + ')', true)

    // Top feature loadings for each of the first k singular vectors u_r
    // (left singular vector: M = U Σ Vᵀ, entry i of u_r is feature i's loading).
    var modesShown = Math.min(k, 6) // keep the card readable
    var loadBox = wrap.append('div').st({
      marginTop: '8px', borderTop: '1px solid #E4E2D8', paddingTop: '6px',
    })
    loadBox.append('div')
      .text('Top features by |uᵣ| on σ₁…σ' + modesShown)
      .st({fontSize: '10px', fontWeight: 600, color: '#444', marginBottom: '4px'})
    renderLatex(
      loadBox,
      'u_r = \\text{left singular vector for }\\sigma_r\\;\\;(|u_{r,i}|\\text{ = loading of feature }i)',
      false
    )
    loadBox.selectAll('.katex').st({fontSize: '0.85em', color: '#666'})
    loadBox.append('div')
      .text('Click a feature to select it (sticky). Hover only previews. Scroll up to inspect in the graph.')
      .st({fontSize: '9px', color: '#777', marginTop: '3px', lineHeight: '1.35'})

    if (yMax < 1e-12) {
      loadBox.append('div')
        .text('All σ ≈ 0 — no edges among the selected nodes (empty adjacency).')
        .st({fontSize: '10px', color: '#a23', marginTop: '4px'})
      return
    }

    function styleLoadRow(rowSel, nodeId) {
      var selected = nodeId && nodeId === visState.clickedId
      rowSel.st({
        color: selected ? '#0D7377' : '#333',
        fontWeight: selected ? 700 : 400,
        background: selected ? '#DCEEEC' : 'transparent',
        borderRadius: '3px',
        padding: '2px 4px',
      })
    }

    for (var r = 0; r < modesShown; r++) {
      var tops = window.svdSpectrum.topLoadings(vectors[r], nodes, 3)
      var block = loadBox.append('div').st({
        marginTop: '5px', padding: '4px 6px', background: r < k ? '#F3FAFA' : '#FAFAF8',
        borderRadius: '4px', border: '1px solid #E8E6DC',
      })
      block.append('div')
        .text('σ' + (r + 1) + ' = ' + sigmas[r].toFixed(3))
        .st({fontSize: '10px', fontWeight: 700, color: '#0D7377', marginBottom: '2px',
          fontFamily: 'ui-monospace, monospace'})
      if (!tops.length) {
        block.append('div').text('—').st({fontSize: '10px', color: '#aaa'})
        continue
      }
      tops.forEach((t, ti) => {
        var row = block.append('div')
          .datum(t.node)
          .st({
            display: 'flex', justifyContent: 'space-between', gap: '6px',
            fontSize: '10px', cursor: 'pointer',
          })
          .on('click', (ev) => {
            ev.preventDefault()
            ev.stopPropagation()
            focusSvdFeature(t.node)
            // Re-style all load rows in this card for sticky highlight.
            loadBox.selectAll('.svd-load-row').each(function (d) {
              styleLoadRow(d3.select(this), d?.nodeId)
            })
          })
          .on('mouseenter', function () {
            if (t.node) utilCg.hoverFeature(visState, renderAll, t.node)
            if (t.node?.nodeId !== visState.clickedId) {
              d3.select(this).st({color: '#0D7377'})
            }
          })
          .on('mouseleave', function () {
            utilCg.unHoverFeature(visState, renderAll)
            styleLoadRow(d3.select(this), t.node?.nodeId)
          })
        row.classed('svd-load-row', 1)
        styleLoadRow(row, t.node?.nodeId)
        row.append('span')
          .text((ti + 1) + '. ' + featureLabel(t.node))
          .st({overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1})
        row.append('span')
          .text((t.loading >= 0 ? '+' : '') + t.loading.toFixed(2))
          .st({fontFamily: 'ui-monospace, monospace', color: '#666', flexShrink: 0})
      })
    }
  }

  function render() {
    // If KaTeX hasn't finished loading yet, retry shortly so equations aren't
    // stuck as raw TeX.
    if (!window.katex || typeof window.katex.render !== 'function') {
      if (!render._katexRetries) render._katexRetries = 0
      if (render._katexRetries < 40) {
        render._katexRetries++
        setTimeout(render, 50)
        return
      }
    } else {
      render._katexRetries = 0
    }

    // Prefer backend full-graph SVD when available.
    if (bundlePath() && svdBundleStatus !== 'ready' && svdBundleStatus !== 'missing' && svdBundleStatus !== 'error') {
      sel.html('').st({
        fontFamily: 'system-ui, sans-serif', fontSize: '12px', color: '#333',
        padding: '10px 14px', background: '#F7F6F1',
      })
      sel.append('div').text('Loading full-graph SVD from backend…')
        .st({color: '#666', fontSize: '12px', padding: '20px 0'})
      ensureSvdBundle(() => render())
      return
    }

    var nodes = selectedNodes()
    var nodeIds = nodes.map(d => d.nodeId)
    var totalTrans = (data.allNodes || data.nodes || []).filter(d => {
      var ft = (d.feature_type || '').toLowerCase()
      return ft.includes('transcoder') && !ft.includes('error')
    }).length
    var usingBundle = !!(svdBundle && nodesFromBundle()?.length >= 2)

    // Use full link set when available so edges among prune-survivors are complete.
    var links = data.allLinks || data.links || []

    if (clusterSim) { clusterSim.stop(); clusterSim = null }

    sel.html('').st({
      fontFamily: 'system-ui, sans-serif',
      fontSize: '12px',
      color: '#333',
      padding: '10px 14px',
      boxSizing: 'border-box',
      overflowY: 'auto',
      height: '100%',
      background: '#F7F6F1',
      borderTop: '1px solid #E4E2D8',
      overflowAnchor: 'none',
    })

    var header = sel.append('div').st({
      display: 'flex', flexWrap: 'wrap', alignItems: 'center',
      gap: '10px', marginBottom: '8px', justifyContent: 'space-between',
    })
    var left = header.append('div')
    left.append('div').text('SVD spectra')
      .st({fontWeight: 600, fontSize: '13px', letterSpacing: '0.02em'})
    left.append('div')
      .text(
        nodeIds.length
          ? (usingBundle
            ? (nodeIds.length + ' / ' + totalTrans + ' transcoder nodes · full graph (backend SVD)'
              + ' · top-' + (svdBundle.k || spectrumKMax) + ' singular vectors loaded')
            : (nodeIds.length + ' / ' + totalTrans + ' transcoder nodes · edge-expanded fallback'
              + (svdBundleStatus === 'missing'
                ? ' · no metadata.svd_bundle (run export-svd)'
                : svdBundleStatus === 'error'
                  ? ' · SVD bundle failed to load'
                  : '')
              + ' · same order in every plot'
              + (nodeIds.length > 72
                ? ' · SVD uses top-' + spectrumKMax + ' (truncated)'
                : '')))
          : 'No transcoder nodes found.'
      )
      .st({fontSize: '10px', color: '#666', marginTop: '2px'})

    // Shared SVD / energy definitions for the panel.
    var defs = left.append('div').st({
      marginTop: '6px', background: '#fff', border: '1px solid #E4E2D8',
      borderRadius: '4px', padding: '6px 8px', maxWidth: '640px',
    })
    renderLatex(
      defs,
      'M = U\\Sigma V^{\\top},\\quad'
        + '\\sigma_1\\ge\\sigma_2\\ge\\cdots\\ge\\sigma_n\\ge 0,'
        + '\\quad'
        + '\\mathrm{energy}(k)=\\dfrac{\\sum_{i=1}^{k}\\sigma_i^{2}}{\\lVert M\\rVert_F^{2}}',
      true
    )
    defs.append('div')
      .text(
        usingBundle
          ? 'Bars from backend full-graph SVD. Teal = top-k in energy(k). Click loadings to select features.'
          : 'Bars: singular values of each matrix M. Teal = top-k used in energy(k). Under each plot, |uᵣ,ᵢ| ranks which features contribute most to that mode — click to select (sticky).'
      )
      .st({fontSize: '9px', color: '#777', marginTop: '2px', lineHeight: '1.35'})

    if (nodeIds.length >= 2) {
      var sliderMax = Math.min(
        spectrumKMax,
        nodeIds.length,
        usingBundle ? (svdBundle.k || spectrumKMax) : nodeIds.length
      )
      rankK = Math.max(1, Math.min(rankK, sliderMax))
      var ctrl = header.append('div').st({
        display: 'flex', alignItems: 'center', gap: '6px',
        background: '#fff', border: '1px solid #E4E2D8', borderRadius: '5px',
        padding: '4px 8px', fontSize: '11px',
      })
      ctrl.append('span').text('rank k')
      ctrl.append('span').text('1').st({color: '#999', fontSize: '10px'})
      var slider = ctrl.append('input')
        .at({type: 'range', min: 1, max: sliderMax, step: 1, value: rankK})
        .st({width: '140px', cursor: 'pointer'})
      ctrl.append('span').text(String(sliderMax))
        .st({color: '#999', fontSize: '10px', fontWeight: 600})
      var kLabel = ctrl.append('span').text(rankK + ' / ' + sliderMax)
        .st({fontWeight: 700, color: '#0D7377', minWidth: '40px'})
      slider.on('input', function () {
        rankK = +this.value
        kLabel.text(rankK + ' / ' + sliderMax)
        drawCharts(nodeIds, links)
      })
    }

    if (nodeIds.length < 2) {
      sel.append('div').text('Need at least 2 transcoder nodes — relax the prune slider.')
        .st({color: '#999', fontSize: '11px', marginTop: '12px'})
      return
    }

    sel.append('div.svd-charts')
    drawCharts(nodeIds, links)

    function drawCharts(ids, linkList) {
      var host = sel.select('.svd-charts')
      if (host.empty()) return
      host.html('')
      matrixPickerStyles = []
      matrixViewRedraws = []
      positionPickerStyles = []

      var idToNode = {}
      ;(data.allNodes || data.nodes || []).forEach(n => { idToNode[n.nodeId] = n })
      var orderedNodes = ids.map(id => idToNode[id]).filter(Boolean)

      var charts = chartsFromBundle()
      var cosine = null
      var cosineNote = ''
      var bundleMode = !!charts

      if (!charts) {
        var geom = (data.metadata || {}).svd_geom
        cosine = window.svdSpectrum.cosineFromGeom(ids, geom)
        var cosineEq =
          'C_{ij} = \\cos(\\mathbf{v}_i, \\mathbf{v}_j),\\quad'
          + '\\mathbf{v}_f = \\operatorname{unit}\\big['
          + '\\widehat{W_{\\mathrm{enc}}}[:,f]\\,\\|\\,'
          + '\\widehat{W_{\\mathrm{dec}}}[f,:]\\big]'
        cosineNote = 'inflow ∥ outflow directions (from verify)'
        if (!cosine) {
          cosine = window.svdSpectrum.linkNeighborhoodCosine(ids, linkList)
          cosineEq =
            'C_{ij} = \\cos(\\mathbf{u}_i, \\mathbf{u}_j),\\quad'
            + '\\mathbf{u}_f = \\operatorname{unit}\\big['
            + '\\widehat{\\mathrm{in}}_f\\,\\|\\,'
            + '\\widehat{\\mathrm{out}}_f\\big]'
          cosineNote = 'fallback: in/out = abs attribution neighborhoods among selected nodes'
        }

        var attr = buildAttributionMatrices(ids, linkList)
        charts = [
          {
            title: 'Cosine similarity',
            equation: cosineEq,
            note: cosineNote,
            matrix: cosine,
          },
          {
            title: 'Unsigned adjacency',
            equation: 'A_{ij} = \\lvert w(i \\rightarrow j)\\rvert',
            note: 'w = attribution edge weight; direction i → j only',
            matrix: attr.unsigned,
          },
          {
            title: 'Signed adjacency',
            equation: 'A_{ij} = w(i \\rightarrow j)',
            note: 'keeps positive / negative attribution sign',
            matrix: attr.signed,
          },
          {
            title: 'Symmetric |A|',
            equation: 'A_{ij} = \\lvert w(i \\rightarrow j)\\rvert + \\lvert w(j \\rightarrow i)\\rvert',
            note: 'undirected strength between i and j',
            matrix: attr.symmetric,
          },
        ]
      }

      // When using the backend bundle, do NOT materialize dense n×n client
      // matrices (that freezes the browser). Spectra use precomputed vectors.
      if (bundleMode) {
        cosine = null
        cosineNote = 'backend full-graph SVD'
      }

      var row = host.append('div').st({
        display: 'flex', flexWrap: 'wrap', gap: '10px', alignItems: 'stretch',
      })

      var svdK = Math.min(spectrumKMax, Math.max(rankK, 8), orderedNodes.length)
      charts.forEach(ch => {
        var svdResult = ch.svdResult || window.svdSpectrum.svd(ch.matrix, {k: svdK})
        var frob2 = ch.frob2 != null
          ? ch.frob2
          : window.svdSpectrum.frobenius2(ch.matrix)
        ch.svdResult = svdResult
        ch.frob2 = frob2
        renderSpectrum(row, ch.title, ch.equation, ch.note, svdResult, orderedNodes, rankK, frob2)
      })

      // After SVDs are attached, lock Matrix onto the first usable chart so
      // Unsigned/Signed with σ₁≈0 never stick the cluster/token panels.
      if (chartIsEmpty(charts[clusterMatrixIdx])) {
        clusterMatrixIdx = firstUsableMatrixIdx(charts)
      }

      if (cosine && (!bundleMode || orderedNodes.length <= 80)) {
        renderCosineMatrixHeatmap(host, cosine, orderedNodes, cosineNote)
      } else if (bundleMode) {
        var note = host.append('div.svd-cosine-matrix').st({
          marginTop: '12px', background: '#fff', border: '1px solid #E4E2D8',
          borderRadius: '6px', padding: '10px 12px',
        })
        note.append('div').text('Node × node cosine similarity')
          .st({fontWeight: 600, fontSize: '12px', marginBottom: '4px'})
        note.append('div')
          .text('Spectra/loadings are the backend full-graph SVD (n=' + orderedNodes.length
            + '). Dense n×n heatmap skipped here — matrices are in graph_files/svd_exports/.')
          .st({fontSize: '10px', color: '#666', lineHeight: '1.35'})
      }

      renderModeClusterGraph(host, charts, orderedNodes, rankK)
      renderComponentTokenView(host, charts, orderedNodes, rankK)
      renderSpectralClusterView(host)
    }
  }

  /** Spectral clustering: affinity W heatmap (cluster-ordered) + λ spectrum + groups. */
  function renderSpectralClusterView(container) {
    var spec = (data.metadata || {}).spectral_clusters
    var wrap = container.append('div.svd-spectral').st({
      marginTop: '12px', background: '#fff', border: '1px solid #E4E2D8',
      borderRadius: '6px', padding: '10px 12px', boxSizing: 'border-box',
    })
    var head = wrap.append('div').st({
      display: 'flex', flexWrap: 'wrap', alignItems: 'flex-start',
      justifyContent: 'space-between', gap: '10px', marginBottom: '8px',
    })
    var titleCol = head.append('div')
    titleCol.append('div').text('Spectral clusters (cosine friendships)')
      .st({fontWeight: 600, fontSize: '12px', color: '#222'})
    titleCol.append('div')
      .text('This view is the friendship graph W, with rows/cols grouped by spectral clustering. Same groups as Computed · Spectral.')
      .st({fontSize: '10px', color: '#666', marginTop: '2px', maxWidth: '640px', lineHeight: '1.35'})

    if (!spec || !spec.by_k) {
      wrap.append('div')
        .text('No metadata.spectral_clusters — run: interpretability-graph export-spectral')
        .st({fontSize: '11px', color: '#999', padding: '8px 0'})
      return
    }

    var kMin = spec.k_min || 2
    var kMax = spec.k_max || kMin
    if (spectralK == null) spectralK = spec.default_k || kMin
    spectralK = Math.max(kMin, Math.min(kMax, spectralK))
    if (spectralFocusGroup != null && spectralFocusGroup >= spectralK) {
      spectralFocusGroup = null
    }

    var ctrl = head.append('div').st({
      display: 'flex', alignItems: 'center', gap: '8px', fontSize: '11px',
      background: '#FBFAF5', border: '1px solid #E4E2D8', borderRadius: '5px',
      padding: '5px 10px', flexWrap: 'wrap',
    })
    ctrl.append('span').text('groups k').st({color: '#666', fontSize: '10px'})
    var slider = ctrl.append('input')
      .at({type: 'range', min: kMin, max: kMax, step: 1, value: spectralK})
      .st({width: '120px', cursor: 'pointer'})
    var kLabel = ctrl.append('span').text(String(spectralK))
      .st({fontWeight: 700, color: '#0D7377', minWidth: '18px'})
    var heatToggle = ctrl.append('div').st({display: 'flex'})
    ;[
      {all: false, label: 'Top 64'},
      {all: true, label: 'All'},
    ].forEach(opt => {
      heatToggle.append('div')
        .text(opt.label)
        .st({
          fontSize: '10px', padding: '2px 7px', border: '1px solid #ccc',
          marginLeft: '-1px', cursor: 'pointer', userSelect: 'none',
          background: spectralHeatShowAll === opt.all ? '#0D7377' : '#fff',
          color: spectralHeatShowAll === opt.all ? '#fff' : '#333',
          borderColor: spectralHeatShowAll === opt.all ? '#0D7377' : '#ccc',
        })
        .on('click', () => {
          if (spectralHeatShowAll === opt.all) return
          spectralHeatShowAll = opt.all
          redraw()
        })
    })
    var clearBtn = ctrl.append('div')
      .text('show all groups')
      .st({
        fontSize: '10px', color: '#0D7377', cursor: 'pointer',
        borderBottom: '1px dotted #0D7377', userSelect: 'none',
        opacity: spectralFocusGroup == null ? 0.35 : 1,
      })
      .on('click', () => {
        spectralFocusGroup = null
        redraw()
      })
    slider.on('input', function () {
      spectralK = +this.value
      kLabel.text(String(spectralK))
      if (spectralFocusGroup != null && spectralFocusGroup >= spectralK) {
        spectralFocusGroup = null
      }
      redraw()
    })

    var body = wrap.append('div.svd-spectral-body')

    function featureLabel(node, id) {
      var feat = featureIdOf(node, id)
      if (!feat) return id || ''
      var ctx = node && node.ctx_idx
      if (ctx == null && id) {
        var parts = String(id).split('_')
        if (parts.length >= 3) ctx = parts[2]
      }
      return ctx != null && ctx !== '' ? '[' + feat + '] · t' + ctx : '[' + feat + ']'
    }

    function ensureAffinity(done) {
      var path = spec.affinity_path
      if (typeof done === 'function') affinityWaiters.push(done)

      function flush(npy) {
        var waiters = affinityWaiters.splice(0)
        waiters.forEach(fn => {
          try { fn(npy) } catch (e) { console.warn(e) }
        })
      }

      if (!path) {
        affinityNpyStatus = 'missing'
        affinityLoadError = 'metadata.spectral_clusters.affinity_path missing'
        flush(null)
        return
      }
      if (affinityNpyStatus === 'ready' && affinityNpyPath === path && affinityNpy) {
        flush(affinityNpy)
        return
      }
      // Already in flight for this path — waiters will be flushed when it finishes.
      if (affinityNpyStatus === 'loading' && affinityNpyPath === path) {
        return
      }

      affinityNpyStatus = 'loading'
      affinityNpyPath = path
      affinityLoadError = null
      affinityNpy = null

      // Bypass util.getFile: its ?query type detection and cache have bitten us.
      // Parse with correct byteOffset (npyjs's buf.slice().buffer bug is easy to hit).
      var url = util.graphDataUrl(path) + '?t=' + Date.now()
      fetch(url, {cache: 'no-store'})
        .then(res => {
          if (!res.ok) throw new Error('HTTP ' + res.status + ' for ' + url)
          return res.arrayBuffer()
        })
        .then(ab => {
          if (!window.npyjs || typeof window.npyjs.parse !== 'function') {
            throw new Error('npyjs not loaded')
          }
          var raw = window.npyjs.parse(ab)
          // npyjs may wrap the whole ArrayBuffer; rebuild a tight typed view.
          var m0 = raw.shape && raw.shape[0]
          var m1 = raw.shape && raw.shape[1]
          var nExpect = (m0 && m1) ? (m0 * m1) : 0
          var data = raw.data
          if (nExpect && data && data.length !== nExpect && ab && ab.byteLength) {
            // Re-parse header and slice the payload with explicit offset/length.
            var buf = new Uint8Array(ab)
            var headerLength = buf[8] + buf[9] * 256
            var offsetBytes = 10 + headerLength
            var payload = buf.buffer.slice(offsetBytes)
            data = new Float32Array(payload)
          }
          if (!data || !raw.shape) throw new Error('npy parse returned empty')
          if (nExpect && data.length !== nExpect) {
            throw new Error('npy size mismatch: got ' + data.length + ' want ' + nExpect)
          }
          var npy = {data: data, shape: raw.shape, dtype: raw.dtype || 'float32'}
          affinityNpy = npy
          affinityNpyStatus = 'ready'
          affinityLoadError = null
          flush(npy)
        })
        .catch(err => {
          console.warn('Affinity W load failed:', url, err)
          affinityNpy = null
          affinityNpyStatus = 'error'
          affinityLoadError = (err && err.message) ? err.message : String(err)
          flush(null)
        })
    }

    function redraw() {
      body.html('')
      var groups = (spec.by_k[String(spectralK)] || []).map(g => g.slice())
      var idToNode = {}
      ;(data.allNodes || data.nodes || []).forEach(n => {
        idToNode[n.nodeId || n.node_id] = n
      })
      var idToGroup = {}
      groups.forEach((g, gi) => {
        g.slice(1).forEach(id => { idToGroup[id] = gi })
      })
      clearBtn.st({opacity: spectralFocusGroup == null ? 0.35 : 1})

      var metaRow = body.append('div').st({
        display: 'flex', flexWrap: 'wrap', gap: '10px 18px',
        fontSize: '10px', color: '#666', marginBottom: '10px',
        fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
      })
      ;[
        ['n', spec.m],
        ['knn', spec.knn || '—'],
        ['nnz(W)', spec.nnz_affinity || '—'],
        ['L', spec.laplacian || 'L_sym'],
        ['embed', spec.row_normalize === false ? 'raw evecs' : (spec.embedding || 'evecs')],
      ].forEach(pair => {
        var cell = metaRow.append('span')
        cell.append('span').text(pair[0] + ' ').st({color: '#999'})
        cell.append('span').text(String(pair[1])).st({color: '#333', fontWeight: 600})
      })

      // —— Math / pipeline (what we actually computed) ——
      var mathBox = body.append('div.svd-spectral-math').st({
        border: '1px solid #E4E2D8', borderRadius: '6px', padding: '10px 12px',
        background: '#FFFEF9', marginBottom: '12px',
      })
      mathBox.append('div').text('What we computed (exact pipeline)')
        .st({fontSize: '11px', fontWeight: 600, marginBottom: '6px', color: '#222'})
      var knnVal = spec.knn || 10
      var steps = [
        {
          n: '1',
          title: 'Nodes + cosine',
          body: 'Take all transcoder features in the SVD/cosine export (n = ' + spec.m
            + '). Cosine C comes from unit enc∥dec directions (precomputed .npy when present):',
          tex: 'C_{ij} = \\mathbf{v}_i^\\top \\mathbf{v}_j \\in [-1,1],\\quad C_{ii}=1.',
        },
        {
          n: '2',
          title: 'Friendships — drop negatives',
          body: 'Signed anti-similarity is discarded entirely (not kept as negative edges):',
          tex: 'W^{(0)}_{ij} = \\max(C_{ij},\\,0),\\qquad W^{(0)}_{ii}=0.',
        },
        {
          n: '3',
          title: 'Mutual kNN sparsify (k = ' + knnVal + ')',
          body: 'Each node keeps only its ' + knnVal
            + ' strongest positive links; then symmetrize. This is the matrix shown in the heatmap:',
          tex: 'A_{ij}=W^{(0)}_{ij}\\cdot\\mathbf{1}[j\\in\\mathcal{N}_{'
            + knnVal + '}(i)],\\quad'
            + 'W=\\tfrac12(A+A^\\top).',
        },
        {
          n: '4',
          title: 'Symmetric normalized Laplacian',
          body: 'Degrees d_i = Σ_j W_{ij}. The operator whose small eigenmodes find cuts of W:',
          tex: 'L_{\\mathrm{sym}} = I - D^{-1/2} W D^{-1/2}.',
        },
        {
          n: '5',
          title: 'Smallest eigenvectors → k-means',
          body: 'For each k = 2…'
            + (spec.k_max || 'K')
            + ', take the first k eigenvectors of L_sym and run k-means on the rows x_i = U_{i,1:k}. Those labels are the groups below and the block order of W.',
          tex: 'x_i = U_{i,1:k}\\in\\mathbb{R}^{k},\\quad'
            + '\\mathrm{labels}=\\mathrm{k\\text{-}means}(\\{x_i\\}).',
        },
      ]
      steps.forEach(step => {
        var row = mathBox.append('div').st({
          display: 'flex', gap: '10px', marginBottom: '8px', alignItems: 'flex-start',
        })
        row.append('div').text(step.n).st({
          flex: 'none', width: '18px', height: '18px', borderRadius: '50%',
          background: '#0D7377', color: '#fff', fontSize: '10px', fontWeight: 700,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          marginTop: '1px',
        })
        var col = row.append('div').st({flex: '1', minWidth: 0})
        col.append('div').text(step.title)
          .st({fontSize: '11px', fontWeight: 600, color: '#333', marginBottom: '2px'})
        col.append('div').text(step.body)
          .st({fontSize: '10px', color: '#555', lineHeight: '1.35', marginBottom: '2px'})
        renderLatex(col, step.tex, true)
        col.selectAll('.svd-eq').st({textAlign: 'left', minHeight: '0', marginBottom: '0'})
        col.selectAll('.katex').st({fontSize: '1.05em'})
      })
      mathBox.append('div')
        .text(
          'Heatmap cells are W_{ij} (not uuᵀ). Blocks follow k-means groups at k='
          + spectralK
          + ', then |influence| inside a group.'
        )
        .st({fontSize: '10px', color: '#666', lineHeight: '1.35', marginTop: '2px'})

      var howBox = body.append('div.svd-spectral-how').st({
        border: '1px solid #E4E2D8', borderRadius: '6px', padding: '10px 12px',
        background: '#FBFAF5', marginBottom: '12px',
      })
      howBox.append('div').text('What this visualization is')
        .st({fontSize: '11px', fontWeight: 600, marginBottom: '4px', color: '#222'})
      howBox.append('div')
        .text(
          'A square heatmap of the sparsified friendship matrix W. Each row and column is one feature. '
          + 'The cell at (i, j) is the edge weight between those two features after dropping negative cosine and keeping only mutual kNN links. '
          + 'Features are sorted into the k=' + spectralK
          + ' spectral groups, so each group is a contiguous block. Color ticks on the axes mark group boundaries. '
          + 'The bar chart is the smallest eigenvalues of L_sym; teal bars are the k coordinates used for clustering. '
          + 'The cards list the features in each group.'
        )
        .st({fontSize: '10px', color: '#555', lineHeight: '1.4', marginBottom: '8px'})
      howBox.append('div').text('How to read it')
        .st({fontSize: '11px', fontWeight: 600, marginBottom: '4px', color: '#222'})
      var howList = howBox.append('div').st({fontSize: '10px', color: '#555', lineHeight: '1.45'})
      ;[
        'Teal = friendship (W > 0). Near-white = no edge. There are no negative cells: anti-similar pairs were dropped.',
        'Bright squares on the diagonal mean a group is internally densely linked — clustering found a real community.',
        'Bright off-diagonal rectangles mean two groups still share many kNN edges; they may be one community split by k, or a leftover cut.',
        'A mostly empty off-diagonal means groups are well separated in W (cheap cuts between them).',
        'Hover a cell for the two feature names and Wᵢⱼ. Click a group chip, legend, or card header to isolate that block; click a feature to select it in the graph.',
        'On the spectrum, a jump after λ_k is a hint that k groups is a natural number of bottlenecks. Click a bar to set k.',
        'Top 64 shows the highest-|influence| features in each group (a preview). All shows every clustered feature.',
      ].forEach(line => {
        var row = howList.append('div').st({display: 'flex', gap: '8px', marginBottom: '3px'})
        row.append('span').text('·').st({color: '#0D7377', fontWeight: 700, flex: 'none'})
        row.append('span').text(line)
      })

      var layout = body.append('div').st({
        display: 'flex', flexWrap: 'wrap', gap: '12px', alignItems: 'stretch',
      })

      // —— Affinity matrix W ——
      var heatBox = layout.append('div.svd-spectral-affinity').st({
        flex: '1 1 420px', minWidth: '300px', maxWidth: '720px',
        border: '1px solid #E4E2D8', borderRadius: '6px', padding: '10px',
        background: '#FBFAF5',
      })
      heatBox.append('div').text('Affinity matrix W (cluster-ordered)')
        .st({fontSize: '11px', fontWeight: 600, marginBottom: '2px'})
      heatBox.append('div')
        .text(
          'Each cell is a friendship weight Wᵢⱼ. Rows and columns are the same features, grouped at k='
          + spectralK
          + '. Look for bright diagonal blocks (communities) and dark gaps between them (cuts). Teal intensity ∝ W.'
        )
        .st({fontSize: '10px', color: '#777', marginBottom: '8px', lineHeight: '1.35'})

      var hoverBar = heatBox.append('div').st({
        minHeight: '36px', marginBottom: '6px', padding: '5px 8px',
        background: 'rgba(255,255,255,0.72)', border: '1px solid #E4E2D8',
        borderRadius: '4px', fontSize: '10px', color: '#555', lineHeight: '1.35',
      })
      hoverBar.append('span')
        .text('Hover a cell — Wᵢⱼ between two features. Click a group chip to isolate its block.')
        .st({color: '#999'})

      var heatHost = heatBox.append('div')

      var pathOk = affinityNpyStatus === 'ready'
        && affinityNpyPath === spec.affinity_path
        && affinityNpy
      var loadFailed = affinityNpyStatus === 'error'
        && affinityNpyPath === spec.affinity_path
      if (!spec.affinity_path) {
        heatHost.append('div')
          .text('No affinity_path — re-run: interpretability-graph export-spectral')
          .st({fontSize: '11px', color: '#999', padding: '16px 0'})
      } else if (loadFailed) {
        heatHost.append('div')
          .text('Failed to load ' + spec.affinity_path
            + (affinityLoadError ? ' — ' + affinityLoadError : ''))
          .st({fontSize: '11px', color: '#a23', padding: '8px 0'})
        heatHost.append('div')
          .text('Retry')
          .st({
            fontSize: '11px', color: '#0D7377', cursor: 'pointer',
            borderBottom: '1px dotted #0D7377', display: 'inline-block',
            marginBottom: '12px',
          })
          .on('click', () => {
            affinityNpyStatus = 'idle'
            affinityLoadError = null
            ensureAffinity(() => redraw())
          })
      } else if (!pathOk) {
        heatHost.append('div').text('Loading W…').st({fontSize: '11px', color: '#888', padding: '16px 0'})
        ensureAffinity(() => redraw())
      } else {
        // Build cluster-ordered node list (optionally top-by-influence)
        var ids = spec.node_ids || []
        var mFull = ids.length
        var shape = affinityNpy.shape || []
        var Wdata = affinityNpy.data
        var mW = shape[0] || Math.round(Math.sqrt(Wdata.length))
        if (mW !== mFull) {
          heatHost.append('div')
            .text('W shape mismatch: npy m=' + mW + ' vs node_ids=' + mFull + ' — re-export-spectral.')
            .st({fontSize: '11px', color: '#a23', padding: '12px 0'})
        } else {
          var idToIdx = {}
          ids.forEach((id, i) => { idToIdx[id] = i })

          // Members per group, sorted by |influence|
          var groupMembers = groups.map((grp, gi) => {
            return grp.slice(1).map(id => {
              var node = idToNode[id]
              return {
                id: id,
                idx: idToIdx[id],
                g: gi,
                node: node,
                infl: node ? Math.abs(+node.influence || 0) : 0,
                label: featureLabel(node, id),
              }
            }).filter(d => d.idx != null)
              .sort((a, b) => b.infl - a.infl)
          })

          var order = []
          if (spectralFocusGroup != null) {
            order = groupMembers[spectralFocusGroup].slice()
          } else {
            groupMembers.forEach(list => { order = order.concat(list) })
          }

          if (!spectralHeatShowAll && order.length > 64) {
            // Keep proportional top influence within each shown group
            if (spectralFocusGroup != null) {
              order = order.slice(0, 64)
            } else {
              var per = Math.max(2, Math.ceil(64 / Math.max(1, groups.length)))
              order = []
              groupMembers.forEach(list => {
                order = order.concat(list.slice(0, per))
              })
              if (order.length > 64) order = order.slice(0, 64)
            }
          }

          var nH = order.length
          if (!nH) {
            heatHost.append('div').text('No features in scope.').st({fontSize: '11px', color: '#999'})
          } else {
            // Quantile scale for W > 0 so sparse matrix stays readable
            var samples = []
            var step = Math.max(1, Math.floor(nH * nH / 8000))
            for (var ai = 0; ai < nH; ai++) {
              var ia = order[ai].idx
              for (var bi = ai; bi < nH; bi += step) {
                var v = +Wdata[ia * mW + order[bi].idx]
                if (v > 0) samples.push(v)
              }
            }
            samples.sort((a, b) => a - b)
            var wScale = samples.length
              ? samples[Math.min(samples.length - 1, Math.floor(samples.length * 0.92))]
              : 1
            wScale = Math.max(wScale, 1e-6)

            var heatPad = {top: 8, right: 8, bottom: 40, left: 48}
            var heatOuter = heatBox.node().getBoundingClientRect().width || 480
            var heatAvail = Math.floor(heatOuter - 20 - heatPad.left - heatPad.right)
            var disp = Math.max(220, Math.min(heatAvail, 640, Math.max(220, nH * 12)))
            var frame = heatHost.append('div').st({
              position: 'relative',
              width: (disp + heatPad.left + heatPad.right) + 'px',
              height: (disp + heatPad.top + heatPad.bottom) + 'px',
              maxWidth: '100%',
            })
            var cnv = frame.append('canvas').node()
            var pixN = Math.min(nH, 900)
            cnv.style.position = 'absolute'
            cnv.style.left = heatPad.left + 'px'
            cnv.style.top = heatPad.top + 'px'
            cnv.style.width = disp + 'px'
            cnv.style.height = disp + 'px'
            cnv.style.imageRendering = nH > 120 ? 'auto' : 'pixelated'
            cnv.style.border = '1px solid #DDD9CC'
            cnv.style.background = '#FBFAF5'
            cnv.style.cursor = 'crosshair'

            function lerpTeal(t) {
              var r0 = 0xF7, g0 = 0xF5, b0 = 0xEE
              var rgb = [0x00, 0x6B, 0x70]
              t = Math.max(0, Math.min(1, t))
              t = Math.pow(t, 0.4)
              return [
                (r0 + (rgb[0] - r0) * t) | 0,
                (g0 + (rgb[1] - g0) * t) | 0,
                (b0 + (rgb[2] - b0) * t) | 0,
              ]
            }

            cnv.width = pixN
            cnv.height = pixN
            var cx = cnv.getContext('2d')
            var im = cx.createImageData(pixN, pixN)
            var d4 = im.data
            for (var py = 0; py < pixN; py++) {
              var yi = Math.min(nH - 1, Math.floor(py * nH / pixN))
              var ia2 = order[yi].idx
              var gi = order[yi].g
              for (var px = 0; px < pixN; px++) {
                var xi = Math.min(nH - 1, Math.floor(px * nH / pixN))
                var w = +Wdata[ia2 * mW + order[xi].idx]
                var t = w / wScale
                var rgb = lerpTeal(t)
                // Dim off-focus blocks when a group is isolated (already filtered)
                // Soft-dim cross-group cells when showing all
                if (spectralFocusGroup == null && order[xi].g !== gi) {
                  // keep natural W (usually ~0); slight cool cast already from teal
                }
                var pix = (py * pixN + px) * 4
                d4[pix] = rgb[0]
                d4[pix + 1] = rgb[1]
                d4[pix + 2] = rgb[2]
                d4[pix + 3] = 255
              }
            }
            cx.putImageData(im, 0, 0)

            // Group boundary overlay
            var runs = []
            order.forEach((d, i) => {
              var last = runs[runs.length - 1]
              if (!last || last.g !== d.g) runs.push({g: d.g, start: i, end: i + 1})
              else last.end = i + 1
            })
            var overlay = frame.append('svg').at({
              width: disp + heatPad.left + heatPad.right,
              height: disp + heatPad.top + heatPad.bottom,
            }).st({
              position: 'absolute', left: 0, top: 0,
              width: (disp + heatPad.left + heatPad.right) + 'px',
              height: (disp + heatPad.top + heatPad.bottom) + 'px',
              pointerEvents: 'none',
            })
            var hg = overlay.append('g').at({
              transform: `translate(${heatPad.left},${heatPad.top})`,
            })
            var scale = disp / nH
            runs.slice(1).forEach(run => {
              var at = run.start * scale
              hg.append('line').at({
                x1: at, x2: at, y1: 0, y2: disp,
                stroke: '#111', strokeOpacity: 0.45, strokeWidth: 1,
              })
              hg.append('line').at({
                x1: 0, x2: disp, y1: at, y2: at,
                stroke: '#111', strokeOpacity: 0.45, strokeWidth: 1,
              })
            })
            // Colored group edge ticks
            runs.forEach(run => {
              var y0 = run.start * scale
              var y1 = run.end * scale
              hg.append('rect').at({
                x: -6, y: y0, width: 4, height: Math.max(1, y1 - y0),
                fill: modeColors[run.g % modeColors.length],
              })
              hg.append('rect').at({
                x: y0, y: disp + 2, width: Math.max(1, y1 - y0), height: 4,
                fill: modeColors[run.g % modeColors.length],
              })
              var mid = (y0 + y1) / 2
              if (y1 - y0 > 14) {
                hg.append('text').text(String(run.g + 1)).at({
                  x: -10, y: mid + 3, textAnchor: 'end',
                  fill: modeColors[run.g % modeColors.length], fontSize: 9, fontWeight: 700,
                })
              }
            })
            hg.append('text').text('feature j (grouped by spectral cluster)').at({
              x: disp / 2, y: disp + 28, textAnchor: 'middle', fill: '#888', fontSize: 9,
            })
            hg.append('text').text('feature i (grouped by spectral cluster)').at({
              transform: 'rotate(-90)',
              x: -disp / 2, y: -34, textAnchor: 'middle', fill: '#888', fontSize: 9,
            })

            // Hit layer for hover
            var hit = frame.append('div').st({
              position: 'absolute',
              left: heatPad.left + 'px', top: heatPad.top + 'px',
              width: disp + 'px', height: disp + 'px',
              cursor: 'crosshair',
            })
            hit.on('mousemove', function (ev) {
              var rect = this.getBoundingClientRect()
              var x = (ev.clientX - rect.left) / rect.width
              var y = (ev.clientY - rect.top) / rect.height
              var xi = Math.max(0, Math.min(nH - 1, Math.floor(x * nH)))
              var yi = Math.max(0, Math.min(nH - 1, Math.floor(y * nH)))
              var a = order[yi]
              var b = order[xi]
              var w = +Wdata[a.idx * mW + b.idx]
              hoverBar.html('')
              var row = hoverBar.append('div').st({
                display: 'flex', gap: '8px', alignItems: 'flex-start',
              })
              row.append('span').st({
                width: '9px', height: '9px', borderRadius: '50%', marginTop: '3px',
                flex: 'none', background: modeColors[a.g % modeColors.length],
              })
              var col = row.append('div')
              var la = a.label.length > 40 ? a.label.slice(0, 38) + '…' : a.label
              var lb = b.label.length > 40 ? b.label.slice(0, 38) + '…' : b.label
              col.append('div').text('W = ' + w.toFixed(4) + (w >= wScale ? ' (saturated)' : ''))
                .st({fontWeight: 600, color: '#222'})
              col.append('div')
                .text('i: [g' + (a.g + 1) + '] ' + la)
                .st({color: '#555'})
              col.append('div')
                .text('j: [g' + (b.g + 1) + '] ' + lb)
                .st({color: '#555'})
            })
            hit.on('mouseleave', () => {
              hoverBar.html('').append('span')
                .text('Hover a cell — Wᵢⱼ between two features. Click a group chip to isolate its block.')
                .st({color: '#999'})
            })
            hit.on('click', function (ev) {
              var rect = this.getBoundingClientRect()
              var y = (ev.clientY - rect.top) / rect.height
              var yi = Math.max(0, Math.min(nH - 1, Math.floor(y * nH)))
              if (order[yi] && order[yi].node) focusSvdFeature(order[yi].node)
            })

            heatHost.append('div')
              .text(
                nH + ' × ' + nH + ' shown'
                + (spectralHeatShowAll ? ' (all in scope)' : ' (top-by-|influence| preview)')
                + ' · color scale ~ P92 of W>0 = ' + wScale.toFixed(3)
                + (spectralFocusGroup != null ? ' · focused group ' + (spectralFocusGroup + 1) : '')
              )
              .st({fontSize: '10px', color: '#888', marginTop: '6px'})

            // Group chips
            var legend = heatHost.append('div').st({
              display: 'flex', flexWrap: 'wrap', gap: '4px', marginTop: '6px',
            })
            groups.forEach((grp, gi) => {
              var on = spectralFocusGroup == null || spectralFocusGroup === gi
              legend.append('div')
                .text((gi + 1) + ' · ' + (grp.length - 1))
                .st({
                  fontSize: '10px', padding: '2px 7px', borderRadius: '10px',
                  cursor: 'pointer', userSelect: 'none',
                  background: on ? modeColors[gi % modeColors.length] : '#EEE',
                  color: on ? '#fff' : '#888',
                  opacity: on ? 1 : 0.55,
                  border: '1px solid ' + (on ? modeColors[gi % modeColors.length] : '#DDD'),
                })
                .on('click', () => {
                  spectralFocusGroup = (spectralFocusGroup === gi) ? null : gi
                  redraw()
                })
            })
          }
        }
      }

      // —— Eigenvalue / eigengap spectrum ——
      var evals = (spec.eigenvalues || []).map(Number).filter(isFinite)
      var eigBox = layout.append('div').st({
        flex: '0 1 200px', minWidth: '160px', maxWidth: '240px',
        border: '1px solid #E4E2D8', borderRadius: '6px', padding: '10px',
        background: '#FBFAF5',
      })
      eigBox.append('div').text('Laplacian spectrum')
        .st({fontSize: '11px', fontWeight: 600, marginBottom: '2px'})
      eigBox.append('div')
        .text('Smallest eigenvalues of L_sym. Each λ is the cost of a successive balanced cut. Teal = the k coordinates used for clustering. A gap after λ_k suggests k groups.')
        .st({fontSize: '10px', color: '#888', marginBottom: '8px', lineHeight: '1.3'})

      if (evals.length) {
        var ew = 200, eh = 300
        var em = {top: 8, right: 10, bottom: 28, left: 36}
        var eiw = ew - em.left - em.right
        var eih = eh - em.top - em.bottom
        var esvg = eigBox.append('svg').at({width: '100%', viewBox: `0 0 ${ew} ${eh}`})
        var eg = esvg.append('g').at({transform: `translate(${em.left},${em.top})`})
        var xE = d3.scaleBand()
          .domain(evals.map((_, i) => i))
          .range([0, eiw])
          .padding(0.2)
        var yE = d3.scaleLinear()
          .domain([0, (d3.max(evals) || 1) * 1.08])
          .range([eih, 0])
          .nice()

        for (var giGap = 0; giGap < evals.length - 1; giGap++) {
          var mid = xE(giGap) + xE.bandwidth()
            + 0.5 * (xE(giGap + 1) - (xE(giGap) + xE.bandwidth()))
          var isGapAtK = (giGap + 1) === spectralK
          eg.append('line').at({
            x1: mid, x2: mid,
            y1: yE(evals[giGap + 1]), y2: yE(evals[giGap]),
            stroke: isGapAtK ? '#C45C26' : '#D8D4C8',
            strokeWidth: isGapAtK ? 2 : 1,
            strokeDasharray: isGapAtK ? null : '2,2',
            opacity: isGapAtK ? 0.9 : 0.45,
          })
        }

        eg.appendMany('rect', evals.map((v, i) => ({v: v, i: i}))).at({
          x: d => xE(d.i),
          y: d => yE(d.v),
          width: xE.bandwidth(),
          height: d => Math.max(0, eih - yE(d.v)),
          fill: d => (d.i < spectralK ? '#0D7377' : '#C8C4B4'),
          fillOpacity: d => (d.i < spectralK ? 0.9 : 0.55),
          rx: 1,
          cursor: 'pointer',
        })
          .on('click', (ev, d) => {
            var nextK = Math.max(kMin, Math.min(kMax, d.i + 1))
            if (nextK === spectralK) return
            spectralK = nextK
            slider.property('value', spectralK)
            kLabel.text(String(spectralK))
            redraw()
          })
          .append('title')
          .text(d => 'λ' + (d.i + 1) + ' = ' + d.v.toFixed(4)
            + ' · click to set k=' + (d.i + 1))

        eg.append('g').at({transform: `translate(0,${eih})`})
          .call(d3.axisBottom(xE).tickFormat(i => (i % 2 === 0 ? String(i + 1) : '')))
          .call(ax => ax.select('.domain').st({stroke: '#CCC'}))
          .selectAll('text').st({fontSize: '8px', fill: '#666'})
        eg.append('g')
          .call(d3.axisLeft(yE).ticks(4))
          .call(ax => ax.select('.domain').st({stroke: '#CCC'}))
          .selectAll('text').st({fontSize: '8px', fill: '#888'})

        var bestGap = -1
        var bestAt = -1
        for (var j = 2; j < Math.min(evals.length, kMax + 1); j++) {
          var ggap = evals[j] - evals[j - 1]
          if (ggap > bestGap) {
            bestGap = ggap
            bestAt = j
          }
        }
        if (bestAt >= 2) {
          eigBox.append('div')
            .text('Largest gap after λ' + bestAt + ' → try k=' + bestAt
              + (bestAt === spectralK ? ' (current)' : ' · click'))
            .st({
              fontSize: '10px', marginTop: '6px',
              color: bestAt === spectralK ? '#0D7377' : '#C45C26',
              cursor: bestAt === spectralK ? 'default' : 'pointer',
            })
            .on('click', () => {
              if (bestAt === spectralK) return
              spectralK = bestAt
              slider.property('value', spectralK)
              kLabel.text(String(spectralK))
              redraw()
            })
        }
      } else {
        eigBox.append('div').text('No eigenvalues stored.').st({fontSize: '11px', color: '#999'})
      }

      // —— Group cards ——
      var listSel = layout.append('div.svd-spectral-groups').st({
        flex: '1 1 260px', minWidth: '220px', maxWidth: '420px',
        display: 'flex', flexDirection: 'column', gap: '6px',
        maxHeight: '520px', overflowY: 'auto',
        paddingRight: '2px',
      })
      listSel.append('div')
        .text('Groups at k=' + spectralK + ' · click a header to isolate that block in the heatmap · click a feature to select it')
        .st({
          fontSize: '11px', fontWeight: 600, marginBottom: '2px',
          position: 'sticky', top: 0, background: '#fff', zIndex: 1, paddingBottom: '4px',
        })

      groups.forEach((grp, gi) => {
        var members = grp.slice(1)
        var focused = spectralFocusGroup == null || spectralFocusGroup === gi
        var card = listSel.append('div.svd-spectral-card')
          .attr('data-gi', gi)
          .st({
            border: '1px solid ' + (spectralFocusGroup === gi
              ? modeColors[gi % modeColors.length] : '#E4E2D8'),
            borderRadius: '5px', padding: '7px 9px',
            background: focused ? '#FAFAF8' : '#F3F2EC',
            opacity: focused ? 1 : 0.4,
          })
        var hdr = card.append('div').st({
          display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px',
          cursor: 'pointer', userSelect: 'none',
        })
          .on('click', () => {
            spectralFocusGroup = (spectralFocusGroup === gi) ? null : gi
            redraw()
          })
        hdr.append('span').st({
          width: '10px', height: '10px', borderRadius: '50%',
          background: modeColors[gi % modeColors.length], display: 'inline-block',
          boxShadow: spectralFocusGroup === gi ? '0 0 0 2px rgba(0,0,0,0.15)' : 'none',
        })
        hdr.append('span').text(grp[0] + ' · ' + members.length)
          .st({fontSize: '11px', fontWeight: 600, flex: '1'})
        hdr.append('span')
          .text(spectralFocusGroup === gi ? 'clear' : 'focus')
          .st({fontSize: '9px', color: '#0D7377'})

        var top = members.slice(0, spectralFocusGroup === gi ? 14 : 5)
        top.forEach(id => {
          var node = idToNode[id]
          var label = featureLabel(node, id)
          if (label.length > 48) label = label.slice(0, 46) + '…'
          card.append('div')
            .text(label)
            .st({
              fontSize: '10px', color: '#444', cursor: node ? 'pointer' : 'default',
              padding: '1px 0', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
            })
            .on('click', () => { if (node) focusSvdFeature(node) })
        })
        if (members.length > top.length) {
          card.append('div').text('… +' + (members.length - top.length) + ' more')
            .st({fontSize: '10px', color: '#999'})
        }
      })
    }

    // Always redraw; ensureAffinity registers waiters so panel re-renders
    // during an in-flight fetch still get notified (avoids stuck "Loading W…").
    redraw()
  }

  ensureSvdBundle(() => render())
  // Recompute when prune reloads the whole CG (formatData); also if thickness etc. fire.
  if (renderAll.pinnedIds) renderAll.pinnedIds.fns['svdPanel'] = render
  if (renderAll.edgeColorMode) renderAll.edgeColorMode.fns['svdPanel'] = render
}

window.init?.()
