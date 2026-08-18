window.initCgSubgraph = function ({visState, renderAll, data, cgSel, opts}) {
  opts = opts || {}
  var mode = opts.mode || 'curated'            // 'curated' | 'pruned' | 'computed'
  var isCurated = mode === 'curated'
  var isComputed = mode === 'computed'         // k-means clusters from backend
  var isPruned = mode === 'pruned'
  var selClass = opts.selClass || '.subgraph'
  var titleLabel = opts.title || 'Subgraph'
  var stateKey = opts.stateKey || mode
  var maxPrunedNodes = opts.maxNodes || 30
  var forceNodesKey = '_selForceNodes_' + stateKey
  var autoClusters = (data.metadata && data.metadata.auto_clusters) || null
  var spectralClusters = (data.metadata && data.metadata.spectral_clusters) || null
  // Computed panel: agglomerative (auto_clusters) vs spectral (spectral_clusters).
  // Persist choice on the function so re-renders keep the toggle.
  if (isComputed && initCgSubgraph._clusterSource == null) {
    initCgSubgraph._clusterSource = spectralClusters ? 'spectral' : 'agglomerative'
  }
  function activeClusterMeta() {
    if (!isComputed) return null
    var src = initCgSubgraph._clusterSource || 'agglomerative'
    if (src === 'spectral' && spectralClusters) return spectralClusters
    if (autoClusters) return autoClusters
    return spectralClusters
  }
  function activeClusterSource() {
    var meta = activeClusterMeta()
    if (!meta) return null
    if (meta === spectralClusters) return 'spectral'
    return 'agglomerative'
  }

  // Live-adjustable number of groups for the computed panel (persists across
  // re-renders so the k-slider stays put).
  var computedK = null
  var clusterMeta0 = activeClusterMeta()
  if (isComputed && clusterMeta0) {
    computedK = clusterMeta0.default_k || clusterMeta0.k_min || 2
  }
  function computedSupernodesForK() {
    var meta = activeClusterMeta()
    if (!meta) return []
    var byK = meta.by_k
    var sns = byK ? (byK[String(computedK)] || byK[String(meta.default_k)] || []) : []
    return sns.map(sn => sn.slice())
  }

  var subgraphSel = cgSel.select(selClass)
  if (subgraphSel.empty()) return
  subgraphSel.datum().resizeFn = renderSubgraph

  var nodeIdToNode = {}
  var sgNodes = []
  var sgLinks = []

  let nodeSel = null
  let memberNodeSel = null
  let simulation = null

  var nodeWidth = 88
  var nodeHeight = 30

  function supernodesToUrl() {
    if (!isCurated) return  // auto-derived panels never write URL state
    util.params.set('supernodes', JSON.stringify(subgraphState.supernodes))
  }

  // Curated panel owns visState.subgraph (persisted, editable). Auto panels keep
  // their own local, non-persisted state: 'pruned' derives nodes from pruning,
  // 'computed' uses the backend k-means clusters (metadata.auto_clusters).
  var subgraphState
  if (!isCurated) {
    subgraphState = {
      sticky: true,
      dagrefy: true,
      supernodes: isComputed ? computedSupernodesForK() : [],
      activeGrouping: {isActive: false, selectedNodeIds: new Set(), isDragging: false},
    }
  } else {
    subgraphState = visState.subgraph = visState.subgraph || {
      sticky: true,
      dagrefy: true,
      supernodes: visState.supernodes || [],
      activeGrouping: {
        isActive: false,
        selectedNodeIds: new Set(),
        isDragging: false,
      }
    }
  }

  // Which node ids the panel shows.
  function getPinnedIds() {
    if (isCurated) return visState.pinnedIds.slice(0, 200)
    if (isComputed) {
      var meta = activeClusterMeta()
      return meta ? meta.node_ids.slice() : []
    }
    // pruned: data.nodes is already pruned by formatData; rank survivors by |influence|.
    var ranked = d3.sort(nodes, d => -(Math.abs(d.influence || 0)))
    var ids = ranked
      .filter(d => d.feature_type !== 'logit')
      .slice(0, maxPrunedNodes)
      .map(d => d.nodeId)
    nodes.filter(d => d.isLogit).forEach(d => { if (!ids.includes(d.nodeId)) ids.push(d.nodeId) })
    return ids
  }

  if (isCurated) d3.select('body')
    .on('keydown.grouping' + stateKey + data.metadata.slug, ev => {
      if (ev.repeat) return
      if (!visState.isEditMode || ev.key != 'g') return
      subgraphState.activeGrouping.isActive = true
      styleNodes()
      
      subgraphSel.classed('is-grouping', true)
    })
    .on('keyup.grouping' + stateKey + data.metadata.slug, ev => {
      if (!visState.isEditMode || ev.key != 'g') return
      if (subgraphState.activeGrouping.selectedNodeIds.size > 1){
        var allSelectedIds = []
        var prevSupernodeLabel = ''
        subgraphState.activeGrouping.selectedNodeIds.forEach(id => {
          var node = nodeIdToNode[id]
          if (!node?.memberNodeIds) return allSelectedIds.push(id)
          prevSupernodeLabel = node.ppClerp

          // if a supernode is selected, remove the previous super node
          subgraphState.supernodes = subgraphState.supernodes.filter(([label, ...nodeIds]) =>
            !nodeIds.every(d => node.memberNodeIds.includes(d))
          )
          // and adds its member nodes to selection
          node.memberNodeIds.forEach(id => allSelectedIds.push(id))
        })

        var label = prevSupernodeLabel || allSelectedIds
          .map(id => nodeIdToNode[id]?.ppClerp)
          .find(d => d) || 'supernode'
        subgraphState.supernodes.push([label, ...new Set(allSelectedIds)])
        supernodesToUrl()
      }
      subgraphState.activeGrouping.isActive = false
      subgraphState.activeGrouping.selectedNodeIds.clear()
      renderSubgraph()
      
      subgraphSel.classed('is-grouping', false)
    })

  let {nodes, links} = data
  // Computed clusters always use the full unpruned graph so pruning the main
  // view never drops auto-cluster members.
  if (isComputed) {
    nodes = data.allNodes || nodes
    links = data.allLinks || links
  }

  function resolveNode(id) {
    if (isComputed && data.allNodes?.idToNode?.[id]) return data.allNodes.idToNode[id]
    return nodes.idToNode?.[id] || nodes.find(d => d.nodeId === id)
  }
  
  function renderSubgraph() {
    var c = d3.conventions({
      sel: subgraphSel.html(''),
      margin: {top: 36, bottom: 16, left: visState.isHideLayer ? 8 : 40, right: 16},
      layers: 'sd',
    })
    // subgraphSel.st({borderTop: '1px solid #eee'})
    
    var titleSel = c.svg.append('text.section-title').text(titleLabel).translate(-16, 1)
    c.svg.append('g.border-path').append('path')
      .at({stroke: '#eee', d: 'M 0 -10 H ' + c.width})


    var [svg, div] = c.layers

    // // set up arrowheads
    // svg.appendMany('marker', [{id: 'mid-negative', color: '#40004b'},{id: 'mid-positive', color: '#00441b'}])
    //   .at({id: d => d.id, orient: 'auto', refX: .1, refY: 1}) // marker-height/marker-width?
    //   .append('path')
    //   .at({d: 'M0,0 V2 L1,1 Z', fill: d => d.color})


    // pick out the subgraph and do supernode surgery
    // Computed panel: resolve members from the unpruned id map and never apply
    // the influence prune — auto clusters are a fixed analysis.
    var pinnedIds = getPinnedIds()
    var thr = parseFloat(visState.pruningThreshold)
    var pinnedNodes
    if (isComputed) {
      var idMap = (data.allNodes && data.allNodes.idToNode) || {}
      pinnedNodes = pinnedIds.map(id => idMap[id] || resolveNode(id)).filter(Boolean)
      // Clear supernode marks only on the nodes we actually show.
      pinnedNodes.forEach(d => { d.supernodeId = null })
    } else {
      nodes.forEach(d => d.supernodeId = null)
      function survivesPruning(d) {
        if (!Number.isFinite(thr) || thr >= 0.995) return true
        return d.feature_type === 'embedding' ||
               d.feature_type === 'logit' ||
               (d.influence != null && d.influence <= thr)
      }
      pinnedNodes = nodes.filter(d => pinnedIds.includes(d.nodeId) && survivesPruning(d))
    }

    // create supernodes and mark their children
    nodeIdToNode = Object.fromEntries(pinnedNodes.map(d => [d.nodeId, d]))
    var supernodes = subgraphState.supernodes
      .map(([label, ...nodeIds], i) => {
        var nodeId = nodeIdToNode[label] ? `supernode-${i}` : label
        var memberNodes = nodeIds
          .map(id => nodeIdToNode[id] || resolveNode(id))
          .filter(d => d)
        memberNodes.forEach(d => {
          d.supernodeId = nodeId
          nodeIdToNode[d.nodeId] = d
        })
  
        var rv = {
          nodeId,
          featureId: `supernode-${i}`,
          ppClerp: label,
          layer: d3.mean(memberNodes, d => +d.layer),
          ctx_idx: d3.mean(memberNodes, d => d.ctx_idx),
          ppLayer: d3.extent(memberNodes, d => +d.layer).join('—'),
          streamIdx: d3.mean(memberNodes, d => d.streamIdx),
          memberNodeIds: nodeIds,
          memberNodes,
          isSuperNode: true,
        }
        nodeIdToNode[rv.nodeId] = rv
  
        return rv
      })
      .filter(d => d.memberNodes.length)
    
    // update clerps — fragile hack if hClerpUpdate changes
    // nodes.forEach(d => d.ppClerp = d.localClerp || d.clerp)
    supernodes.forEach(({ppClerp, memberNodes}) => {
      if (memberNodes.length == 1 && ppClerp == memberNodes[0].ppClerp) return
      
      memberNodes.forEach(d => {
        const nodeClerp = (d.localClerp || d.clerp || '').trim()
        const feat = d.feature != null && d.feature !== ''
          ? String(d.feature)
          : String(d.featureId || d.nodeId || '').split('_')[1] || ''
        const extra = (nodeClerp && nodeClerp !== ppClerp) ? nodeClerp : feat
        d.ppClerp = extra ? `[${ppClerp}] ${extra}` : `[${ppClerp}]`
      })
    })
    
    // inputAbsSumExternalSn: the abs sum of input links from outside the supernode
    pinnedNodes.forEach(d => {
      d.inputAbsSumExternalSn = d3.sum(d.sourceLinks, e => {
        if (!e.sourceNode.supernodeId) return Math.abs(e.weight)
        return e.sourceNode.supernodeId == d.supernodeId ? 0 : Math.abs(e.weight)
      })
      d.sgSnInputWeighting = d.inputAbsSumExternalSn/d.inputAbsSum
    })

    // subgraph plots pinnedNodes not in a supernode and supernodes
    sgNodes = pinnedNodes.filter(d => !d.supernodeId).concat(supernodes)

    // Header count: how many pinned nodes are actually shown vs pinned, and
    // whether the graph is pruned. This matches the main-graph pruning exactly
    // (same influence <= threshold rule as util-cg.formatData).
    var pinnedTotal = new Set(pinnedIds).size
    var shownFeatureNodes = pinnedNodes.length
    var graphIsPruned = Number.isFinite(thr) && thr < 0.995
    var prunedOut = pinnedTotal - shownFeatureNodes
    var titleText = titleLabel + ' · ' + shownFeatureNodes + ' node' + (shownFeatureNodes == 1 ? '' : 's')
    if (supernodes.length) titleText += ' in ' + sgNodes.length + ' groups'
    if (isComputed) {
      var src = activeClusterSource()
      var meta = activeClusterMeta()
      if (!meta) {
        titleText += ' · run `export-spectral` or `verify`'
      } else if (src === 'spectral') {
        titleText += ' · spectral (max cosine→0, kNN=' + (meta.knn || '?') + ', n=' + meta.m
          + (meta.row_normalize === false ? ', raw evecs' : '') + ')'
      } else {
        titleText += ' · cosine agglomerative (n=' + meta.m + ')'
      }
    } else if (isPruned) {
      titleText += graphIsPruned ? ' · top by influence @ ' + thr.toFixed(2) : ' · top by influence (full)'
    } else if (graphIsPruned) {
      titleText += ' · pruned to ' + thr.toFixed(2)
      if (prunedOut > 0) titleText += ' (' + prunedOut + ' pinned hidden)'
    } else {
      titleText += ' · full graph'
    }
    titleSel.text(titleText)

    // Method toggle + live k-slider for computed clusters.
    if (isComputed && (autoClusters || spectralClusters)) {
      var meta = activeClusterMeta()
      subgraphSel.st({position: 'relative'})
      var ctrl = subgraphSel.append('div.k-slider').st({
        position: 'absolute', top: '5px', right: '10px', zIndex: 20,
        display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '4px',
        fontSize: '10px', color: '#555', fontFamily: 'system-ui, sans-serif',
        background: 'rgba(247,246,241,0.92)', border: '1px solid #E4E2D8',
        borderRadius: '5px', padding: '4px 7px',
      })
      if (autoClusters && spectralClusters) {
        var srcRow = ctrl.append('div').st({display: 'flex', gap: '0'})
        ;[
          {id: 'agglomerative', label: 'Hierarchical'},
          {id: 'spectral', label: 'Spectral'},
        ].forEach(opt => {
          var on = (initCgSubgraph._clusterSource || 'spectral') === opt.id
          srcRow.append('div')
            .text(opt.label)
            .st({
              padding: '2px 6px', border: '1px solid #ccc', marginLeft: '-1px',
              cursor: 'pointer', userSelect: 'none',
              background: on ? '#0D7377' : '#fff',
              color: on ? '#fff' : '#333',
              borderColor: on ? '#0D7377' : '#ccc',
              fontSize: '10px',
            })
            .on('click', () => {
              initCgSubgraph._clusterSource = opt.id
              var m2 = activeClusterMeta()
              computedK = m2.default_k || m2.k_min || 2
              subgraphState.supernodes = computedSupernodesForK()
              renderSubgraph()
            })
        })
      }
      if (meta && meta.by_k) {
        var kMin = meta.k_min || 2
        var kMax = meta.k_max || kMin
        computedK = Math.max(kMin, Math.min(kMax, computedK || meta.default_k || kMin))
        var srcNow = activeClusterSource()
        var row = ctrl.append('div').st({display: 'flex', alignItems: 'center', gap: '5px'})
        row.append('span').text(srcNow === 'spectral' ? 'groups (raw evecs)' : 'groups')
        var slider = row.append('input')
          .at({type: 'range', min: kMin, max: kMax, step: 1, value: computedK})
          .st({width: '84px', cursor: 'pointer'})
        var kLabel = row.append('span').text(computedK)
          .st({fontWeight: 700, minWidth: '12px', textAlign: 'right', color: '#0D7377'})
        slider.on('input', function () { kLabel.text(this.value) })
        slider.on('change', function () {
          computedK = +this.value
          subgraphState.supernodes = computedSupernodesForK()
          renderSubgraph()
        })
      }
    }

    sgNodes.forEach(d => {
      // for supernodes, sum up values from member nodes
      if (d.isSuperNode) {
        d.inputAbsSum = d3.sum(d.memberNodes, e => e.inputAbsSum)
        d.inputAbsSumExternalSn = d3.sum(d.memberNodes, e => e.inputAbsSumExternalSn)
      } else {
        d.memberNodes = [d]
      }

      var sum = d3.sum(d.memberNodes, e => e.sgSnInputWeighting)
      d.memberNodes.forEach(e => e.sgSnInputWeighting = e.sgSnInputWeighting/sum)
    })

    // select subgraph links
    sgLinks = links
      .filter(d => nodeIdToNode[d.sourceNode.nodeId] && nodeIdToNode[d.targetNode.nodeId])
      .map(d => ({
        source: d.sourceNode.nodeId,
        target: d.targetNode.nodeId,
        weight: d.weight,
        color: d.color || d.pctInputColor,
        colorClass: d.colorClass,
        intensity: d.intensity,
        ogLink: d,
      }))

    // then remap source/target to supernodes
    sgLinks.forEach(link => {
      if (nodeIdToNode[link.source]?.supernodeId) link.source = nodeIdToNode[link.source].supernodeId
      if (nodeIdToNode[link.target]?.supernodeId) link.target = nodeIdToNode[link.target].supernodeId
    })

    // finally combine parallel links and remove self-links
    sgLinks = d3.nestBy(sgLinks, d => d.source + '-' + d.target)
      .map(links => {
        var weight = d3.sum(links, link => {
          var {inputAbsSumExternalSn, sgSnInputWeighting} = link.ogLink.targetNode
          return link.weight/inputAbsSumExternalSn*sgSnInputWeighting
        })

        // Prefer strongest underlying colored edge's palette color as seed
        var strongest = links[0]
        links.forEach(d => {
          if (Math.abs(d.weight) > Math.abs(strongest.weight)) strongest = d
        })
        return {
          source: links[0].source,
          target: links[0].target,
          weight,
          color: strongest?.color || utilCg.pctInputColorFn(weight),
          colorClass: strongest?.colorClass,
          intensity: strongest?.intensity,
          pctInput: weight,
          pctInputColor: utilCg.pctInputColorFn(weight),
          ogLinks: links
        }
      })
      .filter(d => d.source !== d.target)

    // Re-color aggregate subgraph edges with the active k-color heuristic
    if (window.edgeColoring?.applySubgraphLinks) {
      window.edgeColoring.applySubgraphLinks(sgLinks, visState.edgeColorMode || 'greedy4')
    }
    sgLinks = d3.sort(sgLinks, d => Math.abs(d.weight))

    if (!sgNodes.length) {
      var emptyMsg = 'No pinned nodes — pin features or load a cluster.'
      if (isComputed) emptyMsg = autoClusters
        ? 'No clustered nodes to show.'
        : 'No computed clusters — run `python -m interpretability_graph verify`.'
      else if (isPruned) emptyMsg = 'No nodes survive the current pruning threshold.'
      c.svg.append('text')
        .text(emptyMsg)
        .translate([0, 20]).at({fontSize: 12, fill: '#999'})
      return
    }

    let xScale = d3.scaleLinear()
      .domain(d3.extent(sgNodes.map(d => d.ctx_idx)))
      .range([0, c.width*3/4])
    let yScale = d3.scaleLinear()
      .domain(d3.extent(sgNodes.map(d => d.streamIdx)).toReversed())
      .range([0, c.height - nodeHeight])

    // d3.force is impure, need copy
    // Also want to persist these positions across node changes (per-panel cache)
    const existingNodes = window[forceNodesKey] && new Map(window[forceNodesKey].map(n => [n.node.nodeId, n]))
    window[forceNodesKey] = sgNodes.map(node => {
      const existing = existingNodes?.get(node.nodeId)
      return {
        x: existing ? existing.x : xScale(node.ctx_idx),
        y: existing ? existing.y : yScale(node.streamIdx),
        fx: existing?.fx,
        fy: existing?.fy,
        nodeId: node.nodeId, // for addFeatureEvents
        featureId: node.featureId, // for addFeatureEvents
        node,
        sortedSlug: d3.sort(node.memberNodes.map(d => d.featureIndex).join(' ')),
      }
    })
    

    var selForceNodes = window[forceNodesKey] = d3.sort(window[forceNodesKey], d => d.sortedSlug)
    if (isCurated) window._exportSubgraphPos = function(){
      return selForceNodes.map(d => [d.x/c.width*1000, d.y/c.height*1000].map(Math.round)).flat().join(' ')
    }

    if (simulation) simulation.stop()
    simulation = d3.forceSimulation(selForceNodes)
      .force('link', d3.forceLink(sgLinks).id(d => d.node.nodeId))
      .force('charge', d3.forceManyBody())
      .force('collide', d3.forceCollide(Math.sqrt(nodeHeight ** 2 + nodeWidth ** 2) / 2 + 8))
      .force('container', forceContainer([[-10, 0], [c.width - nodeHeight, c.height - nodeHeight]]))
      .force('x', d3.forceX(d => xScale(d.node.ctx_idx)).strength(.1))
      .force('y', d3.forceY(d => yScale(d.node.streamIdx)).strength(2))

    var svgPaths = svg.appendMany('path.link-path', sgLinks).at({
      fill: 'none',
      markerMid: d => d.weight > 0 ? 'url(#mid-positive)' : 'url(#mid-negative)',
      strokeWidth: d => Math.max(1.2, Math.abs(d.weight) * 18),
      stroke: d => d.color,
      opacity: 0.9,
      strokeLinecap: 'round',
    })

    var edgeLabels = svg.appendMany('text.weight-label', sgLinks)
      // .text(d => d3.format('+.2f')(d.weight))

    simulation.on('tick', renderForce)

    var drag = d3.drag()
      .on('drag', (ev) => {
        // Only when actually dragging, mark as no longer dagre positioned and restart sim
        subgraphState.activeGrouping.isDragging = true
        ev.subject.dagrePositioned = false
        if (!ev.active) simulation.alphaTarget(0.3).restart()
        ev.subject.fx = ev.subject.x = ev.x
        ev.subject.fy = ev.subject.y = ev.y
        renderForce()
      })
      .on('end', (ev) => {
        subgraphState.activeGrouping.isDragging = false
        if (!ev.active) simulation.alphaTarget(0)
        if (!subgraphState.sticky && !ev.subject.dagrePositioned){
          ev.subject.fx = null
          ev.subject.fy = null
        }
      })

    nodeSel = div
      .appendMany('div.supernode-container', selForceNodes)
      .translate(d => [d.x, d.y])
      .st({width: nodeWidth, height: nodeHeight})
      .call(utilCg.addFeatureEvents(visState, renderAll, ev => ev.shiftKey))
      .on('click.group', (ev, d) => {
        var {isActive, selectedNodeIds} = subgraphState.activeGrouping
        if (!isActive) return

        // If it's a child node, use its parent supernode's ID instead
        var nodeId = d.supernodeId || d.nodeId
        selectedNodeIds.has(nodeId) ? selectedNodeIds.delete(nodeId) : selectedNodeIds.add(nodeId)
        
        styleNodes()
        ev.stopPropagation()
        ev.preventDefault()
      })
      .call(drag)

    selForceNodes.forEach(d => {
      if (!d.node.memberNodes) d.node.memberNodes = [d.node]
    })

    var supernodeSel = nodeSel//.filter(d => d.node.isSuperNode)
      .classed('is-supernode', true)
      .st({height: nodeHeight + 12})

    memberNodeSel = supernodeSel.append('div.member-circles')
      .st({
        width: d => d.node.memberNodes.length <= 4 ? 'auto' : 'calc(32px + 12px)', 
        gap: d => d.node.memberNodes.length <= 4 ? 4 : 0,
      })
      .appendMany('div.member-circle', d => d.node.memberNodes)
      .classed('not-clt-feature', d => d.feature_type != 'cross layer transcoder')
      .st({marginLeft: function(d, i) {
          var n = this.parentNode.childNodes.length  
          return n <= 4 ? 0 : i == 0 ? 0 : -((n - 4)*8)/(n - 1)
      }})
      .call(utilCg.addFeatureEvents(visState, renderAll, ev => ev.shiftKey))
      .on('click.stop-parent', ev => {
        if (!subgraphState.activeGrouping.isActive) ev.stopPropagation()
      })  
      .on('mouseover.stop-parent', ev => ev.stopPropagation())
      .at({title: d => d.ppClerp})
      .st({background: d => d.nodeColor || null})

    if (visState.isEditMode) {
      // TODO: enable
      supernodeSel.select('.member-circles')
        .filter(d => d.node.isSuperNode)
        .append('div.ungroup-btn')
        .text('×').st({top: 2, left: -15, position: 'absolute'})
        .on('click', (ev, d) => {
          ev.stopPropagation()
          
          subgraphState.supernodes = subgraphState.supernodes.filter(([label, ...nodeIds]) =>
            !nodeIds.every(id => d.node.memberNodeIds.includes(id))
          )
          supernodesToUrl()
          renderSubgraph()
        })
    }

    var nodeTextSel = nodeSel.append('div.node-text-container')
    nodeTextSel.append('span')
      .text(d => d.node.ppClerp)
      .on('click', (ev, d) => {
        if (!visState.isEditMode) return
        if (!d.node.isSuperNode) return
        // TODO: enable?
        // return
        ev.stopPropagation()

        var spanSel = d3.select(ev.target).st({display: 'none'})
        var input = d3.select(spanSel.node().parentNode).append('input')
          .at({class: 'temp-edit', value: spanSel.text()})
          .on('blur', save)
          .on('keydown', ev => {
            if (ev.key === 'Enter'){
              save()
              input.node().blur()
            }
            ev.stopPropagation()
          })

        input.node().focus()

        function save(){
          var idx = subgraphState.supernodes.findIndex(([label, ...nodeIds]) =>
            nodeIds.every(id => d.node.memberNodeIds.includes(id))
          )
          if (idx >= 0){
            subgraphState.supernodes[idx][0] = input.node().value || 'supernode'
            supernodesToUrl()
            renderSubgraph()
          }
        }
      })


    nodeTextSel.each(function(d) {
      d.textHeight = this.getBoundingClientRect().height || -8
    })

    nodeSel.append('div.clicked-weight.source')
    nodeSel.append('div.clicked-weight.target')
    styleNodes()


    var checkboxes = Object.entries({
      sticky: () => {
        // simulation.alphaTarget(0.3).restart()
        if (!subgraphState.sticky) unsticky()
      },
      dagrefy: () => {
        subgraphState.dagrefy ? dagrefy() : selForceNodes.forEach(d => d.dagrePositioned = null)
      },
    }).map(([key, fn]) => ({key, fn}))


    if (visState.isEditMode) {
      div.append('div.checkbox-container').translate([-c.margin.left, c.margin.bottom])
        .appendMany('label', checkboxes).append('input')
        .at({type: 'checkbox'})
        .property('checked', d => subgraphState[d.key])
        .on('change', function(ev, d){
          subgraphState[d.key] = this.checked
          d.fn()
        })
        .parent().append('span').text(d => d.key)
    }

    checkboxes.forEach(d => d.fn())

    function unsticky(){
      selForceNodes.forEach(d => (d.fx = d.fy = null))
      simulation.alphaTarget(0.3).restart()
      if (subgraphState.dagrefy) {
        subgraphState.dagrefy = false
        d3.select('.checkbox-container').selectAll('input').filter(d => d.key == 'dagrefy').property('checked', 0)
        checkboxes.find(d => d.key == 'dagrefy').fn()
      }
    }

    function dagrefy(){
      // Saved hand-tuned positions only apply to the curated panel.
      if (isCurated && visState.sg_pos){
        var nums = visState.sg_pos.split(' ').map(d => +d)
        selForceNodes.forEach((d, i) => {
          d.fx = d.x = nums[i*2 + 0]/1000*c.width
          d.fy = d.y = nums[i*2 + 1]/1000*c.height
        })

        nodeSel.translate(d => [d.x, d.y])
        styleNodes()
        renderEdges()

        visState.og_sg_pos = visState.sg_pos
        delete visState.sg_pos
      }
      if (isCurated && visState.og_sg_pos) return


      var g = new window.dagre.graphlib.Graph()
      g.setGraph({rankdir: 'BT', nodesep: 20, ranksep: 20})
      g.setDefaultEdgeLabel(() => ({}))

      sgLinks.forEach(d =>{
        if (Math.abs(d.weight) < .003) return
        // set width and height to make dagre return x and y for edges
        g.setEdge(d.source.nodeId, d.target.nodeId, {width: 1, height: 1, labelpos: 'c', weight: Math.abs(d.weight)})
      })
      sgNodes.forEach(d => {
        g.setNode(d.nodeId, {width: nodeWidth, height: nodeHeight})
      })

      window.dagre.layout(g)

      // rescale to fit container
      var xs = d3.scaleLinear([0, g.graph().width], [0, Math.min(c.width, g.graph().width)])
      var ys = d3.scaleLinear([0, g.graph().height], [0, Math.min(c.height, g.graph().height)])

      // flip to make ctx_idx left to right and streamIdx bottom to top
      var w0 = d3.mean(selForceNodes, d =>  g.node(d.nodeId).x*d.node.ctx_idx)
      var w1 = d3.mean(selForceNodes, d => -g.node(d.nodeId).x*d.node.ctx_idx)
      if (w0 < w1) xs.range(xs.range().reverse())

      var w0 = d3.mean(selForceNodes, d => g.node(d.nodeId).y*d.node.streamIdx)
      var w1 = d3.mean(selForceNodes, d => -g.node(d.nodeId).y*d.node.streamIdx)
      if (w0 < w1) ys.range(ys.range().reverse())

      for (var node of window[forceNodesKey]) {
        var pos = g.node(node.nodeId)
        node.fx = node.x = xs(pos.x) - nodeWidth/2
        node.fy = node.y = ys(pos.y) - nodeHeight/2
        node.dagrePositioned = true
      }

      // var curveFactory = d3.line(d => d.x, d => d.y).curve(d3.curveBasis)
      // svgPaths.at({d: d => {
      //   var points = g.edge(d.source.nodeId, d.target.nodeId)?.points
      //   if (!points) return ''
      //   return curveFactory(points.map(p => ({x: xs(p.x), y: ys(p.y)})))
      // }})
      renderEdges()

      // edgeLabels.translate(d => {
      //   var pos = g.edge(d.source.nodeId, d.target.nodeId)
      //   if (!pos) return [-100, -100]
      //   return [xs(pos.x), ys(pos.y)]
      // })
      styleNodes()
    }

    function renderForce(){
      nodeSel.translate(d => [d.x, d.y])

      renderEdges()

      edgeLabels
        .filter(d => !(d.source.dagrePositioned && d.target.dagrePositioned))
        .translate(d => [
          (d.source.x + d.target.x) / 2 + nodeWidth / 2,
          (d.source.y + d.target.y) / 2 + nodeHeight / 2
        ])
    }
    
    function renderEdges(){
      
      // TODO: use actual strokeWidth to spread
      d3.nestBy(sgLinks, d => d.source).forEach(links => {
        // if (links[0].source.nodeId == '6_12890134_-0') debugger
        var numSlots = links[0].source.node.memberNodes.length
        var totalWidth = (Math.min(4, numSlots))*8
        d3.sort(links, d => Math.atan2(d.target.y - d.source.y, d.target.x - d.source.x))
          .forEach((d, i) => d.sourceOffsetX = (i - links.length/2)*totalWidth/links.length)
      })

      d3.nestBy(sgLinks, d => d.target).forEach(links => {
        var numSlots = links[0].target.node.memberNodes.length
        var totalWidth = (Math.min(4, numSlots) + 1)*3
        d3.sort(links, d => -Math.atan2(d.source.y - d.target.y, d.source.x - d.target.x))
          .forEach((d, i) => d.targetOffsetX = (i - links.length/2)*totalWidth/links.length)
      })

      svgPaths.at({
        d: d => {
          var x0 = d.source.x + nodeWidth/2 + d.sourceOffsetX
          var y0 = d.source.y 
          var x1 = d.target.x + nodeWidth/2 + d.targetOffsetX
          var y1 = d.target.y + d.target.textHeight + 28

          return `M${x0},${y0} L${x1},${y1}`
        },
      })
    }
  }


  function styleNodes() {
    if (!nodeSel) return

    nodeSel
      .classed('clicked', d => d.nodeId == visState.clickedId)
      .classed('hovered', d => d.featureId == visState.hoveredId)
      .st({zIndex: d => Math.round(d.x*20 + d.y) + 1000})
      .classed('grouping-selected', d => subgraphState.activeGrouping.selectedNodeIds.has(d.nodeId))

    memberNodeSel
      .classed('clicked', d => d.nodeId == visState.clickedId)
      .classed('hovered', d => d.featureId == visState.hoveredId)
      .st({
        background: d => d.tmpClickedLink?.pctInputColor,
        color: d => utilCg.bgColorToTextColor(d.tmpClickedLink?.pctInputColor)
      })
      // .at({title: d => d3.format('.1%')(d.tmpClickedLink?.pctInput)})



    // style clicked links using supernode adjusted graph when possible
    sgNodes.forEach(d => {
      d.tmpClickedSgSource = d.tmpClickedLink?.sourceNode == d ? d.tmpClickedLink : null
      d.tmpClickedSgTarget = d.tmpClickedLink?.targetNode == d ? d.tmpClickedLink : null
    })

    if (visState.clickedId) {
      sgLinks.forEach(d => {
        if (d.source.nodeId == visState.clickedId) nodeIdToNode[d.target.nodeId].tmpClickedSgTarget = d
        if (d.target.nodeId == visState.clickedId) nodeIdToNode[d.source.nodeId].tmpClickedSgSource = d
      })
    }

    // nodeSel.selectAll('.clicked-weight.source')
    //   .st({display: d => d.node.tmpClickedSgSource ? '' : 'none'})
    //   .filter(d => d.node.tmpClickedSgSource)
    //   .text(d => d3.format('.1%')(d.node.tmpClickedSgSource.pctInput))
    //   .st({
    //     background: d => d.node.tmpClickedSgSource.pctInputColor,
    //     color: d => utilCg.bgColorToTextColor(d.node.tmpClickedSgSource.pctInputColor)
    //   })

    // nodeSel.selectAll('.clicked-weight.target')
    //   .st({display: d => d.node.tmpClickedSgTarget ? '' : 'none'})
    //   .filter(d => d.node.tmpClickedSgTarget)
    //   .text(d => d3.format('.1%')(d.node.tmpClickedSgTarget.pctInput))
    //   .st({
    //     background: d => d.node.tmpClickedSgTarget.pctInputColor,
    //     color: d => utilCg.bgColorToTextColor(d.node.tmpClickedSgTarget.pctInputColor)
    //   })
  }

  var rk = 'subgraph-' + stateKey
  renderAll.hClerpUpdate.fns[rk] = renderSubgraph
  renderAll.pinnedIds.fns[rk] = renderSubgraph
  renderAll.clickedId.fns[rk] = styleNodes
  renderAll.hoveredId.fns[rk] = styleNodes
  renderAll.edgeColorMode.fns[rk] = renderSubgraph
  if (!isCurated && renderAll.thicknessMode) renderAll.thicknessMode.fns[rk] = renderSubgraph

  // Auto-fill the curated panel so it isn't blank on first load (only when the
  // graph ships no pinned nodes / clusters). Auto panels derive their own nodes.
  if (isCurated && !visState.pinnedIds?.length && window.edgeColoring?.fillSubgraph) {
    window.edgeColoring.fillSubgraph(visState, renderAll, data, {maxNodes: 24})
  }

  // https://github.com/1wheel/d3-force-container/blob/master/src/force-container.js
  function forceContainer(bbox) {
    var nodes, strength = 1

    function force(alpha) {
      var i,
          n = nodes.length,
          node,
          x = 0,
          y = 0

      for (i = 0; i < n; ++i) {
        node = nodes[i], x = node.x, y = node.y

        if (x < bbox[0][0]) node.vx += (bbox[0][0] - x)*alpha
        if (y < bbox[0][1]) node.vy += (bbox[0][1] - y)*alpha
        if (x > bbox[1][0]) node.vx += (bbox[1][0] - x)*alpha
        if (y > bbox[1][1]) node.vy += (bbox[1][1] - y)*alpha
      }
    }

    force.initialize = function(_){
      nodes = _
    }

    force.bbox = function(_){
      return arguments.length ? (bbox = +_, force) : bbox
    }
    force.strength = function(_){
      return arguments.length ? (strength = +_, force) : strength
    }

    return force
  }
}

window.init?.()
