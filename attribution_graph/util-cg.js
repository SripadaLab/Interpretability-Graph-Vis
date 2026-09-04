window.utilCg = (function(){
  function clerpUUID(d){
    // Handle case where d.featureId is undefined
    if (!d.featureId) return '🤖' + d.feature;
    
    // Split featureId by underscore and take first two parts
    const parts = d.featureId.split('_');
    return '🤖' + parts[0] + '_' + parts[1];
  }

  function parseClerpUUID(str){
    var [featureIndex] = str.split('🤖')
    return {featureIndex}
  }

  function loadDatapath(urlStr){
    try {
      var url = new URL(urlStr)
      urlStr = url.searchParams.get('datapath') ?? urlStr
    } catch {}
    urlStr = urlStr?.replace('index.html', 'data.json').split('?')[0] || 'data.json'

    try {
      return util.getFile(urlStr)
    } catch (exc) {
      d3.select('body')
        .html(`Couldn't load data from <code>${urlStr}</code>: ${exc}. Maybe you need to specify a <code>?datapath=</code> argument?`)
        .st({color: '#c00', fontSize: '150%', padding: '1em', whiteSpace: 'pre-wrap'})
      throw exc
    }
  }

  function saveHClerpsToLocalStorage(hClerps, scan) {
    const key = `local-clerp-${scan}`
    const hClerpArray = Array.from(hClerps.entries()).filter(d => d[1])
    localStorage.setItem(key, JSON.stringify(hClerpArray));
  }

  function getHClerpsFromLocalStorage(scan) {
    const key = `local-clerp-${scan}`
    // We want to set on load here so that any page load will fix the key.
    if (localStorage.getItem(key) === null) localStorage.setItem(key, '[]')
    const hClerpArray = JSON.parse(localStorage.getItem(key))
      .filter(d => d[0] != clerpUUID({}))
    return new Map(hClerpArray)
  }

  // if we have multiple transcoder (not just one scan), loads all their clerps into one map
  function getAllHClerpsFromLocalStorage(d) {
    if (d.metadata.scan?.startsWith('custom-')){
      // iterate over all layers
      const transcoder_hclerps = d.metadata.transcoder_list.map(id => getHClerpsFromLocalStorage(id))
      return new Map(function*() { 
        for (const map of transcoder_hclerps) {
          yield* map;
        }
      }());
    } else {
      return getHClerpsFromLocalStorage(d.metadata.scan)
    }
  }

  async function deDupHClerps(scan) {
    const remoteClerps = []
    let remoteMap = new Map(remoteClerps.map(d => {
      let key = clerpUUID(d);
      let clerp = d.clerp;
      return [key, clerp];
    }));

    let localClerps = getHClerpsFromLocalStorage(scan)
    let featureLookup = {}
    data.features.forEach(d => featureLookup[clerpUUID(d)] = d)

    // update feature data with current spreadsheet
    // (why is this behind the "copy" button?)
    Array.from(remoteMap).forEach(([key, value]) => {
      if (featureLookup[key]) featureLookup[key].remoteClerp = value
    })

    const deDupArray = Array.from(localClerps)
      .filter(([key, localClerp]) => {
        let remote = remoteMap.get(key);

        // keep only local clerps that are different from remote
        if (!remote) return true
        // gdoc to local storage mangles quotes, don't force strict equality
        function slugify(d){ return d ? d.replace(/['"]/g, '').trim() : ''}
        if (slugify(remote) != slugify(localClerp)) return true

        // if local changes are on remote, delete localClerp and set remoteClerp
        localClerps.delete(key)
        if (featureLookup[key]) featureLookup[key].localClerp = ''
      })

    // copy feature.remoteClerp to node.remoteClerp
    data.nodes?.forEach(node => {
      var feature = data.features.idToFeature[node.featureId]
      node.remoteClerp = feature.remoteClerp
      node.localClerp = feature.localClerp
    })

    saveHClerpsToLocalStorage(new Map(deDupArray), scan)
    return new Map(deDupArray);
  }

  function tabifyHClerps(hClerps) {
    return []
  }

  function hClerpUpdateFn(params, data){
    // update the relevant clerp
    if (params) {
      const [node, hClerp] = params;
      const scan = data.metadata.scan?.startsWith('custom-') ? data.metadata.transcoder_list[node.layer] : data.metadata.scan;
      console.log(`Updating clerp for node ${node.featureId} in scan ${scan}:`, hClerp)
      const localClerps = getHClerpsFromLocalStorage(scan);
      localClerps.set(clerpUUID(node), hClerp)
      saveHClerpsToLocalStorage(localClerps, scan);
    }

    // load all clerps
    const allLocalClerps = getAllHClerpsFromLocalStorage(data);

    data.features.forEach(node => {
      node.localClerp = allLocalClerps.get(clerpUUID(node))
      node.ppClerp = node.localClerp || node.remoteClerp || node.clerp;
    })

    data.nodes?.forEach(node => {
      var feature = data.features.idToFeature[node.featureId]
      if (!feature) return
      node.localClerp = feature.localClerp
      node.ppClerp = feature.ppClerp
    })

    // Save clerps to url param
    const currentFeatureIds = new Set(data.features.map(d => d.featureIndex.toString()))
    const clerps = Array.from(allLocalClerps.entries())
      .map(([key, value]) => [key.split('🤖')[1].split('_')[1], value])
      .filter(d => currentFeatureIds.has(d[0]))
    util.params.set('clerps', JSON.stringify(clerps))
  }

  // Adds virtual logit node showing A-B logit difference based on url param logitDiff=⍽tokenA⍽__vs__⍽tokenB⍽
  function addVirtualDiff(data){
    // Filter out any previous virtual nodes/links
    var nodes = data.nodes.filter(d => !d.isJsVirtual)
    var links = data.links.filter(d => !d.isJsVirtual)
    nodes.forEach(d => d.logitToken = d.clerp?.split(`"`)[1]?.split(`" k(p=`)[0])

    var [logitAStr, logitBStr] = util.params.get('logitDiff')?.split('__vs__') || []
    if (!logitAStr || !logitBStr) return {nodes, links}
    var logitANode = nodes.find(d => d.logitToken == logitAStr)
    var logitBNode = nodes.find(d => d.logitToken == logitBStr)
    if (!logitANode || !logitBNode) return {nodes, links}

    var virtualId = `virtual-diff-${logitAStr}-vs-${logitBStr}`
    var diffNode = {
      ...logitANode,
      node_id: virtualId,
      jsNodeId: virtualId,
      feature: virtualId,
      isJsVirtual: true,
      logitToken: `${logitAStr} - ${logitBStr}`,
      clerp: `Logit diff: ${logitAStr} - ${logitBStr}`,
    }
    nodes.push(diffNode)

    var targetLinks = links.filter(d => d.target == logitANode.node_id || d.target == logitBNode.node_id)
    d3.nestBy(targetLinks, d => d.source).map(sourceLinks => {
      var linkA = sourceLinks.find(d => d.target == logitANode.node_id)
      var linkB = sourceLinks.find(d => d.target == logitBNode.node_id)

      links.push({
        source: sourceLinks[0].source,
        target: diffNode.node_id,
        weight: (linkA?.weight || 0) - (linkB?.weight || 0),
        isJsVirtual: true
      })
    })

    return {nodes, links}
  }

  function replaceWhitespace(token) {
    return token.replaceAll("\n", "⏎").replaceAll("\t", "→").replaceAll("\r", "↵")
  }

  // Decorates and mutates data.json
  // - Adds pointers between node and links
  // - Deletes very common features
  // - Adds data.features and data.byStream
  async function formatData(data, visState){
    var {metadata} = data
    var {nodes, links} = addVirtualDiff(data)

    var py_node_id_to_node = {}
    var idToNode = {}
    var maxLayer = d3.max(nodes.filter(d => d.feature_type != 'logit'), d => +d.layer)

    // Store pre-prune counts for the stats sidebar
    data._rawNodeCount = nodes.length
    data._rawLinkCount = (data.links || data.edges || []).length

    // Pruning-independent thickness reference: p95 of |weight| over the FULL
    // link set (before any node pruning), so an edge keeps the same width
    // regardless of how many nodes are pruned out of view.
    var _absW = links.map(l => Math.abs(l.weight ?? l.pctInput ?? 0)).filter(v => v > 0).sort((a, b) => a - b)
    data._globalWeightP95 = _absW.length
      ? (_absW[Math.min(_absW.length - 1, Math.floor(0.95 * (_absW.length - 1)))] || _absW[_absW.length - 1])
      : 1

    // Pruning is applied AFTER formatting so we can keep a full unpruned copy
    // (data.allNodes / data.allLinks) for the computed-clusters panel. The main
    // graph and curated subgraph still use the pruned data.nodes / data.links.
    var pruneThr = visState.pruningThreshold != null
      ? parseFloat(visState.pruningThreshold) : null
    var willPrune = Number.isFinite(pruneThr) && pruneThr < 0.995

    // update all prompt tokens to replace \n, \t, \r with ⏎, →, ↵
    data.metadata.prompt_tokens = data.metadata.prompt_tokens.map(replaceWhitespace)

    nodes.forEach((d, i) => {
      // To make hover state work across prompts, drop ctx from node id
      d.featureId = `${d.layer}_${d.feature}_${d.ctx_idx}`
      if (visState.clerps && visState.clerps.has(`${d.feature}`)) {
        d.clerp = visState.clerps.get(`${d.feature}`)
      }

      d.active_feature_idx = d.feature
      d.nodeIndex = i
      d.isLogit = d.feature_type === 'logit' || !!d.is_target_logit

      if (d.feature_type == 'logit' && !isNaN(maxLayer)) d.layer = maxLayer + 1
      
      // TODO: does this handle error nodes correctly?
      if (d.feature_type == 'unexplored node' && !d.layer != 'E'){
        d.feature_type = 'cross layer transcoder'
      }
      
      // count from end to align last token on diff prompts
      d.ctx_from_end = data.metadata.prompt_tokens.length - d.ctx_idx

      // add clerp to embed and error nodes
      if (d.feature_type.includes('error')){
        d.isError = true

        if (!d.featureId.includes('__err_idx_')) d.featureId = d.featureId + '__err_idx_' + d.ctx_from_end

        if (d.feature_type == 'mlp reconstruction error' && !d.clerp) {
          d.clerp = `Err: mlp “${util.ppToken(data.metadata.prompt_tokens[d.ctx_idx])}"`
        }
      } else if (d.feature_type == 'embedding'){
        d.clerp = `Emb: “${util.ppToken(data.metadata.prompt_tokens[d.ctx_idx])}"`
      }

      d.url = d.vis_link
      d.isFeature = true

      d.targetLinks = []
      d.sourceLinks = []

      // TODO: switch to featureIndex in graphgen
      d.featureIndex = d.feature
      d.nodeId = d.node_id
      if (d.feature_type == 'logit' && d.clerp) d.logitPct= +d.clerp.split('(p=')[1].split(')')[0]
      idToNode[d.nodeId] = d
      py_node_id_to_node[d.node_id] = d
    })

    // delete features that occur in than 2/3 of tokens
    // TODO: more principled way of filtering them out — maybe by feature density?
    var deletedFeatures = []
    var byFeatureId = d3.nestBy(nodes, d => d.featureId)
    byFeatureId.forEach(feature => {
      if (feature.length > metadata.prompt_tokens.length*2/3){
        deletedFeatures.push(feature)
        feature.forEach(d => {
          delete idToNode[d.nodeId]
          delete py_node_id_to_node[d.node_id]
        })
      }
    })
    if (deletedFeatures.length) console.log({deletedFeatures})
    nodes = nodes.filter(d => idToNode[d.nodeId])
    nodes = d3.sort(nodes, d => +d.layer)

    links = links.filter(d => py_node_id_to_node[d.source] && py_node_id_to_node[d.target])

    // connect links to nodes
    links.forEach(link => {
      link.sourceNode = py_node_id_to_node[link.source]
      link.targetNode = py_node_id_to_node[link.target]

      link.linkId = link.sourceNode.nodeId + '__' + link.targetNode.nodeId

      link.sourceNode.targetLinks.push(link)
      link.targetNode.sourceLinks.push(link)
      link.absWeight = Math.abs(link.weight)
    })
    links = d3.sort(links, d => d.absWeight) 
    

    nodes.forEach(d => {
      d.inputAbsSum = d3.sum(d.sourceLinks, e => Math.abs(e.weight))
      d.sourceLinks.forEach(e => e.pctInput = e.weight/d.inputAbsSum)
      d.inputError = d3.sum(d.sourceLinks.filter(e => e.sourceNode.isError), e => Math.abs(e.weight))
      d.pctInputError = d.inputError/d.inputAbsSum
    })

    // convert layer/probe_location_idx to a streamIdx used to position nodes
    var byStream = d3.nestBy(nodes, d => [d.layer, d.probe_location_idx] + '')
    byStream = d3.sort(byStream, d => d[0].probe_location_idx)
    byStream = d3.sort(byStream, d => d[0].layer == 'E' ? -1 : +d[0].layer)
    byStream.forEach((stream, streamIdx) => {
      stream.forEach(d => {
        d.streamIdx = streamIdx
        d.layerLocationLabel = layerLocationLabel(d.layer, d.probe_location_idx)

        if (!visState.isHideLayer) d.streamIdx = isFinite(d.layer) ? +d.layer + 1 : 0
      })
    })

    // add target_logit_effect__ columns for each logit
    var logitNodeMap = new Map(nodes.filter(d => d.isLogit).map(d => [d.node_id, d.logitToken]))
    nodes.forEach(node => {
      node.targetLinks.forEach(link => {
        if (!logitNodeMap.has(link.target)) return
        node[`target_logit_effect__${logitNodeMap.get(link.target)}`] = link.weight
      })
    })

    // add ppClerp
    await Promise.all(nodes.map(async d => {
      if (!d.clerp) d.clerp = ''
      d.remoteClerp = ''
    }))

    // condense nodes into features, using last occurence of feature if necessary to point to a node
    var features = d3.nestBy(nodes.filter(d => d.isFeature), d => d.featureId)
      .map(d => ({
        featureId: d[0].featureId,
        feature_type: d[0].feature_type,
        clerp: d[0].clerp,
        remoteClerp: d[0].remoteClerp,
        layer: d[0].layer,
        streamIdx: d[0].streamIdx,
        probe_location_idx: d[0].probe_location_idx,
        featureIndex: d[0].featureIndex,
        top_logit_effects: d[0].top_logit_effects,
        bottom_logit_effects: d[0].bottom_logit_effects,
        top_embedding_effects: d[0].top_embedding_effects,
        bottom_embedding_effects: d[0].bottom_embedding_effects,
        url: d[0].url,
        lastNodeId: d.at(-1).nodeId,
        isLogit: d[0].isLogit,
        isError: d[0].isError,
        feature_type: d[0].feature_type,
      }))

    nodes.idToNode = idToNode
    features.idToFeature = Object.fromEntries(features.map(d => [d.featureId, d]))
    links.idToLink = Object.fromEntries(links.map(d => [d.linkId, d]))

    // Full unpruned graph — computed clusters always draw from these so pruning
    // the main view never drops auto-cluster members. Slice so later filters
    // cannot mutate this array in place.
    var allNodes = nodes.slice()
    allNodes.idToNode = Object.assign({}, idToNode)
    var allLinks = links.slice()
    allLinks.idToLink = links.idToLink

    if (willPrune) {
      var prunedNodes = nodes.filter(d =>
        d.feature_type === 'embedding' ||
        d.feature_type === 'logit' ||
        (d.influence != null && d.influence <= pruneThr)
      )
      var keep = Object.fromEntries(prunedNodes.map(d => [d.nodeId, d]))
      var prunedLinks = links.filter(d =>
        keep[d.sourceNode.nodeId] && keep[d.targetNode.nodeId]
      )
      prunedNodes.idToNode = keep
      prunedLinks.idToLink = Object.fromEntries(prunedLinks.map(d => [d.linkId, d]))
      nodes = prunedNodes
      links = prunedLinks
      // Features list for the main sidebar should match the pruned view.
      features = d3.nestBy(nodes.filter(d => d.isFeature), d => d.featureId)
        .map(d => ({
          featureId: d[0].featureId,
          feature_type: d[0].feature_type,
          clerp: d[0].clerp,
          remoteClerp: d[0].remoteClerp,
          layer: d[0].layer,
          streamIdx: d[0].streamIdx,
          probe_location_idx: d[0].probe_location_idx,
          featureIndex: d[0].featureIndex,
          top_logit_effects: d[0].top_logit_effects,
          bottom_logit_effects: d[0].bottom_logit_effects,
          top_embedding_effects: d[0].top_embedding_effects,
          bottom_embedding_effects: d[0].bottom_embedding_effects,
          url: d[0].url,
          lastNodeId: d.at(-1).nodeId,
          isLogit: d[0].isLogit,
          isError: d[0].isError,
          feature_type: d[0].feature_type,
        }))
      features.idToFeature = Object.fromEntries(features.map(d => [d.featureId, d]))
    }

    // Create a copy of the data object with the new properties
    return { ...data, nodes, features, links, byStream, allNodes, allLinks };
  }

  function initBcSync({visState, renderAll}){
    var bcStateSync = window.bcSync = window.bcSync ||  new BroadcastChannel('state-sync')

    function broadcastState(){
      if (!visState.isSyncEnabled) return
      bcStateSync.postMessage({
        pinnedIds: visState.pinnedIds,
        hiddenIds: visState.hiddenIds,
        clickedId: visState.clickedId,
        hoveredId: visState.hoveredId,
        pageUUID: visState.pageUUID,
        isSyncEnabled: visState.isSyncEnabled,
      })
    }

    renderAll.pinnedIds.fns.push(ev => { if (!ev?.skipBroadcast) broadcastState() })
    renderAll.hiddenIds.fns.push(ev => { if (!ev?.skipBroadcast) broadcastState() })
    renderAll.clickedId.fns.push(ev => { if (!ev?.skipBroadcast) broadcastState() })
    renderAll.hoveredId.fns.push(ev => { if (!ev?.skipBroadcast) broadcastState() })

    bcStateSync.onmessage = ev => {
      if (!visState.isSyncEnabled) return
      if (visState.isSyncEnabled != ev.data.isSyncEnabled) return
      if (ev.data.pageUUID == visState.pageUUID) return

      if (JSON.stringify(visState.pinnedIds) != JSON.stringify(ev.data.pinnedIds)){
        visState.pinnedIds = ev.data.pinnedIds
        renderAll.pinnedIds({skipBroadcast: true})
      }

      if (JSON.stringify(visState.hiddenIds) != JSON.stringify(ev.data.hiddenIds)){
        visState.hiddenIds = ev.data.hiddenIds
        renderAll.hiddenIds({skipBroadcast: true})
      }

      if (visState.clickedId != ev.data.clickedId){
        visState.clickedId = ev.data.clickedId
        renderAll.clickedId({skipBroadcast: true})
      }

      if (visState.hoveredId != ev.data.hoveredId){
        visState.hoveredId = ev.data.hoveredId
        renderAll.hoveredId({skipBroadcast: true})
      }
    }
  }

  function addFeatureEvents(visState, renderAll) {
    return function(selection) {
      selection
        .on('mouseover', (ev, d) => {
          if (ev.shiftKey) return
          if (visState.subgraph?.activeGrouping.isActive) return
          if (visState.subgraph?.activeGrouping.isDragging) return
          ev.preventDefault()
          hoverFeature(visState, renderAll, d)
        })
        .on('mouseout', (ev, d) => {
          if (ev.shiftKey) return
          if (visState.subgraph?.activeGrouping.isActive) return
          if (visState.subgraph?.activeGrouping.isDragging) return
          ev.preventDefault()
          unHoverFeature(visState, renderAll)
        })
        .on('click', (ev, d) => {
          if (visState.subgraph?.activeGrouping.isActive) return
          clickFeature(visState, renderAll, d, ev.metaKey || ev.ctrlKey)
        })
    }
  }

  function hoverFeature(visState, renderAll, d) {
    if (d.nodeId.includes('supernode-')) return

    if (visState.hoveredId != d.featureId) {
      visState.hoveredId = d.featureId
      visState.hoveredCtxIdx = d.ctx_idx
      renderAll.hoveredId()
    }
  }

  function unHoverFeature(visState, renderAll) {
    if (visState.hoveredId) {
      visState.hoveredId = null
      visState.hoveredCtxIdx = null
      setTimeout(() => {
        if (!visState.hoveredId) renderAll.hoveredId()
      })
    }
  }
  function togglePinned(visState, renderAll, d) {
    var index = visState.pinnedIds.indexOf(d.nodeId)
    if (index == -1) {
      visState.pinnedIds.push(d.nodeId)
    } else {
      visState.pinnedIds.splice(index, 1)
    }
    util.params.set('pinnedIds', visState.pinnedIds.join(','))
    renderAll.pinnedIds()
  }

  function toggleExpanded(visState, renderAll, d) {
    if (!visState.expandedIds) visState.expandedIds = []
    if (typeof visState.expandedIds === 'string') {
      visState.expandedIds = visState.expandedIds.split(',').filter(Boolean)
    }
    var index = visState.expandedIds.indexOf(d.nodeId)
    if (index == -1) {
      visState.expandedIds.push(d.nodeId)
    } else {
      visState.expandedIds.splice(index, 1)
    }
    util.params.set('expandedIds', visState.expandedIds.join(','))
    renderAll.expandedIds()
  }

  function clickFeature(visState, renderAll, d, metaKey){
    if (d.nodeId.includes('supernode-')) return
    
    if (metaKey && visState.isEditMode) {
      togglePinned(visState, renderAll, d) 
    } else {
      // In expand-edges mode, click toggles edge visibility for this node
      if (visState.expandEdges) {
        toggleExpanded(visState, renderAll, d)
      }
      if (visState.clickedId == d.nodeId) {
        visState.clickedId = null
        visState.clickedCtxIdx = null
      } else {
        visState.clickedId = d.nodeId
        visState.clickedCtxIdx = d.ctx_idx
      }
      visState.hoveredId = null
      visState.hoveredCtxIdx = null
      renderAll.clickedId()
    }
  }

  function showTooltip(ev, d) {
    let tooltipSel = d3.select('.tooltip'),
        x = ev.clientX,
        y = ev.clientY,
        bb = tooltipSel.node().getBoundingClientRect(),
        left = d3.clamp(20, (x-bb.width/2), window.innerWidth - bb.width - 20),
        top = innerHeight > y + 20 + bb.height ? y + 20 : y - bb.height - 20;

    let tooltipHtml = !ev.metaKey ? (d.ppClerp || featureIdLabel(d)) : Object.keys(d)
      .filter(str => typeof d[str] != 'object' && typeof d[str] != 'function' && !keysToSkip.has(str))
      .map(str => {
        var val = d[str]
        if (typeof val == 'number' && !Number.isInteger(val)) val = val.toFixed(6)
        return `<div>${str}: <b>${val}</b></div>`
      })
      .join('')

    tooltipSel
      .style('left', left +'px')
      .style('top', top + 'px')
      .html(tooltipHtml)
      .classed('tooltip-hidden', false)
  }

  function addFeatureTooltip(selection){
    selection
      .call(d3.attachTooltip, d3.select('.tooltip'), [])
      .on('mouseover.tt', (ev, d) => {
        var tooltipHtml = !ev.metaKey ? d.ppClerp : Object.keys(d)
          .filter(str => typeof d[str] != 'object' && typeof d[str] != 'function' && !keysToSkip.has(str))
          .map(str => {
            var val = d[str]
            if (typeof val == 'number' && !Number.isInteger(val)) val = val.toFixed(6)
            return `<div>${str}: <b>${val}</b></div>`
          })
          .join('')

        d3.select('.tooltip').html(tooltipHtml)
      })
  }

  function hideTooltip() {
    d3.select('.tooltip').classed('tooltip-hidden', true);
  }

  function updateFeatureStyles(nodeSel){
    nodeSel.call(classAndRaise('hovered', e => e.featureId == visState.hoveredId))

    var pinnedIdSet = new Set(visState.pinnedIds)
    nodeSel.call(classAndRaise('pinned', d => pinnedIdSet.has(d.nodeId)))

    var hiddenIdSet = new Set(visState.hiddenIds)
    nodeSel.call(classAndRaise('hidden', d => hiddenIdSet.has(d.featureId)))

    if (nodeSel.datum().nodeId){
      nodeSel.call(classAndRaise('clicked', e => e.nodeId === visState.clickedId))
    } else {
      nodeSel.call(classAndRaise('clicked', d => d.featureId == visState.clickedId))
    }
  }

  function classAndRaise(className, filterFn) {
    return sel => {
      sel
        .classed(className, 0)
        .filter(filterFn)
        .classed(className, 1)
        .raise()
    }
  }

  var keysToSkip = new Set([
    'node_id', 'jsNodeId', 'nodeId', 'layerLocationLabel', 'remoteClerp', 'localClerp', 
    'tmpClickedTargetLink', 'tmpClickedLink', 'tmpClickedSourceLink',
    'pos', 'xOffset', 'yOffset', 'sourceLinks', 'targetLinks', 'url', 'vis_link', 'run_idx',
    'featureId', 'active_feature_idx', 'nodeIndex', 'isFeature', 'Distribution',
    'clerp', 'ppClerp', 'is_target_logit', 'token_prob', 'reverse_ctx_idx', 'ctx_from_end', 'feature', 'logitToken',
    'featureIndex', 'streamIdx', 'nodeColor', 'umap_enc_x', 'umap_enc_y', 'umap_dec_x', 'umap_dec_y', 'umap_concat_x', 'umap_concat_y',
  ])
  

  function layerLocationLabel(layer, location) {
    if (layer == 'E') return 'Emb'
    if (layer == 'E1') return 'Lgt'
    if (location === -1) return 'logit'

    // TODO: is stream probe_location_idx no longer be saved out?
    // NOTE: For now, location is literally ProbePointLocation
    return `L${layer}`
  }

  /** Parse ``{layer}_{feature}_{ctx}`` from node_id / featureId. */
  function parseLayerFeatureCtx(raw) {
    var parts = String(raw == null ? '' : raw).split('_')
    if (parts.length >= 2 && /^-?\d+$/.test(parts[1])) {
      return {layer: parts[0], feature: parts[1], ctx: parts[2]}
    }
    return null
  }

  /**
   * Full feature id for display: ``L11/389 · t1``.
   * Layer + feature come from node_id (not the packed ``feature`` field).
   */
  function featureIdLabel(node, id) {
    var raw = id
      || (node && (node.node_id || node.nodeId || node.featureId || node.jsNodeId))
      || ''
    var parsed = parseLayerFeatureCtx(raw)
    if (node && !parsed) {
      parsed = parseLayerFeatureCtx(node.node_id)
        || parseLayerFeatureCtx(node.nodeId)
        || parseLayerFeatureCtx(node.featureId)
        || parseLayerFeatureCtx(node.jsNodeId)
    }
    var layer = parsed && parsed.layer
    if ((layer == null || layer === '') && node && node.layer != null && node.layer !== '') {
      layer = node.layer
    }
    var feat = parsed && parsed.feature
    if ((feat == null || feat === '') && node && node.feature != null && node.feature !== '') {
      feat = String(node.feature)
    }
    var ctx = parsed && parsed.ctx
    if ((ctx == null || ctx === '') && node && node.ctx_idx != null && node.ctx_idx !== '') {
      ctx = node.ctx_idx
    }
    if (feat == null || feat === '' || /^group\s+\d+$/i.test(String(feat))) {
      return raw || '—'
    }
    var loc = layerLocationLabel(layer, node && node.probe_location_idx)
    var out = loc + '/' + feat
    if (ctx != null && ctx !== '') out += ' · t' + ctx
    return out
  }

  function _countKeys(arr) {
    var c = {}
    arr.forEach(function (x) {
      var k = String(x)
      c[k] = (c[k] || 0) + 1
    })
    return Object.keys(c)
      .map(function (k) { return {key: k, n: c[k], frac: c[k] / Math.max(arr.length, 1)} })
      .sort(function (a, b) { return b.n - a.n })
  }

  function _tokenQuote(ctx, tokens) {
    var i = +ctx
    var raw = tokens && tokens[i]
    if (raw == null) return 't' + i
    var pretty = String(window.util && util.ppToken ? util.ppToken(raw) : raw)
    var stripped = pretty.replace(/^\s+|\s+$/g, '')
    var label = stripped
    if (stripped === '') label = 'blank'
    else if (stripped === '"' || stripped === "'" || stripped === '“' || stripped === '”') label = 'quote'
    return '“' + label + '” (t' + i + ')'
  }

  function _isAnswerishToken(ctx, tokens) {
    var i = +ctx
    if (!tokens || !tokens.length || i !== tokens.length - 1) return false
    var stripped = String(tokens[i] == null ? '' : tokens[i]).replace(/^\s+|\s+$/g, '')
    return stripped === '' || stripped === '"' || stripped === "'" || stripped === '(' || stripped === '='
  }

  /**
   * Structural read of a feature group: layer, token, and whether the same
   * feature index repeats across positions. Used on spectral / cluster cards.
   * Clerps are mentioned only when present; shipped graphs often have none.
   */
  function summarizeFeatureGroup(members, opts) {
    opts = opts || {}
    var tokens = opts.tokens || []
    var totalN = +opts.totalN || 0
    var list = (members || []).map(function (x) {
      if (x && typeof x === 'object') return x
      return {node_id: String(x), nodeId: String(x)}
    })
    var n = list.length
    if (!n) return {headline: 'Empty group.', kind: 'empty'}

    var layers = []
    var ctxs = []
    var featKeys = []
    var clerps = []
    list.forEach(function (m) {
      var parsed = parseLayerFeatureCtx(m.node_id)
        || parseLayerFeatureCtx(m.nodeId)
        || parseLayerFeatureCtx(m.featureId)
        || parseLayerFeatureCtx(m.jsNodeId)
        || {}
      var layer = parsed.layer != null && parsed.layer !== '' ? parsed.layer : m.layer
      var feat = parsed.feature != null && parsed.feature !== '' ? parsed.feature : m.feature
      var ctx = parsed.ctx != null && parsed.ctx !== '' ? parsed.ctx : m.ctx_idx
      layers.push(layer)
      if (ctx != null && ctx !== '') ctxs.push(+ctx)
      if (feat != null && feat !== '' && !/^group\s+\d+$/i.test(String(feat))) {
        var loc = layerLocationLabel(layer, m.probe_location_idx)
        featKeys.push(loc + '/' + feat)
      }
      var cl = String(m.localClerp || m.clerp || '').trim()
      if (cl && !/^\[group\s+\d+\]/i.test(cl)) clerps.push(cl)
    })

    var byLayer = _countKeys(layers)
    var byCtx = _countKeys(ctxs)
    var byFeat = _countKeys(featKeys)
    var byClerp = _countKeys(clerps)
    var nums = layers.map(Number).filter(Number.isFinite)
    var layerMin = nums.length ? Math.min.apply(null, nums) : null
    var layerMax = nums.length ? Math.max.apply(null, nums) : null
    var topLayer = byLayer[0]
    var topCtx = byCtx[0]
    var repeats = byFeat.filter(function (f) {
      return f.n >= 3 && (n <= 50 || f.n / n >= 0.12 || (f.n >= 4 && n <= 80))
    })
    var lastCtx = tokens.length ? tokens.length - 1 : (ctxs.length ? Math.max.apply(null, ctxs) : null)
    var leftover = totalN > 0 && n / totalN >= 0.28
      && (!topLayer || topLayer.frac < 0.7)
      && (!topCtx || topCtx.frac < 0.7)
      && !repeats.length
    var split = topCtx && byCtx[1]
      && topCtx.frac >= 0.35 && byCtx[1].frac >= 0.35
      && topCtx.frac + byCtx[1].frac >= 0.8
    var embN = layers.filter(function (l) {
      return l === 'E' || l === 'e' || l === 'Emb'
    }).length

    var parts = []
    var kind = 'mixed'
    if (n === 1) {
      kind = 'single'
      parts.push(
        'Single feature: '
        + (featKeys[0] || 'one node')
        + (topCtx ? ' on ' + _tokenQuote(topCtx.key, tokens) : '')
        + '.'
      )
    } else if (leftover) {
      kind = 'leftover'
      parts.push('Leftover bin — the remainder after tighter cliques peel off.')
    } else if (topLayer && topCtx && topLayer.frac >= 0.85 && topCtx.frac >= 0.85) {
      kind = 'clique'
      parts.push('Tight clique: almost all ' + layerLocationLabel(topLayer.key) + ' on ' + _tokenQuote(topCtx.key, tokens) + '.')
    } else if (split) {
      kind = 'split'
      parts.push(
        'Split between ' + _tokenQuote(topCtx.key, tokens)
        + ' and ' + _tokenQuote(byCtx[1].key, tokens) + '.'
      )
    } else if (topCtx && topCtx.frac >= 0.7) {
      kind = 'token'
      parts.push(Math.round(topCtx.frac * 100) + '% on ' + _tokenQuote(topCtx.key, tokens) + '.')
    } else if (repeats.length) {
      kind = 'tiled'
      parts.push(
        'Same feature across tokens: '
        + repeats.slice(0, 3).map(function (f) { return f.key + ' ×' + f.n }).join(', ')
        + '.'
      )
    } else if (topLayer && topLayer.frac >= 0.7) {
      kind = 'layer'
      parts.push(Math.round(topLayer.frac * 100) + '% at ' + layerLocationLabel(topLayer.key) + '.')
    }

    if (embN === n) {
      parts.push('All embeddings.')
    } else if (embN && kind !== 'leftover') {
      parts.push(embN + ' embedding' + (embN === 1 ? '' : 's') + '.')
    }

    if (kind !== 'clique' && kind !== 'single' && layerMin != null && layerMax != null) {
      if (layerMin === layerMax) {
        parts.push('All ' + layerLocationLabel(layerMin) + '.')
      } else if (layerMax <= 2) {
        parts.push('Early layers (L' + layerMin + '–L' + layerMax + ').')
      } else if (layerMin >= 16) {
        parts.push('Late layers (L' + layerMin + '–L' + layerMax + ').')
      } else if (kind !== 'leftover') {
        parts.push('Layers L' + layerMin + '–L' + layerMax + '.')
      } else {
        parts.push('Spread across L' + layerMin + '–L' + layerMax + '.')
      }
    }

    if (topCtx && kind !== 'clique' && kind !== 'token' && kind !== 'split' && kind !== 'single') {
      if (topCtx.frac >= 0.45) {
        parts.push('Mostly ' + _tokenQuote(topCtx.key, tokens) + '.')
      } else if (byCtx.length && byCtx.length <= 3) {
        parts.push('Tokens: ' + byCtx.map(function (c) { return _tokenQuote(c.key, tokens) }).join(', ') + '.')
      } else if (kind === 'leftover') {
        parts.push('Mostly ' + _tokenQuote(topCtx.key, tokens) + '; mixed positions.')
      }
    }

    if (topCtx && lastCtx != null && +topCtx.key === lastCtx && topCtx.frac >= 0.5) {
      if (
        (layerMin != null && layerMin >= 10)
        || kind === 'token'
        || kind === 'clique'
        || _isAnswerishToken(topCtx.key, tokens)
      ) {
        parts.push('Last-token / pre-logit mass.')
      }
    }

    if (kind === 'tiled' || (repeats.length && kind !== 'leftover' && kind !== 'clique')) {
      if (kind !== 'tiled') {
        parts.push(
          'Repeated IDs: '
          + repeats.slice(0, 3).map(function (f) { return f.key + ' ×' + f.n }).join(', ')
          + '.'
        )
      }
    }

    if (leftover) {
      parts.push('Few repeated feature IDs — not a tight community.')
    }

    if (clerps.length) {
      function clipClerp(s) {
        return s.length > 72 ? s.slice(0, 69) + '…' : s
      }
      if (n <= 6) {
        parts.push(
          'Clerp: '
          + byClerp.slice(0, 4).map(function (c) {
            return clipClerp(c.key) + (c.n > 1 ? ' ×' + c.n : '')
          }).join('; ')
          + '.'
        )
      } else if (byClerp[0] && byClerp[0].frac >= 0.25) {
        parts.push(
          'Clerp: ' + clipClerp(byClerp[0].key)
          + (byClerp[0].n > 1 ? ' ×' + byClerp[0].n : '') + '.'
        )
      }
    }

    if (parts.length <= 1 && byCtx.length && kind !== 'clique' && kind !== 'token') {
      parts.push(
        'Heaviest tokens: '
        + byCtx.slice(0, 2).map(function (c) {
          return _tokenQuote(c.key, tokens) + ' ' + Math.round(c.frac * 100) + '%'
        }).join(', ')
        + '.'
      )
    }

    if (!parts.length) {
      parts.push(n + ' features; mixed layers and tokens.')
    }

    return {headline: parts.join(' '), kind: kind, n: n}
  }

  /** One-line read of a whole k-cut, from per-group summaries. */
  function summarizeSpectralPartition(groupNotes, k) {
    var list = groupNotes || []
    if (!list.length) return ''
    var leftover = list.filter(function (g) { return g.kind === 'leftover' })
    var lastTok = list.filter(function (g) {
      return g.kind === 'token' || g.kind === 'clique'
    })
    var tiled = list.filter(function (g) { return g.kind === 'tiled' || g.kind === 'split' })
    var bits = ['At k=' + k + ',']
    if (leftover.length) {
      bits.push(
        leftover.map(function (g, i) {
          return (g.label || ('group ' + (i + 1))) + ' is a leftover bin (' + g.n + ')'
        }).join(' / ') + '.'
      )
    }
    var notable = lastTok.concat(tiled).slice(0, 4)
    if (notable.length) {
      bits.push(
        'Tighter cuts: '
        + notable.map(function (g) {
          var h = (g.headline || '').split('.')[0]
          return (g.label || 'group') + ' — ' + h
        }).join('; ')
        + '.'
      )
    } else if (!leftover.length) {
      bits.push('Groups are small and mixed; raise or lower k to peel cliques.')
    }
    return bits.join(' ')
  }

  var memoize = fn => {
    var cache = new Map()
    return (...args) => {
      var key = JSON.stringify(args)
      if (cache.has(key)) return cache.get(key)
      var result = fn(...args)
      cache.set(key, result)
      return result
    }
  }

  var bgColorToTextColor = memoize((backgroundColor, light='#fff', dark='#000') => {
    if (!backgroundColor) return ''
    var hsl = d3.hsl(backgroundColor)
    return hsl.l > 0.55 ? dark : light
  })

  // gradient for hover && pinned state
  function addPinnedClickedGradient(svg){
    svg.append('defs').html(`
      <linearGradient id='pinned-clicked-gradient' x1='0' x2='2' gradientUnits='userSpaceOnUse' spreadMethod='repeat'>
        <stop offset='0'    stop-color='#f0f' />
        <stop offset='70%'  stop-color='#f0f' />
        <stop offset='71%'  stop-color='#000' />
        <stop offset='100%' stop-color='#000' />
      </linearGradient>
    `)
  }

  function renderFeatureRow(sel, visState, renderAll, linkKey='tmpClickedLink'){
    sel.st({
      background: d => d[linkKey]?.tmpColor,
      color: d => bgColorToTextColor(d[linkKey]?.tmpColor, '#eee', '#555'),
    })

    // add events in a timeout to avoid connection clicks leading to an instant hover
    setTimeout(() => sel.call(addFeatureEvents(visState, renderAll)), 16)

    let featureIconSel = sel.append('svg')
      .at({width: 10, height: 10})

    let featureIcon = featureIconSel.append('g')

    featureIcon.append('g.default-icon').append('text')
      .text(d => featureTypeToText(d.feature_type))
      .at({
        fontSize: 9,
        textAnchor: 'middle',
        dominantBaseline: 'central',
        dx: 5,
        dy: 4,
      })
      .at({fill: d => d[linkKey]?.tmpColor})


    sel.append('div.label')
      .text(d => d.ppClerp)
      .at({ title: d => d.ppClerp })

    sel
      .filter(d => d[linkKey] && d[linkKey].tmpClickedCtxOffset != 0)
      .append('div.ctx-offset')
      .text(d => d[linkKey].tmpClickedCtxOffset < 0 ? '←' : '→')

    if (!visState.isHideLayer){
      sel.append('div.layer')
        .text(d => d.layerLocationLabel ?? layerLocationLabel(d.layer, d.probe_location_idx));
    }

    sel.append('div.weight')
      .text(d => d[linkKey] ? d3.format('+.3f')(d[linkKey].weight) : '')
  }

  function featureTypeToText(type){
    if (type == 'logit') return '■'
    if (type == 'embedding') return '■'
    if (type === 'mlp reconstruction error') return '◆'
    return '●'
    
  }


  return {
    loadDatapath,
    formatData,
    initBcSync,
    addFeatureEvents,
    hoverFeature,
    unHoverFeature,
    clickFeature,
    togglePinned,
    toggleExpanded,
    layerLocationLabel,
    parseLayerFeatureCtx,
    featureIdLabel,
    summarizeFeatureGroup,
    summarizeSpectralPartition,
    keysToSkip,
    addFeatureTooltip,
    showTooltip,
    hideTooltip,
    updateFeatureStyles,
    memoize,
    bgColorToTextColor,
    addPinnedClickedGradient,
    renderFeatureRow,
    saveHClerpsToLocalStorage,
    getHClerpsFromLocalStorage,
    hClerpUpdateFn,
    deDupHClerps,
    tabifyHClerps,
    featureTypeToText,
  }
})()

window.init?.()
