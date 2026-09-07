window.initCgLinkGraph = function({visState, renderAll, data, cgSel}){
  var {nodes, links, metadata} = data

  var c = d3.conventions({
    sel: cgSel.select('.link-graph').html(''),
    // Roomy top margin so rotated output-logit labels have space (mirrors the
    // bottom margin used for rotated input-token labels).
    margin: {left: visState.isHideLayer ? 12 : 42, bottom: 100, top: 84, right: 16},
    layers: 'sccccs',
  })
  
  c.svgBot = c.layers[0]
  var allCtx = {
    allLinks: c.layers[1],
    pinnedLinks: c.layers[2],
    bgLinks: c.layers[3],
    clickedLinks: c.layers[4]
  }
  c.svg = c.layers[5]

  // Count max number of nodes at each context to create a polylinear x scale
  var earliestCtxWithNodes = d3.min(nodes, d => d.ctx_idx)
  var cumsum = 0
  var ctxCounts = d3.range(d3.max(nodes, d => d.ctx_idx) + 1).map(ctx_idx => {
    if (ctx_idx >= earliestCtxWithNodes) {
      var group = nodes.filter(d => d.ctx_idx === ctx_idx)
      var maxCount = d3.max([1, d3.max(d3.nestBy(group, d => d.streamIdx), e => e.length)])
      cumsum += maxCount
    }
    return {ctx_idx, maxCount, cumsum}
  })

  var xDomain = [-1].concat(ctxCounts.map(d => d.ctx_idx))
  var xRange = [0].concat(ctxCounts.map(d => d.cumsum * c.width / cumsum))
  c.x = d3.scaleLinear().domain(xDomain.map(d => d + 1)).range(xRange)
  
  // TODO: why this was here?
  // var yNumTicks= visState.isHideLayer ? data.byStream.length : 19 
  var logitLayer = data.byStream.find(d => d[0].feature_type == 'logit');
  var yNumTicks = (logitLayer ? parseInt(logitLayer[0].layer) : 26) + 2; // +2 to account for embedding at 0 and layers starting at 1

  c.y = d3.scaleBand(d3.range(yNumTicks), [c.height, 0])

  c.yAxis = d3.axisLeft(c.y)
    .tickValues(d3.range(yNumTicks))
    .tickFormat(i => {
      if (i % 2) return
      
      return i == yNumTicks - 1 ? 'Lgt' : i == 0 ? 'Emb' : 'L' + (i - 1) // Subtract 1 from layer numbers
    })
  
  c.svgBot.append('rect').at({width: c.width, height: c.height, fill: '#F5F4EE'})
  c.svgBot.append('g').appendMany('rect', [0, yNumTicks - 1])
    .at({width: c.width, height: c.y.bandwidth(), y: c.y, fill: '#F0EEE7'})
  
  c.svgBot.append('g').appendMany('path', d3.range(-1, yNumTicks - 1))
    .translate(d => [0, c.y(d + 1)])
    .at({d: `M0,0H${c.width}`, stroke: 'white', strokeWidth: .5})
  
  c.drawAxis(c.svgBot)
  c.svgBot.select('.x').remove()
  c.svgBot.selectAll('.y line').remove()
  if (visState.isHideLayer) c.svgBot.select('.y').remove()
  
  // Spread nodes across each context 
  // d.width is the total amount of px space in each column
  ctxCounts.forEach((d, i) => d.width = c.x(d.ctx_idx + 1) - c.x(ctxCounts[i].ctx_idx))
  
  // if default to 14px padding right, if pad right to center singletons 
  var padR = Math.min(14, d3.min(ctxCounts.slice(1), d => d.width/2)) + 0
  
  // find the tightest spacing between nodes and use for all ctx (but don't go below 28)
  ctxCounts.forEach(d => d.minS = (d.width - padR)/d.maxCount)
  var overallS = Math.max(28, d3.min(ctxCounts, d => d.minS))

  // apply to nodes
  d3.nestBy(nodes, d => [d.ctx_idx, d.streamIdx].join('-')).forEach(ctxLayer => {
    var ctxWidth = c.x(ctxLayer[0].ctx_idx + 1) - c.x(ctxLayer[0].ctx_idx) - padR
    var s = Math.min(overallS, ctxWidth/ctxLayer.length)
    
    // sorting by pinned stacks all the links on top of each other
    // ctxLayer = d3.sort(ctxLayer, d => visState.pinnedIds.includes(d.nodeId) ? -1 : 1)
    ctxLayer = d3.sort(ctxLayer, d => -d.logitPct)
    ctxLayer.forEach((d, i) => {
      d.xOffset = d.feature_type === 'logit' ? ctxWidth - (padR/2 + i*s) : ctxWidth - (padR/2 + i*s)
      d.yOffset = 0
    })
  })
  nodes.forEach(d => d.pos = [
    c.x(d.ctx_idx) + d.xOffset, 
    c.y(d.streamIdx) + c.y.bandwidth()/2 + d.yOffset
  ])

  
  // hover poitns
  var maxHoverDistance = 30
  c.sel
    .on('mousemove', (ev) => {
      if (ev.shiftKey) return
      var [mouseX, mouseY] = d3.pointer(ev)
      var [closestNode, closestDistance] = findClosestPoint(mouseX - c.margin.left, mouseY - c.margin.top, nodes)
      if (closestDistance > maxHoverDistance) {
        utilCg.unHoverFeature(visState, renderAll)
        utilCg.hideTooltip()
      } else if (visState.hoveredId !== closestNode) {
        utilCg.hoverFeature(visState, renderAll, closestNode)
        utilCg.showTooltip(ev, closestNode)
      }
    })
    .on('mouseleave', (ev) => {
      if (ev.shiftKey) return
      utilCg.unHoverFeature(visState, renderAll)
      utilCg.hideTooltip()
    })
    .on('click', (ev) => {
      var [mouseX, mouseY] = d3.pointer(ev)
      var [closestNode, closestDistance] = findClosestPoint(mouseX - c.margin.left, mouseY - c.margin.top, nodes)
      if (closestDistance > maxHoverDistance) {
        visState.clickedId = null
        visState.clickedCtxIdx = null
        renderAll.clickedId()
      } else {
        utilCg.clickFeature(visState, renderAll, closestNode, ev.metaKey || ev.ctrlKey)
      }
    })

  function findClosestPoint(mouseX, mouseY, points) {
    if (points.length === 0) return null

    let closestPoint = points[0]
    let closestDistance = distance(mouseX, mouseY, closestPoint.pos[0], closestPoint.pos[1])

    for (let i = 1; i < points.length; i++){
      var point = points[i]
      var dist = distance(mouseX, mouseY, point.pos[0], point.pos[1])
      if (dist < closestDistance){
        closestPoint = point
        closestDistance = dist
      }
    }
    return [closestPoint, closestDistance]

    function distance(x1, y1, x2, y2) {
      return Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2))
    }
  }

  // set up dom
  var nodeSel = c.svg.appendMany('text.node', nodes)
    .translate(d => d.pos)
    .text(d => utilCg.featureTypeToText(d.feature_type))
    .at({
      fontSize: 9,
      fill: d => d.nodeColor,
      stroke: '#000',
      strokeWidth: .5,
      textAnchor: 'middle',
      dominantBaseline: 'central',
    })
    // .call(utilCg.addFeatureTooltip)
    // .call(utilCg.addFeatureEvents(visState, renderAll, ev => ev.shiftKey))

  var hoverSel = c.svg.appendMany('circle', nodes)
    .translate(d => d.pos)
    .at({r: 6, cy: .5, stroke: '#f0f', strokeWidth: 2, strokeDasharray: '2 2', fill: 'none', display: 'xnone', pointEvents: 'none'})

  links.forEach(d => {
    var [x1, y1] = d.sourceNode.pos
    var [x2, y2] = d.targetNode.pos
    d.pathStr = `M${x1},${y1}L${x2},${y2}`
  })

  
  function drawLinks(links, ctx, strokeWidthOffset=0, colorOverride){
    ctx.clearRect(-c.margin.left, -c.margin.top, c.totalWidth, c.totalHeight)
    d3.sort(links, d => d.strokeWidth).forEach(d => {
      ctx.beginPath()
      ctx.moveTo(d.sourceNode.pos[0], d.sourceNode.pos[1])
      ctx.lineTo(d.targetNode.pos[0], d.targetNode.pos[1])
      ctx.strokeStyle = colorOverride || d.color
      ctx.lineWidth = d.strokeWidth + strokeWidthOffset
      ctx.stroke()
    })
  }

  function filterLinks(featureIds){
    var filteredLinks = []

    featureIds.forEach(nodeId => {
      nodes.filter(n => n.nodeId == nodeId).forEach(node => {
        if (visState.linkType == 'input' || visState.linkType == 'either') {
          Array.prototype.push.apply(filteredLinks, node.sourceLinks)
        }
        if (visState.linkType == 'output' || visState.linkType == 'either') {
          Array.prototype.push.apply(filteredLinks, node.targetLinks)
        }
        if (visState.linkType == 'both') {
          Array.prototype.push.apply(filteredLinks, node.sourceLinks.filter(
            link => visState.pinnedIds.includes(link.sourceNode.nodeId)
          ))
          Array.prototype.push.apply(filteredLinks, node.targetLinks.filter(
            link => visState.pinnedIds.includes(link.targetNode.nodeId)
          ))
        }
      })
    })

    return filteredLinks
  }

  function filterExpandedLinks(){
    var expanded = [...(visState.expandedIds || [])]
    if (!expanded.length) return []
    // Reuse linkType filtering against expanded nodes
    return filterLinks(expanded)
  }

  function focusNodeIds(){
    var ids = [...(visState.pinnedIds || [])]
    if (visState.clickedId) ids.push(visState.clickedId)
    return [...new Set(ids.filter(Boolean))]
  }

  function renderAllLinks(){
    if (visState.expandEdges) {
      // Expand mode: only edges incident to expanded nodes (respects linkType)
      var sorted = d3.sort(filterExpandedLinks(), d => d.intensity ?? d.absWeight ?? 0)
      drawLinks(sorted, allCtx.allLinks)
    } else if (visState.isShowAllLinks) {
      // Draw weak → strong so high-intensity edges sit on top.
      var sorted = d3.sort(links, d => d.intensity ?? d.absWeight ?? 0)
      drawLinks(sorted, allCtx.allLinks)
    } else {
      // Focused view: Input / Output / Either / Both filter edges for pinned + clicked
      var sorted = d3.sort(filterLinks(focusNodeIds()), d => d.intensity ?? d.absWeight ?? 0)
      drawLinks(sorted, allCtx.allLinks)
    }
  }
  renderAll.isShowAllLinks.fns['linkGraph'] = renderAllLinks
  if (renderAll.expandedIds) renderAll.expandedIds.fns['linkGraph'] = renderAllLinks
  if (renderAll.expandEdges) renderAll.expandEdges.fns['linkGraph'] = renderAllLinks
  renderAllLinks()

  function renderPinnedIds(){
    var pinnedOnly = (visState.pinnedIds || []).filter(id => id !== visState.clickedId)
    drawLinks(filterLinks(pinnedOnly), allCtx.pinnedLinks)
    nodeSel.classed('pinned', d => visState.pinnedIds.includes(d.nodeId))
    nodeSel.classed('expanded', d => (visState.expandedIds || []).includes(d.nodeId))
    // Focused (non–show-all) canvas depends on pinned + clicked set
    if (!visState.isShowAllLinks || visState.expandEdges) renderAllLinks()
  }
  renderAll.pinnedIds.fns['linkGraph'] = renderPinnedIds
  if (renderAll.expandedIds) {
    renderAll.expandedIds.fns['linkGraphNodes'] = () => {
      nodeSel.classed('expanded', d => (visState.expandedIds || []).includes(d.nodeId))
      renderAllLinks()
    }
  }

  function renderHiddenIds(){
    var hiddenIdSet = new Set(visState.hiddenIds)
    nodeSel.classed('hidden', d => hiddenIdSet.has(d.featureId))
  }
  renderAll.hiddenIds.fns['linkGraph'] = renderHiddenIds


  function renderClicked(){
    // Respect Links: Input / Output / Either / Both for the focused node
    var clickedLinks = visState.clickedId ? filterLinks([visState.clickedId]) : []

    drawLinks(clickedLinks, allCtx.bgLinks, .05, '#000')
    drawLinks(clickedLinks, allCtx.clickedLinks)
    nodeSel.classed('clicked', e => e.nodeId === visState.clickedId)
    nodeSel.classed('expanded', d => (visState.expandedIds || []).includes(d.nodeId))

    // Pinned overlay: when a node is clicked, still show other pinned edges
    // (excluding the clicked node's own filtered set to avoid double-draw).
    var pinnedOnly = (visState.pinnedIds || []).filter(id => id !== visState.clickedId)
    drawLinks(filterLinks(pinnedOnly), allCtx.pinnedLinks)

    // Always keep the 2-color node fills; connected nodes adopt their edge color
    // only when they appear in the linkType-filtered click set.
    var highlightIds = new Set()
    clickedLinks.forEach(l => {
      highlightIds.add(l.sourceNode.nodeId)
      highlightIds.add(l.targetNode.nodeId)
    })
    nodeSel
      .at({fill: d => d.nodeColor || '#fff'})
      .filter(d => highlightIds.has(d.nodeId) && d.tmpClickedLink?.tmpColor)
      .at({fill: d => d.tmpClickedLink.tmpColor})
      .raise()
  }

  renderAll.clickedId.fns['linkGraph'] = renderClicked
  renderAll.linkType.fns['linkGraph'] = () => {
    util.params.set('linkType', visState.linkType)
    renderAllLinks()
    renderPinnedIds()
    renderClicked()
  }

  renderAll.hoveredId.fns['linkGraph'] = () => {
    hoverSel.st({display: e => e.featureId == visState.hoveredId ? '' : 'none'})
  }

  // Add x axis text/lines
  var promptTicks = data.metadata.prompt_tokens.slice(earliestCtxWithNodes).map((token, i) =>{
    var ctx_idx = i + earliestCtxWithNodes 
    var mNodes = nodes.filter(d => d.ctx_idx == ctx_idx)
    var hasEmbed = mNodes.some(d => d.feature_type == 'embedding')
    return {token, ctx_idx, mNodes, hasEmbed}
  })

  var xTickSel = c.svgBot.appendMany('g.prompt-token', promptTicks)
    .translate(d => [c.x(d.ctx_idx + 1), c.height])
  
  xTickSel.append('path').at({d: `M0,0v${-c.height}`, stroke: '#fff',strokeWidth: 1})
  xTickSel.filter(d => d.hasEmbed).append('path').at({
    stroke: '#B0AEA6',
    d: `M-${padR + 3.5},${-c.y.bandwidth()/2 + 6}V${8}`,
  })
  
  xTickSel.filter(d => d.hasEmbed).append('g').translate([-12, 8])
    .append('text').text(d => d.token) 
    .at({
      x: -5,
      y: 2,
      textAnchor: 'end',
      transform: 'rotate(-45)',
      dominantBaseline: 'middle',
      fontSize: 12,
      // fontSize: (d, i) => c.x(i+1) - c.x(i) < 15 ? 9 : 14,
    })
  
  // Output logits often stack on the final token, so their rotated labels
  // collide. Spread the label anchors with a guaranteed minimum gap (wide enough
  // that -45° text doesn't overlap: perpendicular spacing ≈ 0.7·gap) and draw a
  // leader line from each node up to its (possibly shifted) label.
  var logitNodes = d3.sort(nodes.filter(d => d.feature_type == 'logit'), d => d.pos[0])
  var logitLabelGap = 22
  logitNodes.forEach(d => d.labelX = d.pos[0])
  // Left-to-right pass: push each label right so it clears the previous one.
  for (var li = 1; li < logitNodes.length; li++) {
    if (logitNodes[li].labelX < logitNodes[li - 1].labelX + logitLabelGap) {
      logitNodes[li].labelX = logitNodes[li - 1].labelX + logitLabelGap
    }
  }
  // Shift the whole set left if it ran past the rightmost node.
  var _maxLabelX = d3.max(logitNodes, d => d.pos[0])
  var _overflow = logitNodes.length ? logitNodes[logitNodes.length - 1].labelX - _maxLabelX : 0
  if (_overflow > 0) logitNodes.forEach(d => d.labelX -= _overflow)
  // Clamp the leftmost inside the plot, then re-spread rightwards if needed.
  if (logitNodes.length && logitNodes[0].labelX < 0) {
    var shift = -logitNodes[0].labelX
    logitNodes.forEach(d => d.labelX += shift)
    for (var lj = 1; lj < logitNodes.length; lj++) {
      if (logitNodes[lj].labelX < logitNodes[lj - 1].labelX + logitLabelGap) {
        logitNodes[lj].labelX = logitNodes[lj - 1].labelX + logitLabelGap
      }
    }
  }

  var logitLabelY = -c.y.bandwidth()/2 - 8
  var logitTickSel = c.svgBot.append('g.axis').appendMany('g', logitNodes)
    .translate(d => [d.labelX, d.pos[1]])
  // Leader line: node's real x (at graph top) → spread-out label anchor.
  logitTickSel.append('path').at({
    stroke: '#B0AEA6',
    fill: 'none',
    d: d => `M${d.pos[0] - d.labelX},${-6} L0,${logitLabelY}`,
  })
  logitTickSel.append('g').translate([0, logitLabelY])
    .append('text').text(d => d.logitToken) 
    .at({
      x: 5,
      y: 2,
      textAnchor: 'start',
      transform: 'rotate(-45)',
      dominantBaseline: 'middle',
      fontSize: 12,
    })


  utilCg.addPinnedClickedGradient(c.svg)
}

window.init?.()
