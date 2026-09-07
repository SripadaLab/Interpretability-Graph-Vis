window.initCg = async function (sel, slug, {clickedId, clickedIdCb, isModal, isGridsnap, pruningThreshold, graphData} = {}){
  // Fresh copy every render — getFile caches the parsed JSON, and formatData
  // mutates node objects in place. Without cloning, a prune re-init can see a
  // half-formatted graph and drop computed-cluster members.
  // Local serve: bypass HTTP/memory cache so svd_bundle metadata stays fresh
  // after `export-svd`. Remote CDN graphs can keep the cache.
  var raw = graphData || await util.getFile(
    `./graph_data/${slug}.json`,
    !window.isLocalServing
  )
  var data = JSON.parse(JSON.stringify(raw))
  // Some exports use "edges" instead of "links".
  if (!data.links && data.edges) data.links = data.edges
  if (!data.metadata) data.metadata = {}
  if (!data.metadata.slug) data.metadata.slug = slug
  if (!data.metadata.prompt_tokens) {
    // Minimal fallback so formatData doesn't crash on bare uploads.
    data.metadata.prompt_tokens = (data.metadata.prompt || '').split(/\s+/).filter(Boolean)
    if (!data.metadata.prompt_tokens.length) data.metadata.prompt_tokens = ['']
  }
  
  var visState = {
    pinnedIds: [],
    hiddenIds: [],
    hoveredId: null,
    hoveredNodeId: null,
    hoveredCtxIdx: null,
    clickedId: null, 
    clickedCtxIdx: null,
    linkType: 'either',
    isShowAllLinks: '1', // show colored hairball by default on all graphs
    expandEdges: util.params.get('expandEdges') === '1' ? '1' : '', // opt-in
    edgeColorMode: util.params.get('edgeColorMode') || 'greedy4',
    thicknessMode: util.params.get('thicknessMode') || 'relative', // 'relative' | 'stable'
    isSyncEnabled: '',
    subgraph: null,
    isEditMode: 1,
    isHideLayer: data.metadata.scan == util.scanSlugToName.h35 || data.metadata.scan == util.scanSlugToName.moc,
    graphSchemaVersion: data.metadata?.schema_version || 0,
    sg_pos: '',
    // Only the modal viewer fills the viewport; the standalone page must
    // size from content or grid tiles squash and SVG overflow steals clicks.
    isModal: !!isModal,
    isGridsnap,
    slug: slug, // Store slug for localStorage keys
    pruningThreshold: pruningThreshold,
    expandedIds: [],
    ...(data.qParams ? Object.fromEntries(Object.entries(data.qParams).filter(([_, v]) => v !== null)) : {})
  }
  
  // Get pinnedIds from URL parameters if available (prioritized over localStorage)
  var urlPinnedIds = util.params.get('pinnedIds');
  var urlHiddenIds = util.params.get('hiddenIds');
  var urlExpandedIds = util.params.get('expandedIds');
  if (urlPinnedIds) visState.pinnedIds = urlPinnedIds
  if (urlHiddenIds) visState.hiddenIds = urlHiddenIds
  if (urlExpandedIds) visState.expandedIds = urlExpandedIds

  // Only override the graph's own qParams.supernodes when the URL actually
  // carries a supernodes param — an empty [] is truthy and would otherwise
  // wipe the published manual clusters (Dallas, French, …).
  let urlSupernodes = null;
  try {
    const supernodesParam = util.params.get('supernodes');
    if (supernodesParam) {
      urlSupernodes = JSON.parse(supernodesParam);
    }
  } catch (e) {
    console.error('Error parsing supernodes from URL:', e);
  }
  if (urlSupernodes) visState.supernodes = urlSupernodes

  if (visState.clickedId?.includes('supernode')) delete visState.clickedId
  if (clickedId && clickedId != 'null' && !clickedId.includes('supernode-')) visState.clickedId = clickedId
  if (!visState.clickedId || visState.clickedId == 'null' || visState.clickedId == 'undefined') visState.clickedId = data.nodes.find(d => d.isLogit)?.nodeId
  if (visState.pinnedIds?.replace) visState.pinnedIds = visState.pinnedIds.split(',')
  if (visState.hiddenIds?.replace) visState.hiddenIds = visState.hiddenIds.split(',')
  if (visState.expandedIds?.replace) visState.expandedIds = visState.expandedIds.split(',').filter(Boolean)
  if (!Array.isArray(visState.expandedIds)) visState.expandedIds = []

  // Expand-edges mode: hide the full hairball until nodes are opened
  if (visState.expandEdges) {
    visState.isShowAllLinks = ''
    // Seed expansion with the focused node so something is visible
    if (visState.clickedId && !visState.expandedIds.includes(visState.clickedId)) {
      visState.expandedIds = [visState.clickedId, ...visState.expandedIds]
    }
  }


  // Load clerps from URL params
  const clerpsParam = util.params.get('clerps') || data.qParams.clerps;
  if (clerpsParam) {
    const clerps = JSON.parse(clerpsParam);
    visState.clerps = new Map(clerps);
  }
  data = await utilCg.formatData(data, visState)
  
  var renderAll = util.initRenderAll(['hClerpUpdate', 'clickedId', 'hiddenIds', 'pinnedIds', 'expandedIds', 'linkType', 'isShowAllLinks', 'expandEdges', 'edgeColorMode', 'thicknessMode', 'features', 'isSyncEnabled', 'shouldSortByWeight', 'hoveredId', 'panelVisibility'])

  // 2-color edge + node encoding (greedy / layer parity / sign) with intensity.
  // Mode `prgn` restores the original Anthropic positive/negative PRGn scale.
  function colorGraph() {
    var _linearPctScale = d3.scaleLinear().domain([-.4, .4])
    var _linearTScale = d3.scaleLinear().domain([0, .5, .5, 1]).range([0, .5 - .001, .5 + .001, 1])
    utilCg.pctInputColorFn = d => d3.interpolatePRGn(_linearTScale(_linearPctScale(d)))

    if (window.edgeColoring) {
      window.edgeColoring.apply(data.links, visState.edgeColorMode || 'greedy4', data.nodes, {
        thicknessMode: visState.thicknessMode || 'relative',
        globalWeightP95: data._globalWeightP95,
      })
    } else {
      var widthScale = d3.scaleSqrt().domain([0, 1]).range([.00001, 3])
      data.links.forEach(d => {
        d.strokeWidth = widthScale(Math.abs(d.pctInput))
        d.pctInputColor = utilCg.pctInputColorFn(d.pctInput)
        d.color = d.pctInputColor
      })
      data.nodes.forEach(d => { d.nodeColor = '#fff' })
    }
  }
  colorGraph()
  renderAll.edgeColorMode.fns['recolorLinks'] = () => {
    util.params.set('edgeColorMode', visState.edgeColorMode)
    colorGraph()
    renderAll.isShowAllLinks()
    renderAll.pinnedIds()
    renderAll.clickedId()
  }
  renderAll.thicknessMode.fns['recolorLinks'] = () => {
    util.params.set('thicknessMode', visState.thicknessMode)
    colorGraph()
    renderAll.isShowAllLinks()
    renderAll.pinnedIds()
    renderAll.clickedId()
  }

  renderAll.hClerpUpdate.fns.push(params => utilCg.hClerpUpdateFn(params, data))

  renderAll.hoveredId.fns.push(() => {
    // use hovered node if possible, otherwise use last occurence of feature
    var targetCtxIdx = visState.hoveredCtxIdx ?? 999
    var hoveredNodes = data.nodes.filter(n => n.featureId == visState.hoveredId)
    var node = d3.sort(hoveredNodes, d => Math.abs(d.ctx_idx - targetCtxIdx))[0]
    visState.hoveredNodeId = node?.nodeId
  })

  // set tmpClickedLink w/ strength of all the links connected the clickedNode
  renderAll.clickedId.fns.push(() => {
    clickedIdCb?.(visState.clickedId)

    var node = data.nodes.idToNode[visState.clickedId]
    if (!node){
      // for a clicked supernode, sum over memberNode links to make tmpClickedLink
      if (visState.clickedId?.startsWith('supernode-')) {
        node = {
          nodeId: visState.clickedId,
          memberNodes: visState.subgraph.supernodes[+visState.clickedId.split('-')[1]]
            .slice(1)
            .map(id => data.nodes.idToNode[id])
        }
        node.memberSet = new Set(node.memberNodes.map(d => d.nodeId))

        function combineLinks(links, isSrc) {
          return d3.nestBy(links, d => isSrc ? d.sourceNode.nodeId : d.targetNode.nodeId)
            .map(links => ({
              source: isSrc ? links[0].sourceNode.nodeId : visState.clickedId,
              target: isSrc ? visState.clickedId : links[0].targetNode.nodeId,
              sourceNode: isSrc ? links[0].sourceNode : node,
              targetNode: isSrc ? node : links[0].targetNode,
              weight: d3.sum(links, d => d.weight),
              absWeight: Math.abs(d3.sum(links, d => d.weight))
            }))
        }

        node.sourceLinks = combineLinks(node.memberNodes.flatMap(d => d.sourceLinks), true)
        node.targetLinks = combineLinks(node.memberNodes.flatMap(d => d.targetLinks), false)
      } else {
        return data.nodes.forEach(d => {
          d.tmpClickedLink = null
          d.tmpClickedSourceLink = null
          d.tmpClickedTargetLink = null
        })
      }
    }

    var connectedLinks = [...node.sourceLinks, ...node.targetLinks]
    var max = d3.max(connectedLinks, d => d.absWeight)
    var colorScale = d3.scaleSequential(d3.interpolatePRGn).domain([-max*1.1, max*1.1])

    // allowing supernode links means each node can have a both tmpClickedSourceLink and tmpClickedTargetLink
    // currently we render bidirectional links where possible, falling back to the target side links otherwises
    var nodeIdToSourceLink = {}
    var nodeIdToTargetLink = {}
    var featureIdToLink = {}
    connectedLinks.forEach(link => {
      if (link.sourceNode === node) {
        nodeIdToTargetLink[link.targetNode.nodeId] = link
        featureIdToLink[link.targetNode.featureId] = link
        link.tmpClickedCtxOffset = link.targetNode.ctx_idx - node.ctx_idx
      }
      if (link.targetNode === node) {
        nodeIdToSourceLink[link.sourceNode.nodeId] = link
        featureIdToLink[link.sourceNode.featureId] = link
        link.tmpClickedCtxOffset = link.sourceNode.ctx_idx - node.ctx_idx
      }
      // Use the active edge palette (2-color or PRGn) so connected nodes match links.
      link.tmpColor = link.color || link.pctInputColor
    })

    data.nodes.forEach(d => {
      var link = nodeIdToSourceLink[d.nodeId] || nodeIdToTargetLink[d.nodeId]
      d.tmpClickedLink = link
      d.tmpClickedSourceLink = nodeIdToSourceLink[d.nodeId]
      d.tmpClickedTargetLink = nodeIdToTargetLink[d.nodeId]
    })

    data.features.forEach(d => {
      var link = featureIdToLink[d.featureId]
      d.tmpClickedLink = link
    })
  })

  // Secondary panels that can be closed and reopened from the Panels tabs.
  var dismissiblePanels = [
    {class: 'graph-stats', label: 'Legend'},
    {class: 'cluster-panel', label: 'Clusters'},
    {class: 'node-connections', label: 'Connections'},
    {class: 'feature-detail', label: 'Feature'},
    {class: 'subgraph', label: 'Curated'},
    {class: 'subgraph-pruned', label: 'Computed'},
    {class: 'svd-panel', label: 'SVD'},
  ]
  visState.dismissiblePanels = dismissiblePanels

  var hiddenFromUrl = (util.params.get('hiddenPanels') || '')
    .split(',').map(s => s.trim()).filter(Boolean)
  visState.hiddenPanels = new Set(hiddenFromUrl)

  function persistHiddenPanels() {
    util.params.set('hiddenPanels', [...visState.hiddenPanels].join(','))
  }

  function initGridsnap() {
    // Left: stats legend + clusters | center: graph + curated/pruned subgraphs | right: connections + detail
    var gridData = [
      {cur: {x: 0,  y: 0,  w: 24, h: 3},  class: 'button-container'},
      {cur: {x: 0,  y: 3,  w: 4,  h: 13}, class: 'graph-stats'},
      {cur: {x: 0,  y: 16, w: 4,  h: 13}, class: 'cluster-panel'},
      {cur: {x: 4,  y: 3,  w: 12, h: 14}, class: 'link-graph', resizeFn: makeResizeFn(initCgLinkGraph)},
      {cur: {x: 16, y: 3,  w: 8,  h: 7},  class: 'node-connections'},
      {cur: {x: 16, y: 10, w: 8,  h: 12}, class: 'feature-detail'},
      {cur: {x: 4,  y: 17, w: 6,  h: 12}, class: 'subgraph'},
      {cur: {x: 10, y: 17, w: 6,  h: 12}, class: 'subgraph-pruned'},
      {cur: {x: 0,  y: 29, w: 24, h: 36}, class: 'svd-panel'},
    ].filter(d => d)

    // Seed hide state before gridsnap packs the layout.
    gridData.forEach(d => {
      if (visState.hiddenPanels.has(d.class)) d.hidden = true
    })

    var initFns = [
      initCgButtonContainer,
      initCgGraphStats,
      initCgClusterPanel,
      (a) => initCgSubgraph({...a, opts: {mode: 'curated', selClass: '.subgraph', title: 'Curated subgraph', stateKey: 'curated'}}),
      (a) => initCgSubgraph({...a, opts: {mode: 'computed', selClass: '.subgraph-pruned', title: 'Computed clusters', stateKey: 'computed', maxNodes: 40}}), // Hierarchical or Spectral (metadata)
      initCgSvdPanel,
      initCgLinkGraph,
      initCgNodeConnections,
      initCgFeatureDetail,
    ].filter(d => d)
    
    var gridsnapSel = sel.html('').append('div.gridsnap.cg')
      .classed('is-edit-mode', visState.isGridsnap)
    if (visState.isModal) gridsnapSel.st({width: '100%', height: '100%'})

    
    visState.gridsnap = window.initGridsnap({
      gridData,
      gridSizeY: 56,
      pad: 32,
      sel: gridsnapSel,
      isFullScreenY: false,
      isFillContainer: visState.isModal,
      serializedGrid: ''
    })

    // Keep the toolbar above graph SVGs (they used to steal panel-tab clicks).
    gridsnapSel.selectAll('.grid-item').each(function (d) {
      if (d?.class === 'button-container') d3.select(this).classed('is-toolbar', 1).st({zIndex: 50})
    })

    // Close (×) on dismissible panels — lives on the grid-item so panel
    // html('') redraws don't wipe it. Skip .preview (no datum).
    var byClass = Object.fromEntries(dismissiblePanels.map(p => [p.class, p]))
    gridsnapSel.selectAll('.grid-item').each(function (d) {
      if (!d || !byClass[d.class]) return
      var meta = byClass[d.class]
      var panelId = d.class
      d3.select(this).selectAll('.panel-close').remove()
      d3.select(this).append('button.panel-close')
        .at({type: 'button', title: 'Hide ' + meta.label, 'data-panel': panelId})
        .text('×')
    })

    // Event delegation — survives rebinds and avoids per-button closure issues.
    gridsnapSel.on('click.panelClose', ev => {
      var btn = ev.target.closest?.('.panel-close')
      if (!btn || !gridsnapSel.node().contains(btn)) return
      ev.preventDefault()
      ev.stopPropagation()
      var panelId = btn.getAttribute('data-panel')
      if (panelId) setPanelHidden(panelId, true)
    })

    initFns.forEach(fn => fn({visState, renderAll, data, cgSel: sel}))

    function makeResizeFn(fn){
      return () => {
        fn({visState, renderAll, data, cgSel: sel.select('.gridsnap.cg')})
        Object.values(renderAll).forEach(d => d())
      }
    }
  }

  function setPanelHidden(className, hide) {
    if (!className) return
    if (hide) visState.hiddenPanels.add(className)
    else visState.hiddenPanels.delete(className)
    persistHiddenPanels()
    if (visState.gridsnap) {
      visState.gridsnap.setHidden(className, !!hide)
    } else {
      // Fallback if gridsnap API is missing
      sel.selectAll('.grid-item').each(function (d) {
        if (d && d.class === className) {
          d.hidden = !!hide
          d3.select(this).classed('is-hidden-panel', !!hide)
        }
      })
    }
    renderAll.panelVisibility()
  }
  visState.setPanelHidden = setPanelHidden

  initGridsnap()
  renderAll.hClerpUpdate()
  renderAll.edgeColorMode()
  renderAll.expandEdges()
  renderAll.isShowAllLinks()
  renderAll.linkType()
  renderAll.clickedId()
  renderAll.expandedIds()
  renderAll.pinnedIds()
  renderAll.features()
  renderAll.isSyncEnabled()
  renderAll.hoveredId()
  renderAll.panelVisibility()
}

window.init?.()
