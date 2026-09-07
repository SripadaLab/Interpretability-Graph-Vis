window.initCgButtonContainer = function({visState, renderAll, data, cgSel}){
  var buttonContainer = cgSel.select('.button-container').html('')
    .st({
      marginBottom: '0',
      display: 'flex',
      flexWrap: 'wrap',
      gap: '10px 14px',
      alignItems: 'center',
      padding: '6px 8px',
      rowGap: '10px',
    })

  function addGroup(label) {
    var g = buttonContainer.append('div.btn-group')
      .st({display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap'})
    if (label) {
      g.append('span.btn-group-label')
        .text(label)
        .st({fontSize: '10px', color: '#888', textTransform: 'uppercase', letterSpacing: '0.04em', marginRight: '2px'})
    }
    return g
  }

  var linkGroup = addGroup('Links')
  var linkTypeHelp = {
    input: 'Show edges into the clicked / pinned nodes',
    output: 'Show edges out of the clicked / pinned nodes',
    either: 'Show edges into or out of the clicked / pinned nodes',
    both: 'Show only edges that connect two pinned nodes',
  }
  var linkTypeSel = linkGroup.append('div.link-type-buttons')
    .appendMany('div', ['input', 'output', 'either', 'both'])
    .text(d => d[0].toUpperCase() + d.slice(1).toLowerCase())
    .at({title: d => linkTypeHelp[d]})
    .on('click', (ev, d) => {
      visState.linkType = d
      // Focused link modes are hard to see under the full hairball — drop it
      // so Input / Output / Either / Both visibly filter the graph.
      if (visState.isShowAllLinks && d !== 'either') {
        visState.isShowAllLinks = ''
        renderAll.isShowAllLinks()
      }
      renderAll.linkType()
    })
  renderAll.linkType.fns.push(() => {
    linkTypeSel.classed('active', d => d === visState.linkType)
  })

  var viewGroup = addGroup('View')
  var showAllSel = viewGroup.append('div.toggle-buttons')
    .append('div').text('Show all links')
    .on('click', () => {
      visState.isShowAllLinks = visState.isShowAllLinks ? '' : '1'
      if (visState.isShowAllLinks) {
        visState.expandEdges = ''
        util.params.set('expandEdges', '0')
        renderAll.expandEdges()
        // Full hairball implies either-direction viewing
        if (visState.linkType !== 'either') {
          visState.linkType = 'either'
          renderAll.linkType()
        }
      }
      renderAll.isShowAllLinks()
    })
  renderAll.isShowAllLinks.fns.push(() => {
    showAllSel.classed('active', visState.isShowAllLinks)
  })

  var expandSel = viewGroup.append('div.toggle-buttons')
    .append('div').text('Expand edges')
    .at({title: 'Hide all edges until you click a node to expand it'})
    .on('click', () => {
      visState.expandEdges = visState.expandEdges ? '' : '1'
      util.params.set('expandEdges', visState.expandEdges ? '1' : '0')
      if (visState.expandEdges) {
        visState.isShowAllLinks = ''
        renderAll.isShowAllLinks()
      }
      renderAll.expandEdges()
      renderAll.expandedIds()
    })
  renderAll.expandEdges.fns.push(() => {
    expandSel.classed('active', !!visState.expandEdges)
  })

  viewGroup.append('div.toggle-buttons')
    .append('div').text('Collapse all')
    .on('click', () => {
      visState.expandedIds = []
      util.params.set('expandedIds', '')
      renderAll.expandedIds()
    })

  var thicknessGroup = addGroup('Thickness')
  var thicknessSel = thicknessGroup.append('div.link-type-buttons.thickness-modes')
    .appendMany('div', [
      {id: 'relative', label: 'Relative'},
      {id: 'stable', label: 'Stable'},
    ])
    .text(d => d.label)
    .at({title: d => d.id === 'stable'
      ? 'Edge width uses a fixed full-graph scale — width does not change as you prune nodes'
      : 'Edge width is normalized to the currently-visible edges — rescales as you prune'})
    .on('click', (ev, d) => {
      visState.thicknessMode = d.id
      renderAll.thicknessMode()
    })
  renderAll.thicknessMode.fns.push(() => {
    thicknessSel.classed('active', d => d.id === visState.thicknessMode)
  })

  var colorGroup = addGroup('Color')
  var colorModes = [
    {id: 'greedy4', label: '4-color'},
    {id: 'greedy2', label: '2-color'},
    {id: 'layer_parity', label: 'Layer'},
    {id: 'sign', label: 'Sign'},
    {id: 'prgn', label: 'PRGn'},
  ]
  var colorModeSel = colorGroup.append('div.link-type-buttons.edge-color-modes')
    .appendMany('div', colorModes)
    .text(d => d.label)
    .on('click', (ev, d) => {
      visState.edgeColorMode = d.id
      renderAll.edgeColorMode()
    })
  renderAll.edgeColorMode.fns.push(() => {
    colorModeSel.classed('active', d => d.id === visState.edgeColorMode)
  })

  var sgGroup = addGroup('Subgraph')
  var sgBtns = sgGroup.append('div.toggle-buttons.subgraph-controls')
  sgBtns.append('div').text('Fill')
    .at({title: 'Pin strongest upstream path from clicked node (or top logits)'})
    .on('click', () => {
      if (window.edgeColoring?.fillSubgraph) {
        window.edgeColoring.fillSubgraph(visState, renderAll, data)
      }
    })
  sgBtns.append('div').text('Clear')
    .on('click', () => {
      visState.pinnedIds = []
      if (visState.subgraph) visState.subgraph.supernodes = []
      util.params.set('pinnedIds', '')
      util.params.set('supernodes', '')
      renderAll.pinnedIds()
    })

  // Panel tabs — active = visible; click to hide/show secondary displays.
  var panels = visState.dismissiblePanels || []
  if (panels.length) {
    var panelGroup = addGroup('Panels')
    var panelSel = panelGroup.append('div.link-type-buttons.panel-tabs')
      .appendMany('div', panels)
      .text(d => d.label)
      .at({title: d => 'Show or hide the ' + d.label + ' panel'})
      .on('click', (ev, d) => {
        var hide = !visState.hiddenPanels.has(d.class)
        visState.setPanelHidden?.(d.class, hide)
      })
    renderAll.panelVisibility.fns.push(() => {
      panelSel.classed('active', d => !visState.hiddenPanels.has(d.class))
      panelSel.classed('panel-tab-hidden', d => visState.hiddenPanels.has(d.class))
    })
  }

  var clearGroup = addGroup('')
  clearGroup.append('div.toggle-buttons')
    .appendMany('div', ['Clear pinned', 'Clear clicked'])
    .text(d => d)
    .on('click', (ev, d) => {
      if (d == 'Clear pinned') {
        visState.pinnedIds = []
        util.params.set('pinnedIds', '')
        renderAll.pinnedIds()
      } else {
        visState.clickedId = ''
        renderAll.clickedId()
      }
    })

  cgSel.on('keydown.esc-check', ev => {
    if (ev.key == 'Escape') {
      visState.clickedId = ''
      renderAll.clickedId()
    }
  })

  clearGroup.append('div.toggle-buttons')
    .append('div').text('Reset grid')
    .on('click', () => {
      util.params.set('gridsnap', '')
      window.location.reload()
    })

  var onSyncValue = visState.isSyncEnabled || '1'
  var syncButtonSel = clearGroup.append('div.toggle-buttons')
    .append('div').text('Sync')
    .on('click', () => {
      visState.isSyncEnabled = visState.isSyncEnabled ? '' : onSyncValue
      renderAll.isSyncEnabled()
    })
  renderAll.isSyncEnabled.fns.push(() => {
    syncButtonSel.classed('active', visState.isSyncEnabled)
  })
}

window.init?.()
