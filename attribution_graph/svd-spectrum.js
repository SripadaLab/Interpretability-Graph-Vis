/**
 * Lightweight SVD for small dense matrices (n ≲ 100).
 * Returns singular values σ₁≥…≥σₙ and corresponding left singular vectors
 * (eigenvectors of M Mᵀ), so we can rank which features load on each mode.
 */
window.svdSpectrum = (function () {
  function matMul(A, B, n, m, p) {
    var out = new Array(n)
    for (var i = 0; i < n; i++) {
      out[i] = new Array(p)
      for (var j = 0; j < p; j++) {
        var s = 0
        for (var k = 0; k < m; k++) s += A[i][k] * B[k][j]
        out[i][j] = s
      }
    }
    return out
  }

  function transpose(A) {
    var n = A.length, m = A[0].length
    var T = new Array(m)
    for (var j = 0; j < m; j++) {
      T[j] = new Array(n)
      for (var i = 0; i < n; i++) T[j][i] = A[i][j]
    }
    return T
  }

  /** Jacobi EVD for symmetric S. Returns {values, vectors} with vectors as columns. */
  function jacobiEVD(S, maxIter) {
    var n = S.length
    var A = S.map(row => row.slice())
    // V starts as identity; columns become eigenvectors.
    var V = Array.from({length: n}, (_, i) => {
      var row = new Array(n).fill(0)
      row[i] = 1
      return row
    })
    maxIter = maxIter || 100
    for (var iter = 0; iter < maxIter; iter++) {
      var p = 0, q = 1, max = 0
      for (var i = 0; i < n; i++) {
        for (var j = i + 1; j < n; j++) {
          var v = Math.abs(A[i][j])
          if (v > max) { max = v; p = i; q = j }
        }
      }
      if (max < 1e-12) break
      var app = A[p][p], aqq = A[q][q], apq = A[p][q]
      var tau = (aqq - app) / (2 * apq)
      var t = Math.sign(tau) / (Math.abs(tau) + Math.sqrt(1 + tau * tau))
      if (!isFinite(t)) t = 0
      var c = 1 / Math.sqrt(1 + t * t)
      var s = t * c
      A[p][p] = app - t * apq
      A[q][q] = aqq + t * apq
      A[p][q] = A[q][p] = 0
      for (var k = 0; k < n; k++) {
        if (k === p || k === q) continue
        var aik = A[p][k], aqk = A[q][k]
        A[p][k] = A[k][p] = c * aik - s * aqk
        A[q][k] = A[k][q] = s * aik + c * aqk
      }
      // Rotate eigenvectors
      for (var k = 0; k < n; k++) {
        var vip = V[k][p], viq = V[k][q]
        V[k][p] = c * vip - s * viq
        V[k][q] = s * vip + c * viq
      }
    }
    var values = []
    for (var i = 0; i < n; i++) values.push(A[i][i])
    return {values, vectors: V}
  }

  /**
   * Full SVD spectrum + left singular vectors.
   * For large n, computes only the top-k via subspace iteration (Jacobi is O(n³)).
   * @returns {{sigmas: number[], vectors: number[][], truncated?: boolean, k?: number}}
   *   vectors[r] = length-n array = left singular vector for σ_r (node loadings)
   */
  function svd(A, opts) {
    if (!A || !A.length || !A[0].length) return {sigmas: [], vectors: []}
    var n = A.length, m = A[0].length
    opts = opts || {}
    var kWant = opts.k != null ? opts.k : Math.min(32, n)
    // Full Jacobi only for small matrices.
    if (n <= 72 && kWant >= n) {
      return svdFull(A)
    }
    if (n <= 72) {
      var full = svdFull(A)
      return {
        sigmas: full.sigmas.slice(0, kWant),
        vectors: full.vectors.slice(0, kWant),
        truncated: kWant < n,
        k: kWant,
      }
    }
    return svdTopK(A, kWant)
  }

  function svdFull(A) {
    var n = A.length, m = A[0].length
    var G = matMul(A, transpose(A), n, m, n)
    var {values, vectors: V} = jacobiEVD(G)
    var pairs = []
    for (var j = 0; j < n; j++) {
      var eig = values[j]
      var sigma = Math.sqrt(Math.max(0, eig))
      var col = new Array(n)
      for (var i = 0; i < n; i++) col[i] = V[i][j]
      var maxAbs = 0, maxIdx = 0
      for (var i = 0; i < n; i++) {
        if (Math.abs(col[i]) > maxAbs) { maxAbs = Math.abs(col[i]); maxIdx = i }
      }
      if (col[maxIdx] < 0) col = col.map(x => -x)
      pairs.push({sigma, col})
    }
    pairs.sort((a, b) => b.sigma - a.sigma)
    return {
      sigmas: pairs.map(p => p.sigma),
      vectors: pairs.map(p => p.col),
      truncated: false,
      k: n,
    }
  }

  function matVec(M, v) {
    var n = M.length, m = M[0].length
    var out = new Array(n)
    for (var i = 0; i < n; i++) {
      var s = 0
      var row = M[i]
      for (var j = 0; j < m; j++) s += row[j] * v[j]
      out[i] = s
    }
    return out
  }

  function matTVec(M, v) {
    var n = M.length, m = M[0].length
    var out = new Array(m).fill(0)
    for (var i = 0; i < n; i++) {
      var vi = v[i]
      if (!vi) continue
      var row = M[i]
      for (var j = 0; j < m; j++) out[j] += row[j] * vi
    }
    return out
  }

  function norm2(v) {
    var s = 0
    for (var i = 0; i < v.length; i++) s += v[i] * v[i]
    return Math.sqrt(s)
  }

  function normalizeInPlace(v) {
    var nrm = norm2(v)
    if (nrm < 1e-15) {
      v[0] = 1
      for (var i = 1; i < v.length; i++) v[i] = 0
      return 0
    }
    for (var i = 0; i < v.length; i++) v[i] /= nrm
    return nrm
  }

  /** Thin QR (Gram–Schmidt) on columns of X (n×k). Returns R unused; orthonormalizes X. */
  function qrColumnsInPlace(X) {
    var n = X.length, k = X[0].length
    for (var j = 0; j < k; j++) {
      // v = column j
      var v = new Array(n)
      for (var i = 0; i < n; i++) v[i] = X[i][j]
      for (var p = 0; p < j; p++) {
        var dot = 0
        for (var i = 0; i < n; i++) dot += X[i][p] * v[i]
        for (var i = 0; i < n; i++) v[i] -= dot * X[i][p]
      }
      normalizeInPlace(v)
      for (var i = 0; i < n; i++) X[i][j] = v[i]
    }
  }

  /**
   * Top-k left singular vectors via subspace iteration on A Aᵀ.
   * Enough for spectra / loadings when n is hundreds–thousands.
   */
  function svdTopK(A, k) {
    var n = A.length, m = A[0].length
    k = Math.max(1, Math.min(k || 8, n))
    // Block size slightly larger than k for stability
    var b = Math.min(n, k + (n > 120 ? 2 : 4))
    var X = Array.from({length: n}, () => {
      var row = new Array(b)
      for (var j = 0; j < b; j++) row[j] = Math.random() - 0.5
      return row
    })
    qrColumnsInPlace(X)
    var iters = n > 120 ? 14 : Math.min(40, 8 + Math.ceil(Math.log2(n + 1) * 3))
    for (var it = 0; it < iters; it++) {
      // Y = A (Aᵀ X)
      var Y = Array.from({length: n}, () => new Array(b).fill(0))
      for (var j = 0; j < b; j++) {
        var col = new Array(n)
        for (var i = 0; i < n; i++) col[i] = X[i][j]
        var Atc = matTVec(A, col)
        var Ac = matVec(A, Atc)
        for (var i = 0; i < n; i++) Y[i][j] = Ac[i]
      }
      X = Y
      qrColumnsInPlace(X)
    }
    // Rayleigh quotients: σ² ≈ ||Aᵀ u||² for unit left vector u
    var pairs = []
    for (var j = 0; j < b; j++) {
      var col = new Array(n)
      for (var i = 0; i < n; i++) col[i] = X[i][j]
      var Atu = matTVec(A, col)
      var sigma = norm2(Atu)
      var maxAbs = 0, maxIdx = 0
      for (var i = 0; i < n; i++) {
        if (Math.abs(col[i]) > maxAbs) { maxAbs = Math.abs(col[i]); maxIdx = i }
      }
      if (col[maxIdx] < 0) col = col.map(x => -x)
      pairs.push({sigma, col})
    }
    pairs.sort((a, b) => b.sigma - a.sigma)
    pairs = pairs.slice(0, k)
    return {
      sigmas: pairs.map(p => p.sigma),
      vectors: pairs.map(p => p.col),
      truncated: true,
      k: k,
    }
  }

  function singularValues(A, opts) {
    return svd(A, opts).sigmas
  }

  /** Frobenius energy of top-k singular values. Pass frob2 = ||M||_F² for truncated SVD. */
  function energyFraction(sigmas, k, frob2) {
    if (!sigmas.length) return 0
    var part = 0
    for (var i = 0; i < Math.min(k, sigmas.length); i++) {
      part += sigmas[i] * sigmas[i]
    }
    var total = frob2
    if (total == null) {
      total = 0
      for (var i = 0; i < sigmas.length; i++) total += sigmas[i] * sigmas[i]
    }
    return total > 0 ? part / total : 0
  }

  function frobenius2(A) {
    if (!A || !A.length) return 0
    var s = 0
    for (var i = 0; i < A.length; i++) {
      var row = A[i]
      for (var j = 0; j < row.length; j++) s += row[j] * row[j]
    }
    return s
  }

  /**
   * Top feature loadings for singular vector r.
   * @returns {{node, loading, abs}[]}
   */
  function topLoadings(vector, nodes, topN) {
    topN = topN || 4
    if (!vector || !nodes) return []
    var ranked = nodes.map((node, i) => ({
      node,
      loading: vector[i] || 0,
      abs: Math.abs(vector[i] || 0),
    }))
    ranked.sort((a, b) => b.abs - a.abs)
    // Skip near-zero modes (zero matrix)
    if (!ranked.length || ranked[0].abs < 1e-10) return []
    return ranked.slice(0, topN)
  }

  function linkNeighborhoodCosine(nodeIds, links) {
    var idx = {}
    nodeIds.forEach((id, i) => { idx[id] = i })
    var n = nodeIds.length
    var inflow = Array.from({length: n}, () => new Array(n).fill(0))
    var outflow = Array.from({length: n}, () => new Array(n).fill(0))
    ;(links || []).forEach(l => {
      var s = l.sourceNode?.nodeId || l.source
      var t = l.targetNode?.nodeId || l.target
      if (idx[s] == null || idx[t] == null) return
      var w = Math.abs(l.weight ?? l.pctInput ?? 0)
      if (!(w > 0)) return
      inflow[idx[t]][idx[s]] += w
      outflow[idx[s]][idx[t]] += w
    })
    function rowUnit(M) {
      return M.map((row, i) => {
        var norm = Math.sqrt(row.reduce((a, b) => a + b * b, 0))
        if (norm > 0) return row.map(v => v / norm)
        var e = new Array(n).fill(0)
        e[i % n] = 1
        return e
      })
    }
    var Uin = rowUnit(inflow), Uout = rowUnit(outflow)
    var X = Uin.map((row, i) => {
      var combined = row.concat(Uout[i])
      var norm = Math.sqrt(combined.reduce((a, b) => a + b * b, 0))
      return norm > 0 ? combined.map(v => v / norm) : combined
    })
    var C = Array.from({length: n}, () => new Array(n).fill(0))
    for (var i = 0; i < n; i++) {
      for (var j = i; j < n; j++) {
        var s = 0
        for (var k = 0; k < X[i].length; k++) s += X[i][k] * X[j][k]
        s = Math.max(-1, Math.min(1, s))
        C[i][j] = C[j][i] = s
      }
    }
    return C
  }

  function cosineFromGeom(nodeIds, geom) {
    if (!geom?.node_ids?.length || !geom.cosine) return null
    var gIdx = {}
    geom.node_ids.forEach((id, i) => { gIdx[id] = i })
    if (!nodeIds.every(id => gIdx[id] != null)) return null
    return nodeIds.map(a => nodeIds.map(b => geom.cosine[gIdx[a]][gIdx[b]]))
  }

  /**
   * Build an SV↔feature bipartite graph for cluster visualization.
   * Modes σ₁…σₖ are hubs; each gets its top-N features by |u_r|.
   * Features that appear under multiple modes become a single node with
   * multiple edges (primaryMode = argmax |loading| among those).
   */
  function buildModeGraph(svdResult, nodes, k, topN) {
    topN = topN || 4
    k = Math.max(0, Math.min(k || 0, (svdResult.sigmas || []).length))
    var sigmas = svdResult.sigmas || []
    var vectors = svdResult.vectors || []
    if (!k || !nodes?.length) return {modes: [], features: [], links: []}

    var modeNodes = []
    var featureMap = {} // nodeId -> feature node
    var links = []

    for (var r = 0; r < k; r++) {
      var mode = {
        id: 'mode-' + r,
        type: 'mode',
        r: r,
        label: 'σ' + (r + 1),
        sigma: sigmas[r] || 0,
      }
      modeNodes.push(mode)
      var tops = topLoadings(vectors[r], nodes, topN)
      tops.forEach(t => {
        if (!t.node) return
        var fid = t.node.nodeId
        var abs = t.abs
        var feat = featureMap[fid]
        if (!feat) {
          feat = featureMap[fid] = {
            id: 'feat-' + fid,
            type: 'feature',
            node: t.node,
            nodeId: fid,
            label: (t.node.ppClerp || t.node.clerp || fid || '').trim() || fid,
            primaryMode: r,
            primaryAbs: abs,
            loadings: {},
          }
        }
        feat.loadings[r] = t.loading
        if (abs > feat.primaryAbs) {
          feat.primaryMode = r
          feat.primaryAbs = abs
        }
        links.push({
          source: mode.id,
          target: feat.id,
          mode: r,
          loading: t.loading,
          abs: abs,
        })
      })
    }

    return {
      modes: modeNodes,
      features: Object.values(featureMap),
      links: links,
    }
  }

  /**
   * Summarize a singular vector's |loading| mass by token position (ctx_idx).
   * Used to test whether a component is concentrated on one position vs spanning.
   */
  function componentPositionStats(vector, nodes, tokens) {
    tokens = tokens || []
    var byCtx = {}
    var total = 0
    ;(nodes || []).forEach((node, i) => {
      var loading = vector[i] || 0
      var mass = loading * loading // |u_i|² contribution to unit vector
      var ctx = node.ctx_idx
      if (ctx == null) ctx = -1
      if (!byCtx[ctx]) {
        byCtx[ctx] = {
          ctx_idx: ctx,
          token: tokens[ctx] != null ? tokens[ctx] : ('#' + ctx),
          mass: 0,
          absSum: 0,
          nodes: [],
        }
      }
      byCtx[ctx].mass += mass
      byCtx[ctx].absSum += Math.abs(loading)
      byCtx[ctx].nodes.push({node, loading, abs: Math.abs(loading), i})
      total += mass
    })
    var rows = Object.values(byCtx).sort((a, b) => a.ctx_idx - b.ctx_idx)
    rows.forEach(r => {
      r.frac = total > 0 ? r.mass / total : 0
      r.nodes.sort((a, b) => b.abs - a.abs)
    })
    var dominant = rows.slice().sort((a, b) => b.mass - a.mass)[0] || null
    // Shannon entropy of mass across positions (nats → effective # positions)
    var entropy = 0
    rows.forEach(r => {
      if (r.frac > 0) entropy -= r.frac * Math.log(r.frac)
    })
    var effectivePositions = entropy > 0 ? Math.exp(entropy) : (rows.length ? 1 : 0)
    return {
      total,
      rows,
      dominant,
      dominantFrac: dominant ? dominant.frac : 0,
      nPositions: rows.filter(r => r.mass > 1e-12).length,
      entropy,
      effectivePositions,
      // Hypothesis heuristic: concentrated if ≥60% mass on one ctx and ≤2 active positions
      concentrated: !!(dominant && dominant.frac >= 0.6),
    }
  }

  /**
   * Restrict a precomputed SVD to a subset of nodes (same left-vector
   * coefficients, reindexed). Used for token-position focus so we never
   * re-SVD a sparse/empty among-position adjacency in the browser.
   */
  function projectSvdToNodes(svdResult, orderedNodes, viewNodes) {
    if (!svdResult?.sigmas?.length || !orderedNodes?.length || !viewNodes?.length) {
      return null
    }
    var idxOf = {}
    orderedNodes.forEach((n, i) => {
      var id = n && (n.nodeId || n.node_id)
      if (id != null) idxOf[id] = i
    })
    var indices = []
    viewNodes.forEach(n => {
      var id = n && (n.nodeId || n.node_id)
      if (id != null && idxOf[id] != null) indices.push(idxOf[id])
    })
    if (indices.length < 1) return null
    var vectors = (svdResult.vectors || []).map(v => indices.map(i => (v && v[i]) || 0))
    return {
      sigmas: (svdResult.sigmas || []).slice(),
      vectors: vectors,
      truncated: svdResult.truncated,
      k: svdResult.k,
      projected: true,
    }
  }

  return {
    svd,
    singularValues,
    energyFraction,
    frobenius2,
    topLoadings,
    buildModeGraph,
    componentPositionStats,
    linkNeighborhoodCosine,
    cosineFromGeom,
    projectSvdToNodes,
  }
})()
