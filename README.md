# InterpretabilityGraph

Gemma-2-2B attribution graphs after [Anthropic's circuit tracing methods](https://transformer-circuits.pub/2025/attribution-graphs/methods.html#building-architecture). Edges and nodes use a small palette with intensity from \(|\mathrm{attribution}|\) / \(|\mathrm{influence}|\).

## What this implements

1. Replacement-model / CLT setup (paper § Building an Interpretable Replacement Model) via [circuit-tracer](https://github.com/safety-research/circuit-tracer) and pretrained GemmaScope PLTs or published CLTs (`426K` / `2.5M`).
2. Attribution + pruning to JSON for the layer×token UI.
3. Edge / node coloring: greedy conflict minimization, layer parity, or sign (`interpretability_graph/edge_coloring.py`).
4. Full-graph SVD of cosine and attribution adjacencies — spectra, SV↔feature clusters, component × token ([SVD panel](#svd-panel-spectra-clusters-component--token)).
5. Spectral clustering on that cosine: \(W=\max(C,0)\) after mutual kNN, \(L_{\mathrm{sym}}\) cuts, k-means on raw eigenvector rows; the UI draws \(W\) in group order ([Spectral clustering](#spectral-clustering-on-cosine)).

Training a CLT (~210 H100-hours for 2M features on Gemma-2B) is out of scope. Use published transcoder weights.

## Quick start (offline demo — no GPU)

```bash
cd InterpretabilityGraph
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Write/reset demos and open the UI
interpretability-graph demo
interpretability-graph serve
```

Open [http://127.0.0.1:8041/](http://127.0.0.1:8041/). `serve` maps `/graph_data/` and `/data/` to `graph_files/`. All links are shown by default. Upload JSON keeps the graph in memory; Save writes `qParams` to disk (`POST /save_graph/{slug}`) or downloads an uploaded file. Examples:

| Graph                | Prompt                                                             |
| -------------------- | ------------------------------------------------------------------ |
| **DAG acronym**      | `The National Digital Analytics Group (`                           |
| **3 + 5**            | `<bos>3 + 5 =` (GemmaScope)                                        |
| **Capital of Texas** | `Fact: The capital of the state containing Dallas is` (GemmaScope) |
| **Michael Jordan**   | `Michael Jordan plays the sport of` (GemmaScope)                   |
| **French opposite**  | `Le contraire de "petit" est "` (GemmaScope)                       |

Pull all precomputed graphs (no GPU):

```bash
interpretability-graph fetch
```

Use the toolbar to switch:

| Mode        | Behavior                                                |
| ----------- | ------------------------------------------------------- |
| **4-color** | Default. Least-conflict assignment from a 4-hue palette |
| **2-color** | Same heuristic restricted to teal/coral                 |
| **Layer**   | Color by source-layer parity                            |
| **Sign**    | Teal = positive, coral = negative                       |
| **PRGn**    | Original Anthropic green/purple scale                   |

Pruning at 1.0 shows every loaded node; the subgraph uses the same threshold. The stats sidebar lists feature counts, visible vs loaded nodes/links, and the current palette. Expand edges hides links until a node is clicked.

## Pretrained graphs (no local GPU needed)

```bash
interpretability-graph fetch
interpretability-graph serve
```

`fetch` pulls GemmaScope graphs already attributed (addition, Dallas→Austin, Michael Jordan, French petit→grand) plus the synthetic DAG demo.

## GitHub Pages

The UI is static once SVD / spectral files are written. `pages/` is a folder you can push as a project site (or copy to `docs/` and serve from this repo).

```bash
# optional: fill in missing bundles (needs numpy; may download transcoder weights)
interpretability-graph export-svd
interpretability-graph export-spectral

interpretability-graph pages          # → pages/
# interpretability-graph pages --out docs
```

Push `pages/` as the repo root, or enable Pages on `main` / `/docs`. `.nojekyll` is included. Preview with `python -m http.server 8041 --directory pages`. Save downloads JSON (no write API). Feature-example tiles may still hit Anthropic / Hugging Face CDNs.

## Optional: local attribution CLI

The UI does not run the model. To attribute a new prompt:

1. Accept the [Gemma license](https://huggingface.co/google/gemma-2-2b) and run `hf auth login`.
2. Install: `pip install -e '.[gemma]'`
3. Run:

```bash
interpretability-graph attribute \
  --prompt "The National Digital Analytics Group (" \
  --preset gemma \
  --slug gemma-dag \
  --serve
```

Presets:

| `--preset`       | Transcoders                                       |
| ---------------- | ------------------------------------------------- |
| `gemma`          | GemmaScope PLTs (`mntss/gemma-scope-transcoders`) |
| `gemma-clt-426k` | Cross-layer transcoder 426K                       |
| `gemma-clt-2.5m` | Cross-layer transcoder 2.5M                       |

On low VRAM, add `--offload cpu` or `--dtype float16`.

Re-color an existing JSON without re-attributing:

```bash
interpretability-graph colorize graph_files/gemma-dag.json --mode greedy4
```

## Architecture (CLT)

Each feature at layer $\ell$ reads the residual stream and writes to MLP outputs at layers $\ell\ldots L$:

$$
a^{\ell} = \mathrm{JumpReLU}(W_{\mathrm{enc}}^{\ell} x^{\ell}),
\qquad
\hat{y}^{\ell} = \sum_{\ell'\le \ell} W_{\mathrm{dec}}^{\ell'\to\ell} a^{\ell'}.
$$

The **local replacement model** freezes attention patterns and LN denominators so edges are linear attributions (see `interpretability_graph/architecture.py`).

## Edge / node coloring

`interpretability_graph/edge_coloring.py` implements a **greedy edge k-coloring** (default k=4: teal, coral, ochre, slate):

1. Sort edges by descending weight.
2. At each endpoint, count already-colored incident edges per class.
3. Assign the least-used class (tie → sign cue).
4. Map normalized weight → saturation + alpha; nodes take a weighted-majority class from incident edges.

Classical edge coloring can need $\Delta+1$ colors; this is a cheaper assignment for dense DAGs. The subgraph panel uses the same palette on aggregate links.

## SVD panel (spectra, clusters, component × token)

The SVD tile uses transcoder feature nodes only (no embeddings, logits, or error nodes). Vectors come from `export-svd`; the page does not run full-graph SVD in the browser.

### Export the data (required for full-graph view)

```bash
# Full-graph SVD bundle (σ + left vectors for 4 matrices; default k=32)
interpretability-graph export-svd \
  --slug gemma-michael-jordan \
  --slug gemma-fact-dallas-austin

# Optional: dense cosine matrix export (CSV / NPY)
interpretability-graph export-cosine \
  --slug gemma-michael-jordan \
  --slug gemma-fact-dallas-austin
```

Outputs:

| Path | Contents |
|------|----------|
| `graph_files/svd_exports/{slug}-svd-bundle.json` | `node_ids`, top-$k$ `sigmas` + `vectors` for cosine / unsigned / signed / symmetric, `frob2` |
| `graph_files/svd_exports/{slug}-{matrix}.npy` | Dense $n\times n$ float32 matrices |
| `graph_files/cosine_exports/` | Unthresholded cosine CSV / NPY / labels |
| Graph JSON `metadata.svd_bundle` | Pointer `{path, n, k, matrices}` so the UI can `fetch('/graph_data/…')` |

Without a bundle, the UI falls back to an **edge-expanded sample** (≤250 nodes) and client-side truncated SVD.

### Node set and order

1. Take every graph node with `feature_type` containing `transcoder` (exclude `error`, `logit`, embeddings).
2. Rank by $|\mathrm{influence}|$ descending — this becomes `node_ids` and the index order of every matrix / $\mathbf{u}_r$.
3. Full export uses **all** such nodes (e.g. MJ $n=1174$, Dallas $n=697$).

### Feature vectors → cosine matrix

For each feature $f$ at layer $\ell$, load GemmaScope transcoder weights and form a unit direction:

$$
\mathbf{v}_f
=
\operatorname{unit}\Big[
  \widehat{W_{\mathrm{enc}}}[:,f]
  \,\Vert\,
  \widehat{W_{\mathrm{dec}}}[f,:]
\Big]
$$

(concat of unit encoder column and unit decoder row). Then

$$
C_{ij}=\cos(\mathbf{v}_i,\mathbf{v}_j)=\mathbf{v}_i^\top\mathbf{v}_j\in[-1,1].
$$

If weights are missing (synthetic demos), fall back to cosine of **concatenated unit in/out attribution neighborhood** vectors among selected nodes.

### Attribution adjacency matrices

Over the same `node_ids`, accumulate graph edges with weight $w$ (`weight` or `pctInput`):

| Matrix | Entry |
|--------|--------|
| **Unsigned** | $A_{ij}=\lvert w(i\to j)\rvert$ (directed) |
| **Signed** | $A_{ij}=w(i\to j)$ |
| **Symmetric** | $A_{ij}=\lvert w(i\to j)\rvert+\lvert w(j\to i)\rvert$ |

Only edges with both endpoints in the transcoder node set are included.

### SVD

For each matrix $M\in\{C,\,A_{\mathrm{unsigned}},\,A_{\mathrm{signed}},\,A_{\mathrm{sym}}\}$:

$$
M = U\Sigma V^\top,
\qquad
\sigma_1\ge\sigma_2\ge\cdots\ge 0,
\qquad
\mathbf{u}_r = \text{$r$-th column of $U$}.
$$

- Backend: NumPy economy SVD; store top $k=32$ singular values and **left** vectors.
- Orient each $\mathbf{u}_r$ so its largest-magnitude entry is **positive**.
- Energy in the UI:

$$
\mathrm{energy}(k)
=
\frac{\sum_{i=1}^{k}\sigma_i^{2}}{\lVert M\rVert_{F}^{2}}.
$$

$\sigma_r$ is the strength of mode $r$; $u_{r,i}$ is feature $i$'s loading ($\lVert\mathbf{u}_r\rVert_2=1$). Unsigned and Symmetric $M$ are nonnegative, so the leading $\mathbf{u}$ is usually all $\ge 0$ (Perron). Opposite-sign products $u_i u_j$ (orange in the component matrix) show up mainly on Cosine and Signed.

### UI

Matrix picker: Cosine / Unsigned / Signed / Symmetric. Same `node_ids` order as the bundle.

#### 1. Spectra + top loadings

- Bar chart of $\sigma_1,\ldots,\sigma_k$ (teal = ranks inside current energy $k$).
- Under each spectrum: top features by $\lvert u_{r,i}\rvert$ for $\sigma_1\ldots\sigma_k$ (click → sticky select / pin in the main graph).

#### 2. SV → feature clusters

- Hubs = modes $\sigma_1\ldots\sigma_k$; spokes = top-$N$ features by $\lvert u\rvert$ per mode (slider, default $N=8$).
- Spokes are mode↔feature (from $|u|$).
- **Position** chips filter features. With a bundle, precomputed $\mathbf{u}_r$ is restricted to those nodes.

#### 3. Component × token position

Mass $p_t$ says whether mode $r$ sits on one prompt position (`ctx_idx`) or spreads. Re-blocking the component matrix by layer asks the same thing about depth.

For selected component $r$ and vector $\mathbf{u}_r$:

$$
p_t
=
\sum_{i=1}^{n}
u_{r,i}^{2}\,
\mathbf{1}_{\{\mathrm{ctx}_i=t\}},
\qquad
\sum_t p_t = 1,
\qquad
t^{\star}=\arg\max_t p_t.
$$

$$
H=-\sum_t p_t\log p_t,
\qquad
e^{H}=\text{effective number of positions}.
$$

Label: concentrated / local if $p_{t^{\star}}\ge 0.6$, else position-spanning.

- Mass bars $p_t$ (teal = $t^{\star}$).
- Scatter: $x=\mathrm{ctx}_i$, $y=\mathrm{layer}_i$, radius $\propto\lvert u_{r,i}\rvert$.
- Component matrix $(UU^{\top})_{ij}=u_{r,i}u_{r,j}$ (teal $>0$, orange $<0$; quantile-scaled $|u|$), blocked by token, layer, or both nested.

**Component matrix controls**

| Control | Behavior |
|---------|----------|
| **Top 28 / All** | Top 28 = strongest $\lvert u\rvert$ globally, then laid out for display. All = every feature in scope (full component). |
| **Group by: token** (default) | Blocks by `ctx_idx`. Bright diagonal at $t^\star$ = those features co-load. |
| **Group by: layer** | Blocks by layer. One bright diagonal block = one depth; bright off-diagonal = two layers co-loading. |
| **Group by: token → layer** | Token blocks, layer sub-blocks (heavy / light dashed rules). |
| **Group by: layer → token** | Layer blocks, token sub-blocks (heavy / light rules swapped). |
| **Within block: unsorted** (default) | Arrival order (`node_ids` = $|\mathrm{influence}|$ rank). Independent of $\mathbf{u}_r$. |
| **Within block: by \|u\|** | Inside the innermost block, strongest $\lvert u\rvert$ first. |
| **Within block: by signed u** | Signed $u$ descending (+pole → −pole). |
| **Position** | One `ctx` (filters nodes; $p_t$ still from the full $\mathbf{u}_r$). |
| **PNG** | 2× export (cells redrawn, not upscaled). Caption has slug, prompt, $\sigma_r$, matrix, grouping, within-block order, feature counts. Example: `gemma-michael-jordan-sigma1-cosine-similarity-layertoken-none.png`. |

Grouping permutes rows and columns only. The full $n\times n$ $M$ is not drawn (see `.npy` / `cosine_exports/`).

### Spectral clustering on cosine

Same $C$ as above. Build $W=\mathrm{kNN}(\max(C,0))$, take the smallest eigenpairs of $L_{\mathrm{sym}}$, then k-means on the stacked rows.

```bash
# Prefers graph_files/svd_exports/{slug}-cosine.npy when present (all transcoder nodes)
interpretability-graph export-spectral \
  --slug gemma-michael-jordan \
  --slug gemma-fact-dallas-austin \
  --knn 10 \
  --k-max 12
```

Writes `metadata.spectral_clusters` (`by_k` matches agglomerative `auto_clusters`: `[["group 1", nodeId, …], …]`) and `graph_files/svd_exports/{slug}-affinity.npy`.

#### Pipeline

1. **Nodes.** Transcoder features from the cosine / SVD export, or `--max-nodes N` by $|\mathrm{influence}|$. Shipped graphs: $n=1174$ (Michael Jordan), $n=697$ (Dallas).
2. **Cosine.** $C_{ij}=\mathbf{v}_i^\top\mathbf{v}_j\in[-1,1]$ from the same enc∥dec unit directions. Prefer the precomputed `.npy`.
3. **Affinity.**
   $$W^{(0)}_{ij}=\max(C_{ij},\,0),\qquad W^{(0)}_{ii}=0.$$
4. **Mutual kNN.** Keep each node's `knn` strongest positive links (default 10). Symmetrize $W=\tfrac12(A+A^\top)$ (one-sided edges keep half weight). Isolated nodes reconnect to their best positive partner so $d_i>0$.

   Later cuts only see these edges. $W_{ij}=0$ has no cut cost and does not couple $u_i$ and $u_j$.

5. **Symmetric normalized Laplacian.** $d_i=\sum_j W_{ij}$, $D=\mathrm{diag}(d)$,
   $$
   L_{\mathrm{sym}}=I-D^{-1/2}WD^{-1/2},
   \qquad
   (L_{\mathrm{sym}})_{ij}=-\frac{W_{ij}}{\sqrt{d_i d_j}}\ \ (i\neq j).
   $$
   $L_{\mathrm{sym}}$ is real symmetric (`eigh`). For $u\in\mathbb{R}^n$,
   $$
   R(u)=\frac{u^\top L_{\mathrm{sym}} u}{u^\top u}
   $$
   is the degree-normalized jump of $u$ across $W$. Eigenpairs $(\lambda,u)$ minimize $R$ in order, orthogonal to earlier $u$. This is a normalized-cut relaxation (plain min-cut usually peels one low-degree node).

   - $\lambda_1\approx 0$, $u^{(1)}$ nearly constant.
   - $\lambda_2,u^{(2)}$: cheapest balanced 2-way split (usually $+$ / $-$).
   - Later pairs: more bottlenecks.

   Michael Jordan, first twelve $\lambda$: $0,\,0.025,\,0.039,\,\ldots$. Largest gaps after $\lambda_1$ sit after $\lambda_2$ and $\lambda_8$. Slider $k=12$ uses every computed mode.

6. **Stack and k-means.** For each $k=2\ldots k_{\max}$,
   $$
   x_i=U_{i,1:k}\in\mathbb{R}^{k}.
   $$
   k-means++ (several restarts) on those rows. Rows are not renormalized, so $|x_i|$ counts in the distance. Centroids are k-means means. Labels go to supernodes and to the row/column order of $W$.

7. **Store.** `by_k["k"]`, `eigenvalues`, `row_normalize: false`, `affinity_path`, `knn`, `m`. Default slider $k=\min(k_{\max},\mathrm{round}(n/5))$ (12 on these graphs).

#### UI

Heatmap of $W$, rows/columns by k-means label then $|\mathrm{influence}|$. Teal $=W_{ij}$; white $=$ no kNN edge. All entries $\ge 0$.

White between two diagonal blocks: the cut ran through missing kNN edges. Teal between blocks: labels split a friendship.

| Control | Role |
|---------|------|
| **groups $k$** / spectrum bars | How many leading eigenvectors are stacked before k-means. $W$ is fixed; only the permutation changes. Click a $\lambda$ bar or the gap hint. Teal bars are the stacked coordinates. |
| **Top 64 / All** | Features (transcoder nodes). All = every clustered feature. Top 64 = highest-$\lvert\mathrm{influence}\rvert$ in each group ($\lceil 64/k\rceil$ per group, then clipped). Legend chips = full group sizes; footer (`62 × 62 shown`) = heatmap size. |
| **Group chips / cards** | Isolate one block. Same labels as **Computed · Spectral**. Click a feature to select it in the main graph. |

**Computed** subgraph: Hierarchical = agglomerative `auto_clusters` from `verify`; Spectral = this export. Both sliders read `by_k`.

Michael Jordan, $k=5$, Top 64: sizes $\approx 841 / 273 / 36 / 14 / 10$. The last three are small kNN cliques; the first two are large and pale. Most features land in group 1. Off-diagonal $W$ is mostly empty.

At $k=12$: ~670 features in a remainder with almost no internal $W$, plus several 10–35-node cliques (40–100% linked inside). Mean within-block $W$ is hundreds of times the between-block mean, driven by those cliques.

SVD **Top 28 / All** is a different control: it subsets by $|u_r|$ for one mode.

| | SVD | Spectral |
|--|-----|----------|
| Matrix | Dense $C$ (or an attribution adjacency) | Sparse $W=\mathrm{kNN}(\max(C,0))$ |
| Spectrum | Largest $\sigma_r$ of $M$ | Smallest $\lambda_r$ of $L_{\mathrm{sym}}$ |
| Heatmap | $u_r u_r^\top$ | $W_{ij}$ |
| Groups | SV↔feature spokes | k-means on stacked Laplacian rows |

### Code map

| Piece | Location |
|-------|----------|
| Cosine + candidates | `interpretability_graph/verify_supernodes.py` (`build_feature_cosine`) |
| SVD export | `interpretability_graph/svd_export.py` |
| Spectral clustering | `interpretability_graph/spectral_clusters.py` |
| Pages export | `interpretability_graph/pages_export.py` |
| CLI | `interpretability-graph export-svd` / `export-cosine` / `export-spectral` / `pages` / `verify` |
| UI panel | `frontend/attribution_graph/init-cg-svd-panel.js` |
| Computed / spectral subgraph | `frontend/attribution_graph/init-cg-subgraph.js` |
| Client SVD helpers | `frontend/attribution_graph/svd-spectrum.js` |

## Layout

```
interpretability_graph/   # Python package (attribute, colorize, serve, export-svd, export-spectral, pages, …)
frontend/                 # Patched Anthropic attribution-graph UI (+ SVD / spectral panel)
graph_files/              # Exported / demo JSON graphs (+ metadata.spectral_clusters)
  svd_exports/            # SVD bundles, cosine / affinity .npy
  cosine_exports/         # Optional dense cosine CSV/NPY
pages/                    # Static GitHub Pages snapshot (`interpretability-graph pages`)
```

## Cite

Ameisen et al., *Circuit Tracing: Revealing Computational Graphs in Language Models*, 2025.  
Hanna & Piotrowski et al., `circuit-tracer`.