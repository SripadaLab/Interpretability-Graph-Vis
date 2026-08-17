# InterpretabilityGraph (static)

Self-contained attribution-graph UI. SVD bundles and spectral \(W\) are precomputed;
this folder is just HTML/JS/JSON/NPY.

## GitHub Pages

1. Push this folder as the repo root (or copy it to `docs/` on `main`).
2. Settings → Pages → Deploy from a branch → `main` / root (or `/docs`).
3. Site URL is `https://<user>.github.io/<repo>/`.

`.nojekyll` is required so GitHub does not run Jekyll.

Preview locally: `python -m http.server 8041` in this directory, then open
http://127.0.0.1:8041/

Save downloads JSON (no write API). Upload JSON still works in-memory.
Core UI JS is vendored under `lib/`. Feature-example tiles may still hit
Anthropic / Hugging Face.

Rebuild from the parent repo:

```
interpretability-graph pages
```
