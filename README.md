# InterpretabilityGraph (static)

Self-contained attribution-graph UI. SVD bundles and spectral \(W\) are precomputed;
this folder is just HTML/JS/JSON/NPY.

## GitHub Pages

Do **not** use GitHub's default "Jekyll" Pages workflow (`jekyll-build-pages`).
This folder is already static; that action is unnecessary and often 429s.

Either:

- Settings → Pages → Deploy from a branch → `main` / root, or
- Settings → Pages → GitHub Actions, keep `.github/workflows/pages.yml`
  (static upload, no Jekyll). Delete `jekyll-gh-pages.yml` if GitHub added it.

Site URL: `https://<user>.github.io/<repo>/`. `.nojekyll` is included.

Preview locally: `python -m http.server 8041` in this directory, then open
http://127.0.0.1:8041/

Save downloads JSON (no write API). Upload JSON still works in-memory.
Core UI JS is vendored under `lib/`. Feature-example tiles may still hit
Anthropic / Hugging Face.

Rebuild from the parent repo:

```
interpretability-graph pages
```
