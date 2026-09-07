"""Build a static GitHub Pages folder (frontend + precomputed graph assets)."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
DEFAULT_DATA = ROOT / "graph_files"
DEFAULT_OUT = ROOT / "pages"

# Copied into pages/ — not used by the main attribution-graph UI.
_SKIP_FRONTEND = {
    "addition",
    "feature-view.html",
    "prettier.config.js",
    "README.md",
    "LICENSE",
}

def _frontend_ignore(directory: str, names: list[str]) -> list[str]:
    skip = []
    for name in names:
        if name in _SKIP_FRONTEND or name.startswith("."):
            skip.append(name)
    return skip


def _needed_relpaths(graph_dir: Path) -> list[Path]:
    """Graph JSON + metadata pointers the UI actually fetches."""
    needed: list[Path] = []
    for p in sorted(graph_dir.glob("*.json")):
        needed.append(p)
        if p.name == "graph-metadata.json":
            continue
        try:
            md = json.loads(p.read_text()).get("metadata") or {}
        except json.JSONDecodeError:
            logger.warning("Skip unreadable %s", p.name)
            continue
        bundle = md.get("svd_bundle") or {}
        if bundle.get("path"):
            needed.append(graph_dir / bundle["path"])
        spec = md.get("spectral_clusters") or {}
        if spec.get("affinity_path"):
            needed.append(graph_dir / spec["affinity_path"])
        full = md.get("svd_geom_full") or {}
        for key in ("csv", "labels", "meta"):
            if full.get(key):
                needed.append(graph_dir / full[key])
    # unique, existing only
    seen: set[Path] = set()
    out: list[Path] = []
    for src in needed:
        src = src.resolve()
        if src in seen or not src.is_file():
            if src not in seen and not src.is_file():
                logger.warning("Missing referenced file: %s", src)
            continue
        seen.add(src)
        out.append(src)
    return out


def _patch_index_html(path: Path) -> None:
    text = path.read_text()
    if "window.isStaticPages = true" not in text:
        text = text.replace(
            "window.isLocalServing = true;",
            "window.isLocalServing = true;\n  window.isStaticPages = true;",
            1,
        )
    path.write_text(text)


def _write_pages_readme(out_dir: Path) -> None:
    (out_dir / "README.md").write_text(
        r"""# InterpretabilityGraph (static)

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
"""
    )


def _write_pages_workflow(out_dir: Path) -> None:
    wf = out_dir / ".github" / "workflows" / "pages.yml"
    wf.parent.mkdir(parents=True, exist_ok=True)
    wf.write_text(
        """# Static site — do not use actions/jekyll-build-pages (unneeded, and 429s).
name: Deploy Pages

on:
  push:
    branches: [main, master]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: true

jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with:
          path: .
      - id: deployment
        uses: actions/deploy-pages@v4
"""
    )


def export_pages(
    data_dir: str | Path = DEFAULT_DATA,
    out_dir: str | Path = DEFAULT_OUT,
    *,
    compute: bool = False,
    slugs: list[str] | None = None,
) -> Path:
    """Copy the UI and precomputed assets into ``out_dir``."""
    from interpretability_graph.serve import _sync_metadata

    data_dir = Path(data_dir).resolve()
    out_dir = Path(out_dir).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"No graph directory: {data_dir}")

    if compute:
        _try_compute(data_dir, slugs)

    _sync_metadata(data_dir)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    shutil.copytree(
        FRONTEND,
        out_dir,
        dirs_exist_ok=True,
        ignore=_frontend_ignore,
    )

    _patch_index_html(out_dir / "index.html")
    (out_dir / ".nojekyll").write_text("")
    _write_pages_readme(out_dir)
    _write_pages_workflow(out_dir)

    graph_out = out_dir / "graph_data"
    graph_out.mkdir(parents=True, exist_ok=True)
    data_out = out_dir / "data"
    data_out.mkdir(parents=True, exist_ok=True)

    copied = 0
    bytes_copied = 0
    for src in _needed_relpaths(data_dir):
        rel = src.relative_to(data_dir)
        dest = graph_out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied += 1
        bytes_copied += dest.stat().st_size
        if src.name == "graph-metadata.json":
            shutil.copy2(src, data_out / "graph-metadata.json")

    (out_dir / "404.html").write_text(
        "<!DOCTYPE html><meta charset='utf-8'>"
        "<title>Not found</title>"
        "<p>Missing file. This is a static site — try "
        "<a href='./'>the graph UI</a>.</p>\n"
    )

    mb = bytes_copied / (1024 * 1024)
    logger.info("Wrote %s (%d files, %.1f MB of graph data)", out_dir, copied, mb)
    return out_dir


def _try_compute(data_dir: Path, slugs: list[str] | None) -> None:
    try:
        import numpy  # noqa: F401
    except ImportError:
        logger.warning("--compute needs numpy; skipping SVD / spectral export")
        return

    missing_svd: list[str] = []
    missing_spec: list[str] = []
    for p in sorted(data_dir.glob("*.json")):
        if p.name == "graph-metadata.json":
            continue
        slug = p.stem
        if slugs and slug not in slugs:
            continue
        md = json.loads(p.read_text()).get("metadata") or {}
        if not md.get("svd_bundle"):
            missing_svd.append(slug)
        if not (md.get("spectral_clusters") or {}).get("by_k"):
            missing_spec.append(slug)

    if missing_svd:
        from interpretability_graph.svd_export import export_svd_bundles

        logger.info("export-svd for %s", ", ".join(missing_svd))
        try:
            export_svd_bundles(data_dir, slugs=missing_svd)
        except Exception:
            logger.exception("export-svd failed")

    if missing_spec:
        from interpretability_graph.spectral_clusters import export_spectral_clusters

        logger.info("export-spectral for %s", ", ".join(missing_spec))
        try:
            export_spectral_clusters(data_dir, slugs=missing_spec)
        except Exception:
            logger.exception("export-spectral failed")
