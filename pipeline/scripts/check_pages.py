"""Load the static pages/ UI and assert panels actually render."""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

PAGES = Path(__file__).resolve().parent.parent / "pages"
SLUGS = [
    "gemma-dag-demo",
    "gemma-addition",
    "gemma-fact-dallas-austin",
    "gemma-michael-jordan",
    "gemma-small-big-fr",
]


def wait_http(url: str, tries: int = 40) -> None:
    for _ in range(tries):
        try:
            urllib.request.urlopen(url, timeout=0.3)
            return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError(f"server did not start: {url}")


def check_slug(page, base: str, slug: str) -> list[str]:
    errs: list[str] = []
    bad: list[str] = []
    page.on("pageerror", lambda exc: errs.append(f"pageerror: {exc}"))
    page.on(
        "response",
        lambda res: bad.append(f"{res.status} {res.url}")
        if res.status >= 400
        and any(s in res.url for s in ("graph_data", "/data/", ".js", ".css", ".npy", "katex"))
        and "huggingface.co" not in res.url
        and "cloudfront.net" not in res.url
        and "/features/" not in res.url
        else None,
    )
    page.goto(f"{base}?slug={slug}", wait_until="domcontentloaded", timeout=60_000)
    page.wait_for_selector(".gridsnap", timeout=60_000)
    page.wait_for_timeout(4000)

    text = page.inner_text("body")
    if "Loading full-graph SVD" in text:
        errs.append("SVD still loading")
    if "Loading W…" in text or "Loading W..." in text:
        errs.append("affinity W still loading")

    n_svg = page.locator("svg").count()
    n_canvas = page.locator("canvas").count()
    if n_svg < 3:
        errs.append(f"only {n_svg} svg(s)")
    if n_canvas < 1:
        errs.append("no canvas (heatmap)")
    if page.locator(".link-graph").count() == 0:
        errs.append("no link-graph")

    svd = page.locator(".svd-panel")
    svd_text = svd.inner_text() if svd.count() else ""
    if "SVD spectra" not in svd_text:
        errs.append("SVD spectra heading missing")
    if "Spectral clusters" not in svd_text:
        errs.append("Spectral clusters heading missing")
    if "Couldn't load" in svd_text:
        errs.append("SVD panel error text")
    if "σ" not in svd_text and "sigma" not in svd_text.lower() and "σ₁" not in svd_text:
        # spectra still draw even if the heading uses KaTeX
        if "energy" not in svd_text.lower() and "rank k" not in svd_text:
            errs.append("SVD spectrum controls missing")
    if page.locator(".svd-spectral canvas, .svd-spectral-affinity canvas").count() < 1:
        if "Spectral clusters" in svd_text and n_canvas < 1:
            errs.append("no spectral W canvas")

    sel = page.locator("select.graph-prompt-select")
    if sel.count() == 0 or slug not in (sel.input_value() or ""):
        errs.append(f"dropdown {sel.input_value() if sel.count() else None}")

    computed = page.locator(".subgraph-pruned")
    if computed.count() and "run `export-spectral`" in computed.inner_text():
        errs.append("computed panel missing cluster export")

    errs.extend(bad[:8])
    print(f"{'OK' if not errs else 'FAIL'}  {slug}  svgs={n_svg} canvases={n_canvas}")
    for e in errs:
        print(f"     {e}")
    return errs


def main() -> int:
    proc = subprocess.Popen(
        [sys.executable, "-m", "http.server", "8766", "--bind", "127.0.0.1", "--directory", str(PAGES)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    rc = 0
    try:
        wait_http("http://127.0.0.1:8766/")
        with sync_playwright() as p:
            browser = p.chromium.launch()
            for slug in SLUGS:
                page = browser.new_page()
                if check_slug(page, "http://127.0.0.1:8766/", slug):
                    rc = 1
                page.close()
            browser.close()
    finally:
        proc.terminate()
        proc.wait(timeout=5)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
