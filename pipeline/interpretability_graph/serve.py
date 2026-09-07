"""Local HTTP server that serves the enhanced attribution-graph frontend.

Graphs are precomputed (Neuronpedia / GemmaScope). On-device attribution is
available via the CLI (`interpretability-graph attribute`) but not the UI.
"""

from __future__ import annotations

import json
import logging
import shutil
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import TCPServer
from urllib.parse import unquote

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
DEFAULT_DATA = ROOT / "graph_files"


class ReusableTCPServer(TCPServer):
    allow_reuse_address = True


class GraphHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, data_dir: Path, frontend_dir: Path, **kwargs):
        self.data_dir = data_dir
        self.frontend_dir = frontend_dir
        super().__init__(*args, directory=str(frontend_dir), **kwargs)

    def log_message(self, format, *args):
        logger.info("%s - %s", self.address_string(), format % args)

    def _json_response(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        path = self.path.split("?")[0]
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode() or "{}")
        except json.JSONDecodeError:
            return self._json_response(400, {"error": "Invalid JSON"})

        if path.startswith("/save_graph/"):
            slug = unquote(path[len("/save_graph/") :])
            local = self.data_dir / f"{slug}.json"
            if not local.exists():
                return self._json_response(404, {"error": f"Unknown slug {slug}"})
            try:
                with local.open() as f:
                    graph = json.load(f)
                if "qParams" in payload:
                    graph["qParams"] = payload["qParams"]
                with local.open("w") as f:
                    json.dump(graph, f)
                return self._json_response(200, {"ok": True})
            except Exception as exc:
                return self._json_response(500, {"error": str(exc)})

        return self._json_response(404, {"error": "Unknown endpoint"})

    def do_GET(self):
        path = self.path.split("?")[0]

        if path.startswith("/graph_data/") or path.startswith("/data/"):
            prefix = "/graph_data/" if path.startswith("/graph_data/") else "/data/"
            rel = path[len(prefix) :]
            local = self.data_dir / rel
            if not local.exists() and rel == "graph-metadata.json":
                local = self.frontend_dir / "data" / "graph-metadata.json"
            if not local.exists():
                if rel == "graph-metadata.json":
                    _sync_metadata(self.data_dir)
                    local = self.data_dir / "graph-metadata.json"
                if not local.exists():
                    self.send_error(404, f"Missing {local}")
                    return
            data = local.read_bytes()
            ctype = "application/json" if local.suffix == ".json" else "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            # Local graphs/bundles change after export-svd / verify — don't let
            # the browser keep a stale JSON without metadata.svd_bundle.
            # JSON + matrix dumps change after export-svd / export-spectral.
            if local.suffix in (".json", ".npy", ".npz"):
                self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return
        return super().do_GET()

    def do_HEAD(self):
        # SimpleHTTP's default HEAD 404s nested graph_data paths; mirror GET.
        path = self.path.split("?")[0]
        if path.startswith("/graph_data/") or path.startswith("/data/"):
            prefix = "/graph_data/" if path.startswith("/graph_data/") else "/data/"
            rel = path[len(prefix) :]
            local = self.data_dir / rel
            if not local.exists() and rel == "graph-metadata.json":
                local = self.frontend_dir / "data" / "graph-metadata.json"
            if not local.exists():
                self.send_error(404, f"Missing {local}")
                return
            self.send_response(200)
            ctype = "application/json" if local.suffix == ".json" else "application/octet-stream"
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(local.stat().st_size))
            self.send_header("Access-Control-Allow-Origin", "*")
            if local.suffix in (".json", ".npy", ".npz"):
                self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return
        return super().do_HEAD()

    def end_headers(self):
        # Local frontend JS/CSS change often during development — never let the
        # browser keep a stale init-cg-svd-panel.js that shows empty SVD panels.
        path = self.path.split("?")[0]
        if path.endswith((".js", ".css", ".html")) or path in ("/", "/index.html"):
            self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()


def _sync_metadata(data_dir: Path) -> None:
    """Ensure frontend/data/graph-metadata.json lists graphs (canonical order first)."""
    from interpretability_graph.demo_graph import CANONICAL_SLUGS

    by_slug: dict[str, dict] = {}
    for p in data_dir.glob("*.json"):
        if p.name == "graph-metadata.json":
            continue
        with p.open() as f:
            g = json.load(f)
        md = g.get("metadata", {})
        slug = md.get("slug", p.stem)
        by_slug[slug] = {
            "slug": slug,
            "prompt": md.get("prompt", p.stem),
            "scan": md.get("scan", "unknown"),
            "node_threshold": 1.0 if md.get("node_threshold") is not None else None,
            "title_prefix": md.get("title_prefix", ""),
        }

    graphs = []
    for slug in CANONICAL_SLUGS:
        if slug in by_slug:
            graphs.append(by_slug.pop(slug))
    graphs.extend(sorted(by_slug.values(), key=lambda g: g["slug"]))

    meta = {"graphs": graphs}
    meta_path = data_dir / "graph-metadata.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)
    dest = FRONTEND / "data"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(meta_path, dest / "graph-metadata.json")


def serve(
    data_dir: str | Path = DEFAULT_DATA,
    port: int = 8041,
    open_browser: bool = True,
) -> None:
    data_dir = Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    _sync_metadata(data_dir)

    handler = partial(GraphHandler, data_dir=data_dir, frontend_dir=FRONTEND)
    with ReusableTCPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://127.0.0.1:{port}/"
        logger.info("Serving attribution graphs at %s (data: %s)", url, data_dir)
        print(f"\n  Attribution graph UI → {url}")
        print(f"  Graph JSON directory → {data_dir}\n")
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
