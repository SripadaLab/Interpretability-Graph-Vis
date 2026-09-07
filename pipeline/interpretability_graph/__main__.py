"""CLI: interpretability-graph demo | attribute | serve | colorize | pages"""

from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        prog="interpretability-graph",
        description=(
            "Gemma-2B attribution graphs (Anthropic circuit tracing) with "
            "2-color edge encoding + intensity gradients."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_demo = sub.add_parser(
        "demo",
        help="Reset canonical demos (DAG acronym, 3+5, capital-of-Texas) with clean 2-coloring",
    )
    p_demo.add_argument("--out", default="graph_files", help="Output directory")

    p_serve = sub.add_parser("serve", help="Serve the local visualization UI")
    p_serve.add_argument("--data", default="graph_files", help="Graph JSON directory")
    p_serve.add_argument("--port", type=int, default=8041)
    p_serve.add_argument("--no-browser", action="store_true")

    p_color = sub.add_parser("colorize", help="Annotate an existing graph JSON with 2-colors")
    p_color.add_argument("path", help="Path to graph JSON")
    p_color.add_argument(
        "--mode",
        choices=["greedy4", "greedy2", "layer_parity", "sign"],
        default="greedy4",
    )

    p_fetch = sub.add_parser(
        "fetch",
        help="Download pretrained GemmaScope graphs from Neuronpedia and 2-color them",
    )
    p_fetch.add_argument(
        "--slug",
        action="append",
        dest="slugs",
        help="Graph slug (repeatable). Defaults to featured demos.",
    )

    p_verify = sub.add_parser(
        "verify",
        help="Check manual supernodes (cosine) + compute auto k-means clusters",
    )
    p_verify.add_argument("--graph-dir", default="graph_files")
    p_verify.add_argument(
        "--no-write", action="store_true", help="don't write results into graph JSON"
    )
    p_verify.add_argument(
        "--max-nodes", type=int, default=40,
        help="auto-clustering: number of top-influence features to cluster",
    )
    p_verify.add_argument(
        "--k", type=int, default=None,
        help="auto-clustering: default number of groups the slider starts at",
    )

    p_cos = sub.add_parser(
        "export-cosine",
        help="Export node×node cosine similarity matrices (CSV + labels JSON + .npy)",
    )
    p_cos.add_argument("--graph-dir", default="graph_files")
    p_cos.add_argument(
        "--out",
        default=None,
        help="Output directory (default: <graph-dir>/cosine_exports)",
    )
    p_cos.add_argument(
        "--slug",
        action="append",
        dest="slugs",
        help="Graph slug (repeatable). Defaults to all graphs.",
    )
    p_cos.add_argument(
        "--no-recompute",
        action="store_true",
        help="Reuse existing svd_geom instead of rebuilding from transcoder weights",
    )
    p_cos.add_argument(
        "--max-nodes",
        type=int,
        default=0,
        help="Top-influence features to include (0 = all transcoder nodes in the graph)",
    )

    p_spec = sub.add_parser(
        "export-spectral",
        help=(
            "Spectral clustering on cosine: max(C,0) friendships, kNN sparsify, "
            "normalized Laplacian, k-means on raw evecs (no row-normalize) → metadata.spectral_clusters"
        ),
    )
    p_spec.add_argument("--graph-dir", default="graph_files")
    p_spec.add_argument(
        "--slug",
        action="append",
        dest="slugs",
        help="Graph slug (repeatable). Defaults to all graphs.",
    )
    p_spec.add_argument(
        "--max-nodes",
        type=int,
        default=0,
        help="Top-influence features (0 = all transcoder nodes / use svd cosine .npy)",
    )
    p_spec.add_argument(
        "--k-max",
        type=int,
        default=20,
        help="Largest number of groups to precompute (default 20)",
    )
    p_spec.add_argument(
        "--knn",
        type=int,
        default=10,
        help="Mutual kNN: keep this many strongest positive friendships per node",
    )
    p_spec.add_argument("--seed", type=int, default=0)

    p_svd = sub.add_parser(
        "export-svd",
        help="Precompute full-graph SVD (all transcoder nodes) for the UI to load",
    )
    p_svd.add_argument("--graph-dir", default="graph_files")
    p_svd.add_argument(
        "--out",
        default=None,
        help="Output directory (default: <graph-dir>/svd_exports)",
    )
    p_svd.add_argument(
        "--slug",
        action="append",
        dest="slugs",
        help="Graph slug (repeatable). Defaults to all graphs.",
    )
    p_svd.add_argument(
        "--k",
        type=int,
        default=32,
        help="Number of singular values / left vectors to store (default 32)",
    )

    p_interp = sub.add_parser(
        "assign-interpretations",
        help=(
            "Semantic interpretation assignment: pull a Neuronpedia explanation for "
            "every feature node (node_id <layer>_<idx>_<ctx>) into node.clerp"
        ),
    )
    p_interp.add_argument("--graph-dir", default="graph_files")
    p_interp.add_argument(
        "--slug",
        action="append",
        dest="slugs",
        help="Graph slug (repeatable). Defaults to all graphs.",
    )
    p_interp.add_argument(
        "--model-id",
        default="gemma-2-2b",
        help="Neuronpedia model id (default gemma-2-2b)",
    )
    p_interp.add_argument(
        "--workers", type=int, default=8, help="Concurrent API requests (default 8)"
    )
    p_interp.add_argument(
        "--refresh",
        action="store_true",
        help="Refetch features already present in the explanation cache",
    )

    p_themes = sub.add_parser(
        "export-theme-requests",
        help=(
            "Bundle each distinct spectral group's feature interpretations into "
            "JSONL for an LLM to summarize into themes"
        ),
    )
    p_themes.add_argument("--graph-dir", default="graph_files")
    p_themes.add_argument("--out", default="graph_files/themes/requests.jsonl")
    p_themes.add_argument(
        "--slug",
        action="append",
        dest="slugs",
        help="Graph slug (repeatable). Defaults to all graphs.",
    )

    p_apply = sub.add_parser(
        "apply-themes",
        help=(
            "Fold LLM theme summaries ({group_id: {theme, summary}}) into "
            "metadata.spectral_clusters.themes"
        ),
    )
    p_apply.add_argument("themes", help="JSON file of group_id → {theme, summary}")
    p_apply.add_argument("--graph-dir", default="graph_files")

    p_pages = sub.add_parser(
        "pages",
        help="Build a static GitHub Pages folder (UI + precomputed graph assets)",
    )
    p_pages.add_argument("--data", default="graph_files", help="Graph JSON directory")
    p_pages.add_argument(
        "--out",
        default="pages",
        help="Output folder (default: pages/). Use docs/ to serve from this repo.",
    )
    p_pages.add_argument(
        "--slug",
        action="append",
        dest="slugs",
        help="With --compute, only these slugs. Default: all graphs.",
    )
    p_pages.add_argument(
        "--compute",
        action="store_true",
        help="Run export-svd / export-spectral for graphs that lack them (needs numpy)",
    )

    p_attr = sub.add_parser(
        "attribute",
        help="Run real Gemma-2B attribution (needs GPU/MPS + HF access to gemma-2-2b)",
    )
    p_attr.add_argument("-p", "--prompt", required=True)
    p_attr.add_argument(
        "-t",
        "--preset",
        default="gemma",
        help="gemma | gemma-clt-426k | gemma-clt-2.5m | HF repo id",
    )
    p_attr.add_argument("--slug", default="gemma-run")
    p_attr.add_argument("--graph_file_dir", default="graph_files")
    p_attr.add_argument("-o", "--graph_output_path", default=None)
    p_attr.add_argument("--dtype", default="bfloat16")
    p_attr.add_argument("--batch_size", type=int, default=128)
    p_attr.add_argument("--offload", choices=["cpu", "disk"], default=None)
    p_attr.add_argument(
        "--mode",
        choices=["greedy4", "greedy2", "layer_parity", "sign"],
        default="greedy4",
    )
    p_attr.add_argument("--serve", action="store_true", help="Start UI after attribution")
    p_attr.add_argument("--port", type=int, default=8041)

    args = parser.parse_args(argv)

    if args.cmd == "demo":
        from interpretability_graph.demo_graph import reset_canonical_graphs

        paths = reset_canonical_graphs(args.out)
        print(f"Reset {len(paths)} canonical graph(s):")
        for p in paths:
            print(f"  → {p}")
        print("Run: interpretability-graph serve")
        return 0

    if args.cmd == "serve":
        from interpretability_graph.demo_graph import reset_canonical_graphs
        from interpretability_graph.serve import serve
        from pathlib import Path

        data = Path(args.data)
        if not any(p for p in data.glob("*.json") if p.name != "graph-metadata.json"):
            print("No graphs found; resetting canonical demos…")
            reset_canonical_graphs(data)
        serve(data_dir=data, port=args.port, open_browser=not args.no_browser)
        return 0

    if args.cmd == "colorize":
        from interpretability_graph.attribute import annotate_graph_json

        annotate_graph_json(args.path, mode=args.mode)
        return 0

    if args.cmd == "fetch":
        from interpretability_graph.fetch_graphs import fetch_pretrained_graphs

        paths = fetch_pretrained_graphs(args.slugs)
        print(f"Wrote {len(paths)} graph(s) — run: interpretability-graph serve")
        return 0

    if args.cmd == "verify":
        from interpretability_graph.verify_supernodes import verify_all, _format_report

        report = verify_all(
            args.graph_dir, write=not args.no_write,
            max_nodes=args.max_nodes, k=args.k,
        )
        print(_format_report(report))
        return 0

    if args.cmd == "export-cosine":
        from interpretability_graph.verify_supernodes import export_cosine_matrices

        paths = export_cosine_matrices(
            args.graph_dir,
            out_dir=args.out,
            slugs=args.slugs,
            recompute=not args.no_recompute,
            max_nodes=args.max_nodes if args.max_nodes > 0 else None,
        )
        print(f"Wrote {len(paths)} file(s):")
        for p in paths:
            print(f"  → {p}")
        return 0

    if args.cmd == "export-svd":
        from interpretability_graph.svd_export import export_svd_bundles

        paths = export_svd_bundles(
            args.graph_dir,
            out_dir=args.out,
            slugs=args.slugs,
            k=args.k,
        )
        print(f"Wrote {len(paths)} SVD bundle(s):")
        for p in paths:
            print(f"  → {p}")
        return 0

    if args.cmd == "export-spectral":
        from interpretability_graph.spectral_clusters import export_spectral_clusters

        paths = export_spectral_clusters(
            args.graph_dir,
            slugs=args.slugs,
            max_nodes=args.max_nodes if args.max_nodes > 0 else None,
            k_max=args.k_max,
            knn=args.knn,
            seed=args.seed,
        )
        print(f"Wrote spectral_clusters into {len(paths)} graph(s):")
        for p in paths:
            print(f"  → {p}")
        return 0

    if args.cmd == "assign-interpretations":
        from interpretability_graph.neuronpedia import assign_interpretations

        paths = assign_interpretations(
            args.graph_dir,
            slugs=args.slugs,
            model_id=args.model_id,
            workers=args.workers,
            refresh=args.refresh,
        )
        print(f"Wrote interpretations into {len(paths)} graph(s):")
        for p in paths:
            print(f"  → {p}")
        return 0

    if args.cmd == "export-theme-requests":
        from interpretability_graph.group_themes import write_requests

        path = write_requests(args.out, args.graph_dir, slugs=args.slugs)
        print(f"  → {path}")
        return 0

    if args.cmd == "apply-themes":
        from interpretability_graph.group_themes import apply_themes

        paths = apply_themes(args.themes, args.graph_dir)
        print(f"Applied themes to {len(paths)} graph(s):")
        for p in paths:
            print(f"  → {p}")
        return 0

    if args.cmd == "pages":
        from interpretability_graph.pages_export import export_pages

        out = export_pages(
            args.data,
            args.out,
            compute=args.compute,
            slugs=args.slugs,
        )
        print(f"Static site → {out}")
        print("Preview:  python -m http.server 8041 --directory", out)
        print("GitHub Pages: push this folder as the repo root, or copy to docs/")
        return 0

    if args.cmd == "attribute":
        from interpretability_graph.attribute import attribute_prompt
        from interpretability_graph.serve import serve

        path = attribute_prompt(
            args.prompt,
            preset=args.preset,
            slug=args.slug,
            graph_file_dir=args.graph_file_dir,
            graph_output_path=args.graph_output_path,
            dtype=args.dtype,
            batch_size=args.batch_size,
            offload=args.offload,
            color_mode=args.mode,
        )
        print(f"Graph ready → {path}")
        if args.serve:
            serve(data_dir=args.graph_file_dir, port=args.port)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
