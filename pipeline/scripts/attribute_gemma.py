#!/usr/bin/env python3
"""Attribute a prompt on Gemma-2-2B (requires: pip install -e '.[gemma]')."""

import argparse

from interpretability_graph.attribute import attribute_prompt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", "-p", required=True)
    p.add_argument("--preset", "-t", default="gemma")
    p.add_argument("--slug", default="gemma-run")
    p.add_argument("--offload", choices=["cpu", "disk"], default=None)
    args = p.parse_args()
    path = attribute_prompt(
        args.prompt,
        preset=args.preset,
        slug=args.slug,
        offload=args.offload,
    )
    print(path)


if __name__ == "__main__":
    main()
