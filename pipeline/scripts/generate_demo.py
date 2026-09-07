#!/usr/bin/env python3
"""Convenience script: generate demo graph and print serve instructions."""

from interpretability_graph.demo_graph import write_demo_graph

if __name__ == "__main__":
    path = write_demo_graph("graph_files")
    print(f"Demo graph → {path}")
    print("Start UI:  python -m interpretability_graph serve")
