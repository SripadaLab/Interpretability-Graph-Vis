#!/usr/bin/env python3
"""Download featured Gemma-2-2B graphs (pretrained GemmaScope) and 2-color them."""

from interpretability_graph.fetch_graphs import fetch_pretrained_graphs

if __name__ == "__main__":
    fetch_pretrained_graphs()
