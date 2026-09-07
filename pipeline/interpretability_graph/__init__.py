"""Local Gemma-2B attribution graphs with 2-color edge visualization.

Implements the circuit-tracing pipeline from Ameisen et al. (2025) using
pretrained Gemma-2-2B cross-layer / per-layer transcoders via circuit-tracer,
plus a conflict-minimizing 2-color edge encoding with intensity gradients so
dense attribution graphs remain readable when all links are shown.
"""

__version__ = "0.1.0"

from interpretability_graph.edge_coloring import (
    EdgeColorAssignment,
    assign_edge_2colors,
    color_rgba,
)

__all__ = [
    "EdgeColorAssignment",
    "assign_edge_2colors",
    "color_rgba",
    "__version__",
]
