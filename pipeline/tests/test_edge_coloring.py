"""Unit tests for edge 2-coloring (no GPU / circuit-tracer required)."""

from interpretability_graph.edge_coloring import (
    annotate_graph_links,
    assign_edge_2colors,
    conflict_rate,
    color_rgba,
)
from interpretability_graph.demo_graph import build_demo_graph


def test_color_rgba_intensity_monotonic_alpha():
    weak = color_rgba(0, 0.0)
    strong = color_rgba(0, 1.0)
    assert weak[3] < strong[3]
    assert strong[:3] != weak[:3]


def test_greedy2_prefers_different_colors_at_star():
    # Star: one hub with three strong edges → at most one same-color conflict pair ideally,
    # but with 2 colors perfect coloring of K_{1,3} line graph needs 3 colors; greedy still
    # balances the two classes.
    edges = [
        {"id": "a", "source": "hub", "target": "t1", "weight": 1.0},
        {"id": "b", "source": "hub", "target": "t2", "weight": 0.9},
        {"id": "c", "source": "hub", "target": "t3", "weight": 0.8},
    ]
    out = assign_edge_2colors(edges, mode="greedy2")
    classes = [a.color_class for a in out]
    assert set(classes) == {0, 1}
    assert classes.count(0) in (1, 2)
    assert classes.count(1) in (1, 2)


def test_layer_parity():
    edges = [
        {"id": "a", "source": "s", "target": "t", "weight": 1.0, "source_layer": 2},
        {"id": "b", "source": "s", "target": "u", "weight": 1.0, "source_layer": 3},
    ]
    out = assign_edge_2colors(edges, mode="layer_parity")
    assert out[0].color_class == 0
    assert out[1].color_class == 1


def test_demo_graph_annotated():
    g = build_demo_graph()
    assert len(g["links"]) > 20
    assert "edge_coloring" in g["metadata"]
    assert all("color_class" in e and "rgba" in e for e in g["links"])
    rate = g["metadata"]["edge_coloring"]["conflict_rate"]
    assert 0 <= rate <= 1


def test_annotate_preserves_weights():
    links = [{"source": "a", "target": "b", "weight": 0.5}]
    nodes = [{"node_id": "a", "layer": "1"}, {"node_id": "b", "layer": "2"}]
    out = annotate_graph_links(links, nodes, mode="greedy2")
    assert out[0]["weight"] == 0.5
    assert out[0]["color"].startswith("rgba(")
