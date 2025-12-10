"""Unit tests for the broadcasting module."""

import math

import pytest

import networkx as nx


def test_example_tree_broadcast():
    """
    Test the BROADCAST algorithm on the example in the paper titled: "Information Dissemination in Trees"
    """
    edge_list = [
        (0, 1),
        (1, 2),
        (2, 7),
        (3, 4),
        (5, 4),
        (4, 7),
        (6, 7),
        (7, 9),
        (8, 9),
        (9, 13),
        (13, 14),
        (14, 15),
        (14, 16),
        (14, 17),
        (13, 11),
        (11, 10),
        (11, 12),
        (13, 18),
        (18, 19),
        (18, 20),
    ]
    G = nx.Graph(edge_list)
    b_T, b_C = nx.tree_broadcast_center(G)
    assert b_T == 6
    assert b_C == {13, 9}
    # test broadcast time from specific vertex
    assert nx.tree_broadcast_time(G, 17) == 8
    assert nx.tree_broadcast_time(G, 3) == 9
    # test broadcast time of entire tree
    assert nx.tree_broadcast_time(G) == 10


@pytest.mark.parametrize("n", range(2, 12))
def test_path_broadcast(n):
    G = nx.path_graph(n)
    b_T, b_C = nx.tree_broadcast_center(G)
    assert b_T == math.ceil(n / 2)
    assert b_C == {
        math.ceil(n / 2),
        n // 2,
        math.ceil(n / 2 - 1),
        n // 2 - 1,
    }
    assert nx.tree_broadcast_time(G) == n - 1


def test_empty_graph_broadcast():
    H = nx.empty_graph(1)
    b_T, b_C = nx.tree_broadcast_center(H)
    assert b_T == 0
    assert b_C == {0}
    assert nx.tree_broadcast_time(H) == 0


@pytest.mark.parametrize("n", range(4, 12))
def test_star_broadcast(n):
    G = nx.star_graph(n)
    b_T, b_C = nx.tree_broadcast_center(G)
    assert b_T == n
    assert b_C == set(G.nodes())
    assert nx.tree_broadcast_time(G) == b_T


@pytest.mark.parametrize("n", range(2, 8))
def test_binomial_tree_broadcast(n):
    G = nx.binomial_tree(n)
    b_T, b_C = nx.tree_broadcast_center(G)
    assert b_T == n
    assert b_C == {0, 2 ** (n - 1)}
    assert nx.tree_broadcast_time(G) == 2 * n - 1


@pytest.mark.parametrize("fn", [nx.tree_broadcast_center, nx.tree_broadcast_time])
@pytest.mark.parametrize("graph_type", [nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph])
def test_raises_graph_type(fn, graph_type):
    """Check that broadcast functions properly raise for directed and multigraph types."""
    G = nx.path_graph(5, create_using=graph_type)
    with pytest.raises(nx.NetworkXNotImplemented, match=r"not implemented for"):
        fn(G)


@pytest.mark.parametrize("fn", [nx.tree_broadcast_center, nx.tree_broadcast_time])
@pytest.mark.parametrize("gen", [nx.empty_graph, nx.cycle_graph])
def test_raises_not_tree(fn, gen):
    """Check that broadcast functions properly raise for nontree graphs."""
    G = gen(5)
    with pytest.raises(nx.NotATree, match=r"not a tree"):
        fn(G)


def test_raises_node_not_in_G():
    """Check that `tree_broadcast_time` properly raises for invalid nodes."""
    G = nx.path_graph(5)
    with pytest.raises(nx.NodeNotFound, match=r"node.*not in G"):
        nx.tree_broadcast_time(G, node=10)
