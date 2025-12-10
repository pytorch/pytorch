from itertools import permutations

import pytest

import networkx as nx
from networkx.algorithms.community import (
    greedy_node_swap_bipartition,
    kernighan_lin_bisection,
)


def assert_partition_equal(x, y):
    assert set(map(frozenset, x)) == set(map(frozenset, y))


def test_partition():
    G = nx.barbell_graph(3, 0)
    split = kernighan_lin_bisection(G)
    assert_partition_equal(split, [{0, 1, 2}, {3, 4, 5}])


def test_partition_argument():
    G = nx.barbell_graph(3, 0)
    partition = [{0, 1, 2}, {3, 4, 5}]
    split = kernighan_lin_bisection(G, partition)
    assert_partition_equal(split, partition)


def test_partition_argument_non_integer_nodes():
    G = nx.Graph([("A", "B"), ("A", "C"), ("B", "C"), ("C", "D")])
    partition = ({"A", "B"}, {"C", "D"})
    split = kernighan_lin_bisection(G, partition)
    assert_partition_equal(split, partition)


def test_seed_argument():
    G = nx.barbell_graph(3, 0)
    split = kernighan_lin_bisection(G, seed=1)
    assert_partition_equal(split, [{0, 1, 2}, {3, 4, 5}])


def test_non_disjoint_partition():
    with pytest.raises(nx.NetworkXError):
        G = nx.barbell_graph(3, 0)
        partition = ({0, 1, 2}, {2, 3, 4, 5})
        kernighan_lin_bisection(G, partition)


def test_kernighan_lin_too_many_blocks():
    with pytest.raises(nx.NetworkXError):
        G = nx.barbell_graph(3, 0)
        partition = ({0, 1}, {2}, {3, 4, 5})
        kernighan_lin_bisection(G, partition)


def test_multigraph():
    G = nx.cycle_graph(4)
    M = nx.MultiGraph(G.edges())
    M.add_edges_from(G.edges())
    M.remove_edge(1, 2)
    for labels in permutations(range(4)):
        mapping = dict(zip(M, labels))
        A, B = kernighan_lin_bisection(nx.relabel_nodes(M, mapping), seed=0)
        assert_partition_equal(
            [A, B], [{mapping[0], mapping[1]}, {mapping[2], mapping[3]}]
        )


def test_max_iter_argument():
    G = nx.Graph(
        [
            ("A", "B", {"weight": 1}),
            ("A", "C", {"weight": 2}),
            ("A", "D", {"weight": 3}),
            ("A", "E", {"weight": 2}),
            ("A", "F", {"weight": 4}),
            ("B", "C", {"weight": 1}),
            ("B", "D", {"weight": 4}),
            ("B", "E", {"weight": 2}),
            ("B", "F", {"weight": 1}),
            ("C", "D", {"weight": 3}),
            ("C", "E", {"weight": 2}),
            ("C", "F", {"weight": 1}),
            ("D", "E", {"weight": 4}),
            ("D", "F", {"weight": 3}),
            ("E", "F", {"weight": 2}),
        ]
    )
    partition = ({"A", "B", "C"}, {"D", "E", "F"})
    split = kernighan_lin_bisection(G, partition, max_iter=1)
    assert_partition_equal(split, ({"A", "F", "C"}, {"D", "E", "B"}))


def test_weight_function():
    G = nx.cycle_graph(4)

    def my_weight(u, v, d):
        if u == 2 and v == 3:
            return None
        return u + v

    split = kernighan_lin_bisection(G, weight=my_weight)
    assert_partition_equal(split, ({1, 2}, {0, 3}))


## Test Spectral Modularity Bipartition


def test_spectral_bipartition():
    pytest.importorskip("scipy")
    G = nx.barbell_graph(3, 0)
    split = nx.community.spectral_modularity_bipartition(G)
    soln = ({3, 4, 5}, {0, 1, 2})
    assert set(map(frozenset, split)) == set(map(frozenset, soln))


def test_karate_club():
    pytest.importorskip("scipy")
    G = nx.karate_club_graph()
    MrHi = {v for v, club in G.nodes.data("club") if club == "Mr. Hi"}
    Officer = {v for v, club in G.nodes.data("club") if club == "Officer"}
    split = nx.community.spectral_modularity_bipartition(G)

    # spectral method misplaces member 8
    MrHi.remove(8)
    Officer.add(8)
    soln = (MrHi, Officer)
    assert set(map(frozenset, split)) == set(map(frozenset, soln))


## Test Node Swap Greedy Bipartition


def test_greedy_bipartition():
    G = nx.barbell_graph(3, 0)
    split = nx.community.greedy_node_swap_bipartition(G)
    soln = ({0, 1, 2}, {3, 4, 5})
    assert set(map(frozenset, split)) == set(map(frozenset, soln))


def test_greedy_non_disjoint_partition():
    G = nx.barbell_graph(3, 0)
    split = ({0, 1, 2}, {2, 3, 4, 5})
    with pytest.raises(nx.NetworkXError):
        nx.community.greedy_node_swap_bipartition(G, init_split=split)


def test_greedy_node_swap_too_many_blocks():
    G = nx.barbell_graph(3, 0)
    split = ({0, 1}, {2}, {3, 4, 5})
    with pytest.raises(nx.NetworkXError):
        nx.community.greedy_node_swap_bipartition(G, init_split=split)


def test_greedy_multigraph_disallowed():
    with pytest.raises(nx.NetworkXNotImplemented):
        nx.community.greedy_node_swap_bipartition(nx.MultiGraph())
