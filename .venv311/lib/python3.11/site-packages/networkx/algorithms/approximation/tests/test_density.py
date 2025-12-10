import pytest

import networkx as nx
import networkx.algorithms.approximation as approx


def close_cliques_example(d=12, D=300, h=24, k=2):
    """
    Hard example from Harb, Elfarouk, Kent Quanrud, and Chandra Chekuri.
    "Faster and scalable algorithms for densest subgraph and decomposition."
    Advances in Neural Information Processing Systems 35 (2022): 26966-26979.
    """
    Kh = nx.complete_graph(h)
    KdD = nx.complete_bipartite_graph(d, D)
    G = nx.disjoint_union_all([KdD] + [Kh for _ in range(k)])
    best_density = d * D / (d + D)  # of the complete bipartite graph
    return G, best_density, set(KdD.nodes)


@pytest.mark.parametrize("iterations", (1, 3))
@pytest.mark.parametrize("n", range(4, 7))
@pytest.mark.parametrize("method", ("greedy++", "fista"))
def test_star(n, iterations, method):
    if method == "fista":
        pytest.importorskip("numpy")

    G = nx.star_graph(n)
    # The densest subgraph of a star network is the entire graph.
    # The peeling algorithm would peel all the vertices with degree 1,
    # and so should discover the densest subgraph in one iteration!
    d, S = approx.densest_subgraph(G, iterations=iterations, method=method)

    assert d == pytest.approx(G.number_of_edges() / G.number_of_nodes())
    assert S == set(G)  # The entire graph!


@pytest.mark.parametrize("method", ("greedy++", "fista"))
def test_greedy_plus_plus_complete_graph(method):
    if method == "fista":
        pytest.importorskip("numpy")

    G = nx.complete_graph(4)
    # The density of a complete graph network is the entire graph: C(4, 2)/4
    # where C(n, 2) is n*(n-1)//2. The peeling algorithm would find
    # the densest subgraph in one iteration!
    d, S = approx.densest_subgraph(G, iterations=1, method=method)

    assert d == pytest.approx(6 / 4)  # The density, 4/5=0.8.
    assert S == {0, 1, 2, 3}  # The entire graph!


def test_greedy_plus_plus_close_cliques():
    G, best_density, densest_set = close_cliques_example()
    # NOTE: iterations=185 fails to ID the densest subgraph
    greedy_pp, S_pp = approx.densest_subgraph(G, iterations=186, method="greedy++")

    assert greedy_pp == pytest.approx(best_density)
    assert S_pp == densest_set


def test_fista_close_cliques():
    pytest.importorskip("numpy")
    G, best_density, best_set = close_cliques_example()
    # NOTE: iterations=12 fails to ID the densest subgraph
    density, dense_set = approx.densest_subgraph(G, iterations=13, method="fista")

    assert density == pytest.approx(best_density)
    assert dense_set == best_set


def bipartite_and_clique_example(d=5, D=200, k=2):
    """
    Hard example from: Boob, Digvijay, Yu Gao, Richard Peng, Saurabh Sawlani,
    Charalampos Tsourakakis, Di Wang, and Junxing Wang. "Flowless: Extracting
    densest subgraphs without flow computations." In Proceedings of The Web
    Conference 2020, pp. 573-583. 2020.
    """
    B = nx.complete_bipartite_graph(d, D)
    H = [nx.complete_graph(d + 2) for _ in range(k)]
    G = nx.disjoint_union_all([B] + H)

    best_density = d * D / (d + D)  # of the complete bipartite graph
    correct_one_round_density = (2 * d * D + (d + 1) * (d + 2) * k) / (
        2 * d + 2 * D + 2 * k * (d + 2)
    )
    best_subgraph = set(B.nodes)
    return G, best_density, best_subgraph, correct_one_round_density


def test_greedy_plus_plus_bipartite_and_clique():
    G, best_density, best_subgraph, correct_one_iter_density = (
        bipartite_and_clique_example()
    )
    one_round_density, S_one = approx.densest_subgraph(
        G, iterations=1, method="greedy++"
    )
    assert one_round_density == pytest.approx(correct_one_iter_density)
    assert S_one == set(G.nodes)

    ten_round_density, S_ten = approx.densest_subgraph(
        G, iterations=10, method="greedy++"
    )
    assert ten_round_density == pytest.approx(best_density)
    assert S_ten == best_subgraph


def test_fista_bipartite_and_clique():
    pytest.importorskip("numpy")
    G, best_density, best_subgraph, _ = bipartite_and_clique_example()

    ten_round_density, S_ten = approx.densest_subgraph(G, iterations=10, method="fista")
    assert ten_round_density == pytest.approx(best_density)
    assert S_ten == best_subgraph


def test_fista_big_dataset():
    pytest.importorskip("numpy")
    G, best_density, best_subgraph = close_cliques_example(d=30, D=2000, h=60, k=20)

    # Note: iterations=12 fails to identify densest subgraph
    density, dense_set = approx.densest_subgraph(G, iterations=13, method="fista")

    assert density == pytest.approx(best_density)
    assert dense_set == best_subgraph


@pytest.mark.parametrize("iterations", (1, 3))
def test_greedy_plus_plus_edgeless_cornercase(iterations):
    G = nx.Graph()
    assert approx.densest_subgraph(G, iterations=iterations, method="greedy++") == (
        0,
        set(),
    )
    G.add_nodes_from(range(4))
    assert approx.densest_subgraph(G, iterations=iterations, method="greedy++") == (
        0,
        set(),
    )


@pytest.mark.parametrize("labels", ((1, 2, 3), ("a", "b", "c")))
def test_gh_8271(labels):
    """Test for graphs with nonstandard node labels."""
    pytest.importorskip("numpy")
    G = nx.complete_graph(labels)
    assert approx.densest_subgraph(G, method="fista") == (1, set(labels))
