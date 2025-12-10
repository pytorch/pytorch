import pytest

import networkx as nx


def test_dominating_set():
    G = nx.gnp_random_graph(100, 0.1)
    D = nx.dominating_set(G)
    assert nx.is_dominating_set(G, D)
    D = nx.dominating_set(G, start_with=0)
    assert nx.is_dominating_set(G, D)


def test_complete():
    """In complete graphs each node is a dominating set.
    Thus the dominating set has to be of cardinality 1.
    """
    K4 = nx.complete_graph(4)
    assert len(nx.dominating_set(K4)) == 1
    K5 = nx.complete_graph(5)
    assert len(nx.dominating_set(K5)) == 1


def test_raise_dominating_set():
    with pytest.raises(nx.NetworkXError):
        G = nx.path_graph(4)
        D = nx.dominating_set(G, start_with=10)


def test_is_dominating_set():
    G = nx.path_graph(4)
    d = {1, 3}
    assert nx.is_dominating_set(G, d)
    d = {0, 2}
    assert nx.is_dominating_set(G, d)
    d = {1}
    assert not nx.is_dominating_set(G, d)


def test_wikipedia_is_dominating_set():
    """Example from https://en.wikipedia.org/wiki/Dominating_set"""
    G = nx.cycle_graph(4)
    G.add_edges_from([(0, 4), (1, 4), (2, 5)])
    assert nx.is_dominating_set(G, {4, 3, 5})
    assert nx.is_dominating_set(G, {0, 2})
    assert nx.is_dominating_set(G, {1, 2})


def test_is_connected_dominating_set():
    G = nx.path_graph(4)
    D = {1, 2}
    assert nx.is_connected_dominating_set(G, D)
    D = {1, 3}
    assert not nx.is_connected_dominating_set(G, D)
    D = {2, 3}
    assert nx.is_connected(nx.subgraph(G, D))
    assert not nx.is_connected_dominating_set(G, D)


def test_null_graph_connected_dominating_set():
    G = nx.Graph()
    assert 0 == len(nx.connected_dominating_set(G))


def test_single_node_graph_connected_dominating_set():
    G = nx.Graph()
    G.add_node(1)
    CD = nx.connected_dominating_set(G)
    assert nx.is_connected_dominating_set(G, CD)


def test_raise_disconnected_graph_connected_dominating_set():
    with pytest.raises(nx.NetworkXError):
        G = nx.Graph()
        G.add_node(1)
        G.add_node(2)
        nx.connected_dominating_set(G)


def test_complete_graph_connected_dominating_set():
    K5 = nx.complete_graph(5)
    assert 1 == len(nx.connected_dominating_set(K5))
    K7 = nx.complete_graph(7)
    assert 1 == len(nx.connected_dominating_set(K7))


def test_docstring_example_connected_dominating_set():
    G = nx.Graph(
        [
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (2, 7),
            (3, 8),
            (4, 9),
            (5, 10),
            (6, 11),
            (7, 12),
            (8, 12),
            (9, 12),
            (10, 12),
            (11, 12),
        ]
    )
    assert {1, 2, 3, 4, 5, 6, 7} == nx.connected_dominating_set(G)


@pytest.mark.parametrize("seed", [1, 13, 29])
@pytest.mark.parametrize("n,k,p", [(10, 3, 0.2), (100, 10, 0.7), (1000, 50, 0.5)])
def test_connected_watts_strogatz_graph_connected_dominating_set(n, k, p, seed):
    G = nx.connected_watts_strogatz_graph(n, k, p, seed=seed)
    D = nx.connected_dominating_set(G)
    assert nx.is_connected_dominating_set(G, D)
