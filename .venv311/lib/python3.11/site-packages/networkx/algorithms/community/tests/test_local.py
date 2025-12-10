import pytest

import networkx as nx


def test_greedy_source_expansion_karate_club():
    G = nx.karate_club_graph()

    community = nx.community.greedy_source_expansion(G, source=16)

    expected = {0, 4, 5, 6, 10, 16}

    assert community == expected


def test_greedy_source_expansion_cutoff():
    G = nx.karate_club_graph()

    community = nx.community.greedy_source_expansion(G, source=16, cutoff=3)

    assert community == {5, 6, 16}


def test_greedy_source_expansion_invalid_method():
    G = nx.karate_club_graph()

    with pytest.raises(ValueError):
        nx.community.greedy_source_expansion(G, source=16, cutoff=3, method="invalid")


def test_greedy_source_expansion_connected_component():
    G_edges = [(0, 2), (0, 1), (1, 0), (2, 1), (2, 0), (3, 4), (4, 3)]
    G = nx.Graph(G_edges)
    expected = {0, 1, 2}
    community = nx.community.greedy_source_expansion(G, source=0)
    assert community == expected


def test_greedy_source_expansion_directed_graph():
    G_edges = [
        (0, 2),
        (0, 1),
        (1, 0),
        (2, 1),
        (2, 0),
        (3, 4),
        (4, 3),
        (4, 5),
        (5, 3),
        (5, 6),
        (0, 6),
    ]
    G = nx.DiGraph(G_edges)

    expected = {0, 1, 2, 6}
    community = nx.community.greedy_source_expansion(G, source=0)
    assert community == expected


def test_greedy_source_expansion_multigraph():
    G = nx.MultiGraph(nx.karate_club_graph())
    G.add_edge(0, 1)
    G.add_edge(0, 9)

    expected = {0, 4, 5, 6, 10, 16}

    community = nx.community.greedy_source_expansion(G, source=16)

    assert community == expected


def test_greedy_source_expansion_empty_graph():
    G = nx.Graph()
    G.add_nodes_from(range(5))
    expected = {0}
    assert nx.community.greedy_source_expansion(G, source=0) == expected
