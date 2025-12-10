import pytest

import networkx as nx


def test_square_clustering_adjacent_squares():
    G = nx.Graph([(1, 2), (1, 3), (2, 4), (3, 4), (3, 5), (4, 6), (5, 6)])
    # Corner nodes: C_4 == 0.5, central face nodes: C_4 = 1 / 3
    expected = {1: 0.5, 2: 0.5, 3: 1 / 3, 4: 1 / 3, 5: 0.5, 6: 0.5}
    assert nx.square_clustering(G) == expected


def test_square_clustering_2d_grid():
    G = nx.grid_2d_graph(3, 3)
    # Central node: 4 squares out of 20 potential
    expected = {
        (0, 0): 1 / 3,
        (0, 1): 0.25,
        (0, 2): 1 / 3,
        (1, 0): 0.25,
        (1, 1): 0.2,
        (1, 2): 0.25,
        (2, 0): 1 / 3,
        (2, 1): 0.25,
        (2, 2): 1 / 3,
    }
    assert nx.square_clustering(G) == expected


def test_square_clustering_multiple_squares_non_complete():
    """An example where all nodes are part of all squares, but not every node
    is connected to every other."""
    G = nx.Graph([(0, 1), (0, 2), (1, 3), (2, 3), (1, 4), (2, 4), (1, 5), (2, 5)])
    expected = {n: 1 for n in G}
    assert nx.square_clustering(G) == expected


class TestTriangles:
    def test_empty(self):
        G = nx.Graph()
        assert list(nx.triangles(G).values()) == []

    def test_path(self):
        G = nx.path_graph(10)
        assert list(nx.triangles(G).values()) == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        assert nx.triangles(G) == {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0,
        }

    def test_cubical(self):
        G = nx.cubical_graph()
        assert list(nx.triangles(G).values()) == [0, 0, 0, 0, 0, 0, 0, 0]
        assert nx.triangles(G, 1) == 0
        assert list(nx.triangles(G, [1, 2]).values()) == [0, 0]
        assert nx.triangles(G, 1) == 0
        assert nx.triangles(G, [1, 2]) == {1: 0, 2: 0}

    def test_k5(self):
        G = nx.complete_graph(5)
        assert list(nx.triangles(G).values()) == [6, 6, 6, 6, 6]
        assert sum(nx.triangles(G).values()) / 3 == 10
        assert nx.triangles(G, 1) == 6
        G.remove_edge(1, 2)
        assert list(nx.triangles(G).values()) == [5, 3, 3, 5, 5]
        assert nx.triangles(G, 1) == 3
        G.add_edge(3, 3)  # ignore self-edges
        assert list(nx.triangles(G).values()) == [5, 3, 3, 5, 5]
        assert nx.triangles(G, 3) == 5


def test_all_triangles_non_integer_nodes():
    G = nx.Graph()
    G.add_edges_from(
        [
            ("a", "b"),
            ("b", "c"),
            ("c", "a"),  # triangle: a-b-c
        ]
    )
    expected = {frozenset({"a", "b", "c"})}
    assert {frozenset(t) for t in nx.all_triangles(G)} == expected


def test_all_triangles_overlapping():
    G = nx.Graph()
    G.add_edges_from(
        [
            (0, 1),
            (1, 2),
            (2, 0),  # triangle: 0-1-2
            (0, 2),
            (2, 3),
            (3, 0),  # triangle: 0-2-3
        ]
    )
    expected = {frozenset({0, 1, 2}), frozenset({0, 2, 3})}
    assert {frozenset(t) for t in nx.all_triangles(G)} == expected


def test_all_triangles_subset():
    G = nx.Graph()
    G.add_edges_from(
        [
            (0, 1),
            (1, 2),
            (2, 0),  # triangle: 0-1-2
            (2, 3),
            (3, 4),
            (4, 2),  # triangle: 2-3-4
        ]
    )
    assert {frozenset(t) for t in nx.all_triangles(G, nbunch=[0, 1])} == {
        frozenset({0, 1, 2})
    }


def test_all_triangles_subset_empty():
    G = nx.Graph()
    G.add_edges_from(
        [
            (0, 1),
            (1, 2),
            (2, 0),  # triangle: 0-1-2
            (2, 3),
            (3, 4),
            (4, 2),  # triangle: 2-3-4
            (5, 2),
        ]
    )
    assert list(nx.all_triangles(G, nbunch=[5])) == []


def test_all_triangles_no_triangles():
    G = nx.path_graph(4)
    assert list(nx.all_triangles(G)) == []


def test_all_triangles_complete_graph_exact():
    G = nx.complete_graph(4)

    expected = {
        frozenset({0, 1, 2}),
        frozenset({0, 1, 3}),
        frozenset({0, 2, 3}),
        frozenset({1, 2, 3}),
    }

    assert {frozenset(t) for t in nx.all_triangles(G)} == expected


def test_all_triangles_directed_graph():
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    with pytest.raises(nx.NetworkXNotImplemented):
        list(nx.all_triangles(G))


@pytest.mark.parametrize("graph_type", [nx.Graph, nx.MultiGraph])
def test_all_triangles_multiedges(graph_type):
    G = graph_type()
    G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 2)])
    assert {frozenset(t) for t in nx.all_triangles(G)} == {frozenset({0, 1, 2})}


class TestDirectedClustering:
    def test_clustering(self):
        G = nx.DiGraph()
        assert list(nx.clustering(G).values()) == []
        assert nx.clustering(G) == {}

    def test_path(self):
        G = nx.path_graph(10, create_using=nx.DiGraph())
        assert list(nx.clustering(G).values()) == [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        assert nx.clustering(G) == {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0,
        }
        assert nx.clustering(G, 0) == 0

    def test_k5(self):
        G = nx.complete_graph(5, create_using=nx.DiGraph())
        assert list(nx.clustering(G).values()) == [1, 1, 1, 1, 1]
        assert nx.average_clustering(G) == 1
        G.remove_edge(1, 2)
        assert list(nx.clustering(G).values()) == [
            11 / 12,
            1,
            1,
            11 / 12,
            11 / 12,
        ]
        assert nx.clustering(G, [1, 4]) == {1: 1, 4: 11 / 12}
        G.remove_edge(2, 1)
        assert list(nx.clustering(G).values()) == [
            5 / 6,
            1,
            1,
            5 / 6,
            5 / 6,
        ]
        assert nx.clustering(G, [1, 4]) == {1: 1, 4: 0.83333333333333337}
        assert nx.clustering(G, 4) == 5 / 6

    def test_triangle_and_edge(self):
        G = nx.cycle_graph(3, create_using=nx.DiGraph())
        G.add_edge(0, 4)
        assert nx.clustering(G)[0] == 1 / 6


class TestDirectedWeightedClustering:
    @classmethod
    def setup_class(cls):
        global np
        np = pytest.importorskip("numpy")

    def test_clustering(self):
        G = nx.DiGraph()
        assert list(nx.clustering(G, weight="weight").values()) == []
        assert nx.clustering(G) == {}

    def test_path(self):
        G = nx.path_graph(10, create_using=nx.DiGraph())
        assert list(nx.clustering(G, weight="weight").values()) == [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        assert nx.clustering(G, weight="weight") == {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0,
        }

    def test_k5(self):
        G = nx.complete_graph(5, create_using=nx.DiGraph())
        assert list(nx.clustering(G, weight="weight").values()) == [1, 1, 1, 1, 1]
        assert nx.average_clustering(G, weight="weight") == 1
        G.remove_edge(1, 2)
        assert list(nx.clustering(G, weight="weight").values()) == [
            11 / 12,
            1,
            1,
            11 / 12,
            11 / 12,
        ]
        assert nx.clustering(G, [1, 4], weight="weight") == {1: 1, 4: 11 / 12}
        G.remove_edge(2, 1)
        assert list(nx.clustering(G, weight="weight").values()) == [
            5 / 6,
            1,
            1,
            5 / 6,
            5 / 6,
        ]
        assert nx.clustering(G, [1, 4], weight="weight") == {
            1: 1,
            4: 0.83333333333333337,
        }

    def test_triangle_and_edge(self):
        G = nx.cycle_graph(3, create_using=nx.DiGraph())
        G.add_edge(0, 4, weight=2)
        assert nx.clustering(G)[0] == 1 / 6
        # Relaxed comparisons to allow graphblas-algorithms to pass tests
        np.testing.assert_allclose(nx.clustering(G, weight="weight")[0], 1 / 12)
        np.testing.assert_allclose(nx.clustering(G, 0, weight="weight"), 1 / 12)


class TestWeightedClustering:
    @classmethod
    def setup_class(cls):
        global np
        np = pytest.importorskip("numpy")

    def test_clustering(self):
        G = nx.Graph()
        assert list(nx.clustering(G, weight="weight").values()) == []
        assert nx.clustering(G) == {}

    def test_path(self):
        G = nx.path_graph(10)
        assert list(nx.clustering(G, weight="weight").values()) == [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        assert nx.clustering(G, weight="weight") == {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0,
        }

    def test_cubical(self):
        G = nx.cubical_graph()
        assert list(nx.clustering(G, weight="weight").values()) == [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        assert nx.clustering(G, 1) == 0
        assert list(nx.clustering(G, [1, 2], weight="weight").values()) == [0, 0]
        assert nx.clustering(G, 1, weight="weight") == 0
        assert nx.clustering(G, [1, 2], weight="weight") == {1: 0, 2: 0}

    def test_k5(self):
        G = nx.complete_graph(5)
        assert list(nx.clustering(G, weight="weight").values()) == [1, 1, 1, 1, 1]
        assert nx.average_clustering(G, weight="weight") == 1
        G.remove_edge(1, 2)
        assert list(nx.clustering(G, weight="weight").values()) == [
            5 / 6,
            1,
            1,
            5 / 6,
            5 / 6,
        ]
        assert nx.clustering(G, [1, 4], weight="weight") == {
            1: 1,
            4: 0.83333333333333337,
        }

    def test_triangle_and_edge(self):
        G = nx.cycle_graph(3)
        G.add_edge(0, 4, weight=2)
        assert nx.clustering(G)[0] == 1 / 3
        np.testing.assert_allclose(nx.clustering(G, weight="weight")[0], 1 / 6)
        np.testing.assert_allclose(nx.clustering(G, 0, weight="weight"), 1 / 6)

    def test_triangle_and_signed_edge(self):
        G = nx.cycle_graph(3)
        G.add_edge(0, 1, weight=-1)
        G.add_edge(3, 0, weight=0)
        assert nx.clustering(G)[0] == 1 / 3
        assert nx.clustering(G, weight="weight")[0] == -1 / 3


class TestClustering:
    @classmethod
    def setup_class(cls):
        pytest.importorskip("numpy")

    def test_clustering(self):
        G = nx.Graph()
        assert list(nx.clustering(G).values()) == []
        assert nx.clustering(G) == {}

    def test_path(self):
        G = nx.path_graph(10)
        assert list(nx.clustering(G).values()) == [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        assert nx.clustering(G) == {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0,
        }

    def test_cubical(self):
        G = nx.cubical_graph()
        assert list(nx.clustering(G).values()) == [0, 0, 0, 0, 0, 0, 0, 0]
        assert nx.clustering(G, 1) == 0
        assert list(nx.clustering(G, [1, 2]).values()) == [0, 0]
        assert nx.clustering(G, 1) == 0
        assert nx.clustering(G, [1, 2]) == {1: 0, 2: 0}

    def test_k5(self):
        G = nx.complete_graph(5)
        assert list(nx.clustering(G).values()) == [1, 1, 1, 1, 1]
        assert nx.average_clustering(G) == 1
        G.remove_edge(1, 2)
        assert list(nx.clustering(G).values()) == [
            5 / 6,
            1,
            1,
            5 / 6,
            5 / 6,
        ]
        assert nx.clustering(G, [1, 4]) == {1: 1, 4: 0.83333333333333337}

    def test_k5_signed(self):
        G = nx.complete_graph(5)
        assert list(nx.clustering(G).values()) == [1, 1, 1, 1, 1]
        assert nx.average_clustering(G) == 1
        G.remove_edge(1, 2)
        G.add_edge(0, 1, weight=-1)
        assert list(nx.clustering(G, weight="weight").values()) == [
            1 / 6,
            -1 / 3,
            1,
            3 / 6,
            3 / 6,
        ]


class TestTransitivity:
    def test_transitivity(self):
        G = nx.Graph()
        assert nx.transitivity(G) == 0

    def test_path(self):
        G = nx.path_graph(10)
        assert nx.transitivity(G) == 0

    def test_cubical(self):
        G = nx.cubical_graph()
        assert nx.transitivity(G) == 0

    def test_k5(self):
        G = nx.complete_graph(5)
        assert nx.transitivity(G) == 1
        G.remove_edge(1, 2)
        assert nx.transitivity(G) == 0.875


class TestSquareClustering:
    def test_clustering(self):
        G = nx.Graph()
        assert list(nx.square_clustering(G).values()) == []
        assert nx.square_clustering(G) == {}

    def test_path(self):
        G = nx.path_graph(10)
        assert list(nx.square_clustering(G).values()) == [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        assert nx.square_clustering(G) == {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0,
        }

    def test_cubical(self):
        G = nx.cubical_graph()
        assert list(nx.square_clustering(G).values()) == [
            1 / 3,
            1 / 3,
            1 / 3,
            1 / 3,
            1 / 3,
            1 / 3,
            1 / 3,
            1 / 3,
        ]
        assert list(nx.square_clustering(G, [1, 2]).values()) == [1 / 3, 1 / 3]
        assert nx.square_clustering(G, [1])[1] == 1 / 3
        assert nx.square_clustering(G, 1) == 1 / 3
        assert nx.square_clustering(G, [1, 2]) == {1: 1 / 3, 2: 1 / 3}

    def test_k5(self):
        G = nx.complete_graph(5)
        assert list(nx.square_clustering(G).values()) == [1, 1, 1, 1, 1]

    def test_bipartite_k5(self):
        G = nx.complete_bipartite_graph(5, 5)
        assert list(nx.square_clustering(G).values()) == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    def test_lind_square_clustering(self):
        """Test C4 for figure 1 Lind et al (2005)"""
        G = nx.Graph(
            [
                (1, 2),
                (1, 3),
                (1, 6),
                (1, 7),
                (2, 4),
                (2, 5),
                (3, 4),
                (3, 5),
                (6, 7),
                (7, 8),
                (6, 8),
                (7, 9),
                (7, 10),
                (6, 11),
                (6, 12),
                (2, 13),
                (2, 14),
                (3, 15),
                (3, 16),
            ]
        )
        G1 = G.subgraph([1, 2, 3, 4, 5, 13, 14, 15, 16])
        G2 = G.subgraph([1, 6, 7, 8, 9, 10, 11, 12])
        assert nx.square_clustering(G, [1])[1] == 3 / 43
        assert nx.square_clustering(G1, [1])[1] == 2 / 6
        assert nx.square_clustering(G2, [1])[1] == 1 / 5

    def test_peng_square_clustering(self):
        """Test eq2 for figure 1 Peng et al (2008)"""
        # Example graph from figure 1b
        G = nx.Graph([(1, 2), (1, 3), (2, 4), (3, 4), (3, 5), (3, 6)])
        # From table 1, row 2
        expected = {1: 1 / 3, 2: 1, 3: 0.2, 4: 1 / 3, 5: 0, 6: 0}
        assert nx.square_clustering(G) == expected

    def test_self_loops_square_clustering(self):
        G = nx.path_graph(5)
        assert nx.square_clustering(G) == {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        G.add_edges_from([(0, 0), (1, 1), (2, 2)])
        assert nx.square_clustering(G) == {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}


class TestAverageClustering:
    @classmethod
    def setup_class(cls):
        pytest.importorskip("numpy")

    def test_empty(self):
        G = nx.Graph()
        with pytest.raises(ZeroDivisionError):
            nx.average_clustering(G)

    def test_average_clustering(self):
        G = nx.cycle_graph(3)
        G.add_edge(2, 3)
        assert nx.average_clustering(G) == (1 + 1 + 1 / 3) / 4
        assert nx.average_clustering(G, count_zeros=True) == (1 + 1 + 1 / 3) / 4
        assert nx.average_clustering(G, count_zeros=False) == (1 + 1 + 1 / 3) / 3
        assert nx.average_clustering(G, [1, 2, 3]) == (1 + 1 / 3) / 3
        assert nx.average_clustering(G, [1, 2, 3], count_zeros=True) == (1 + 1 / 3) / 3
        assert nx.average_clustering(G, [1, 2, 3], count_zeros=False) == (1 + 1 / 3) / 2

    def test_average_clustering_signed(self):
        G = nx.cycle_graph(3)
        G.add_edge(2, 3)
        G.add_edge(0, 1, weight=-1)
        assert nx.average_clustering(G, weight="weight") == (-1 - 1 - 1 / 3) / 4
        assert (
            nx.average_clustering(G, weight="weight", count_zeros=True)
            == (-1 - 1 - 1 / 3) / 4
        )
        assert (
            nx.average_clustering(G, weight="weight", count_zeros=False)
            == (-1 - 1 - 1 / 3) / 3
        )


class TestDirectedAverageClustering:
    @classmethod
    def setup_class(cls):
        pytest.importorskip("numpy")

    def test_empty(self):
        G = nx.DiGraph()
        with pytest.raises(ZeroDivisionError):
            nx.average_clustering(G)

    def test_average_clustering(self):
        G = nx.cycle_graph(3, create_using=nx.DiGraph())
        G.add_edge(2, 3)
        assert nx.average_clustering(G) == (1 + 1 + 1 / 3) / 8
        assert nx.average_clustering(G, count_zeros=True) == (1 + 1 + 1 / 3) / 8
        assert nx.average_clustering(G, count_zeros=False) == (1 + 1 + 1 / 3) / 6
        assert nx.average_clustering(G, [1, 2, 3]) == (1 + 1 / 3) / 6
        assert nx.average_clustering(G, [1, 2, 3], count_zeros=True) == (1 + 1 / 3) / 6
        assert nx.average_clustering(G, [1, 2, 3], count_zeros=False) == (1 + 1 / 3) / 4


class TestGeneralizedDegree:
    def test_generalized_degree(self):
        G = nx.Graph()
        assert nx.generalized_degree(G) == {}

    def test_path(self):
        G = nx.path_graph(5)
        assert nx.generalized_degree(G, 0) == {0: 1}
        assert nx.generalized_degree(G, 1) == {0: 2}

    def test_cubical(self):
        G = nx.cubical_graph()
        assert nx.generalized_degree(G, 0) == {0: 3}

    def test_k5(self):
        G = nx.complete_graph(5)
        assert nx.generalized_degree(G, 0) == {3: 4}
        G.remove_edge(0, 1)
        assert nx.generalized_degree(G, 0) == {2: 3}
        assert nx.generalized_degree(G, [1, 2]) == {1: {2: 3}, 2: {2: 2, 3: 2}}
        assert nx.generalized_degree(G) == {
            0: {2: 3},
            1: {2: 3},
            2: {2: 2, 3: 2},
            3: {2: 2, 3: 2},
            4: {2: 2, 3: 2},
        }
