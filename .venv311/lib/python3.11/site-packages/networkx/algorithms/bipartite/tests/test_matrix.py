import itertools

import pytest

import networkx as nx
from networkx.algorithms import bipartite
from networkx.utils import edges_equal

np = pytest.importorskip("numpy")
sp = pytest.importorskip("scipy")


class TestBiadjacencyMatrix:
    def test_biadjacency_matrix_weight(self):
        G = nx.path_graph(5)
        G.add_edge(0, 1, weight=2, other=4)
        X = [1, 3]
        Y = [0, 2, 4]
        M = bipartite.biadjacency_matrix(G, X, weight="weight")
        assert M[0, 0] == 2
        M = bipartite.biadjacency_matrix(G, X, weight="other")
        assert M[0, 0] == 4

    def test_biadjacency_matrix(self):
        tops = [2, 5, 10]
        bots = [5, 10, 15]
        for i in range(len(tops)):
            G = bipartite.random_graph(tops[i], bots[i], 0.2)
            top = [n for n, d in G.nodes(data=True) if d["bipartite"] == 0]
            M = bipartite.biadjacency_matrix(G, top)
            assert M.shape[0] == tops[i]
            assert M.shape[1] == bots[i]

    def test_biadjacency_matrix_order(self):
        G = nx.path_graph(5)
        G.add_edge(0, 1, weight=2)
        X = [3, 1]
        Y = [4, 2, 0]
        M = bipartite.biadjacency_matrix(G, X, Y, weight="weight")
        assert M[1, 2] == 2

    def test_biadjacency_matrix_empty_graph(self):
        G = nx.empty_graph(2)
        M = nx.bipartite.biadjacency_matrix(G, [0])
        assert np.array_equal(M.toarray(), np.array([[0]]))

    def test_null_graph(self):
        with pytest.raises(nx.NetworkXError):
            bipartite.biadjacency_matrix(nx.Graph(), [])

    def test_empty_graph(self):
        with pytest.raises(nx.NetworkXError):
            bipartite.biadjacency_matrix(nx.Graph([(1, 0)]), [])

    def test_duplicate_row(self):
        with pytest.raises(nx.NetworkXError):
            bipartite.biadjacency_matrix(nx.Graph([(1, 0)]), [1, 1])

    def test_duplicate_col(self):
        with pytest.raises(nx.NetworkXError):
            bipartite.biadjacency_matrix(nx.Graph([(1, 0)]), [0], [1, 1])

    def test_format_keyword(self):
        with pytest.raises(nx.NetworkXError):
            bipartite.biadjacency_matrix(nx.Graph([(1, 0)]), [0], format="foo")

    def test_from_biadjacency_roundtrip(self):
        B1 = nx.path_graph(5)
        M = bipartite.biadjacency_matrix(B1, [0, 2, 4])
        B2 = bipartite.from_biadjacency_matrix(M)
        assert nx.is_isomorphic(B1, B2)

    def test_from_biadjacency_weight(self):
        M = sp.sparse.csc_array([[1, 2], [0, 3]])
        B = bipartite.from_biadjacency_matrix(M)
        assert edges_equal(B.edges(), [(0, 2), (0, 3), (1, 3)])
        B = bipartite.from_biadjacency_matrix(M, edge_attribute="weight")
        e = [(0, 2, {"weight": 1}), (0, 3, {"weight": 2}), (1, 3, {"weight": 3})]
        assert edges_equal(B.edges(data=True), e)

    def test_from_biadjacency_multigraph(self):
        M = sp.sparse.csc_array([[1, 2], [0, 3]])
        B = bipartite.from_biadjacency_matrix(M, create_using=nx.MultiGraph())
        assert edges_equal(B.edges(), [(0, 2), (0, 3), (0, 3), (1, 3), (1, 3), (1, 3)])

    @pytest.mark.parametrize(
        "row_order,column_order,create_using",
        itertools.product(
            (None, ("a", "b"), (25, (0, 5, 10))),
            (None, ("c", "d"), (26, (0, 5, 10))),
            (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph),
        ),
    )
    def test_from_biadjacency_nodelist(self, row_order, column_order, create_using):
        M = sp.sparse.csc_array([[1, 2], [0, 3]])
        B_default = bipartite.from_biadjacency_matrix(M, create_using=create_using())
        B = bipartite.from_biadjacency_matrix(
            M,
            create_using=create_using(),
            row_order=row_order,
            column_order=column_order,
        )

        row_order = row_order if row_order else list(range(M.shape[0]))
        column_order = (
            column_order
            if column_order
            else list(range(M.shape[0], M.shape[0] + M.shape[1]))
        )

        top_map = dict(enumerate(row_order))

        bottom_map = {idx + M.shape[0]: node for idx, node in enumerate(column_order)}

        def map_edges(edges):
            return [(top_map[u], bottom_map[v]) for u, v in edges]

        mapped_edges = map_edges(B_default.edges())
        assert edges_equal(mapped_edges, B.edges())

    def test_invalid_from_biadjacency_nodelist(self):
        M = sp.sparse.csc_array([[1, 2], [0, 3]])
        # For when top nodelist has the wrong length
        row_order_invalid = ["a", "b", "c"]
        # For when bottom nodelist has the wrong length
        column_order_invalid = ["c", "d", "e"]
        with pytest.raises(ValueError):
            bipartite.from_biadjacency_matrix(
                M,
                create_using=nx.MultiGraph(),
                row_order=row_order_invalid,
            )
        with pytest.raises(ValueError):
            bipartite.from_biadjacency_matrix(
                M,
                create_using=nx.MultiGraph(),
                column_order=column_order_invalid,
            )
