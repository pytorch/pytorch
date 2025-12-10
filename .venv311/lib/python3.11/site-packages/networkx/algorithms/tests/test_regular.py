import pytest

import networkx as nx
import networkx.algorithms.regular as reg
import networkx.generators as gen


class TestKFactor:
    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_k_factor_cycle(self, n):
        g = nx.cycle_graph(n)
        kf = nx.k_factor(g, 2)
        assert g.edges == kf.edges
        assert g.nodes == kf.nodes

    @pytest.mark.parametrize("k", range(3))
    def test_k_factor_grid(self, k):
        g = nx.grid_2d_graph(4, 4)
        kf = nx.k_factor(g, k)
        assert g.nodes == kf.nodes
        assert all(g.has_edge(*e) for e in kf.edges)
        assert nx.is_k_regular(kf, k)

    @pytest.mark.parametrize("k", range(6))
    def test_k_factor_complete(self, k):
        g = nx.complete_graph(6)
        kf = nx.k_factor(g, k)
        assert g.nodes == kf.nodes
        assert all(g.has_edge(*e) for e in kf.edges)
        assert nx.is_k_regular(kf, k)

    def test_k_factor_degree(self):
        g = nx.grid_2d_graph(4, 4)
        with pytest.raises(nx.NetworkXUnfeasible, match=r"degree less than"):
            nx.k_factor(g, 3)

    def test_k_factor_no_matching(self):
        g = nx.hexagonal_lattice_graph(4, 4)
        # Perfect matching doesn't exist for 4,4 hexagonal lattice graph
        with pytest.raises(nx.NetworkXUnfeasible, match=r"no perfect matching"):
            nx.k_factor(g, 2)

    @pytest.mark.parametrize("graph_type", [nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph])
    def test_k_factor_not_implemented(self, graph_type):
        with pytest.raises(nx.NetworkXNotImplemented, match=r"not implemented for"):
            nx.k_factor(graph_type(), 2)


class TestIsRegular:
    @pytest.mark.parametrize(
        "graph,expected",
        [
            (nx.cycle_graph(4), True),
            (nx.complete_graph(5), True),
            (nx.path_graph(5), False),
            (nx.lollipop_graph(5, 5), False),
            (nx.cycle_graph(3, create_using=nx.DiGraph), True),
            (nx.Graph([(0, 1)]), True),
            (nx.DiGraph([(0, 1)]), False),
            (nx.MultiGraph([(0, 1), (0, 1)]), True),
            (nx.MultiDiGraph([(0, 1), (0, 1)]), False),
        ],
    )
    def test_is_regular(self, graph, expected):
        assert reg.is_regular(graph) == expected

    def test_is_regular_empty_graph_raises(self):
        G = nx.Graph()
        with pytest.raises(nx.NetworkXPointlessConcept, match="Graph has no nodes"):
            nx.is_regular(G)


class TestIsKRegular:
    def test_is_k_regular1(self):
        g = gen.cycle_graph(4)
        assert reg.is_k_regular(g, 2)
        assert not reg.is_k_regular(g, 3)

    def test_is_k_regular2(self):
        g = gen.complete_graph(5)
        assert reg.is_k_regular(g, 4)
        assert not reg.is_k_regular(g, 3)
        assert not reg.is_k_regular(g, 6)

    def test_is_k_regular3(self):
        g = gen.lollipop_graph(5, 5)
        assert not reg.is_k_regular(g, 5)
        assert not reg.is_k_regular(g, 6)
