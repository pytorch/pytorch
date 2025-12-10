from functools import partial

import pytest

import networkx as nx
from networkx.algorithms import isomorphism as iso

# Convenience functions for testing that the behavior of `could_be_isomorphic`
# with the "properties" kwarg is equivalent to the corresponding function (i.e.
# nx.fast_could_be_isomorphic or nx.faster_could_be_isomorphic)
fast_cbi = partial(nx.could_be_isomorphic, properties="dt")
faster_cbi = partial(nx.could_be_isomorphic, properties="d")


def test_graph_could_be_isomorphic_variants_deprecated():
    G1 = nx.Graph([(1, 2), (1, 3), (1, 5), (2, 3)])
    G2 = nx.Graph([(10, 20), (20, 30), (10, 30), (10, 50)])
    with pytest.deprecated_call():  # graph_could_be_isomorphic
        result = nx.isomorphism.isomorph.graph_could_be_isomorphic(G1, G2)
    assert nx.could_be_isomorphic(G1, G2) == result
    with pytest.deprecated_call():  # fast_graph_could_be_isomorphic
        result = nx.isomorphism.isomorph.fast_graph_could_be_isomorphic(G1, G2)
    assert nx.fast_could_be_isomorphic(G1, G2) == result
    with pytest.deprecated_call():
        result = nx.isomorphism.isomorph.faster_graph_could_be_isomorphic(G1, G2)
    assert nx.faster_could_be_isomorphic(G1, G2) == result


@pytest.mark.parametrize("atlas_ids", [(699, 706), (864, 870)])
def test_could_be_isomorphic_combined_properties(atlas_ids):
    """There are two pairs of graphs from the graph atlas that have the same
    combined degree-triangle distribution, but a different maximal clique
    distribution. See gh-7852."""
    G, H = (nx.graph_atlas(idx) for idx in atlas_ids)

    assert not nx.is_isomorphic(G, H)

    # Degree only
    assert nx.faster_could_be_isomorphic(G, H)
    assert nx.could_be_isomorphic(G, H, properties="d")
    # Degrees & triangles
    assert nx.fast_could_be_isomorphic(G, H)
    assert nx.could_be_isomorphic(G, H, properties="dt")
    # Full properties table (degrees, triangles, cliques)
    assert not nx.could_be_isomorphic(G, H)
    assert not nx.could_be_isomorphic(G, H, properties="dtc")
    # For these two cases, the clique distribution alone is enough to verify
    # the graphs can't be isomorphic
    assert not nx.could_be_isomorphic(G, H, properties="c")


def test_could_be_isomorphic_individual_vs_combined_dt():
    """A test case where G and H have identical degree and triangle distributions,
    but are different when compared together"""
    G = nx.Graph([(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (3, 4), (4, 5), (4, 6)])
    H = G.copy()
    # Modify graphs to produce different clique distributions
    G.add_edge(0, 7)
    H.add_edge(4, 7)
    assert nx.could_be_isomorphic(G, H, properties="d")
    assert nx.could_be_isomorphic(G, H, properties="t")
    assert not nx.could_be_isomorphic(G, H, properties="dt")
    assert not nx.could_be_isomorphic(G, H, properties="c")


class TestIsomorph:
    @classmethod
    def setup_class(cls):
        cls.G1 = nx.Graph([[1, 2], [1, 3], [1, 5], [2, 3]])
        cls.G2 = nx.Graph([[10, 20], [20, 30], [10, 30], [10, 50]])
        cls.G3 = nx.Graph([[1, 2], [1, 3], [1, 5], [2, 5]])
        cls.G4 = nx.Graph([[1, 2], [1, 3], [1, 5], [2, 4]])
        cls.G5 = nx.Graph([[1, 2], [1, 3]])
        cls.G6 = nx.Graph([[10, 20], [20, 30], [10, 30], [10, 50], [20, 50]])

    def test_could_be_isomorphic(self):
        assert iso.could_be_isomorphic(self.G1, self.G2)
        assert iso.could_be_isomorphic(self.G1, self.G3)
        assert not iso.could_be_isomorphic(self.G1, self.G4)
        assert iso.could_be_isomorphic(self.G3, self.G2)
        assert not iso.could_be_isomorphic(self.G1, self.G6)

    @pytest.mark.parametrize("fn", (iso.fast_could_be_isomorphic, fast_cbi))
    def test_fast_could_be_isomorphic(self, fn):
        assert fn(self.G3, self.G2)
        assert not fn(self.G3, self.G5)
        assert not fn(self.G1, self.G6)

    @pytest.mark.parametrize("fn", (iso.faster_could_be_isomorphic, faster_cbi))
    def test_faster_could_be_isomorphic(self, fn):
        assert fn(self.G3, self.G2)
        assert not fn(self.G3, self.G5)
        assert not fn(self.G1, self.G6)

    def test_is_isomorphic(self):
        assert iso.is_isomorphic(self.G1, self.G2)
        assert not iso.is_isomorphic(self.G1, self.G4)
        assert iso.is_isomorphic(self.G1.to_directed(), self.G2.to_directed())
        assert not iso.is_isomorphic(self.G1.to_directed(), self.G4.to_directed())
        with pytest.raises(
            nx.NetworkXError, match="Graphs G1 and G2 are not of the same type."
        ):
            iso.is_isomorphic(self.G1.to_directed(), self.G1)
