"""
Unit tests for WROM algorithm generator in generators/nonisomorphic_trees.py
"""

import pytest

import networkx as nx
from networkx.utils import edges_equal


def test_nonisomorphic_tree_negative_order():
    with pytest.raises(ValueError, match="order must be non-negative"):
        nx.number_of_nonisomorphic_trees(-1)
    with pytest.raises(ValueError, match="order must be non-negative"):
        next(nx.nonisomorphic_trees(-1))


def test_nonisomorphic_tree_order_0():
    assert nx.number_of_nonisomorphic_trees(0) == 0
    assert list(nx.nonisomorphic_trees(0)) == []


def test_nonisomorphic_tree_order_1():
    assert nx.number_of_nonisomorphic_trees(1) == 1
    nit_list = list(nx.nonisomorphic_trees(1))
    assert len(nit_list) == 1
    G = nit_list[0]
    assert nx.utils.graphs_equal(G, nx.empty_graph(1))


@pytest.mark.parametrize("n", range(5))
def test_nonisomorphic_tree_low_order_agreement(n):
    """Ensure all the order<2 'special cases' are consistent."""
    assert len(list(nx.nonisomorphic_trees(n))) == nx.number_of_nonisomorphic_trees(n)


class TestGeneratorNonIsomorphicTrees:
    def test_tree_structure(self):
        # test for tree structure for nx.nonisomorphic_trees()
        def f(x):
            return list(nx.nonisomorphic_trees(x))

        for i in f(6):
            assert nx.is_tree(i)
        for i in f(8):
            assert nx.is_tree(i)

    def test_nonisomorphism(self):
        # test for nonisomorphism of trees for nx.nonisomorphic_trees()
        def f(x):
            return list(nx.nonisomorphic_trees(x))

        trees = f(6)
        for i in range(len(trees)):
            for j in range(i + 1, len(trees)):
                assert not nx.is_isomorphic(trees[i], trees[j])
        trees = f(8)
        for i in range(len(trees)):
            for j in range(i + 1, len(trees)):
                assert not nx.is_isomorphic(trees[i], trees[j])

    def test_number_of_nonisomorphic_trees(self):
        # http://oeis.org/A000055
        assert nx.number_of_nonisomorphic_trees(2) == 1
        assert nx.number_of_nonisomorphic_trees(3) == 1
        assert nx.number_of_nonisomorphic_trees(4) == 2
        assert nx.number_of_nonisomorphic_trees(5) == 3
        assert nx.number_of_nonisomorphic_trees(6) == 6
        assert nx.number_of_nonisomorphic_trees(7) == 11
        assert nx.number_of_nonisomorphic_trees(8) == 23
        assert nx.number_of_nonisomorphic_trees(9) == 47
        assert nx.number_of_nonisomorphic_trees(10) == 106
        assert nx.number_of_nonisomorphic_trees(20) == 823065
        assert nx.number_of_nonisomorphic_trees(30) == 14830871802

    def test_nonisomorphic_trees(self):
        def f(x):
            return list(nx.nonisomorphic_trees(x))

        assert edges_equal(f(3)[0].edges(), [(0, 1), (0, 2)])
        assert edges_equal(f(4)[0].edges(), [(0, 1), (0, 3), (1, 2)])
        assert edges_equal(f(4)[1].edges(), [(0, 1), (0, 2), (0, 3)])
