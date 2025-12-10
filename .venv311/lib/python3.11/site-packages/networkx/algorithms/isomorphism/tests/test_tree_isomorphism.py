import random
import time

import pytest

import networkx as nx


def _check_isomorphism(t1, t2, isomorphism):
    assert nx.is_directed(t1) == nx.is_directed(t2)
    # Apply mapping and check for equality
    H = nx.relabel_nodes(t1, dict(isomorphism))
    return nx.utils.graphs_equal(t2, H)


@pytest.mark.parametrize("graph_constructor", (nx.DiGraph, nx.MultiGraph))
def test_tree_isomorphism_raises_on_directed_and_multigraphs(graph_constructor):
    t1 = graph_constructor([(0, 1)])
    t2 = graph_constructor([(1, 2)])
    with pytest.raises(nx.NetworkXNotImplemented):
        nx.isomorphism.tree_isomorphism(t1, t2)


def test_input_not_tree():
    tree = nx.Graph([(0, 1), (0, 2)])
    not_tree = nx.cycle_graph(3)

    # tree_isomorphism
    with pytest.raises(nx.NetworkXError, match="t1 is not a tree"):
        nx.isomorphism.tree_isomorphism(not_tree, tree)
    with pytest.raises(nx.NetworkXError, match="t2 is not a tree"):
        nx.isomorphism.tree_isomorphism(tree, not_tree)

    # rooted_tree_isomorphism
    with pytest.raises(nx.NetworkXError, match="t1 is not a tree"):
        nx.isomorphism.rooted_tree_isomorphism(not_tree, 0, tree, 0)
    with pytest.raises(nx.NetworkXError, match="t2 is not a tree"):
        nx.isomorphism.rooted_tree_isomorphism(tree, 0, not_tree, 0)


def test_hardcoded():
    # define a test problem
    edges_1 = [
        ("a", "b"),
        ("a", "c"),
        ("a", "d"),
        ("b", "e"),
        ("b", "f"),
        ("e", "j"),
        ("e", "k"),
        ("c", "g"),
        ("c", "h"),
        ("g", "m"),
        ("d", "i"),
        ("f", "l"),
    ]

    edges_2 = [
        ("v", "y"),
        ("v", "z"),
        ("u", "x"),
        ("q", "u"),
        ("q", "v"),
        ("p", "t"),
        ("n", "p"),
        ("n", "q"),
        ("n", "o"),
        ("o", "r"),
        ("o", "s"),
        ("s", "w"),
    ]

    # there are two possible correct isomorphisms
    # it currently returns isomorphism1
    # but the second is also correct
    isomorphism1 = [
        ("a", "n"),
        ("b", "q"),
        ("c", "o"),
        ("d", "p"),
        ("e", "v"),
        ("f", "u"),
        ("g", "s"),
        ("h", "r"),
        ("i", "t"),
        ("j", "y"),
        ("k", "z"),
        ("l", "x"),
        ("m", "w"),
    ]

    # could swap y and z
    isomorphism2 = [
        ("a", "n"),
        ("b", "q"),
        ("c", "o"),
        ("d", "p"),
        ("e", "v"),
        ("f", "u"),
        ("g", "s"),
        ("h", "r"),
        ("i", "t"),
        ("j", "z"),
        ("k", "y"),
        ("l", "x"),
        ("m", "w"),
    ]

    t1 = nx.Graph()
    t1.add_edges_from(edges_1)
    root1 = "a"

    t2 = nx.Graph()
    t2.add_edges_from(edges_2)
    root2 = "n"

    isomorphism = sorted(nx.isomorphism.rooted_tree_isomorphism(t1, root1, t2, root2))

    # is correct by hand
    assert isomorphism in (isomorphism1, isomorphism2)

    # check algorithmically
    assert _check_isomorphism(t1, t2, isomorphism)

    # try again as digraph
    t1 = nx.DiGraph()
    t1.add_edges_from(edges_1)
    root1 = "a"

    t2 = nx.DiGraph()
    t2.add_edges_from(edges_2)
    root2 = "n"

    isomorphism = sorted(nx.isomorphism.rooted_tree_isomorphism(t1, root1, t2, root2))

    # is correct by hand
    assert isomorphism in (isomorphism1, isomorphism2)

    # check algorithmically
    assert _check_isomorphism(t1, t2, isomorphism)


# NOTE: number of nonisomorphic_trees grows very rapidly - do not increase n
# further without marking "slow"
@pytest.mark.parametrize("n", range(2, 15))
def test_tree_isomorphic_all_non_isomorphic_trees_relabeled(n):
    """Tests every non-isomorphic tree with `n` nodes is isomorphic with a
    copy of itself with an arbitrary node-remapping."""
    for tree in nx.nonisomorphic_trees(n):
        nodes = list(tree)
        random.shuffle(shuffled := nodes.copy())
        node_mapping = dict(zip(nodes, shuffled))
        # Shuffle the edge list to ensure no dependence on edge order
        random.shuffle(
            new_edges := [
                # Randomly order edges to ensure no dependence on node order within edges
                (node_mapping[u], node_mapping[v])
                if random.randint(0, 1)
                else (node_mapping[v], node_mapping[u])
                for (u, v) in tree.edges
            ]
        )
        relabeled = nx.Graph(new_edges)
        # Does not necessarily have to be the same as node_mapping
        iso_mapping = nx.isomorphism.tree_isomorphism(tree, relabeled)

        assert iso_mapping != []
        assert _check_isomorphism(tree, relabeled, iso_mapping)


def test_trivial_rooted_tree_isomorphism():
    t1 = nx.Graph()
    t1.add_node("a")

    t2 = nx.Graph()
    t2.add_node("n")

    assert nx.isomorphism.rooted_tree_isomorphism(t1, "a", t2, "n") == [("a", "n")]


def test_rooted_tree_isomorphism_different_order():
    t1 = nx.Graph([("a", "b"), ("a", "c")])
    t2 = nx.Graph([("a", "b")])
    assert nx.isomorphism.tree_isomorphism(t1, t2) == []


# NOTE: number of nonisomorphic_trees grows very rapidly - do not increase n
# further without marking "slow"
@pytest.mark.parametrize("n", range(4, 12))
def test_tree_isomorphism_all_non_isomorphic_pairs(n):
    test_trees = list(nx.nonisomorphic_trees(n))
    assert all(
        nx.isomorphism.tree_isomorphism(test_trees[i], test_trees[j]) == []
        for i in range(len(test_trees) - 1)
        for j in range(i + 1, len(test_trees))
    )


def test_long_paths_graphs():
    """Smoke test for potential RecursionError. See gh-7945."""
    G = nx.path_graph(10_000)
    nx.isomorphism.rooted_tree_isomorphism(G, 0, G, 0) == [(n, n) for n in G]
