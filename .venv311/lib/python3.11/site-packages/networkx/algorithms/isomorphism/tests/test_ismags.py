"""
Tests for ISMAGS isomorphism algorithm.
"""

import random

import pytest

import networkx as nx
from networkx.algorithms import isomorphism as iso

graph_classes = [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]


def _matches_to_sets(matches):
    """
    Helper function to facilitate comparing collections of dictionaries in
    which order does not matter.
    """
    return {frozenset(m.items()) for m in matches}


graph_examples = [
    # node_data, edge_data, [id used in name for the test]
    pytest.param([0, 1, 2, 3], [(0, 0)], id="isolated-nodes-and-selfloops"),
    pytest.param([], nx.star_graph(3).edges, id="3-star"),
    pytest.param(
        # 6-cycle with 2-paths stuck onto nodes 0, 2, 4 (stretched symmetry)
        [],
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 0),
            (0, 6),
            (6, 7),
            (2, 8),
            (8, 9),
            (4, 10),
            (10, 11),
        ],
        id="sun:6-cycle-with-2-path-rays",
    ),
    # 0-1-2-3-5
    #  /     \
    # 4       6
    pytest.param([], [(0, 1), (1, 2), (1, 4), (2, 3), (3, 5), (3, 6)], id="tree"),
    pytest.param([], nx.petersen_graph().edges, id="petersen_graph"),
    # Example Fig 3 from Houbraken, et al (ISMAGS paper)
    pytest.param(
        [], nx.cycle_graph([1, 2, 4, 3]).edges, id="houbraken-ismags-paper-fig3"
    ),
    pytest.param(
        # path with node labels
        [
            (0, {"name": "a"}),
            (1, {"name": "a"}),
            (2, {"name": "b"}),
            (3, {"name": "b"}),
            (4, {"name": "a"}),
            (5, {"name": "a"}),
        ],
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        id="path-with-node-labels",
    ),
    pytest.param(
        # 5 - 4 \     / 12 - 13
        #        0 - 3
        # 9 - 8 /     \ 16 - 17
        # Assume 0 and 3 are coupled and no longer equivalent.
        # Coupling node 4 to 8 means that 5 and 9
        # are no longer equivalent, pushing them in their own partitions.
        # So, [{5}, {9}] is no longer considered equivalent to {13, 17}.
        # Minimal example with this trait. Adding all permutations of
        # same-size parts at each step finds the symmetry.
        [],
        [
            (0, 3),
            (3, 0),  # added to provide symmetry for DiGraphs
            (0, 4),
            (4, 5),
            (0, 8),
            (8, 9),
            (3, 12),
            (12, 13),
            (3, 16),
            (16, 17),
        ],
        id="gh8055-tricky-case",
    ),
    pytest.param(
        [], nx.path_graph([1, 2, 3, "a", "b", "c"]).edges, id="unsortable-nodes"
    ),
    # Example from Katebi, 2012, Fig 1-3.
    # Node order specified to (almost) match their DFS order
    pytest.param(
        [3, 5, 6, 4, 2, 1, 0],
        set(nx.cycle_graph(4).edges) | set(nx.cycle_graph(range(4, 7)).edges),
        id="katebi-paper-fig2",
    ),
    pytest.param(
        [], [(0, 1), (1, 2), (2, 3), (3, 6), (2, 4), (4, 5)], id="len-2-rays-tri-star"
    ),
    # Example of refining permutations with two different length parts at the same time.
    # Underlying shape is a 4-cycle and 2-path. Multiedges make all nodes degree-3
    # Full simple graph is then obtained by extending each edge as a path thru 1 node.
    #                   0
    # Underlying      // \     4       When 0->0 coupling occurs,
    # MultiGraph     1    3    \\\     refining {1, 2, 3, 4, 5}
    #                 \  //      5     refined parts [{1}, {3}, {2, 4, 5}]
    #                   2              with different parts having different lengths.
    #               0
    #              /|\         4
    #             6 7 8       /|\
    # Full:       |/  |      / | \
    #             1   3     12 13 14
    #             |  / \     \ | /
    #             9 10 11     \|/
    #              \ | /       5
    #                2
    # Nodes 0-5 are the degree-3 nodes.
    # Nodes 6-14 are degree 2 nodes on paths between the degree-3 nodes.
    pytest.param(
        [],
        [
            (0, 6),
            (0, 7),
            (0, 8),
            (1, 6),
            (1, 7),
            (1, 9),
            (2, 9),
            (2, 10),
            (2, 11),
            (3, 8),
            (3, 10),
            (3, 11),
            (4, 12),
            (4, 13),
            (4, 14),
            (5, 12),
            (5, 13),
            (5, 14),
        ],
        id="refining-parts-finds-different-lengths",
    ),
    # Underlying structure from previous example
    pytest.param(
        [],
        [(0, 1), (0, 1), (1, 2), (2, 3), (2, 3), (3, 0), (4, 5), (4, 5), (4, 5)],
        id="basic-structure-for-refining-parts-test",
    ),
]


@pytest.mark.parametrize("graph_constructor", graph_classes)
class TestSelfIsomorphism:
    @pytest.mark.parametrize(["node_data", "edge_data"], graph_examples)
    def test_self_isomorphism(self, graph_constructor, node_data, edge_data):
        """
        For some small, symmetric graphs, make sure that 1) they are isomorphic
        to themselves, and 2) that only the identity mapping is found.
        """
        graph = graph_constructor()
        graph.add_nodes_from(node_data)
        graph.add_edges_from(edge_data)

        ismags = iso.ISMAGS(
            graph, graph, node_match=iso.categorical_node_match("name", None)
        )
        assert ismags.is_isomorphic()
        assert ismags.is_isomorphic(symmetry=True)
        assert ismags.subgraph_is_isomorphic()
        ismags_answer = list(ismags.subgraph_isomorphisms_iter(symmetry=True))
        assert ismags_answer == [{n: n for n in graph.nodes}]


class TestSubgraphIsomorphism:
    def test_isomorphism_4_sun(self):
        g1 = nx.cycle_graph(4)
        g2 = nx.cycle_graph(4)
        g2.add_edges_from(list(zip(g2, range(4, 8))))
        ismags = iso.ISMAGS(g2, g1)
        assert list(ismags.subgraph_isomorphisms_iter(symmetry=True)) == [
            {n: n for n in g1.nodes}
        ]
        assert sum(1 for _ in ismags.subgraph_isomorphisms_iter(symmetry=False)) == 8

    def test_isomorphism_path_in_tristar(self):
        g1 = nx.path_graph(3)

        g2 = g1.copy()
        g2.add_edge(1, 3)

        ismags = iso.ISMAGS(g2, g1)
        matches = ismags.subgraph_isomorphisms_iter(symmetry=True)
        expected_symmetric = [
            {0: 0, 1: 1, 2: 2},
            {0: 0, 1: 1, 3: 2},
            {2: 0, 1: 1, 3: 2},
        ]
        assert _matches_to_sets(matches) == _matches_to_sets(expected_symmetric)

        matches = ismags.subgraph_isomorphisms_iter(symmetry=False)
        expected_asymmetric = [
            {0: 2, 1: 1, 2: 0},
            {0: 2, 1: 1, 3: 0},
            {2: 2, 1: 1, 3: 0},
        ]
        assert _matches_to_sets(matches) == _matches_to_sets(
            expected_symmetric + expected_asymmetric
        )

    def test_labeled_nodes(self):
        g1 = nx.cycle_graph(3)
        g1.nodes[1]["attr"] = True

        g2 = g1.copy()
        g2.add_edge(1, 3)
        ismags = iso.ISMAGS(g2, g1, node_match=lambda x, y: x == y)
        matches = ismags.subgraph_isomorphisms_iter(symmetry=True)
        expected_symmetric = [{0: 0, 1: 1, 2: 2}]
        assert _matches_to_sets(matches) == _matches_to_sets(expected_symmetric)

        matches = ismags.subgraph_isomorphisms_iter(symmetry=False)
        expected_asymmetric = [{0: 2, 1: 1, 2: 0}]
        assert _matches_to_sets(matches) == _matches_to_sets(
            expected_symmetric + expected_asymmetric
        )

    def test_labeled_edges(self):
        g1 = nx.Graph()
        nx.add_cycle(g1, range(3))
        g1.edges[1, 2]["attr"] = True

        g2 = g1.copy()
        g2.add_edge(1, 3)
        ismags = iso.ISMAGS(g2, g1, edge_match=lambda x, y: x == y)
        matches = ismags.subgraph_isomorphisms_iter(symmetry=True)
        expected_symmetric = [{0: 0, 1: 1, 2: 2}]
        assert _matches_to_sets(matches) == _matches_to_sets(expected_symmetric)

        matches = ismags.subgraph_isomorphisms_iter(symmetry=False)
        expected_asymmetric = [{1: 2, 0: 0, 2: 1}]
        assert _matches_to_sets(matches) == _matches_to_sets(
            expected_symmetric + expected_asymmetric
        )

    def test_exceptions_for_bad_match_functions(self):
        def non_transitive_match(attrs1, attrs2):
            return abs(attrs1["freq"] - attrs2["freq"]) <= 1

        def simple_non_commutative_match(attrs1, attrs2):
            return attrs1["freq"] == 1 + attrs2["freq"]

        def non_commutative_match(attrs1, attrs2):
            # red matches red and green
            # green and blue only match themselves
            if attrs2["color"] == "red":
                return attrs2["color"] in {"red", "green"}
            else:
                return attrs1["color"] == attrs2["color"]

        G1 = nx.Graph()
        G1.add_node(0, color="red", freq=0)
        G1.add_node(1, color="red", freq=1)
        G1.add_node(2, color="blue", freq=2)

        G2 = nx.Graph()
        G2.add_node("A", color="red", freq=0)
        G2.add_node("B", color="green", freq=1)
        G2.add_node("C", color="blue", freq=2)

        with pytest.raises(nx.NetworkXError, match="\nInvalid partition"):
            iso.ISMAGS(G1, G2, node_match=non_transitive_match)

        with pytest.raises(nx.NetworkXError, match="\nInvalid partition"):
            iso.ISMAGS(G1, G2, node_match=simple_non_commutative_match)

        with pytest.raises(nx.NetworkXError, match="\nInvalid partition"):
            iso.ISMAGS(G1, G2, node_match=non_commutative_match)


def test_noncomparable_nodes():
    node1 = object()
    node2 = object()
    node3 = object()

    # Graph
    G = nx.path_graph([node1, node2, node3])
    gm = iso.ISMAGS(G, G)
    assert gm.is_isomorphic()
    assert gm.subgraph_is_isomorphic()

    # DiGraph
    G = nx.path_graph([node1, node2, node3], create_using=nx.DiGraph)
    H = nx.path_graph([node3, node2, node1], create_using=nx.DiGraph)
    dgm = iso.ISMAGS(G, H)
    assert dgm.is_isomorphic()
    assert dgm.is_isomorphic(symmetry=True)
    assert dgm.subgraph_is_isomorphic()


@pytest.mark.parametrize("graph_constructor", graph_classes)
def test_selfloop(graph_constructor):
    # Simple test for graphs with selfloops
    g1 = graph_constructor([(0, 1), (0, 2), (1, 2), (1, 3), (2, 2), (2, 4)])
    nodes = range(5)
    rng = random.Random(42)

    for _ in range(3):
        new_nodes = list(nodes)
        rng.shuffle(new_nodes)
        d = dict(zip(nodes, new_nodes))
        g2 = nx.relabel_nodes(g1, d)
        assert iso.ISMAGS(g1, g2).is_isomorphic()


class TestWikipediaExample:
    # example in wikipedia is g1a and g2b
    # 1 have letter nodes, 2 have number nodes
    # b have some edges reversed vs a (undirected still isomorphic)
    # reversed edges marked with comment `#`
    # isomorphism = {'a': 1, 'g': 2, 'b': 3, 'c': 6, 'h': 4, 'i': 5, 'j': 7, 'd': 8}

    # Nodes 'a', 'b', 'c' and 'd' form a column.
    # Nodes 'g', 'h', 'i' and 'j' form a column.
    g1a_edges = [
        ["a", "g"],
        ["a", "h"],  # edge direction swapped from g1b
        ["a", "i"],
        ["b", "g"],  # edge direction swapped from g1b
        ["b", "h"],
        ["b", "j"],
        ["c", "g"],  # edge direction swapped from g1b
        ["c", "i"],  # edge direction swapped from g1b
        ["c", "j"],
        ["d", "h"],  # edge direction swapped from g1b
        ["d", "i"],
        ["d", "j"],  # edge direction swapped from g1b
    ]

    g1b_edges = [
        ["a", "g"],
        ["h", "a"],  # edge direction swapped from g1a
        ["a", "i"],
        ["g", "b"],  # edge direction swapped from g1a
        ["b", "h"],
        ["b", "j"],
        ["g", "c"],  # edge direction swapped from g1a
        ["i", "c"],  # edge direction swapped from g1a
        ["c", "j"],
        ["h", "d"],  # edge direction swapped from g1a
        ["d", "i"],
        ["j", "d"],  # edge direction swapped from g1a
    ]

    g2b_edges = [
        [1, 2],
        [1, 4],  # edge direction swapped from g2a
        [1, 5],
        [3, 2],  # edge direction swapped from g2a
        [3, 4],
        [3, 7],
        [6, 2],  # edge direction swapped from g2a
        [6, 5],  # edge direction swapped from g2a
        [6, 7],
        [8, 4],  # edge direction swapped from g2a
        [8, 5],
        [8, 7],  # edge direction swapped from g2a
    ]

    # Nodes 1,2,3,4 form the clockwise corners of a large square.
    # Nodes 5,6,7,8 form the clockwise corners of a small square
    g2a_edges = [
        [1, 2],
        [4, 1],  # edge direction swapped from g2b
        [1, 5],
        [2, 3],  # edge direction swapped from g2b
        [3, 4],
        [3, 7],
        [2, 6],  # edge direction swapped from g2b
        [5, 6],  # edge direction swapped from g2b
        [6, 7],
        [4, 8],  # edge direction swapped from g2b
        [8, 5],
        [7, 8],  # edge direction swapped from g2b
    ]

    @pytest.mark.parametrize("graph_constructor", [nx.Graph, nx.MultiGraph])
    def test_graph(self, graph_constructor):
        g1a = graph_constructor(self.g1a_edges)
        g1b = graph_constructor(self.g1b_edges)
        g2a = graph_constructor(self.g2a_edges)
        g2b = graph_constructor(self.g2b_edges)
        assert iso.ISMAGS(g1a, g1b).is_isomorphic()
        assert iso.ISMAGS(g1a, g2a).is_isomorphic()
        assert iso.ISMAGS(g1a, g2b).is_isomorphic()

        assert iso.ISMAGS(g1a, nx.path_graph(range(5))).subgraph_is_isomorphic()
        assert not iso.ISMAGS(g1a, nx.path_graph(range(6))).subgraph_is_isomorphic()

    @pytest.mark.parametrize("graph_constructor", [nx.DiGraph, nx.MultiDiGraph])
    def test_digraph(self, graph_constructor):
        g1a = graph_constructor(self.g1a_edges)
        g1b = graph_constructor(self.g1b_edges)
        g2a = graph_constructor(self.g2a_edges)
        g2b = graph_constructor(self.g2b_edges)
        assert iso.ISMAGS(g1a, g2b).is_isomorphic()
        assert iso.ISMAGS(g1b, g2a).is_isomorphic()
        assert not iso.ISMAGS(g1a, g1b).is_isomorphic()
        assert not iso.ISMAGS(g2a, g2b).is_isomorphic()
        assert not iso.ISMAGS(g1a, g2a).is_isomorphic()
        assert not iso.ISMAGS(g1b, g2b).is_isomorphic()

        P2 = nx.path_graph(range(2), create_using=graph_constructor)
        assert iso.ISMAGS(g1a, P2).subgraph_is_isomorphic()
        P3 = nx.path_graph(range(3), create_using=graph_constructor)
        assert not iso.ISMAGS(g1a, P3).subgraph_is_isomorphic()


class TestLargestCommonSubgraph:
    def test_mcis(self):
        # Example graphs from DOI: 10.1002/spe.588
        graph1 = nx.Graph()
        graph1.add_edges_from([(1, 2), (2, 3), (2, 4), (3, 4), (4, 5)])
        graph1.nodes[1]["color"] = 0

        graph2 = nx.Graph()
        graph2.add_edges_from(
            [(1, 2), (2, 3), (2, 4), (3, 4), (3, 5), (5, 6), (5, 7), (6, 7)]
        )
        graph2.nodes[1]["color"] = 1
        graph2.nodes[6]["color"] = 2
        graph2.nodes[7]["color"] = 2

        ismags = iso.ISMAGS(
            graph1, graph2, node_match=iso.categorical_node_match("color", None)
        )
        assert list(ismags.subgraph_isomorphisms_iter(symmetry=True)) == []
        assert list(ismags.subgraph_isomorphisms_iter(symmetry=False)) == []
        found_mcis = _matches_to_sets(ismags.largest_common_subgraph())
        expected = _matches_to_sets(
            [{2: 2, 3: 4, 4: 3, 5: 5}, {2: 4, 3: 2, 4: 3, 5: 5}]
        )
        assert expected == found_mcis

        ismags = iso.ISMAGS(
            graph2, graph1, node_match=iso.categorical_node_match("color", None)
        )
        assert list(ismags.subgraph_isomorphisms_iter(symmetry=True)) == []
        assert list(ismags.subgraph_isomorphisms_iter(symmetry=False)) == []
        found_mcis = _matches_to_sets(ismags.largest_common_subgraph())
        # Same answer, but reversed.
        expected = _matches_to_sets(
            [{2: 2, 3: 4, 4: 3, 5: 5}, {4: 2, 2: 3, 3: 4, 5: 5}]
        )
        assert expected == found_mcis

    def test_symmetry_mcis(self):
        graph1 = nx.Graph()
        nx.add_path(graph1, range(4))

        graph2 = nx.Graph()
        nx.add_path(graph2, range(3))
        graph2.add_edge(1, 3)

        # Only the symmetry of graph2 is taken into account here.
        ismags1 = iso.ISMAGS(
            graph1, graph2, node_match=iso.categorical_node_match("color", None)
        )
        assert list(ismags1.subgraph_isomorphisms_iter(symmetry=True)) == []
        found_mcis = _matches_to_sets(ismags1.largest_common_subgraph())
        expected = _matches_to_sets([{0: 0, 1: 1, 2: 2}, {1: 0, 3: 2, 2: 1}])
        assert expected == found_mcis

        # Only the symmetry of graph1 is taken into account here.
        ismags2 = iso.ISMAGS(
            graph2, graph1, node_match=iso.categorical_node_match("color", None)
        )
        assert list(ismags2.subgraph_isomorphisms_iter(symmetry=True)) == []
        found_mcis = _matches_to_sets(ismags2.largest_common_subgraph())
        expected = _matches_to_sets(
            [
                {3: 2, 0: 0, 1: 1},
                {2: 0, 0: 2, 1: 1},
                {3: 0, 0: 2, 1: 1},
                {3: 0, 1: 1, 2: 2},
                {0: 0, 1: 1, 2: 2},
                {2: 0, 3: 2, 1: 1},
            ]
        )

        assert expected == found_mcis

        found_mcis1 = _matches_to_sets(ismags1.largest_common_subgraph(symmetry=False))
        found_mcis2 = ismags2.largest_common_subgraph(symmetry=False)
        found_mcis2 = [{v: k for k, v in d.items()} for d in found_mcis2]
        found_mcis2 = _matches_to_sets(found_mcis2)

        expected = _matches_to_sets(
            [
                {3: 2, 1: 3, 2: 1},
                {2: 0, 0: 2, 1: 1},
                {1: 2, 3: 3, 2: 1},
                {3: 0, 1: 3, 2: 1},
                {0: 2, 2: 3, 1: 1},
                {3: 0, 1: 2, 2: 1},
                {2: 0, 0: 3, 1: 1},
                {0: 0, 2: 3, 1: 1},
                {1: 0, 3: 3, 2: 1},
                {1: 0, 3: 2, 2: 1},
                {0: 3, 1: 1, 2: 2},
                {0: 0, 1: 1, 2: 2},
            ]
        )
        assert expected == found_mcis1
        assert expected == found_mcis2


def is_isomorphic(G, SG, edge_match=None, node_match=None):
    return iso.ISMAGS(G, SG, node_match, edge_match).is_isomorphic()


class TestDiGraphISO:
    def test_wikipedia_graph(self):
        edges1 = [
            (1, 5),
            (1, 2),
            (1, 4),
            (3, 2),
            (6, 2),
            (3, 4),
            (7, 3),
            (4, 8),
            (5, 8),
            (6, 5),
            (6, 7),
            (7, 8),
        ]
        mapped = {1: "a", 2: "h", 3: "d", 4: "i", 5: "g", 6: "b", 7: "j", 8: "c"}

        G1 = nx.DiGraph(edges1)
        G2 = nx.relabel_nodes(G1, mapped)

        result = next(nx.isomorphism.ISMAGS(G1, G2).find_isomorphisms())
        assert result == mapped

        # Change the direction of an edge
        G1.remove_edge(1, 5)
        G1.add_edge(5, 1)
        result = list(nx.isomorphism.ISMAGS(G1, G2).find_isomorphisms())
        assert result == []

    def test_non_isomorphic_same_degree_sequence(self):
        r"""
                G1                           G2
        1-------------2              1-------------2
        | \           |              | \           |
        |  5-------6  |              |  5-------6  |
        |  |       |  |              |  |       |  |
        |  8-------7  |              |  8-------7  |
        | /           |              |           \ |
        4-------------3              4-------------3
        """
        edges1 = [
            (1, 5),
            (1, 2),
            (4, 1),
            (3, 2),
            (3, 4),
            (5, 8),
            (6, 5),
            (6, 7),
            (7, 8),
            (4, 8),
        ]
        edges2 = [
            (1, 5),
            (1, 2),
            (4, 1),
            (3, 2),
            (3, 4),
            (5, 8),
            (6, 5),
            (6, 7),
            (8, 7),
            (3, 7),
        ]

        G1 = nx.DiGraph(edges1)
        G2 = nx.DiGraph(edges2)
        assert not is_isomorphic(G1, G2)

    def test_is_isomorphic(self):
        G1 = nx.Graph([[1, 2], [1, 3], [1, 5], [2, 3]])
        G2 = nx.Graph([[10, 20], [20, 30], [10, 30], [10, 50]])
        G4 = nx.Graph([[1, 2], [1, 3], [1, 5], [2, 4]])
        assert is_isomorphic(G1, G2)
        assert not is_isomorphic(G1, G4)
        assert is_isomorphic(G1.to_directed(), G2.to_directed())
        assert not is_isomorphic(G1.to_directed(), G4.to_directed())
        with pytest.raises(
            ValueError, match="Directed and undirected graphs cannot be compared."
        ):
            is_isomorphic(G1.to_directed(), G1)


@pytest.mark.parametrize("graph_class", graph_classes)
def test_simple_node_match(graph_class):
    g1 = graph_class([(0, 0), (0, 1), (1, 0)])
    g2 = g1.copy()
    nm = iso.numerical_node_match("size", 1)
    assert is_isomorphic(g1, g2, node_match=nm)

    g2.nodes[0]["size"] = 3
    assert not is_isomorphic(g1, g2, node_match=nm)


@pytest.mark.parametrize("graph_class", graph_classes)
def test_simple_node_and_edge_match(graph_class):
    g1 = graph_class()
    g1.add_weighted_edges_from([(0, 0, 1.2), (0, 1, 1.4), (1, 0, 1.6)])
    g2 = g1.copy()
    nm = iso.numerical_node_match("size", 1)
    if g1.is_multigraph():
        em = iso.numerical_multiedge_match("weight", 1)
    else:
        em = iso.numerical_edge_match("weight", 1)
    assert is_isomorphic(g1, g2, node_match=nm, edge_match=em)

    g2.nodes[0]["size"] = 3
    assert not is_isomorphic(g1, g2, node_match=nm, edge_match=em)

    g2 = g1.copy()
    if g1.is_multigraph():
        g2.edges[0, 1, 0]["weight"] = 2.1
    else:
        g2.edges[0, 1]["weight"] = 2.1
    assert not is_isomorphic(g1, g2, node_match=nm, edge_match=em)

    g2 = g1.copy()
    g2.nodes[0]["size"] = 3
    if g1.is_multigraph():
        g2.edges[0, 1, 0]["weight"] = 2.1
    else:
        g2.edges[0, 1]["weight"] = 2.1
    assert not is_isomorphic(g1, g2, node_match=nm, edge_match=em)


@pytest.mark.parametrize("graph_class", graph_classes)
def test_simple_edge_match(graph_class):
    # 16 simple tests
    w = "weight"
    edges = [(0, 0, 1), (0, 0, 1.5), (0, 1, 2), (1, 0, 3)]
    g1 = graph_class()
    g1.add_weighted_edges_from(edges)
    g2 = g1.copy()
    if g1.is_multigraph():
        em = iso.numerical_multiedge_match("weight", 1)
    else:
        em = iso.numerical_edge_match("weight", 1)
    assert is_isomorphic(g1, g2, edge_match=em)

    for mod1, mod2 in [(False, True), (True, False), (True, True)]:
        # mod1 tests a regular edge weight difference
        # mod2 tests a selfloop weight difference
        if g1.is_multigraph():
            if mod1:
                data1 = {0: {"weight": 10}}
            if mod2:
                data2 = {0: {"weight": 1}, 1: {"weight": 2.5}}
        else:
            if mod1:
                data1 = {"weight": 10}
            if mod2:
                data2 = {"weight": 2.5}

        g2 = g1.copy()
        if mod1:
            if not g1.is_directed():
                g2._adj[1][0] = data1
                g2._adj[0][1] = data1
            else:
                g2._succ[1][0] = data1
                g2._pred[0][1] = data1
        if mod2:
            if not g1.is_directed():
                g2._adj[0][0] = data2
            else:
                g2._succ[0][0] = data2
                g2._pred[0][0] = data2

        assert not is_isomorphic(g1, g2, edge_match=em)


@pytest.mark.parametrize("graph_class", graph_classes)
def test_weightkey(graph_class):
    g1 = graph_class()
    g2 = graph_class()
    if g1.is_multigraph():
        edge_match = iso.numerical_multiedge_match
    else:
        edge_match = iso.numerical_edge_match

    g1.add_edge("A", "B", weight=1)
    g2.add_edge("C", "D", weight=0)

    assert nx.is_isomorphic(g1, g2)
    em = edge_match("nonexistent attribute", 1)
    assert nx.is_isomorphic(g1, g2, edge_match=em)
    em = edge_match("weight", 1)
    assert not nx.is_isomorphic(g1, g2, edge_match=em)

    g2 = graph_class()
    g2.add_edge("C", "D")
    assert nx.is_isomorphic(g1, g2, edge_match=em)
