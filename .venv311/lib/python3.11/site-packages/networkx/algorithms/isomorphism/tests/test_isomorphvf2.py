"""
Tests for VF2 isomorphism algorithm.
"""

import importlib.resources
import random
import struct

import pytest

import networkx as nx
from networkx.algorithms import isomorphism as iso


class TestWikipediaExample:
    # Source: https://en.wikipedia.org/wiki/Graph_isomorphism

    # Nodes 'a', 'b', 'c' and 'd' form a column.
    # Nodes 'g', 'h', 'i' and 'j' form a column.
    # isomorphism: {'a': 1, 'g': 2, 'b': 3, 'c': 6, 'h': 4, 'i': 5, 'j': 7, 'd': 8}
    g1edges = [
        ["a", "g"],
        ["a", "h"],
        ["a", "i"],
        ["b", "g"],
        ["b", "h"],
        ["b", "j"],
        ["c", "g"],
        ["c", "i"],
        ["c", "j"],
        ["d", "h"],
        ["d", "i"],
        ["d", "j"],
    ]

    # Nodes 1,2,3,4 form the clockwise corners of a large square.
    # Nodes 5,6,7,8 form the clockwise corners of a small square
    g2edges = [
        [1, 2],
        [3, 2],
        [3, 4],
        [1, 4],
        [6, 5],
        [6, 7],
        [8, 7],
        [8, 5],
        [1, 5],
        [6, 2],
        [3, 7],
        [8, 4],
    ]

    @pytest.mark.parametrize(
        ["graph_class", "numb_maps"], [(nx.Graph, 48), (nx.DiGraph, 24)]
    )
    def test_morphic_and_number_of_mappings(self, graph_class, numb_maps):
        g1 = graph_class(self.g1edges)
        g2 = graph_class(self.g2edges)
        Matcher = iso.DiGraphMatcher if g1.is_directed() else iso.GraphMatcher
        gm = Matcher(g1, g2)
        assert gm.is_isomorphic()
        assert gm.subgraph_is_monomorphic()
        assert gm.subgraph_is_isomorphic()

        mapping = list(gm.mapping.items())
        # this mapping is only one of the 48 possibilities
        all_mappings = list(gm.isomorphisms_iter())
        assert len(all_mappings) == numb_maps
        assert dict(mapping) in all_mappings

    @pytest.mark.parametrize("graph_class", [nx.Graph, nx.DiGraph])
    def test_subgraph(self, graph_class):
        g1 = graph_class(self.g1edges)
        g2 = graph_class(self.g2edges)
        g3 = g2.subgraph([1, 2, 3, 4])
        Matcher = iso.DiGraphMatcher if g1.is_directed() else iso.GraphMatcher
        gm = Matcher(g1, g3)
        assert not gm.is_isomorphic()
        assert gm.subgraph_is_isomorphic()
        assert gm.subgraph_is_monomorphic()

    @pytest.mark.parametrize("graph_class", [nx.Graph, nx.DiGraph])
    def test_subgraph_mono(self, graph_class):
        g1 = graph_class(self.g1edges)
        g2 = graph_class(["ag", "cg", "ci", "di", "bg"])
        Matcher = iso.DiGraphMatcher if g1.is_directed() else iso.GraphMatcher
        gm = Matcher(g1, g2)
        assert not gm.is_isomorphic()
        assert not gm.subgraph_is_isomorphic()
        assert gm.subgraph_is_monomorphic()

        # note: this shows need to recreate the matcher for each graph change
        # Also checking not monomorphic
        g2.add_edge("c", "a")
        gm = Matcher(g1, g2)
        assert not gm.subgraph_is_monomorphic()


class TestVF2GraphDB:
    # https://web.archive.org/web/20090303210205/http://amalfi.dis.unina.it/graph/db/

    @staticmethod
    def create_graph(filename):
        """Creates a Graph instance from the filename."""

        # The file is assumed to be in the format from the VF2 graph database.
        # Each file is composed of 16-bit numbers (unsigned short int).
        # So we will want to read 2 bytes at a time.

        # We can read the number as follows:
        #   number = struct.unpack('<H', file.read(2))
        # This says, expect the data in little-endian encoding
        # as an unsigned short int and unpack 2 bytes from the file.

        fh = open(filename, mode="rb")

        # Grab the number of nodes.
        # Node numeration is 0-based, so the first node has index 0.
        nodes = struct.unpack("<H", fh.read(2))[0]

        graph = nx.Graph()
        for from_node in range(nodes):
            # Get the number of edges.
            edges = struct.unpack("<H", fh.read(2))[0]
            for edge in range(edges):
                # Get the terminal node.
                to_node = struct.unpack("<H", fh.read(2))[0]
                graph.add_edge(from_node, to_node)

        fh.close()
        return graph

    def test_graph(self):
        head = importlib.resources.files("networkx.algorithms.isomorphism.tests")
        g1 = self.create_graph(head / "iso_r01_s80.A99")
        g2 = self.create_graph(head / "iso_r01_s80.B99")
        gm = iso.GraphMatcher(g1, g2)
        assert gm.is_isomorphic()

    def test_subgraph(self):
        # A is the subgraph
        # B is the full graph
        head = importlib.resources.files("networkx.algorithms.isomorphism.tests")
        subgraph = self.create_graph(head / "si2_b06_m200.A99")
        graph = self.create_graph(head / "si2_b06_m200.B99")
        gm = iso.GraphMatcher(graph, subgraph)
        assert not gm.is_isomorphic()
        assert gm.subgraph_is_isomorphic()
        assert gm.subgraph_is_monomorphic()

    def test_subgraph_monomorphic(self):
        # A is the subgraph
        # B is the full graph
        head = importlib.resources.files("networkx.algorithms.isomorphism.tests")
        subgraph = self.create_graph(head / "si2_b06_m200.A99")
        graph = self.create_graph(head / "si2_b06_m200.B99")
        subgraph.remove_edge(29, 28)
        gm = iso.GraphMatcher(graph, subgraph)
        assert not gm.is_isomorphic()
        assert not gm.subgraph_is_isomorphic()
        assert gm.subgraph_is_monomorphic()

        # now add new edge to make sg no longer monomorphic
        subgraph.add_edge(29, 2)
        assert not gm.subgraph_is_monomorphic()


class TestAtlas:
    @classmethod
    def setup_class(cls):
        global atlas
        from networkx.generators import atlas

        cls.GAG = atlas.graph_atlas_g()

    def test_graph_atlas(self):
        rng = random.Random(42)
        # Atlas = nx.graph_atlas_g()[0:208] # 208, 6 nodes or less
        Atlas = self.GAG[0:100]
        alphabet = list(range(26))
        for graph in Atlas:
            nlist = list(graph)
            labels = alphabet[: len(nlist)]
            for _ in range(10):
                rng.shuffle(labels)
                d = dict(zip(nlist, labels))
                relabel = nx.relabel_nodes(graph, d)
                gm = iso.GraphMatcher(graph, relabel)
                assert gm.is_isomorphic()


def test_multiedge():
    # Simple test for multigraphs
    # Need something much more rigorous
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (9, 10),
        (10, 11),
        (10, 11),
        (11, 12),
        (11, 12),
        (12, 13),
        (12, 13),
        (13, 14),
        (13, 14),
        (14, 15),
        (14, 15),
        (15, 16),
        (15, 16),
        (16, 17),
        (16, 17),
        (17, 18),
        (17, 18),
        (18, 19),
        (18, 19),
        (19, 0),
        (19, 0),
    ]
    nodes = list(range(20))

    rng = random.Random(42)

    for g1 in [nx.MultiGraph(), nx.MultiDiGraph()]:
        g1.add_edges_from(edges)
        for _ in range(10):
            new_nodes = list(nodes)
            rng.shuffle(new_nodes)
            d = dict(zip(nodes, new_nodes))
            g2 = nx.relabel_nodes(g1, d)
            if not g1.is_directed():
                gm = iso.GraphMatcher(g1, g2)
            else:
                gm = iso.DiGraphMatcher(g1, g2)
            assert gm.is_isomorphic()
            # Testing if monomorphism works in multigraphs
            assert gm.subgraph_is_monomorphic()


@pytest.mark.parametrize("G1", [nx.Graph(), nx.MultiGraph()])
@pytest.mark.parametrize("G2", [nx.Graph(), nx.MultiGraph()])
def test_matcher_raises(G1, G2):
    undirected_matchers = [iso.GraphMatcher, iso.MultiGraphMatcher]
    directed_matchers = [iso.DiGraphMatcher, iso.MultiDiGraphMatcher]

    for matcher in undirected_matchers:
        matcher(G1, G2)

        msg = r"\(Multi-\)GraphMatcher\(\) not defined for directed graphs"
        with pytest.raises(nx.NetworkXError, match=msg):
            matcher(G1.to_directed(), G2.to_directed())

    for matcher in directed_matchers:
        matcher(G1.to_directed(), G2.to_directed())

        msg = r"\(Multi-\)DiGraphMatcher\(\) not defined for undirected graphs"
        with pytest.raises(nx.NetworkXError, match=msg):
            matcher(G1, G2)

    for matcher in undirected_matchers + directed_matchers:
        msg = r"G1 and G2 must have the same directedness"
        with pytest.raises(nx.NetworkXError, match=msg):
            matcher(G1, G2.to_directed())
        with pytest.raises(nx.NetworkXError, match=msg):
            matcher(G1.to_directed(), G2)


def test_selfloop():
    # Simple test for graphs with selfloops
    edges = [
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (2, 2),
        (2, 4),
        (3, 1),
        (3, 2),
        (4, 2),
        (4, 5),
        (5, 4),
    ]
    nodes = list(range(6))

    rng = random.Random(42)

    for g1 in [nx.Graph(), nx.DiGraph()]:
        g1.add_edges_from(edges)
        for _ in range(100):
            new_nodes = list(nodes)
            rng.shuffle(new_nodes)
            d = dict(zip(nodes, new_nodes))
            g2 = nx.relabel_nodes(g1, d)
            if not g1.is_directed():
                gm = iso.GraphMatcher(g1, g2)
            else:
                gm = iso.DiGraphMatcher(g1, g2)
            assert gm.is_isomorphic()


def test_selfloop_mono():
    # Simple test for graphs with selfloops
    edges0 = [
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (2, 4),
        (3, 1),
        (3, 2),
        (4, 2),
        (4, 5),
        (5, 4),
    ]
    edges = edges0 + [(2, 2)]
    nodes = list(range(6))

    rng = random.Random(42)

    for g1 in [nx.Graph(), nx.DiGraph()]:
        g1.add_edges_from(edges)
        for _ in range(100):
            new_nodes = list(nodes)
            rng.shuffle(new_nodes)
            d = dict(zip(nodes, new_nodes))
            g2 = nx.relabel_nodes(g1, d)
            g2.remove_edges_from(nx.selfloop_edges(g2))
            if not g1.is_directed():
                gm = iso.GraphMatcher(g2, g1)
            else:
                gm = iso.DiGraphMatcher(g2, g1)
            assert not gm.subgraph_is_monomorphic()


def test_isomorphism_iter1():
    # As described in:
    # http://groups.google.com/group/networkx-discuss/browse_thread/thread/2ff65c67f5e3b99f/d674544ebea359bb?fwc=1
    g1 = nx.DiGraph(["AB", "BC"])
    g2 = nx.DiGraph(["YZ"])
    g3 = nx.DiGraph(["ZY"])
    gm12 = iso.DiGraphMatcher(g1, g2)
    gm13 = iso.DiGraphMatcher(g1, g3)
    x = list(gm12.subgraph_isomorphisms_iter())
    y = list(gm13.subgraph_isomorphisms_iter())
    assert {"A": "Y", "B": "Z"} in x
    assert {"B": "Y", "C": "Z"} in x
    assert {"A": "Z", "B": "Y"} in y
    assert {"B": "Z", "C": "Y"} in y
    assert len(x) == len(y)
    assert len(x) == 2


def test_monomorphism_iter1():
    g1 = nx.DiGraph(["AB", "BC", "CA"])
    g2 = nx.DiGraph(["XY", "YZ"])
    gm12 = iso.DiGraphMatcher(g1, g2)
    x = list(gm12.subgraph_monomorphisms_iter())
    assert {"A": "X", "B": "Y", "C": "Z"} in x
    assert {"A": "Y", "B": "Z", "C": "X"} in x
    assert {"A": "Z", "B": "X", "C": "Y"} in x
    assert len(x) == 3
    gm21 = iso.DiGraphMatcher(g2, g1)
    # Check if StopIteration exception returns False
    assert not gm21.subgraph_is_monomorphic()


def test_isomorphism_iter2():
    # Path
    for L in range(2, 10):
        g1 = nx.path_graph(L)
        gm = iso.GraphMatcher(g1, g1)
        s = len(list(gm.isomorphisms_iter()))
        assert s == 2
    # Cycle
    for L in range(3, 10):
        g1 = nx.cycle_graph(L)
        gm = iso.GraphMatcher(g1, g1)
        s = len(list(gm.isomorphisms_iter()))
        assert s == 2 * L


def test_multiple():
    # Verify that we can use the graph matcher multiple times
    edges = [("A", "B"), ("B", "A"), ("B", "C")]
    for g1, g2 in [(nx.Graph(), nx.Graph()), (nx.DiGraph(), nx.DiGraph())]:
        g1.add_edges_from(edges)
        g2.add_edges_from(edges)
        g3 = nx.subgraph(g2, ["A", "B"])
        if not g1.is_directed():
            gmA = iso.GraphMatcher(g1, g2)
            gmB = iso.GraphMatcher(g1, g3)
        else:
            gmA = iso.DiGraphMatcher(g1, g2)
            gmB = iso.DiGraphMatcher(g1, g3)
        assert gmA.is_isomorphic()

        g2.remove_node("C")
        if not g1.is_directed():
            gmA = iso.GraphMatcher(g1, g2)
        else:
            gmA = iso.DiGraphMatcher(g1, g2)
        assert gmA.subgraph_is_isomorphic()
        assert gmB.subgraph_is_isomorphic()
        assert gmA.subgraph_is_monomorphic()
        assert gmB.subgraph_is_monomorphic()


def test_noncomparable_nodes():
    node1 = object()
    node2 = object()
    node3 = object()

    # Graph
    G = nx.path_graph([node1, node2, node3])
    gm = iso.GraphMatcher(G, G)
    assert gm.is_isomorphic()
    assert gm.subgraph_is_monomorphic()

    # DiGraph
    G = nx.path_graph([node1, node2, node3], create_using=nx.DiGraph)
    H = nx.path_graph([node3, node2, node1], create_using=nx.DiGraph)
    dgm = iso.DiGraphMatcher(G, H)
    assert dgm.is_isomorphic()
    assert dgm.subgraph_is_monomorphic()


def test_monomorphism_edge_match():
    G = nx.DiGraph()
    G.add_node(1)
    G.add_node(2)
    G.add_edge(1, 2, label="A")
    G.add_edge(2, 1, label="B")
    G.add_edge(2, 2, label="C")

    SG = nx.DiGraph()
    SG.add_node(5)
    SG.add_node(6)
    SG.add_edge(5, 6, label="A")

    gm = iso.DiGraphMatcher(G, SG, edge_match=iso.categorical_edge_match("label", None))
    assert gm.subgraph_is_monomorphic()
    G.edges[1, 2]["label"] = None
    assert not gm.subgraph_is_monomorphic()


@pytest.mark.parametrize(
    ("e1", "e2", "isomorphic", "subgraph_is_isomorphic"),
    [
        ([(0, 1), (0, 2)], [(0, 1), (0, 2), (1, 2)], False, False),
        ([(0, 1), (0, 2)], [(0, 1), (0, 2), (2, 1)], False, False),
        ([(0, 1), (1, 2)], [(0, 1), (1, 2), (2, 0)], False, False),
        ([(0, 1)], [(0, 1), (1, 2), (2, 0)], False, False),
        ([(0, 1), (1, 2), (2, 0)], [(0, 1)], False, True),
        ([(0, 1), (1, 2)], [(0, 1), (2, 0), (2, 1)], False, False),
        ([(0, 1), (0, 2), (1, 2)], [(0, 1), (0, 2), (2, 1)], True, True),
        ([(0, 1), (0, 2), (1, 2)], [(0, 1), (1, 2), (2, 0)], False, False),
        (
            [(0, 0), (0, 1), (1, 2), (2, 1)],
            [(0, 1), (1, 0), (1, 2), (2, 0)],
            False,
            False,
        ),
        (
            [(0, 0), (1, 0), (1, 2), (2, 1)],
            [(0, 1), (0, 2), (1, 0), (1, 2)],
            False,
            False,
        ),
        (
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)],
            [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
            False,
            False,
        ),
    ],
)
def test_three_node(e1, e2, isomorphic, subgraph_is_isomorphic):
    """Test some edge cases distilled from arbitrary search of the input space."""
    G1 = nx.DiGraph(e1)
    G2 = nx.DiGraph(e2)
    gm = iso.DiGraphMatcher(G1, G2)
    assert gm.is_isomorphic() == isomorphic
    assert gm.subgraph_is_isomorphic() == subgraph_is_isomorphic
