from torch.package._digraph import DiGraph
from torch.testing._internal.common_utils import run_tests

try:
    from .common import PackageTestCase
except ImportError:
    # Support the case where we run this file directly.
    from common import PackageTestCase


class TestDiGraph(PackageTestCase):
    """Test the DiGraph structure we use to represent dependencies in PackageExporter"""

    def test_successors(self):
        g = DiGraph()
        g.add_edge("foo", "bar")
        g.add_edge("foo", "baz")
        g.add_node("qux")

        self.assertIn("bar", list(g.successors("foo")))
        self.assertIn("baz", list(g.successors("foo")))
        self.assertEqual(len(list(g.successors("qux"))), 0)

    def test_successor_not_in_graph(self):
        g = DiGraph()
        with self.assertRaises(ValueError):
            g.successors("not in graph")

    def test_node_attrs(self):
        g = DiGraph()
        g.add_node("foo", my_attr=1, other_attr=2)
        self.assertEqual(g.nodes["foo"]["my_attr"], 1)
        self.assertEqual(g.nodes["foo"]["other_attr"], 2)

    def test_node_attr_update(self):
        g = DiGraph()
        g.add_node("foo", my_attr=1)
        self.assertEqual(g.nodes["foo"]["my_attr"], 1)

        g.add_node("foo", my_attr="different")
        self.assertEqual(g.nodes["foo"]["my_attr"], "different")

    def test_edges(self):
        g = DiGraph()
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.add_edge(1, 3)
        g.add_edge(4, 5)

        edge_list = list(g.edges)
        self.assertEqual(len(edge_list), 4)

        self.assertIn((1, 2), edge_list)
        self.assertIn((2, 3), edge_list)
        self.assertIn((1, 3), edge_list)
        self.assertIn((4, 5), edge_list)

    def test_iter(self):
        g = DiGraph()
        g.add_node(1)
        g.add_node(2)
        g.add_node(3)

        nodes = set()
        for n in g:
            nodes.add(n)

        self.assertEqual(nodes, set([1, 2, 3]))

    def test_contains(self):
        g = DiGraph()
        g.add_node("yup")

        self.assertTrue("yup" in g)
        self.assertFalse("nup" in g)

    def test_contains_non_hashable(self):
        g = DiGraph()
        self.assertFalse([1, 2, 3] in g)


if __name__ == "__main__":
    run_tests()
