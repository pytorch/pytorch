# Owner(s): ["module: inductor"]
import operator

import torch
import torch.fx as fx
from torch._inductor.augmented_graph_helper import AugmentedGraphHelper
from torch.testing._internal.common_utils import TestCase
from torch.utils._ordered_set import OrderedSet


class TestAugmentedGraphHelper(TestCase):
    """Test suite for AugmentedGraphHelper dependency and merge management."""

    def setUp(self):
        """Create a simple graph structure for testing."""
        super().setUp()
        # Create a torch.fx.Graph with multiple nodes
        self.graph = fx.Graph()

        # Create placeholder nodes (inputs)
        self.x = self.graph.placeholder("x")
        self.y = self.graph.placeholder("y")

        # Create computation nodes with specific names for easy reference
        self.node_a = self.graph.call_function(
            torch.add, args=(self.x, self.y), name="A"
        )
        self.node_b = self.graph.call_function(
            torch.mul, args=(self.node_a, self.x), name="B"
        )
        self.node_c = self.graph.call_function(
            torch.sub, args=(self.node_a, self.y), name="C"
        )
        self.node_d = self.graph.call_function(
            torch.div, args=(self.node_b, self.node_c), name="D"
        )
        self.node_e = self.graph.call_function(
            operator.neg, args=(self.node_d,), name="E"
        )
        self.node_f = self.graph.call_function(torch.abs, args=(self.node_e,), name="F")
        self.node_g = self.graph.call_function(
            torch.relu, args=(self.node_f,), name="G"
        )
        self.node_h = self.graph.call_function(
            torch.sigmoid, args=(self.node_g,), name="H"
        )

        # Create output
        self.graph.output(self.node_h)

        # Create a mapping of nodes by name for easier access in tests
        self.nodes = {}
        for node in self.graph.nodes:
            if hasattr(node, "name") and node.name in [
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
            ]:
                self.nodes[node.name] = node

        # Get all nodes and compute ancestors
        self.all_nodes = list(self.graph.nodes)
        self.node_ancestors = self._collect_node_ancestors(self.graph)

        # Create tracker with ancestors
        self.tracker = AugmentedGraphHelper(
            self.graph, node_ancestors=self.node_ancestors
        )

    def _collect_node_ancestors(
        self, graph: fx.Graph
    ) -> dict[fx.Node, OrderedSet[fx.Node]]:
        """Collect all ancestors for each node."""
        from collections import defaultdict

        from torch.utils._ordered_set import OrderedSet

        ancestors: dict[fx.Node, OrderedSet[fx.Node]] = defaultdict(OrderedSet)
        for node in graph.nodes:
            for input_node in node.all_input_nodes:
                ancestors[node].add(input_node)
                ancestors[node] |= ancestors[input_node]
        return ancestors

    def get_deps(self, node):
        """Helper to get dependencies for a node."""
        return list(getattr(node, "args", []))

    # ========== Basic Functionality Tests ==========

    def test_initial_state(self):
        """Test that nodes start as singletons."""
        for node in self.all_nodes:
            merge_set = self.tracker.merge_sets[node]
            self.assertEqual(merge_set, {node})
            self.assertEqual(len(merge_set), 1)

    def test_simple_merge(self):
        """Test merging two nodes."""
        node_a = self.nodes["A"]
        node_b = self.nodes["B"]

        self.merge_nodes(self.tracker, [node_a, node_b])

        # Both should be in same merge set
        self.assertEqual(self.tracker.merge_sets[node_a], {node_a, node_b})
        self.assertEqual(self.tracker.merge_sets[node_b], {node_a, node_b})
        self.assertEqual(
            self.tracker.merge_sets[node_a], self.tracker.merge_sets[node_b]
        )

    def test_transitive_merge(self):
        """Test merging already merged nodes."""
        node_a = self.nodes["A"]
        node_b = self.nodes["B"]
        node_c = self.nodes["C"]
        node_d = self.nodes["D"]

        # Merge A-B and C-D separately
        for node in node_b, node_c, node_d:
            self.tracker.merge_to_set(node_a, node)

        expected_set = {node_a, node_b, node_c, node_d}
        for node in [node_a, node_b, node_c, node_d]:
            self.assertEqual(self.tracker.merge_sets[node], expected_set)

    def merge_nodes(self, tracker, nodes):
        for n in nodes[1:]:
            tracker.merge_to_set(nodes[0], n)

    def test_unmerge_node(self):
        """Test removing a node from its merge set."""
        node_a = self.nodes["A"]
        node_b = self.nodes["B"]
        node_c = self.nodes["C"]

        # Merge all three
        self.merge_nodes(self.tracker, [node_a, node_b, node_c])
        self.assertEqual(len(self.tracker.merge_sets[node_a]), 3)

        # Unmerge B
        self.tracker.unmerge_node(node_b)

        # B should be singleton
        self.assertEqual(self.tracker.merge_sets[node_b], {node_b})

        # A and C should still be together
        self.assertEqual(self.tracker.merge_sets[node_a], {node_a, node_c})
        self.assertEqual(self.tracker.merge_sets[node_c], {node_a, node_c})

    def test_unmerge_from_singleton(self):
        """Test unmerging a node that's already singleton."""
        node_a = self.nodes["A"]

        # Should be no-op
        self.tracker.unmerge_node(node_a)
        self.assertEqual(self.tracker.merge_sets[node_a], {node_a})

    # ========== Dependency Propagation Tests ==========

    def test_merged_deps_collection(self):
        """Test that dependencies are collected from all merged nodes."""
        node_a = self.nodes["A"]
        node_b = self.nodes["B"]
        node_c = self.nodes["C"]

        # B already depends on A (and x) from graph construction
        # C already depends on A (and y) from graph construction

        # Merge B and C
        self.merge_nodes(self.tracker, [node_b, node_c])

        # Get merged deps for B - should include deps from both B and C
        deps = self.tracker.get_merged_deps(node_b)

        # Should include all dependencies from both nodes
        self.assertIn(node_a, deps)  # From both B and C
        self.assertIn(self.x, deps)  # From B
        self.assertIn(self.y, deps)  # From C

    def test_extra_deps_with_merge(self):
        """Test extra dependencies work correctly with merged nodes."""
        node_a = self.nodes["A"]
        node_b = self.nodes["B"]
        node_c = self.nodes["C"]
        node_d = self.nodes["D"]

        # Add extra dep from A to C
        self.tracker.add_extra_dep(n=node_a, dep=node_c)

        # Merge A and B
        self.merge_nodes(self.tracker, [node_a, node_b])

        # Add extra dep from D to the merged node (via B)
        self.tracker.add_extra_dep(n=node_d, dep=node_b)

        # D should depend on B through extra deps
        deps = self.tracker.get_merged_deps(node_d)
        self.assertIn(node_b, deps)

        # A should still have its dep on C
        deps = self.tracker.get_merged_deps(node_a)
        self.assertIn(node_c, deps)

    # ========== Path Finding Tests ==========

    def test_has_path_direct(self):
        """Test path finding for direct dependencies."""
        # In our graph: B depends on A
        node_a = self.nodes["A"]
        node_b = self.nodes["B"]

        self.assertTrue(self.tracker.has_path(node_a, node_b))
        self.assertFalse(self.tracker.has_path(node_b, node_a))

    def test_has_path_transitive(self):
        """Test path finding through multiple nodes."""
        # In our graph: A -> B -> D and A -> C -> D -> E
        node_a = self.nodes["A"]
        node_e = self.nodes["E"]

        self.assertTrue(self.tracker.has_path(node_a, node_e))
        self.assertFalse(self.tracker.has_path(node_e, node_a))

    def test_has_path_through_merge(self):
        """Test path finding when nodes are merged."""
        # Create a new graph for this specific test
        graph2 = fx.Graph()
        x2 = graph2.placeholder("x")
        a2 = graph2.call_function(torch.neg, args=(x2,), name="A2")
        b2 = graph2.call_function(torch.abs, args=(a2,), name="B2")
        c2 = graph2.call_function(torch.relu, args=(x2,), name="C2")
        d2 = graph2.call_function(torch.sigmoid, args=(c2,), name="D2")
        graph2.output(d2)

        tracker2 = AugmentedGraphHelper(graph2)

        # Initially no path from B2 to D2
        self.assertFalse(tracker2.has_path(b2, d2))

        # Merge B2 and C2
        tracker2.merge_to_set(b2, c2)

        # Now there should be a path B2/C2 -> D2
        self.assertTrue(tracker2.has_path(b2, d2))

    def test_has_path_with_extra_deps(self):
        """Test path finding with extra dependencies."""

        graph2 = fx.Graph()
        x2 = graph2.placeholder("x")
        a2 = graph2.call_function(torch.neg, args=(x2,), name="A2")
        b2 = graph2.call_function(torch.abs, args=(a2,), name="B2")
        c2 = graph2.call_function(torch.relu, args=(x2,), name="C2")
        d2 = graph2.call_function(torch.sigmoid, args=(c2,), name="D2")
        graph2.output(d2)

        tracker2 = AugmentedGraphHelper(graph2)

        # Initially no path from B2 to D2
        self.assertFalse(tracker2.has_path(b2, d2))

        tracker2.add_extra_dep(n=c2, dep=b2)

        # Now there should be a path B2/C2 -> D2
        self.assertTrue(tracker2.has_path(b2, d2))

    # ========== Cycle Detection Tests ==========

    def test_no_cycle_in_dag(self):
        """Test that DAG has no cycles."""
        # Our original graph is a DAG, should have no cycles
        self.assertFalse(self.tracker.has_cycle())

    def test_simple_cycle_detection(self):
        """Test detection of simple cycle."""
        # Create a graph with a cycle
        graph3 = fx.Graph()
        x3 = graph3.placeholder("x")

        # We can't create true cycles in fx.Graph directly,
        # but we can simulate with extra_deps
        a3 = graph3.call_function(torch.neg, args=(x3,))
        b3 = graph3.call_function(torch.abs, args=(a3,))
        c3 = graph3.call_function(torch.relu, args=(b3,))
        graph3.output(c3)

        tracker3 = AugmentedGraphHelper(graph3)
        self.assertFalse(tracker3.has_cycle())

        # Add extra dep to create cycle: a3 -> c3
        tracker3.add_extra_dep(n=a3, dep=c3)

        self.assertTrue(tracker3.has_cycle())

    def test_cycle_through_merge(self):
        """Test that merging can create cycles."""
        # Create specific graph for this test
        graph4 = fx.Graph()
        x4 = graph4.placeholder("x")
        a4 = graph4.call_function(torch.neg, args=(x4,))
        b4 = graph4.call_function(torch.abs, args=(a4,))
        c4 = graph4.call_function(torch.relu, args=(x4,))
        d4 = graph4.call_function(torch.sigmoid, args=(c4,))
        graph4.output(d4)

        tracker4 = AugmentedGraphHelper(graph4)

        # Add extra dep d4 -> a4
        tracker4.add_extra_dep(n=a4, dep=d4)

        # Now: a4 -> b4, c4 -> d4 -> a4
        # Merging b4 and c4 would create cycle
        tracker4.merge_to_set(b4, c4)

        self.assertTrue(tracker4.has_cycle())

    def test_cycle_with_extra_deps(self):
        """Test cycle detection with extra dependencies."""
        node_a = self.nodes["A"]
        node_b = self.nodes["B"]

        # B already depends on A naturally
        # Add reverse dependency to create cycle
        self.tracker.add_extra_dep(n=node_a, dep=node_b)

        self.assertTrue(self.tracker.has_cycle())

    def test_multiple_merge_unmerge(self):
        """Test sequence of merge and unmerge operations."""
        nodes = [self.nodes[c] for c in ["A", "B", "C", "D", "E"]]

        # Merge A, B, C
        self.merge_nodes(self.tracker, nodes[:3])
        self.assertEqual(len(self.tracker.merge_sets[nodes[0]]), 3)

        # Merge D, E
        self.merge_nodes(self.tracker, nodes[3:5])
        self.assertEqual(len(self.tracker.merge_sets[nodes[3]]), 2)

        # Merge the two groups via B and D
        try:
            self.merge_nodes(self.tracker, [nodes[1], nodes[3]])
            thrown = False
        except AssertionError:
            thrown = True
        self.assertTrue(thrown)

        # Unmerge C
        self.tracker.unmerge_node(nodes[2])
        self.assertEqual(len(self.tracker.merge_sets[nodes[0]]), 2)
        self.assertEqual(self.tracker.merge_sets[nodes[2]], {nodes[2]})

        # Unmerge A
        self.tracker.unmerge_node(nodes[0])
        self.assertEqual(self.tracker.merge_sets[nodes[0]], {nodes[0]})
        self.assertEqual(len(self.tracker.merge_sets[nodes[1]]), 1)

    # ========== Dependency Transfer Tests ==========

    def test_transfer_with_cross_deps(self):
        """Test transfer when erased nodes depend on each other."""
        # old_start -> old_wait, both get replaced
        # Should become: new_start -> new_wait
        graph = fx.Graph()
        x = graph.placeholder("x")
        old_start = graph.call_function(torch.relu, args=(x,), name="old_start")
        old_wait = graph.call_function(torch.abs, args=(x,), name="old_wait")
        compute = graph.call_function(torch.neg, args=(old_wait,), name="compute")
        graph.output(compute)

        tracker = AugmentedGraphHelper(graph)

        # Add cross-dependency: old_start -> old_wait
        tracker.add_extra_dep(n=old_wait, dep=old_start)
        # Add extra dep: compute -> old_wait
        tracker.add_extra_dep(n=compute, dep=old_wait)

        # Create replacements
        new_start = graph.call_function(torch.sigmoid, args=(x,), name="new_start")
        new_wait = graph.call_function(torch.tanh, args=(x,), name="new_wait")

        # Transfer both at once
        tracker.transfer_erased_node_deps({old_start: new_start, old_wait: new_wait})

        # new_wait should depend on new_start (cross-dep redirected correctly)
        self.assertIn(new_start, tracker.extra_deps[new_wait])

        # compute should depend on new_wait
        self.assertIn(new_wait, tracker.extra_deps[compute])

        # Old nodes should be cleaned up
        self.assertEqual(len(tracker.extra_deps[old_start]), 0)
        self.assertEqual(len(tracker.extra_deps[old_wait]), 0)
        self.assertEqual(len(tracker.extra_uses[old_start]), 0)
        self.assertEqual(len(tracker.extra_uses[old_wait]), 0)

    def test_transfer_preserves_external_deps(self):
        """Test that external dependencies are preserved correctly."""
        # external1 -> old1, old2 -> external2
        # Should become: external1 -> new1, new2 -> external2
        graph = fx.Graph()
        x = graph.placeholder("x")
        external1 = graph.call_function(torch.relu, args=(x,), name="external1")
        old1 = graph.call_function(torch.abs, args=(x,), name="old1")
        old2 = graph.call_function(torch.neg, args=(x,), name="old2")
        external2 = graph.call_function(torch.sigmoid, args=(x,), name="external2")
        graph.output(external2)

        tracker = AugmentedGraphHelper(graph)

        # Add deps: old1 -> external1, external2 -> old2
        tracker.add_extra_dep(n=old1, dep=external1)
        tracker.add_extra_dep(n=external2, dep=old2)

        # Create new nodes
        new1 = graph.call_function(torch.tanh, args=(x,), name="new1")
        new2 = graph.call_function(torch.exp, args=(x,), name="new2")

        # Transfer
        tracker.transfer_erased_node_deps({old1: new1, old2: new2})

        self.assertIn(external1, tracker.extra_deps[new1])

        self.assertIn(new2, tracker.extra_deps[external2])
        self.assertNotIn(old2, tracker.extra_deps[external2])

    def test_transfer_with_merge_sets(self):
        """Test transfer when nodes have merge sets."""
        graph = fx.Graph()
        x = graph.placeholder("x")
        old_a = graph.call_function(torch.relu, args=(x,), name="old_a")
        old_b = graph.call_function(torch.abs, args=(x,), name="old_b")
        dep = graph.call_function(torch.neg, args=(x,), name="dep")
        user = graph.call_function(torch.sigmoid, args=(x,), name="user")
        graph.output(user)

        tracker = AugmentedGraphHelper(graph)

        # Merge old_a and old_b
        tracker.merge_to_set(old_a, old_b)

        # Add deps: old_a -> dep, user -> old_a
        tracker.add_extra_dep(n=old_a, dep=dep)
        tracker.add_extra_dep(n=user, dep=old_a)

        # Create new node
        new = graph.call_function(torch.tanh, args=(x,), name="new")

        # Transfer (only need to specify one from merge set)
        tracker.transfer_erased_node_deps({old_a: new})

        # new should have dep on dep
        self.assertIn(dep, tracker.extra_deps[new])

        # user should depend on new
        self.assertIn(new, tracker.extra_deps[user])

        # Both old nodes should be cleaned up
        self.assertEqual(len(tracker.extra_deps[old_a]), 0)
        self.assertEqual(len(tracker.extra_deps[old_b]), 0)

    def test_transfer_multiple_merge_sets_with_chain(self):
        """Test transferring multiple merge sets that depend on each other.

        Setup:
            node1 (singleton)
            node2, node3 (merged)
            other_node (singleton)
            node4, node5 (merged)

        Dependencies:
            node2 -> node1
            other_node -> node3
            node4 -> other_node

        Transfer:
            (node2, node3) -> new_2_3
            (node4, node5) -> new_4_5

        Expected:
            new_2_3 -> node1
            other_node -> new_2_3
            new_4_5 -> other_node
        """
        graph = fx.Graph()
        x = graph.placeholder("x")

        # Create nodes
        node1 = graph.call_function(torch.relu, args=(x,), name="node1")
        node2 = graph.call_function(torch.abs, args=(x,), name="node2")
        node3 = graph.call_function(torch.neg, args=(x,), name="node3")
        other_node = graph.call_function(torch.sigmoid, args=(x,), name="other_node")
        node4 = graph.call_function(torch.tanh, args=(x,), name="node4")
        node5 = graph.call_function(torch.exp, args=(x,), name="node5")
        graph.output(other_node)

        tracker = AugmentedGraphHelper(graph)

        # Merge node2 and node3
        tracker.merge_to_set(node2, node3)

        # Merge node4 and node5
        tracker.merge_to_set(node4, node5)

        # Add dependencies
        tracker.add_extra_dep(n=node2, dep=node1)  # node2 -> node1
        tracker.add_extra_dep(n=other_node, dep=node3)  # other_node -> node3
        tracker.add_extra_dep(n=node4, dep=other_node)  # node4 -> other_node

        # Create replacement nodes
        new_2_3 = graph.call_function(torch.sin, args=(x,), name="new_2_3")
        new_4_5 = graph.call_function(torch.cos, args=(x,), name="new_4_5")

        # Transfer both merge sets atomically
        tracker.transfer_erased_node_deps(
            {
                node2: new_2_3,  # This will transfer both node2 and node3
                node4: new_4_5,  # This will transfer both node4 and node5
            }
        )

        # Verify: new_2_3 should depend on node1
        self.assertIn(node1, tracker.extra_deps[new_2_3])

        # Verify: other_node should depend on new_2_3 (not node3)
        self.assertIn(new_2_3, tracker.extra_deps[other_node])
        self.assertNotIn(node3, tracker.extra_deps[other_node])

        # Verify: new_4_5 should depend on other_node
        self.assertIn(other_node, tracker.extra_deps[new_4_5])

        # Verify: old nodes are cleaned up
        self.assertEqual(len(tracker.extra_deps[node2]), 0)
        self.assertEqual(len(tracker.extra_deps[node3]), 0)
        self.assertEqual(len(tracker.extra_deps[node4]), 0)
        self.assertEqual(len(tracker.extra_deps[node5]), 0)

        # Verify: bidirectional consistency
        self.assertIn(new_2_3, tracker.extra_uses[node1])
        self.assertIn(other_node, tracker.extra_uses[new_2_3])
        self.assertIn(new_4_5, tracker.extra_uses[other_node])


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
