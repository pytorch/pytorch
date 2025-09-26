import unittest
from collections import OrderedDict, defaultdict
from unittest.mock import Mock, MagicMock
import torch.fx as fx
import torch
import operator
from torch._inductor.augmented_graph_helper import AugmentedGraphHelper

class TestAugmentedGraphHelper(unittest.TestCase):
    """Test suite for AugmentedGraphHelper dependency and merge management."""
    
    def setUp(self):
        """Create a simple graph structure for testing."""
        # Create a torch.fx.Graph with multiple nodes
        self.graph = fx.Graph()
        
        # Create placeholder nodes (inputs)
        self.x = self.graph.placeholder("x")
        self.y = self.graph.placeholder("y")
        
        # Create computation nodes with specific names for easy reference
        self.node_a = self.graph.call_function(torch.add, args=(self.x, self.y), name="A")
        self.node_b = self.graph.call_function(torch.mul, args=(self.node_a, self.x), name="B")
        self.node_c = self.graph.call_function(torch.sub, args=(self.node_a, self.y), name="C")
        self.node_d = self.graph.call_function(torch.div, args=(self.node_b, self.node_c), name="D")
        self.node_e = self.graph.call_function(operator.neg, args=(self.node_d,), name="E")
        self.node_f = self.graph.call_function(torch.abs, args=(self.node_e,), name="F")
        self.node_g = self.graph.call_function(torch.relu, args=(self.node_f,), name="G")
        self.node_h = self.graph.call_function(torch.sigmoid, args=(self.node_g,), name="H")
        
        # Create output
        self.graph.output(self.node_h)
        
        # Create a mapping of nodes by name for easier access in tests
        self.nodes = {}
        for node in self.graph.nodes:
            if hasattr(node, 'name') and node.name in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
                self.nodes[node.name] = node
        
        # Get all nodes and create tracker
        self.all_nodes = list(self.graph.nodes)
        self.tracker = AugmentedGraphHelper(self.graph)
        
    def get_deps(self, node):
        """Helper to get dependencies for a node."""
        return list(node.args) if hasattr(node, 'args') else []
    
    # ========== Basic Functionality Tests ==========
    
    def test_initial_state(self):
        """Test that nodes start as singletons."""
        for node in self.all_nodes:
            merge_set = self.tracker.merge_sets[node]
            self.assertEqual(merge_set, {node})
            self.assertEqual(len(merge_set), 1)

    def test_simple_merge(self):
        """Test merging two nodes."""
        node_a = self.nodes['A']
        node_b = self.nodes['B']
        
        self.tracker.merge_nodes([node_a, node_b])
        
        # Both should be in same merge set
        self.assertEqual(self.tracker.merge_sets[node_a], {node_a, node_b})
        self.assertEqual(self.tracker.merge_sets[node_b], {node_a, node_b})
        self.assertEqual(self.tracker.merge_sets[node_a], self.tracker.merge_sets[node_b])
    
    def test_transitive_merge(self):
        """Test merging already merged nodes."""
        node_a = self.nodes['A']
        node_b = self.nodes['B']
        node_c = self.nodes['C']
        node_d = self.nodes['D']
        
        # Merge A-B and C-D separately
        self.tracker.merge_nodes([node_a, node_b])
        self.tracker.merge_nodes([node_c, node_d])
        
        # Now merge B with C (should merge all 4)
        self.tracker.merge_nodes([node_b, node_c])
        
        expected_set = {node_a, node_b, node_c, node_d}
        for node in [node_a, node_b, node_c, node_d]:
            self.assertEqual(self.tracker.merge_sets[node], expected_set)
    
    def test_unmerge_node(self):
        """Test removing a node from its merge set."""
        node_a = self.nodes['A']
        node_b = self.nodes['B']
        node_c = self.nodes['C']
        
        # Merge all three
        self.tracker.merge_nodes([node_a, node_b, node_c])
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
        node_a = self.nodes['A']
        
        # Should be no-op
        self.tracker.unmerge_node(node_a)
        self.assertEqual(self.tracker.merge_sets[node_a], {node_a})
    
    # ========== Dependency Propagation Tests ==========
    
    def test_merged_deps_collection(self):
        """Test that dependencies are collected from all merged nodes."""
        node_a = self.nodes['A']
        node_b = self.nodes['B']
        node_c = self.nodes['C']
        
        # B already depends on A (and x) from graph construction
        # C already depends on A (and y) from graph construction
        
        # Merge B and C
        self.tracker.merge_nodes([node_b, node_c])
        
        # Get merged deps for B - should include deps from both B and C
        deps = self.tracker.get_merged_deps(node_b)
        
        # Should include all dependencies from both nodes
        self.assertIn(node_a, deps)  # From both B and C
        self.assertIn(self.x, deps)  # From B
        self.assertIn(self.y, deps)  # From C
    
    def test_extra_deps_with_merge(self):
        """Test extra dependencies work correctly with merged nodes."""
        node_a = self.nodes['A']
        node_b = self.nodes['B']
        node_c = self.nodes['C']
        node_d = self.nodes['D']
        
        # Add extra dep from A to C
        self.tracker.add_extra_dep(node_a, node_c)
        
        # Merge A and B
        self.tracker.merge_nodes([node_a, node_b])
        
        # Add extra dep from D to the merged node (via B)
        self.tracker.add_extra_dep(node_d, node_b)
        
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
        node_a = self.nodes['A']
        node_b = self.nodes['B']
        
        self.assertTrue(self.tracker.has_path(node_a, node_b))
        self.assertFalse(self.tracker.has_path(node_b, node_a))
    
    def test_has_path_transitive(self):
        """Test path finding through multiple nodes."""
        # In our graph: A -> B -> D and A -> C -> D -> E
        node_a = self.nodes['A']
        node_e = self.nodes['E']
        
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
        tracker2.merge_nodes([b2, c2])
        
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

        tracker2.add_extra_dep(c2, b2)
        
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
        tracker3.add_extra_dep(a3, c3)

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
        tracker4.add_extra_dep(a4, d4)
        
        # Now: a4 -> b4, c4 -> d4 -> a4
        # Merging b4 and c4 would create cycle
        tracker4.merge_nodes([b4, c4])
        
        self.assertTrue(tracker4.has_cycle())
    
    
    def test_cycle_with_extra_deps(self):
        """Test cycle detection with extra dependencies."""
        node_a = self.nodes['A']
        node_b = self.nodes['B']
        
        # B already depends on A naturally
        # Add reverse dependency to create cycle
        self.tracker.add_extra_dep(node_a, node_b)
        
        self.assertTrue(self.tracker.has_cycle())
    
    def test_multiple_merge_unmerge(self):
        """Test sequence of merge and unmerge operations."""
        nodes = [self.nodes[c] for c in ['A', 'B', 'C', 'D', 'E']]
        
        # Merge A, B, C
        self.tracker.merge_nodes(nodes[:3])
        self.assertEqual(len(self.tracker.merge_sets[nodes[0]]), 3)
        
        # Merge D, E
        self.tracker.merge_nodes(nodes[3:5])
        self.assertEqual(len(self.tracker.merge_sets[nodes[3]]), 2)
        
        # Merge the two groups via B and D
        self.tracker.merge_nodes([nodes[1], nodes[3]])
        self.assertEqual(len(self.tracker.merge_sets[nodes[0]]), 5)
        
        # Unmerge C
        self.tracker.unmerge_node(nodes[2])
        self.assertEqual(len(self.tracker.merge_sets[nodes[0]]), 4)
        self.assertEqual(self.tracker.merge_sets[nodes[2]], {nodes[2]})
        
        # Unmerge A
        self.tracker.unmerge_node(nodes[0])
        self.assertEqual(self.tracker.merge_sets[nodes[0]], {nodes[0]})
        self.assertEqual(len(self.tracker.merge_sets[nodes[1]]), 3)


if __name__ == '__main__':
    unittest.main()
