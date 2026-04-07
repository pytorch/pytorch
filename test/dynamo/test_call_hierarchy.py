# Owner(s): ["module: dynamo"]
"""Tests for call_hierarchy metadata on FX nodes.
call_hierarchy is a structured metadata field that provides a unified
module+function call hierarchy for each FX node, built during Dynamo's
tx chain walk. It enables compiler backends to produce profiler-quality
hierarchy strings without parsing the unstructured stack_trace string.
"""

import torch
import torch._dynamo.config
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase, run_tests


def forward(x):
    """Standalone function named 'forward' — not a module method."""
    return x * 3


def helper_fn(x):
    return x * 2 + 1


class SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)


class NestedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = SimpleLinear()
        self.layer2 = nn.Linear(4, 2)

    def forward(self, x):
        x = self.layer1(x)
        return self.layer2(x)


class ModelWithHelper(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        x = self.linear(x)
        x = helper_fn(x)
        return x


class SharedModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(4, 4)

    def forward(self, x):
        x = self.shared(x)
        x = self.shared(x)
        return x


class PureFunctional(nn.Module):
    def forward(self, x):
        return x * 2 + 1


class ModelCallingForwardFunc(nn.Module):
    """Calls a standalone function named 'forward'."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        x = self.linear(x)
        x = forward(x)
        return x


def _get_nodes_with_call_hierarchy(model, x):
    """Compile model and return nodes that have call_hierarchy metadata."""
    nodes = []

    def backend(gm, example_inputs):
        for node in gm.graph.nodes:
            if "call_hierarchy" in node.meta:
                nodes.append(node)
        return gm

    compiled = torch.compile(model, backend=backend)
    compiled(x)
    return nodes


class TestCallHierarchy(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        self._orig = torch._dynamo.config.record_call_hierarchy
        torch._dynamo.config.record_call_hierarchy = True

    def tearDown(self):
        torch._dynamo.config.record_call_hierarchy = self._orig
        torch._dynamo.reset()
        super().tearDown()

    def test_disabled_by_default(self):
        """No call_hierarchy when config flag is off."""
        torch._dynamo.config.record_call_hierarchy = False
        model = SimpleLinear()
        x = torch.randn(2, 4)
        nodes = _get_nodes_with_call_hierarchy(model, x)
        self.assertEqual(len(nodes), 0)

    def test_populated_for_module_ops(self):
        model = SimpleLinear()
        x = torch.randn(2, 4)
        nodes = _get_nodes_with_call_hierarchy(model, x)
        self.assertTrue(len(nodes) > 0)

    def test_module_entry_structure(self):
        model = SimpleLinear()
        x = torch.randn(2, 4)
        nodes = _get_nodes_with_call_hierarchy(model, x)

        for node in nodes:
            for entry in node.meta["call_hierarchy"]:
                self.assertIn("type", entry)
                if entry["type"] == "module":
                    self.assertIn("class", entry)
                    self.assertIn("attr", entry)
                    self.assertIn("count", entry)

    def test_function_entry_structure(self):
        model = ModelWithHelper()
        x = torch.randn(2, 4)
        nodes = _get_nodes_with_call_hierarchy(model, x)

        has_function = False
        for node in nodes:
            for entry in node.meta["call_hierarchy"]:
                if entry["type"] == "function":
                    self.assertIn("name", entry)
                    self.assertIn("count", entry)
                    has_function = True
        self.assertTrue(has_function)

    def test_nested_exact_hierarchy(self):
        """Assert exact module nesting for a known model."""
        model = NestedModel()
        x = torch.randn(2, 4)
        nodes = _get_nodes_with_call_hierarchy(model, x)

        # Find a node from the inner linear of layer1 (SimpleLinear.linear).
        # Its hierarchy should contain SimpleLinear > Linear.
        found = False
        for node in nodes:
            hierarchy = node.meta["call_hierarchy"]
            classes = [e["class"] for e in hierarchy if e["type"] == "module"]
            if "SimpleLinear" in classes and "Linear" in classes:
                idx = classes.index("SimpleLinear")
                self.assertEqual(classes[idx:], ["SimpleLinear", "Linear"])
                found = True
                break
        self.assertTrue(found, "Expected SimpleLinear > Linear nesting")

    def test_helper_function_appears(self):
        model = ModelWithHelper()
        x = torch.randn(2, 4)
        nodes = _get_nodes_with_call_hierarchy(model, x)

        func_names = set()
        for node in nodes:
            for entry in node.meta["call_hierarchy"]:
                if entry["type"] == "function":
                    func_names.add(entry["name"])
        self.assertIn("helper_fn", func_names)

    def test_no_lself_prefix(self):
        model = SimpleLinear()
        x = torch.randn(2, 4)
        nodes = _get_nodes_with_call_hierarchy(model, x)

        for node in nodes:
            for entry in node.meta["call_hierarchy"]:
                if entry["type"] == "module":
                    self.assertNotIn("L['self']", entry["attr"])

    def test_pure_functional_no_module_hierarchy(self):
        """A module with no sub-modules produces no module-type entries
        in call_hierarchy."""
        model = PureFunctional()
        x = torch.randn(2, 4)
        nodes = _get_nodes_with_call_hierarchy(model, x)

        for node in nodes:
            module_entries = [
                e for e in node.meta["call_hierarchy"] if e["type"] == "module"
            ]
            self.assertEqual(len(module_entries), 0)

    def test_shared_module_call_count(self):
        """Shared module called twice should have count > 0 on second call."""
        model = SharedModule()
        x = torch.randn(2, 4)
        nodes = _get_nodes_with_call_hierarchy(model, x)

        counts = []
        for node in nodes:
            for entry in node.meta["call_hierarchy"]:
                if entry["type"] == "module" and entry["class"] == "Linear":
                    counts.append(entry["count"])
        # We should see at least one invocation with count=0 and one with count=1
        self.assertIn(0, counts)
        self.assertIn(1, counts)

    def test_sibling_modules_both_detected(self):
        """Two sibling modules at the same depth should both appear as module
        entries (regression test for depth-based detection bug)."""
        model = NestedModel()
        x = torch.randn(2, 4)
        nodes = _get_nodes_with_call_hierarchy(model, x)

        # layer2 is a sibling of layer1 at the same depth.  Nodes from layer2
        # should have a hierarchy containing NestedModel > Linear with
        # attr="layer2".
        found_layer2 = False
        for node in nodes:
            hierarchy = node.meta["call_hierarchy"]
            for entry in hierarchy:
                if (
                    entry["type"] == "module"
                    and entry["class"] == "Linear"
                    and entry["attr"] == "layer2"
                ):
                    found_layer2 = True
                    break
        self.assertTrue(found_layer2, "layer2 (sibling module) not detected")

    def test_function_count_zero_indexed(self):
        """Function call counts should be 0-indexed, matching module convention."""
        model = ModelWithHelper()
        x = torch.randn(2, 4)
        nodes = _get_nodes_with_call_hierarchy(model, x)

        for node in nodes:
            for entry in node.meta["call_hierarchy"]:
                if entry["type"] == "function" and entry["name"] == "helper_fn":
                    self.assertEqual(entry["count"], 0)

    def test_export_no_crash(self):
        """call_hierarchy should not interfere with torch.export."""
        model = NestedModel()
        x = torch.randn(2, 4)
        exported = torch.export.export(model, (x,))
        # Verify export succeeds and produces a valid graph.
        self.assertIsNotNone(exported)
        result = exported.module()(x)
        expected = model(x)
        self.assertEqual(result, expected)

    def test_graph_break_preserves_hierarchy(self):
        """Nodes in both subgraphs around a graph break get valid hierarchy."""

        class ModelWithBreak(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(4, 4)
                self.linear2 = nn.Linear(4, 4)

            def forward(self, x):
                x = self.linear1(x)
                torch._dynamo.graph_break()
                x = self.linear2(x)
                return x

        model = ModelWithBreak()
        x = torch.randn(2, 4)

        all_nodes = []

        def backend(gm, example_inputs):
            for node in gm.graph.nodes:
                if "call_hierarchy" in node.meta:
                    all_nodes.append(node)
            return gm

        compiled = torch.compile(model, backend=backend)
        compiled(x)

        # Should have nodes from both subgraphs (before and after break).
        attrs = set()
        for node in all_nodes:
            for entry in node.meta["call_hierarchy"]:
                if entry["type"] == "module":
                    attrs.add(entry["attr"])
        self.assertIn("linear1", attrs)
        self.assertIn("linear2", attrs)
        # All entries should be structurally valid.
        for node in all_nodes:
            for entry in node.meta["call_hierarchy"]:
                self.assertIn(entry["type"], ("module", "function"))

    def test_module_forward_not_double_counted(self):
        """Module's forward method should appear only as a module entry,
        not duplicated as a function entry."""
        model = SimpleLinear()
        x = torch.randn(2, 4)
        nodes = _get_nodes_with_call_hierarchy(model, x)

        for node in nodes:
            func_names = [
                e["name"]
                for e in node.meta["call_hierarchy"]
                if e["type"] == "function"
            ]
            self.assertNotIn("forward", func_names)

    def test_standalone_forward_function_captured(self):
        """A standalone function named 'forward' (not a module method)
        should appear as a function entry, not be suppressed."""
        model = ModelCallingForwardFunc()
        x = torch.randn(2, 4)
        nodes = _get_nodes_with_call_hierarchy(model, x)

        found = False
        for node in nodes:
            for entry in node.meta["call_hierarchy"]:
                if entry["type"] == "function" and entry["name"] == "forward":
                    found = True
                    break
        self.assertTrue(found, "standalone forward() function not captured")


if __name__ == "__main__":
    run_tests()