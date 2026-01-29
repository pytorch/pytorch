# Owner(s): ["module: fx"]

import torch
from torch.fx import symbolic_trace
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import run_tests, TestCase


class TestFxGraphPrint(TestCase):
    """Tests for GraphModule.print_readable() with node_info."""

    def test_print_readable_with_callable_annotation(self):
        """Test print_readable with a callable node_info."""

        def forward(x):
            return torch.sin(x) + torch.cos(x)

        x = torch.randn(3)
        gm = make_fx(forward)(x)

        # Test with callable that returns node name
        output = gm.print_readable(
            print_output=False, node_info=lambda n: f"[{n.name}]"
        )

        # The annotation should appear in the generated code
        self.assertIn("[sin]", output)
        self.assertIn("[cos]", output)
        self.assertIn("[add]", output)

    def test_print_readable_with_callable_returning_empty(self):
        """Test print_readable with a callable that returns empty string for some nodes."""

        def forward(x):
            return torch.sin(x) + torch.cos(x)

        x = torch.randn(3)
        gm = make_fx(forward)(x)

        # Only annotate sin nodes
        def annotate_sin_only(node):
            if "sin" in node.name:
                return "[SIN_NODE]"
            return ""

        output = gm.print_readable(print_output=False, node_info=annotate_sin_only)

        self.assertIn("[SIN_NODE]", output)

    def test_print_readable_with_list_of_meta_keys(self):
        """Test print_readable with a list of meta keys to include."""

        def forward(x):
            return torch.sin(x) + torch.cos(x)

        x = torch.randn(3)
        gm = make_fx(forward)(x)

        # Manually add some meta keys to nodes for testing
        call_fn_idx = 0
        for node in gm.graph.nodes:
            if node.op == "call_function":
                node.meta["my_custom_key"] = f"value_{call_fn_idx}"
                node.meta["another_key"] = call_fn_idx * 10
                call_fn_idx += 1

        output = gm.print_readable(
            print_output=False, node_info=["my_custom_key", "another_key"]
        )

        # The actual meta values should appear in the generated code
        self.assertIn("my_custom_key: value_0", output)
        self.assertIn("another_key: 0", output)
        self.assertIn("my_custom_key: value_1", output)
        self.assertIn("another_key: 10", output)
        self.assertIn("my_custom_key: value_2", output)
        self.assertIn("another_key: 20", output)

    def test_print_readable_with_list_missing_keys(self):
        """Test print_readable with a list containing keys that don't exist in meta."""

        def forward(x):
            return torch.sin(x)

        x = torch.randn(3)
        gm = make_fx(forward)(x)

        # Add only one key
        for node in gm.graph.nodes:
            if node.op == "call_function":
                node.meta["existing_key"] = "exists"

        output = gm.print_readable(
            print_output=False,
            node_info=["existing_key", "nonexistent_key"],
        )

        # Only existing_key should appear
        self.assertIn("existing_key: exists", output)
        self.assertNotIn("nonexistent_key", output)

    def test_print_readable_with_seq_nr(self):
        """Test print_readable with seq_nr in node.meta."""

        def forward(x):
            y = torch.sin(x)
            z = torch.cos(x)
            return y + z

        x = torch.randn(3)
        gm = make_fx(forward)(x)

        # Manually set seq_nr for testing (normally set during tracing)
        seq_nr = 0
        for node in gm.graph.nodes:
            if node.op == "call_function":
                node.meta["seq_nr"] = seq_nr
                seq_nr += 1

        output = gm.print_readable(print_output=False, node_info=["seq_nr"])

        # seq_nr values should appear in the generated code
        self.assertIn("seq_nr: 0", output)
        self.assertIn("seq_nr: 1", output)
        self.assertIn("seq_nr: 2", output)

    def test_print_readable_none_annotation(self):
        """Test print_readable with node_info=None (default behavior)."""

        def forward(x):
            return torch.sin(x)

        x = torch.randn(3)
        gm = make_fx(forward)(x)

        # Default behavior - no custom annotation
        output = gm.print_readable(print_output=False, node_info=None)

        # Should still work without errors
        self.assertIn("sin", output)

    def test_print_readable_callable_with_meta_access(self):
        """Test print_readable with a callable that accesses node.meta."""

        def forward(x):
            return torch.sin(x) * 2

        x = torch.randn(3)
        gm = make_fx(forward)(x)

        # Add seq_nr and other meta
        call_fn_idx = 0
        for node in gm.graph.nodes:
            if node.op == "call_function":
                node.meta["seq_nr"] = call_fn_idx
                node.meta["importance"] = "high" if call_fn_idx == 0 else "low"
                call_fn_idx += 1

        # Callable that formats both seq_nr and importance
        def format_meta(node):
            seq_nr = node.meta.get("seq_nr", "N/A")
            importance = node.meta.get("importance", "N/A")
            return f"seq={seq_nr}, importance={importance}"

        output = gm.print_readable(print_output=False, node_info=format_meta)

        self.assertIn("seq=0, importance=high", output)
        self.assertIn("seq=1, importance=low", output)

    def test_print_readable_with_nested_modules(self):
        """Test print_readable with nested GraphModules."""

        class Inner(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x)

        class Outer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = Inner()

            def forward(self, x):
                return self.inner(torch.sin(x))

        model = Outer()
        traced = symbolic_trace(model)

        # Add meta to nodes
        for node in traced.graph.nodes:
            if node.op == "call_function":
                node.meta["layer"] = "outer"

        output = traced.print_readable(print_output=False, node_info=["layer"])

        self.assertIn("layer: outer", output)

    def test_print_readable_annotation_ordering(self):
        """Test that annotation appears before other meta info in the comment."""

        def forward(x):
            return torch.sin(x)

        x = torch.randn(3)
        gm = make_fx(forward)(x)

        # Add custom meta that would also appear via annotation
        for node in gm.graph.nodes:
            if node.op == "call_function":
                node.meta["custom"] = {"test": "value"}

        output = gm.print_readable(print_output=False, node_info=lambda n: "[FIRST]")

        # The callable annotation should appear before the custom annotation
        # Find the line with both annotations
        for line in output.split("\n"):
            if "[FIRST]" in line and "Annotation:" in line:
                first_pos = line.find("[FIRST]")
                annotation_pos = line.find("Annotation:")
                self.assertLess(
                    first_pos,
                    annotation_pos,
                    "node_info result should appear before Annotation",
                )
                break

    def test_print_readable_with_empty_list(self):
        """Test print_readable with an empty list of meta keys."""

        def forward(x):
            return torch.sin(x)

        x = torch.randn(3)
        gm = make_fx(forward)(x)

        # Empty list should not add any annotation
        output = gm.print_readable(print_output=False, node_info=[])

        # Should work without errors
        self.assertIn("sin", output)

    def test_print_readable_callable_returns_none(self):
        """Test print_readable with a callable that returns None for some nodes."""

        def forward(x):
            return torch.sin(x) + torch.cos(x)

        x = torch.randn(3)
        gm = make_fx(forward)(x)

        # Callable that returns None for non-sin nodes
        def annotate_or_none(node):
            if "sin" in node.name:
                return "[HAS_SIN]"
            return None

        output = gm.print_readable(print_output=False, node_info=annotate_or_none)

        self.assertIn("[HAS_SIN]", output)

    def test_print_readable_multiple_meta_keys_formatting(self):
        """Test that multiple meta keys are formatted correctly with commas."""

        def forward(x):
            return torch.sin(x)

        x = torch.randn(3)
        gm = make_fx(forward)(x)

        for node in gm.graph.nodes:
            if node.op == "call_function":
                node.meta["key1"] = "val1"
                node.meta["key2"] = "val2"
                node.meta["key3"] = "val3"

        output = gm.print_readable(
            print_output=False, node_info=["key1", "key2", "key3"]
        )

        # All three keys should appear comma-separated
        self.assertIn("key1: val1", output)
        self.assertIn("key2: val2", output)
        self.assertIn("key3: val3", output)
        # Check comma separation
        self.assertIn(", ", output)

    def test_print_readable_seq_nr_with_multiple_ops(self):
        """Test print_readable with seq_nr across multiple different operations."""

        def forward(x):
            a = torch.sin(x)
            b = torch.cos(x)
            c = torch.relu(a + b)
            return c

        x = torch.randn(3)
        gm = make_fx(forward)(x)

        # Manually set seq_nr
        seq_nr = 100
        for node in gm.graph.nodes:
            if node.op == "call_function":
                node.meta["seq_nr"] = seq_nr
                seq_nr += 1

        output = gm.print_readable(print_output=False, node_info=["seq_nr"])

        # Check that seq_nr values appear
        self.assertIn("seq_nr: 100", output)
        self.assertIn("seq_nr: 101", output)
        self.assertIn("seq_nr: 102", output)
        self.assertIn("seq_nr: 103", output)


if __name__ == "__main__":
    run_tests()
