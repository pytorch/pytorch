# Owner(s): ["module: dsl-native-ops"]

from unittest.mock import MagicMock, patch

import torch._native.registry as registry_module
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


@skipIfTorchDynamo("Registry tests don't need dynamo compilation")
class TestRegistry(TestCase):
    """Tests for the torch._native.registry module."""

    def setUp(self):
        """Clean up registry state before each test."""
        self.registry = registry_module

        # Store original state for restoration
        self._original_libs = dict(self.registry._libs)
        self._original_graphs = dict(self.registry._graphs)
        self._original_dsl_name_to_lib_graph = {
            k: list(v) for k, v in self.registry._dsl_name_to_lib_graph.items()
        }
        self._original_dispatch_key_to_lib_graph = {
            k: list(v) for k, v in self.registry._dispatch_key_to_lib_graph.items()
        }
        self._original_op_symbol_to_lib_graph = {
            k: list(v) for k, v in self.registry._op_symbol_to_lib_graph.items()
        }

        # Store original filter state
        self._original_filter_state = (
            set(self.registry._filter_state._dsl_names),
            set(self.registry._filter_state._op_symbols),
            set(self.registry._filter_state._dispatch_keys),
        )

        # Clear global state
        self.registry._libs.clear()
        self.registry._graphs.clear()
        self.registry._dsl_name_to_lib_graph.clear()
        self.registry._dispatch_key_to_lib_graph.clear()
        self.registry._op_symbol_to_lib_graph.clear()

        # Clear filter state to ensure clean start
        self.registry._filter_state._dsl_names.clear()
        self.registry._filter_state._op_symbols.clear()
        self.registry._filter_state._dispatch_keys.clear()

    def tearDown(self):
        """Restore original registry state after each test."""
        if hasattr(self, "registry"):
            # Restore original state
            self.registry._libs.clear()
            self.registry._libs.update(self._original_libs)

            self.registry._graphs.clear()
            self.registry._graphs.update(self._original_graphs)

            # Properly restore mapping dictionaries with new list instances
            self.registry._dsl_name_to_lib_graph.clear()
            for k, v in self._original_dsl_name_to_lib_graph.items():
                self.registry._dsl_name_to_lib_graph[k] = list(v)

            self.registry._dispatch_key_to_lib_graph.clear()
            for k, v in self._original_dispatch_key_to_lib_graph.items():
                self.registry._dispatch_key_to_lib_graph[k] = list(v)

            self.registry._op_symbol_to_lib_graph.clear()
            for k, v in self._original_op_symbol_to_lib_graph.items():
                self.registry._op_symbol_to_lib_graph[k] = list(v)

            # Restore filter state
            self.registry._filter_state._dsl_names.clear()
            self.registry._filter_state._op_symbols.clear()
            self.registry._filter_state._dispatch_keys.clear()
            self.registry._filter_state._dsl_names.update(
                self._original_filter_state[0]
            )
            self.registry._filter_state._op_symbols.update(
                self._original_filter_state[1]
            )
            self.registry._filter_state._dispatch_keys.update(
                self._original_filter_state[2]
            )

    # Keep essential existing tests
    def test_override_node_dataclass(self):
        """Test _OverrideNode dataclass creation and defaults."""

        def test_fn(x):
            return x

        node = self.registry._OverrideNode("test_dsl", "add.Tensor", "CPU", test_fn)
        self.assertEqual(node.dsl_name, "test_dsl")
        self.assertEqual(node.op_symbol, "add.Tensor")
        self.assertEqual(node.dispatch_key, "CPU")
        self.assertEqual(node.override_fn, test_fn)
        self.assertFalse(node.unconditional_override)
        self.assertTrue(node.active)

    @patch("torch.library.Library")
    def test_register_op_override_basic(self, mock_library_cls):
        """Test basic register_op_override functionality."""

        def impl_fn(x):
            return x

        mock_lib = MagicMock()
        mock_library_cls.return_value = mock_lib

        self.registry.register_op_override(
            "test_backend", "aten", "add.Tensor", "CPU", impl_fn
        )

        key = ("add.Tensor", "CPU")
        self.assertEqual(len(self.registry._graphs[key]), 1)
        node = self.registry._graphs[key][0]
        self.assertEqual(node.dsl_name, "test_backend")
        self.assertEqual(node.override_fn, impl_fn)

    @patch("torch.library.Library")
    def test_deregister_op_overrides_basic(self, mock_library_cls):
        """Test basic deregister_op_overrides functionality."""

        def impl_fn(x):
            return x

        mock_lib = MagicMock()
        mock_library_cls.return_value = mock_lib

        # Register first
        self.registry.register_op_override(
            "test_backend", "aten", "mul.Tensor", "CPU", impl_fn
        )

        key = ("mul.Tensor", "CPU")
        self.assertTrue(self.registry._graphs[key][0].active)

        # Then deregister
        self.registry.deregister_op_overrides(disable_dsl_names="test_backend")
        self.assertFalse(self.registry._graphs[key][0].active)

    # NEW FUNCTIONALITY TESTS - ONLY THE ESSENTIAL ONES

    def test_reorder_graphs_from_user_function_basic(self):
        """Test basic graph reordering functionality."""
        # Set up test data
        key = ("test_reorder.Tensor", "CPU")

        def impl_fn(x):
            return x

        # Create nodes in specific order
        nodes = [
            self.registry._OverrideNode("dsl_c", "test_reorder.Tensor", "CPU", impl_fn),
            self.registry._OverrideNode("dsl_a", "test_reorder.Tensor", "CPU", impl_fn),
            self.registry._OverrideNode("dsl_b", "test_reorder.Tensor", "CPU", impl_fn),
        ]
        self.registry._graphs[key] = nodes

        # Define alphabetical ordering function
        def alphabetical_order(op_symbol, dispatch_key, graph):
            return sorted(graph, key=lambda n: n.dsl_name)

        # Apply reordering
        self.registry.reorder_graphs_from_user_function(alphabetical_order)

        # Verify alphabetical order
        reordered_graph = self.registry._graphs[key]
        actual_names = [node.dsl_name for node in reordered_graph]
        self.assertEqual(actual_names, ["dsl_a", "dsl_b", "dsl_c"])

    def test_reorder_graphs_from_user_function_error_handling(self):
        """Test error handling in graph reordering."""
        # Set up test data
        key = ("test_error.Tensor", "CPU")

        def impl_fn(x):
            return x

        node = self.registry._OverrideNode(
            "test_dsl", "test_error.Tensor", "CPU", impl_fn
        )
        original_graph = [node]
        self.registry._graphs[key] = original_graph.copy()

        # Define failing ordering function
        def failing_order_fn(op_symbol, dispatch_key, graph):
            raise ValueError("Test exception")

        # Should handle the exception gracefully
        with self.assertLogs("torch._native.registry", level="WARNING") as log:
            self.registry.reorder_graphs_from_user_function(failing_order_fn)

        # Verify warning was logged and original graph preserved
        self.assertEqual(len(log.records), 1)
        self.assertIn("Graph transformation failed", log.records[0].getMessage())
        self.assertEqual(self.registry._graphs[key], original_graph)

    def test_get_user_ordering_fn_env_var_not_set(self):
        """Test behavior when environment variable is not set."""
        with patch.dict("os.environ", {}, clear=True):
            from torch._native import get_user_ordering_fn

            get_user_ordering_fn.cache_clear()
            result = get_user_ordering_fn()
            self.assertIsNone(result)

    def test_get_user_ordering_fn_invalid_path(self):
        """Test handling of invalid environment variable paths."""
        with patch.dict(
            "os.environ",
            {"TORCH_PYTHON_NATIVE_USER_GRAPH_ORDER_FN": "nonexistent.module.function"},
        ):
            from torch._native import get_user_ordering_fn

            get_user_ordering_fn.cache_clear()

            with self.assertRaises(ValueError) as cm:
                get_user_ordering_fn()
            self.assertIn("Could not resolve", str(cm.exception))

    def test_integration_reorder_and_register(self):
        """Integration test: reorder then register functionality."""

        def impl_fn1(x):
            return x + 1

        def impl_fn2(x):
            return x + 2

        # Register multiple overrides
        self.registry.register_op_override(
            "backend_z", "aten", "test.Tensor", "CPU", impl_fn1
        )
        self.registry.register_op_override(
            "backend_a", "aten", "test.Tensor", "CPU", impl_fn2
        )

        key = ("test.Tensor", "CPU")

        # Verify initial order
        initial_names = [node.dsl_name for node in self.registry._graphs[key]]
        self.assertEqual(initial_names, ["backend_z", "backend_a"])

        # Reorder alphabetically
        def alphabetical_order(op_symbol, dispatch_key, graph):
            return sorted(graph, key=lambda n: n.dsl_name)

        self.registry.reorder_graphs_from_user_function(alphabetical_order)

        # Verify reordered
        final_names = [node.dsl_name for node in self.registry._graphs[key]]
        self.assertEqual(final_names, ["backend_a", "backend_z"])


if __name__ == "__main__":
    run_tests()
