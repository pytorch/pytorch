# Owner(s): ["module: dsl-native-ops"]

from unittest.mock import MagicMock, patch

import torch._native.registry as registry_module
import torch.library
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

    def test_override_node_dataclass(self):
        """Test _OverrideNode dataclass creation and defaults."""

        def test_fn(x):
            return x

        # Test with minimal arguments
        node = self.registry._OverrideNode("test_dsl", "add.Tensor", "CPU", test_fn)
        self.assertEqual(node.dsl_name, "test_dsl")
        self.assertEqual(node.op_symbol, "add.Tensor")
        self.assertEqual(node.dispatch_key, "CPU")
        self.assertEqual(node.override_fn, test_fn)
        self.assertFalse(node.unconditional_override)
        self.assertTrue(node.active)

        # Test with all arguments
        def override_fn(x):
            return x * 2

        node = self.registry._OverrideNode(
            "another_dsl",
            "mul.Tensor",
            "CUDA",
            override_fn,
            unconditional_override=True,
            active=False,
        )
        self.assertEqual(node.dsl_name, "another_dsl")
        self.assertEqual(node.op_symbol, "mul.Tensor")
        self.assertEqual(node.dispatch_key, "CUDA")
        self.assertEqual(node.override_fn, override_fn)
        self.assertTrue(node.unconditional_override)
        self.assertFalse(node.active)

    def test_get_or_create_library_caching(self):
        """Test _get_or_create_library creates and caches Library instances."""
        # Test library creation and caching
        lib1 = self.registry._get_or_create_library("add.Tensor", "CPU")
        self.assertIsInstance(lib1, torch.library.Library)

        # Should return cached instance
        lib2 = self.registry._get_or_create_library("add.Tensor", "CPU")
        self.assertIs(lib1, lib2)

        # Different key should create different instance
        lib3 = self.registry._get_or_create_library("add.Tensor", "CUDA")
        self.assertIsNot(lib1, lib3)

        # Check that libraries are stored in _libs
        self.assertIn(("add.Tensor", "CPU"), self.registry._libs)
        self.assertIn(("add.Tensor", "CUDA"), self.registry._libs)

    def test_resolve_iterable(self):
        """Test _resolve_iterable with various input types."""
        test_cases = [
            (None, []),
            ("single_string", ["single_string"]),
            (["item1", "item2", "item3"], ["item1", "item2", "item3"]),
        ]

        for input_val, expected in test_cases:
            with self.subTest(input_val=input_val):
                result = list(self.registry._resolve_iterable(input_val))
                self.assertEqual(result, expected)

    def test_filter_functionality(self):
        """Test _filter function with various filter criteria."""
        # Test no filters raises error
        with self.assertRaises(ValueError) as cm:
            self.registry._filter("dsl", "op", "key")
        self.assertIn("Must pass 1+ of filter_", str(cm.exception))

        # Test different filter types
        filter_test_cases = [
            ("dsl_names", "test_dsl", "op", "key", "test_dsl", True),
            ("dsl_names", "test_dsl", "op", "key", ["test_dsl", "other"], True),
            ("dsl_names", "test_dsl", "op", "key", "other_dsl", False),
            ("op_symbols", "dsl", "test_op", "key", "test_op", True),
            ("op_symbols", "dsl", "test_op", "key", ["test_op", "other"], True),
            ("op_symbols", "dsl", "test_op", "key", "other_op", False),
            ("dispatch_keys", "dsl", "op", "test_key", "test_key", True),
            ("dispatch_keys", "dsl", "op", "test_key", ["test_key", "other"], True),
            ("dispatch_keys", "dsl", "op", "test_key", "other_key", False),
        ]

        for filter_type, dsl, op, key, filter_val, expected in filter_test_cases:
            with self.subTest(
                filter_type=filter_type, filter_val=filter_val, expected=expected
            ):
                kwargs = {f"filter_{filter_type}": filter_val}
                result = self.registry._filter(dsl, op, key, **kwargs)
                self.assertEqual(result, expected)

    def test_update_registration_maps(self):
        """Test _update_registration_maps updates all mapping dictionaries."""
        key = ("test_op", "CPU")
        self.registry._update_registration_maps("test_dsl", "test_op", "CPU", key)

        # Check that all mappings were updated
        self.assertIn("test_dsl", self.registry._dsl_name_to_lib_graph)
        self.assertIn("test_op", self.registry._op_symbol_to_lib_graph)
        self.assertIn("CPU", self.registry._dispatch_key_to_lib_graph)

        # Check that the key is in the mapping values
        self.assertIn(key, self.registry._dsl_name_to_lib_graph["test_dsl"])
        self.assertIn(key, self.registry._op_symbol_to_lib_graph["test_op"])
        self.assertIn(key, self.registry._dispatch_key_to_lib_graph["CPU"])

        # Test appending to existing entry
        key2 = ("test_op2", "CUDA")
        self.registry._update_registration_maps("test_dsl", "test_op2", "CUDA", key2)

        # Should have both keys for test_dsl
        self.assertEqual(len(self.registry._dsl_name_to_lib_graph["test_dsl"]), 2)
        self.assertIn(key, self.registry._dsl_name_to_lib_graph["test_dsl"])
        self.assertIn(key2, self.registry._dsl_name_to_lib_graph["test_dsl"])

    def test__build_key_set(self):
        """Test __build_key_set creates correct set of keys."""
        # Set up test data
        key1 = ("op1", "CPU")
        key2 = ("op2", "CUDA")
        self.registry._dsl_name_to_lib_graph["dsl1"] = [key1]
        self.registry._op_symbol_to_lib_graph["op2"] = [key2]
        self.registry._dispatch_key_to_lib_graph["CPU"] = [key1]

        # Test single filter
        result = self.registry._build_key_set("dsl1", None, None)
        self.assertEqual(result, {key1})

        # Test multiple filters
        result = self.registry._build_key_set("dsl1", "op2", None)
        self.assertEqual(result, {key1, key2})

    @patch("torch.library.Library")
    def test_register_op_override_variants(self, mock_library_cls):
        """Test register_op_override with different parameter combinations."""
        mock_lib = MagicMock()
        mock_library_cls.return_value = mock_lib

        def impl_fn(x):
            return x

        # Test 1: Basic registration
        self.registry.register_op_override(
            backend="test_backend",
            lib_symbol="aten",
            op_symbol="add.Tensor",
            dispatch_key="CPU",
            impl=impl_fn,
        )

        key = ("add.Tensor", "CPU")
        self.assertIn(key, self.registry._graphs)
        self.assertEqual(len(self.registry._graphs[key]), 1)

        node = self.registry._graphs[key][0]
        self.assertIsInstance(node, self.registry._OverrideNode)
        self.assertEqual(node.dsl_name, "test_backend")
        self.assertEqual(node.override_fn, impl_fn)
        self.assertFalse(node.unconditional_override)
        self.assertTrue(node.active)

        mock_lib.impl.assert_called_with(
            "add.Tensor", impl_fn, "CPU", with_keyset=True, allow_override=False
        )

        # Test 2: Unconditional override
        mock_lib.reset_mock()
        self.registry.register_op_override(
            backend="test_backend2",
            lib_symbol="aten",
            op_symbol="mul.Tensor",
            dispatch_key="CPU",
            impl=impl_fn,
            unconditional_override=True,
        )

        key2 = ("mul.Tensor", "CPU")
        node2 = self.registry._graphs[key2][0]
        self.assertTrue(node2.unconditional_override)

        mock_lib.impl.assert_called_with(
            "mul.Tensor", impl_fn, "CPU", with_keyset=False, allow_override=False
        )

        # Test 3: Allow multiple overrides
        mock_lib.reset_mock()
        self.registry.register_op_override(
            backend="test_backend3",
            lib_symbol="aten",
            op_symbol="div.Tensor",
            dispatch_key="CPU",
            impl=impl_fn,
            allow_multiple_override=True,
        )

        mock_lib.impl.assert_called_with(
            "div.Tensor", impl_fn, "CPU", with_keyset=True, allow_override=True
        )

    @patch("torch.library.Library")
    def testderegister_op_overrides(self, mock_library_cls):
        """Test deregister_op_overrides functionality."""
        # Set up test data
        key = ("add.Tensor", "CPU")

        def test_fn(x):
            return x

        node1 = self.registry._OverrideNode(
            "dsl1", "add.Tensor", "CPU", test_fn, active=True
        )
        node2 = self.registry._OverrideNode(
            "dsl2", "add.Tensor", "CPU", test_fn, active=True
        )
        self.registry._graphs[key] = [node1, node2]

        mock_old_lib = MagicMock()
        mock_new_lib = MagicMock()
        self.registry._libs[key] = mock_old_lib
        mock_library_cls.return_value = mock_new_lib

        # Set up mappings for __build_key_set to work
        self.registry._dsl_name_to_lib_graph["dsl1"] = [key]

        self.registry.deregister_op_overrides(disable_dsl_names="dsl1")

        # Check that old library was removed and new one created
        self.assertIn(key, self.registry._libs)  # Should have new library

        # Check node states - dsl1 should be inactive, dsl2 should be active
        self.assertFalse(node1.active)
        self.assertTrue(node2.active)

        # Check that only non-filtered node was re-registered
        mock_new_lib.impl.assert_called_once_with(
            "add.Tensor",
            node2.override_fn,
            "CPU",
            with_keyset=True,
            allow_override=True,
        )

    def test_print_override_graphs(self):
        """Test _print_override_graphs with different print_inactive settings."""
        # Set up test data
        key = ("add.Tensor", "CPU")

        def test_fn(x):
            return x

        active_node = self.registry._OverrideNode(
            "active_dsl", "add.Tensor", "CPU", test_fn, active=True
        )
        inactive_node = self.registry._OverrideNode(
            "inactive_dsl", "add.Tensor", "CPU", test_fn, active=False
        )
        self.registry._graphs[key] = [active_node, inactive_node]

        # Test 1: Default behavior (active only)
        with patch("builtins.print") as mock_print:
            self.registry._print_override_graphs()

        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("op='add.Tensor'" in call for call in print_calls))
        self.assertTrue(any("active_dsl" in call for call in print_calls))
        self.assertFalse(any("inactive_dsl" in call for call in print_calls))

        # Test 2: Include inactive nodes
        with patch("builtins.print") as mock_print:
            self.registry._print_override_graphs(print_inactive=True)

        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("active_dsl" in call for call in print_calls))
        self.assertTrue(any("inactive_dsl" in call for call in print_calls))
        self.assertTrue(any("active=True" in call for call in print_calls))
        self.assertTrue(any("active=False" in call for call in print_calls))

    @patch("torch.library.Library")
    def test_integration_register_and_deregister(self, mock_library_cls):
        """Integration test for register and deregister workflow."""

        def impl_fn1(x):
            return x + 1

        def impl_fn2(x):
            return x + 2

        mock_lib = MagicMock()
        mock_library_cls.return_value = mock_lib

        # Register two overrides
        self.registry.register_op_override(
            "dsl1", "aten", "add.Tensor", "CPU", impl_fn1
        )
        self.registry.register_op_override(
            "dsl2", "aten", "add.Tensor", "CPU", impl_fn2
        )

        # Check both are registered
        key = ("add.Tensor", "CPU")
        self.assertEqual(len(self.registry._graphs[key]), 2)
        self.assertTrue(all(node.active for node in self.registry._graphs[key]))

        # Deregister one
        self.registry.deregister_op_overrides(disable_dsl_names="dsl1")

        # Check that dsl1 is inactive, dsl2 is still active
        nodes = self.registry._graphs[key]
        dsl1_node = next(n for n in nodes if n.dsl_name == "dsl1")
        dsl2_node = next(n for n in nodes if n.dsl_name == "dsl2")

        self.assertFalse(dsl1_node.active)
        self.assertTrue(dsl2_node.active)

    def _setup_override_chain(
        self, backends, op_symbol, dispatch_key, mock_library_cls
    ):
        """Helper method to set up a chain of overrides for testing."""
        mock_lib = MagicMock()
        mock_library_cls.return_value = mock_lib

        impl_fns = []
        for backend in backends:

            def make_impl_fn(backend_name):
                def impl_fn(dispatch_keys, x, y):
                    return (backend_name, x, y)

                return impl_fn

            impl_fn = make_impl_fn(backend)
            impl_fns.append(impl_fn)

            self.registry.register_op_override(
                backend,
                "aten",
                op_symbol,
                dispatch_key,
                impl_fn,
                allow_multiple_override=True,
            )

        return mock_lib, impl_fns

    @patch("torch.library.Library")
    def test_override_chain_operations(self, mock_library_cls):
        """Test override chain creation and various removal scenarios."""
        # Test 1: Basic chain creation
        backends = ["backend1", "backend2", "backend3"]
        op_symbol = "mul.Tensor"
        dispatch_key = "CUDA"
        key = (op_symbol, dispatch_key)

        mock_lib, impl_fns = self._setup_override_chain(
            backends, op_symbol, dispatch_key, mock_library_cls
        )

        self.assertEqual(len(self.registry._graphs[key]), 3)
        nodes = self.registry._graphs[key]
        for i, backend in enumerate(backends):
            self.assertEqual(nodes[i].dsl_name, backend)
        self.assertTrue(all(node.active for node in nodes))

        # Test 2: Remove middle override
        self.registry.deregister_op_overrides(disable_dsl_names="backend2")
        nodes = self.registry._graphs[key]
        backend1_node = next(n for n in nodes if n.dsl_name == "backend1")
        backend2_node = next(n for n in nodes if n.dsl_name == "backend2")
        backend3_node = next(n for n in nodes if n.dsl_name == "backend3")

        self.assertTrue(backend1_node.active)
        self.assertFalse(backend2_node.active)
        self.assertTrue(backend3_node.active)

        # Test 3: Multiple removal (setup fresh chain)
        backends_multi = ["backend1", "backend2", "backend3", "backend4", "backend5"]
        op_symbol_multi = "div.Tensor"
        key_multi = (op_symbol_multi, dispatch_key)

        mock_lib2, impl_fns2 = self._setup_override_chain(
            backends_multi, op_symbol_multi, dispatch_key, mock_library_cls
        )

        self.registry.deregister_op_overrides(
            disable_dsl_names=["backend2", "backend4"]
        )
        nodes_multi = self.registry._graphs[key_multi]

        expected_active = {"backend1", "backend3", "backend5"}
        expected_inactive = {"backend2", "backend4"}

        for node in nodes_multi:
            if node.dsl_name in expected_active:
                self.assertTrue(node.active, f"{node.dsl_name} should be active")
            elif node.dsl_name in expected_inactive:
                self.assertFalse(node.active, f"{node.dsl_name} should be inactive")

        # Test 4: Remove all (using a fresh op to avoid interference)
        op_symbol_all = "sub.Tensor"
        key_all = (op_symbol_all, dispatch_key)
        backends_all = ["backend1", "backend2", "backend3"]

        mock_lib3, impl_fns3 = self._setup_override_chain(
            backends_all, op_symbol_all, dispatch_key, mock_library_cls
        )

        self.registry.deregister_op_overrides(disable_dsl_names=backends_all)
        nodes_all = self.registry._graphs[key_all]
        self.assertTrue(all(not node.active for node in nodes_all))

    @patch("torch.library.Library")
    def test_deregister_by_op_symbol_affects_all_backends(self, mock_library_cls):
        """Test deregistering by op_symbol affects all backends for that operation."""
        mock_lib = MagicMock()
        mock_library_cls.return_value = mock_lib

        # Register multiple backends for same operation
        def impl_fn1(dispatch_keys, x, y):
            return ("triton", x, y)

        def impl_fn2(dispatch_keys, x, y):
            return ("cutedsl", x, y)

        self.registry.register_op_override(
            "triton",
            "aten",
            "mul.Tensor",
            "CUDA",
            impl_fn1,
            allow_multiple_override=True,
        )
        self.registry.register_op_override(
            "cutedsl",
            "aten",
            "mul.Tensor",
            "CUDA",
            impl_fn2,
            allow_multiple_override=True,
        )

        # Also register same backends for different operation
        def add_impl_fn1(dispatch_keys, x, y):
            return ("triton", x, y)

        def add_impl_fn2(dispatch_keys, x, y):
            return ("cutedsl", x, y)

        self.registry.register_op_override(
            "triton",
            "aten",
            "add.Tensor",
            "CUDA",
            add_impl_fn1,
            allow_multiple_override=True,
        )
        self.registry.register_op_override(
            "cutedsl",
            "aten",
            "add.Tensor",
            "CUDA",
            add_impl_fn2,
            allow_multiple_override=True,
        )

        # Remove by op_symbol should affect all backends for that op only
        self.registry.deregister_op_overrides(disable_op_symbols="mul.Tensor")

        # Check mul.Tensor overrides are inactive
        mul_key = ("mul.Tensor", "CUDA")
        mul_nodes = self.registry._graphs[mul_key]
        self.assertTrue(all(not node.active for node in mul_nodes))

        # Check add.Tensor overrides are still active
        add_key = ("add.Tensor", "CUDA")
        add_nodes = self.registry._graphs[add_key]
        self.assertTrue(all(node.active for node in add_nodes))

    @patch("torch.library.Library")
    def test_deregister_by_dispatch_key_affects_all_operations(self, mock_library_cls):
        """Test deregistering by dispatch_key affects all operations for that key."""
        mock_lib = MagicMock()
        mock_library_cls.return_value = mock_lib

        # Register same backend for multiple operations and dispatch keys
        def impl_fn_cuda(dispatch_keys, x, y):
            return ("triton_cuda", x, y)

        def impl_fn_cpu(dispatch_keys, x, y):
            return ("triton_cpu", x, y)

        # mul.Tensor on CUDA and CPU
        self.registry.register_op_override(
            "triton",
            "aten",
            "mul.Tensor",
            "CUDA",
            impl_fn_cuda,
            allow_multiple_override=True,
        )
        self.registry.register_op_override(
            "triton",
            "aten",
            "mul.Tensor",
            "CPU",
            impl_fn_cpu,
            allow_multiple_override=True,
        )

        # add.Tensor on CUDA and CPU
        self.registry.register_op_override(
            "triton",
            "aten",
            "add.Tensor",
            "CUDA",
            impl_fn_cuda,
            allow_multiple_override=True,
        )
        self.registry.register_op_override(
            "triton",
            "aten",
            "add.Tensor",
            "CPU",
            impl_fn_cpu,
            allow_multiple_override=True,
        )

        # Remove by dispatch_key should affect all operations for CUDA only
        self.registry.deregister_op_overrides(disable_dispatch_keys="CUDA")

        # Check CUDA overrides are inactive
        mul_cuda_key = ("mul.Tensor", "CUDA")
        add_cuda_key = ("add.Tensor", "CUDA")

        mul_cuda_nodes = self.registry._graphs[mul_cuda_key]
        add_cuda_nodes = self.registry._graphs[add_cuda_key]

        self.assertTrue(all(not node.active for node in mul_cuda_nodes))
        self.assertTrue(all(not node.active for node in add_cuda_nodes))

        # Check CPU overrides are still active
        mul_cpu_key = ("mul.Tensor", "CPU")
        add_cpu_key = ("add.Tensor", "CPU")

        mul_cpu_nodes = self.registry._graphs[mul_cpu_key]
        add_cpu_nodes = self.registry._graphs[add_cpu_key]

        self.assertTrue(all(node.active for node in mul_cpu_nodes))
        self.assertTrue(all(node.active for node in add_cpu_nodes))

    @patch("torch.library.Library")
    def test_complex_multi_criteria_deregistration(self, mock_library_cls):
        """Test deregistration with multiple criteria (dsl_names + op_symbols + dispatch_keys)."""
        mock_lib = MagicMock()
        mock_library_cls.return_value = mock_lib

        # Create a complex scenario with multiple backends, ops, and dispatch keys
        backends = ["triton", "cutedsl", "openvino"]
        ops = ["mul.Tensor", "add.Tensor", "div.Tensor"]
        dispatch_keys = ["CUDA", "CPU"]

        # Register all combinations
        for backend in backends:
            for op in ops:
                for dispatch_key in dispatch_keys:

                    def make_impl_fn(b, o, d):
                        def impl_fn(dk, x, y):
                            return (f"{b}_{o}_{d}", x, y)

                        return impl_fn

                    impl_fn = make_impl_fn(backend, op, dispatch_key)
                    self.registry.register_op_override(
                        backend,
                        "aten",
                        op,
                        dispatch_key,
                        impl_fn,
                        allow_multiple_override=True,
                    )

        # Complex deregistration:
        # - Disable triton backend (affects all ops/dispatch_keys for triton)
        # - Disable mul.Tensor operation (affects all backends/dispatch_keys for mul.Tensor)
        # - Disable CPU dispatch key (affects all backends/ops for CPU)
        self.registry.deregister_op_overrides(
            disable_dsl_names="triton",
            disable_op_symbols="mul.Tensor",
            disable_dispatch_keys="CPU",
        )

        # Check results:
        # 1. All triton overrides should be inactive
        # 2. All mul.Tensor overrides should be inactive
        # 3. All CPU overrides should be inactive
        # 4. Only cutedsl/openvino + add.Tensor/div.Tensor + CUDA should remain active

        for op in ops:
            for dispatch_key in dispatch_keys:
                key = (op, dispatch_key)
                if key in self.registry._graphs:
                    nodes = self.registry._graphs[key]
                    for node in nodes:
                        should_be_active = (
                            node.dsl_name != "triton"  # triton should be inactive
                            and op != "mul.Tensor"  # mul.Tensor should be inactive
                            and dispatch_key != "CPU"  # CPU should be inactive
                        )

                        self.assertEqual(
                            node.active,
                            should_be_active,
                            f"Node {node.dsl_name}/{op}/{dispatch_key} active state incorrect",
                        )

    def test_integration_with_real_torch_library(self):
        """Integration test using real torch.library.Library to ensure actual PyTorch integration works."""
        # Use a unique operation name to avoid conflicts
        test_op = "test_registry_integration.Tensor"

        try:
            # Create real implementation functions
            def impl1(dispatch_keys, x):
                return x + 1

            def impl2(dispatch_keys, x):
                return x + 2

            # Register with real torch.library.Library (no mocking)
            self.registry.register_op_override(
                "test_backend1",
                "aten",
                test_op,
                "CPU",
                impl1,
                allow_multiple_override=True,
            )

            self.registry.register_op_override(
                "test_backend2",
                "aten",
                test_op,
                "CPU",
                impl2,
                allow_multiple_override=True,
            )

            # Verify registry state
            key = (test_op, "CPU")
            self.assertIn(key, self.registry._graphs)
            self.assertEqual(len(self.registry._graphs[key]), 2)

            # Test deregistration
            self.registry.deregister_op_overrides(disable_dsl_names="test_backend1")

            # Verify backend1 is inactive, backend2 is active
            nodes = self.registry._graphs[key]
            backend1_node = next(n for n in nodes if n.dsl_name == "test_backend1")
            backend2_node = next(n for n in nodes if n.dsl_name == "test_backend2")

            self.assertFalse(backend1_node.active)
            self.assertTrue(backend2_node.active)

            # Verify library was actually created
            self.assertIn(key, self.registry._libs)
            self.assertIsInstance(self.registry._libs[key], torch.library.Library)

        except Exception as e:
            # If this fails, it might reveal issues that mocked tests miss
            self.fail(
                f"Integration test failed, suggesting mocking may hide real issues: {e}"
            )
        finally:
            # Clean up - remove our test registrations
            if key in self.registry._libs:
                del self.registry._libs[key]
            if key in self.registry._graphs:
                del self.registry._graphs[key]
            # Clean up mappings
            for mapping in [
                self.registry._dsl_name_to_lib_graph,
                self.registry._op_symbol_to_lib_graph,
                self.registry._dispatch_key_to_lib_graph,
            ]:
                keys_to_remove = []
                for k, v in mapping.items():
                    if key in v:
                        v.remove(key)
                        if not v:  # Remove empty lists
                            keys_to_remove.append(k)
                for k in keys_to_remove:
                    del mapping[k]

    def test_integration_registry_state_consistency_after_operations(self):
        """Integration test: verify registry state remains consistent after complex operations."""
        test_op = "test_consistency.Tensor"
        key = (test_op, "CPU")

        try:
            # Perform a series of registrations and deregistrations
            backends = ["consistency1", "consistency2", "consistency3", "consistency4"]

            # Initial registration
            for backend in backends:

                def make_impl_fn(b):
                    def impl_fn(dispatch_keys, x):
                        return x.clone() + hash(b) % 100

                    return impl_fn

                impl_fn = make_impl_fn(backend)
                self.registry.register_op_override(
                    backend,
                    "aten",
                    test_op,
                    "CPU",
                    impl_fn,
                    allow_multiple_override=True,
                )

            # Partial deregistration
            # Note: PyTorch may warn about kernel override (but only shows warning once per session)
            self.registry.deregister_op_overrides(
                disable_dsl_names=["consistency2", "consistency4"]
            )

            # Verify intermediate state
            nodes = self.registry._graphs[key]
            active_backends = {node.dsl_name for node in nodes if node.active}
            inactive_backends = {node.dsl_name for node in nodes if not node.active}

            self.assertEqual(active_backends, {"consistency1", "consistency3"})
            self.assertEqual(inactive_backends, {"consistency2", "consistency4"})

            # Re-register one that was deregistered
            def new_impl(dispatch_keys, x):
                return x.clone() + 999

            self.registry.register_op_override(
                "consistency2",
                "aten",
                test_op,
                "CPU",
                new_impl,
                allow_multiple_override=True,
            )

            # Verify final state - consistency2 should appear twice now (old inactive + new active)
            nodes = self.registry._graphs[key]
            consistency2_nodes = [n for n in nodes if n.dsl_name == "consistency2"]

            # Should have 2 nodes for consistency2: one inactive (old) and one active (new)
            self.assertEqual(len(consistency2_nodes), 2)
            active_consistency2_nodes = [n for n in consistency2_nodes if n.active]
            inactive_consistency2_nodes = [
                n for n in consistency2_nodes if not n.active
            ]

            self.assertEqual(len(active_consistency2_nodes), 1)
            self.assertEqual(len(inactive_consistency2_nodes), 1)

            # Verify mappings are still consistent
            self.assertIn("consistency2", self.registry._dsl_name_to_lib_graph)
            self.assertIn(test_op, self.registry._op_symbol_to_lib_graph)
            self.assertIn("CPU", self.registry._dispatch_key_to_lib_graph)

            # Verify all mapping entries point to the correct key
            for mapping in [
                self.registry._dsl_name_to_lib_graph,
                self.registry._op_symbol_to_lib_graph,
                self.registry._dispatch_key_to_lib_graph,
            ]:
                for key_list in mapping.values():
                    if key in key_list:
                        # Each mapping should contain valid keys
                        for k in key_list:
                            self.assertIsInstance(k, tuple)
                            self.assertEqual(len(k), 2)

        finally:
            self._cleanup_test_registration(key)

    def test_filter_state_initialization_and_check_enabled(self):
        """Test _FilterState initialization and check_enabled functionality."""

        def test_fn(x):
            return x

        # Test initialization
        filter_state = self.registry._FilterState()
        self.assertIsInstance(filter_state._dsl_names, set)
        self.assertIsInstance(filter_state._op_symbols, set)
        self.assertIsInstance(filter_state._dispatch_keys, set)
        self.assertEqual(len(filter_state._dsl_names), 0)
        self.assertEqual(len(filter_state._op_symbols), 0)
        self.assertEqual(len(filter_state._dispatch_keys), 0)

        # Test empty filters enable all nodes
        node = self.registry._OverrideNode("test_dsl", "add.Tensor", "CPU", test_fn)
        self.assertTrue(filter_state.check_enabled(node))

        # Test DSL filtering
        filter_state._dsl_names.add("filtered_dsl")
        filtered_node = self.registry._OverrideNode(
            "filtered_dsl", "add.Tensor", "CPU", test_fn
        )
        enabled_node = self.registry._OverrideNode(
            "allowed_dsl", "add.Tensor", "CPU", test_fn
        )
        self.assertFalse(filter_state.check_enabled(filtered_node))
        self.assertTrue(filter_state.check_enabled(enabled_node))

        # Test op symbol filtering
        filter_state._dsl_names.clear()
        filter_state._op_symbols.add("mul.Tensor")
        filtered_node = self.registry._OverrideNode(
            "test_dsl", "mul.Tensor", "CPU", test_fn
        )
        enabled_node = self.registry._OverrideNode(
            "test_dsl", "add.Tensor", "CPU", test_fn
        )
        self.assertFalse(filter_state.check_enabled(filtered_node))
        self.assertTrue(filter_state.check_enabled(enabled_node))

        # Test dispatch key filtering
        filter_state._op_symbols.clear()
        filter_state._dispatch_keys.add("CUDA")
        filtered_node = self.registry._OverrideNode(
            "test_dsl", "add.Tensor", "CUDA", test_fn
        )
        enabled_node = self.registry._OverrideNode(
            "test_dsl", "add.Tensor", "CPU", test_fn
        )
        self.assertFalse(filter_state.check_enabled(filtered_node))
        self.assertTrue(filter_state.check_enabled(enabled_node))

    def test_filter_state_update_operations(self):
        """Test _FilterState.update operations for adding and removing values."""
        filter_state = self.registry._FilterState()

        # Test adding single values
        filter_state.update("dsl1", "add.Tensor", "CPU")
        self.assertIn("dsl1", filter_state._dsl_names)
        self.assertIn("add.Tensor", filter_state._op_symbols)
        self.assertIn("CPU", filter_state._dispatch_keys)

        # Test adding multiple values
        filter_state.update(["dsl2", "dsl3"], ["mul.Tensor"], ["CUDA"])
        self.assertEqual(filter_state._dsl_names, {"dsl1", "dsl2", "dsl3"})
        self.assertEqual(filter_state._op_symbols, {"add.Tensor", "mul.Tensor"})
        self.assertEqual(filter_state._dispatch_keys, {"CPU", "CUDA"})

        # Test removing values
        filter_state.update(["dsl1", "dsl3"], "mul.Tensor", "CUDA", remove_keys=True)
        self.assertEqual(filter_state._dsl_names, {"dsl2"})
        self.assertEqual(filter_state._op_symbols, {"add.Tensor"})
        self.assertEqual(filter_state._dispatch_keys, {"CPU"})

        # Test None values (should be no-op)
        original_state = (
            set(filter_state._dsl_names),
            set(filter_state._op_symbols),
            set(filter_state._dispatch_keys),
        )
        filter_state.update(None, None, None)
        current_state = (
            set(filter_state._dsl_names),
            set(filter_state._op_symbols),
            set(filter_state._dispatch_keys),
        )
        self.assertEqual(original_state, current_state)

    def test_key_set_building_functions(self):
        """Test key set building functions for filter operations."""
        # Set up test mappings
        key1 = ("add.Tensor", "CPU")
        key2 = ("mul.Tensor", "CUDA")
        key3 = ("div.Tensor", "CPU")

        self.registry._dsl_name_to_lib_graph["dsl1"] = [key1, key2]
        self.registry._op_symbol_to_lib_graph["add.Tensor"] = [key1]
        self.registry._dispatch_key_to_lib_graph["CPU"] = [key1, key3]

        # Test _build_key_set with single criteria
        key_set = self.registry._build_key_set("dsl1", None, None)
        self.assertEqual(key_set, {key1, key2})

        # Test _build_key_set with multiple criteria (union)
        key_set = self.registry._build_key_set("dsl1", "add.Tensor", "CPU")
        expected_keys = {key1, key2, key3}  # Union of all matching keys
        self.assertEqual(key_set, expected_keys)

        # Test FilterState.build_disable_key_set
        filter_state = self.registry._FilterState()
        filter_state._dsl_names.add("dsl1")
        filter_state._dispatch_keys.add("CPU")

        key_set = filter_state.build_disable_key_set()
        # Should include keys from both dsl1 and CPU filters
        expected_keys = {key1, key2, key3}
        self.assertEqual(key_set, expected_keys)

    @patch("torch.library.Library")
    def test_reenable_op_overrides_scenarios(self, mock_library_cls):
        """Test reenable_op_overrides with different scenarios."""
        mock_lib = MagicMock()
        mock_library_cls.return_value = mock_lib

        def test_fn(x):
            return x

        # Test 1: Basic single re-enable
        key1 = ("add.Tensor", "CPU")
        node1 = self.registry._OverrideNode(
            "dsl1", "add.Tensor", "CPU", test_fn, active=False
        )
        node2 = self.registry._OverrideNode(
            "dsl2", "add.Tensor", "CPU", test_fn, active=True
        )

        self.registry._graphs[key1] = [node1, node2]
        self.registry._dsl_name_to_lib_graph["dsl1"] = [key1]
        self.registry._filter_state._dsl_names.add("dsl1")
        self.registry._libs[key1] = mock_lib

        self.registry.reenable_op_overrides(enable_dsl_names="dsl1")

        self.assertNotIn("dsl1", self.registry._filter_state._dsl_names)
        self.assertTrue(node1.active)
        self.assertTrue(node2.active)
        self.assertEqual(mock_lib.impl.call_count, 2)

        # Test 2: Multiple criteria re-enable
        mock_lib.reset_mock()
        key2 = ("mul.Tensor", "CUDA")
        node3 = self.registry._OverrideNode(
            "dsl3", "mul.Tensor", "CUDA", test_fn, active=False
        )

        self.registry._graphs[key2] = [node3]
        self.registry._op_symbol_to_lib_graph["mul.Tensor"] = [key2]
        self.registry._filter_state._op_symbols.add("mul.Tensor")

        self.registry.reenable_op_overrides(enable_op_symbols="mul.Tensor")

        self.assertNotIn("mul.Tensor", self.registry._filter_state._op_symbols)
        self.assertTrue(node3.active)

        # Test 3: Partial re-enable with remaining filters
        key3 = ("div.Tensor", "CPU")
        node4 = self.registry._OverrideNode(
            "filtered_dsl", "div.Tensor", "CPU", test_fn, active=False
        )
        node5 = self.registry._OverrideNode(
            "allowed_dsl", "div.Tensor", "CPU", test_fn, active=False
        )

        self.registry._graphs[key3] = [node4, node5]
        self.registry._dsl_name_to_lib_graph["allowed_dsl"] = [key3]
        self.registry._filter_state._dsl_names.update(["filtered_dsl", "allowed_dsl"])

        mock_lib_new = MagicMock()
        mock_library_cls.return_value = mock_lib_new
        self.registry._libs[key3] = mock_lib_new

        self.registry.reenable_op_overrides(enable_dsl_names="allowed_dsl")

        self.assertIn("filtered_dsl", self.registry._filter_state._dsl_names)
        self.assertNotIn("allowed_dsl", self.registry._filter_state._dsl_names)
        self.assertFalse(node4.active)  # Still filtered
        self.assertTrue(node5.active)  # Re-enabled

        mock_lib_new.impl.assert_called_once()

    def test_filter_state_miscellaneous(self):
        """Test miscellaneous FilterState functionality: string representation and global integration."""
        # Test string representation
        filter_state = self.registry._FilterState()
        filter_state._dsl_names.update(["dsl1", "dsl2"])
        filter_state._op_symbols.add("add.Tensor")
        filter_state._dispatch_keys.add("CPU")

        try:
            str_repr = str(filter_state)
            self.assertIn("Filter State:", str_repr)
        except Exception as e:
            self.fail(f"FilterState.__str__() raised an exception: {e}")

        # Test global filter state integration
        self.assertIsInstance(self.registry._filter_state, self.registry._FilterState)

        filter_state_ref1 = self.registry._filter_state
        self.registry._filter_state._dsl_names.add("test_dsl")
        filter_state_ref2 = self.registry._filter_state

        self.assertIs(filter_state_ref1, filter_state_ref2)
        self.assertIn("test_dsl", filter_state_ref2._dsl_names)

    def _cleanup_test_registration(self, key):
        """Helper method to clean up test registrations."""
        # Clean up registry state
        if key in self.registry._libs:
            del self.registry._libs[key]
        if key in self.registry._graphs:
            del self.registry._graphs[key]

        # Clean up mappings
        for mapping in [
            self.registry._dsl_name_to_lib_graph,
            self.registry._op_symbol_to_lib_graph,
            self.registry._dispatch_key_to_lib_graph,
        ]:
            keys_to_remove = []
            for k, v in list(
                mapping.items()
            ):  # Use list() to avoid dict changed during iteration
                if key in v:
                    v.remove(key)
                if not v:  # Remove empty lists
                    keys_to_remove.append(k)
            for k in keys_to_remove:
                if k in mapping:
                    del mapping[k]


if __name__ == "__main__":
    run_tests()
