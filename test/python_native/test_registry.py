# Owner(s): ["module: dsl-native-ops"]

from unittest.mock import MagicMock, patch

import torch
import torch._native.registry as registry_module
from torch._subclasses.fake_tensor import FakeTensorMode
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

        def cond_fn(x):
            return True

        def impl_fn(x):
            return x

        node = self.registry._OverrideNode(
            "test_dsl", "add.Tensor", "CPU", cond_fn, impl_fn, "test_node"
        )
        self.assertEqual(node.dsl_name, "test_dsl")
        self.assertEqual(node.op_symbol, "add.Tensor")
        self.assertEqual(node.dispatch_key, "CPU")
        self.assertEqual(node.cond_fn, cond_fn)
        self.assertEqual(node.impl_fn, impl_fn)
        self.assertFalse(node.unconditional_override)
        self.assertTrue(node.active)

    @patch("torch.library.Library")
    def test_register_op_override_basic(self, mock_library_cls):
        """Test basic register_op_override functionality."""

        def cond_fn(x):
            return True

        def impl_fn(x):
            return x

        mock_lib = MagicMock()
        mock_library_cls.return_value = mock_lib

        self.registry.register_op_override(
            "test_backend", "aten", "add.Tensor", "CPU", cond_fn, impl_fn
        )

        key = ("add.Tensor", "CPU")
        self.assertEqual(len(self.registry._graphs[key]), 1)
        node = self.registry._graphs[key][0]
        self.assertEqual(node.dsl_name, "test_backend")
        self.assertEqual(node.cond_fn, cond_fn)
        self.assertEqual(node.impl_fn, impl_fn)

    @patch("torch.library.Library")
    def test_deregister_op_overrides_basic(self, mock_library_cls):
        """Test basic deregister_op_overrides functionality."""

        def cond_fn(x):
            return True

        def impl_fn(x):
            return x

        mock_lib = MagicMock()
        mock_library_cls.return_value = mock_lib

        # Register first
        self.registry.register_op_override(
            "test_backend", "aten", "mul.Tensor", "CPU", cond_fn, impl_fn
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

        def cond_fn(x):
            return True

        def impl_fn(x):
            return x

        # Create nodes in specific order
        nodes = [
            self.registry._OverrideNode(
                "dsl_c", "test_reorder.Tensor", "CPU", cond_fn, impl_fn, "node_c"
            ),
            self.registry._OverrideNode(
                "dsl_a", "test_reorder.Tensor", "CPU", cond_fn, impl_fn, "node_a"
            ),
            self.registry._OverrideNode(
                "dsl_b", "test_reorder.Tensor", "CPU", cond_fn, impl_fn, "node_b"
            ),
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

        def cond_fn(x):
            return True

        def impl_fn(x):
            return x

        node = self.registry._OverrideNode(
            "test_dsl", "test_error.Tensor", "CPU", cond_fn, impl_fn, "test_node"
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

        def cond_fn(x):
            return True

        def impl_fn1(x):
            return x + 1

        def impl_fn2(x):
            return x + 2

        # Register multiple overrides
        self.registry.register_op_override(
            "backend_z", "aten", "test.Tensor", "CPU", cond_fn, impl_fn1
        )
        self.registry.register_op_override(
            "backend_a", "aten", "test.Tensor", "CPU", cond_fn, impl_fn2
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

    def test_disallowed_dispatch_key_rejected(self):
        """Overrides installed at Meta / Composite* would loop through the
        fake kernel's redispatch — `register_op_override` must reject them.
        """

        def cond(*a, **k):
            return True

        def impl(*a, **k):
            return None

        for bad_key in (
            "Meta",
            "CompositeImplicitAutograd",
            "CompositeExplicitAutograd",
        ):
            with self.assertRaisesRegex(ValueError, "dispatch_key="):
                self.registry.register_op_override(
                    "test_dsl", "aten", "mul.Tensor", bad_key, cond, impl
                )

    def test_cond_none_without_unconditional_override_rejected(self):
        """cond=None is only valid when unconditional_override=True."""

        def impl(*a, **k):
            return None

        with self.assertRaisesRegex(ValueError, "cond must be provided"):
            self.registry.register_op_override(
                "test_dsl", "aten", "mul.Tensor", "CPU", None, impl
            )


@skipIfTorchDynamo("Runtime registry tests exercise the dispatcher directly")
class TestRegistryRuntime(TestCase):
    """End-to-end runtime tests that exercise the real dispatcher.

    These tests register overrides on real aten ops and therefore must fully
    tear down any dispatcher-visible state in tearDown, otherwise a leaked
    router will poison other tests in the same process.
    """

    def setUp(self):
        self.registry = registry_module

        self._saved = {
            "graphs": dict(self.registry._graphs),
            "libs": dict(self.registry._libs),
            "aten_override_libs": dict(self.registry._aten_override_libs),
            "def_libs": dict(self.registry._def_libs),
            "defined_native_ops": set(self.registry._defined_native_ops),
            "dsl_map": {
                k: list(v) for k, v in self.registry._dsl_name_to_lib_graph.items()
            },
            "op_map": {
                k: list(v) for k, v in self.registry._op_symbol_to_lib_graph.items()
            },
            "dk_map": {
                k: list(v) for k, v in self.registry._dispatch_key_to_lib_graph.items()
            },
            "node_id_counter": self.registry._node_id_counter,
        }

        self.registry._graphs.clear()
        self.registry._dsl_name_to_lib_graph.clear()
        self.registry._op_symbol_to_lib_graph.clear()
        self.registry._dispatch_key_to_lib_graph.clear()

    def tearDown(self):
        # Destroy any aten overrides the test installed so the dispatcher
        # returns to its pre-test state.
        for lib in list(self.registry._aten_override_libs.values()):
            lib._destroy()
        self.registry._aten_override_libs.clear()

        # Restore aten override libs (none expected from other tests, but
        # be defensive).
        self.registry._aten_override_libs.update(self._saved["aten_override_libs"])

        # _native namespace DEF libraries and the ops defined on them persist
        # for the lifetime of the process (torch.library has no "undefine"),
        # so we deliberately do not destroy them — later tests will just
        # hit the `name in _defined_native_ops` short-circuit.
        # We also leave the fake kernels registered for the same reason.

        # Restore the rest.
        self.registry._graphs.clear()
        self.registry._graphs.update(self._saved["graphs"])
        self.registry._libs.clear()
        self.registry._libs.update(self._saved["libs"])
        self.registry._dsl_name_to_lib_graph.clear()
        for k, v in self._saved["dsl_map"].items():
            self.registry._dsl_name_to_lib_graph[k] = list(v)
        self.registry._op_symbol_to_lib_graph.clear()
        for k, v in self._saved["op_map"].items():
            self.registry._op_symbol_to_lib_graph[k] = list(v)
        self.registry._dispatch_key_to_lib_graph.clear()
        for k, v in self._saved["dk_map"].items():
            self.registry._dispatch_key_to_lib_graph[k] = list(v)

    def _install(self, op_symbol, dispatch_key):
        """Build the graph then push it through the real registration path."""
        self.registry._register_overrides_from_graph(
            op_symbol,
            dispatch_key,
            self.registry._graphs[(op_symbol, dispatch_key)],
        )

    def test_cond_false_falls_through_to_native(self):
        """cond=False must transparently invoke the captured native kernel."""
        sentinel_called = [False]

        def cond(*a, **k):
            return False

        def impl(a, b):
            sentinel_called[0] = True
            return torch.zeros_like(a)

        self.registry.register_op_override(
            "test_dsl", "aten", "mul.Tensor", "CPU", cond, impl
        )
        self._install("mul.Tensor", "CPU")

        a = torch.tensor([2.0, 3.0])
        b = torch.tensor([4.0, 5.0])
        # Call the overload directly; `a * b` goes through overload
        # resolution which can dispatch to mul.Scalar for mixed args.
        out = torch.ops.aten.mul.Tensor(a, b)
        self.assertTrue(torch.equal(out, torch.tensor([8.0, 15.0])))
        self.assertFalse(sentinel_called[0])

    def test_cond_true_routes_to_impl(self):
        """cond=True must route the call to the registered impl."""

        def cond(*a, **k):
            return True

        def impl(a, b):
            return torch.full_like(a, 42.0)

        self.registry.register_op_override(
            "test_dsl", "aten", "mul.Tensor", "CPU", cond, impl
        )
        self._install("mul.Tensor", "CPU")

        a = torch.tensor([2.0, 3.0])
        b = torch.tensor([4.0, 5.0])
        out = torch.ops.aten.mul.Tensor(a, b)
        self.assertTrue(torch.equal(out, torch.tensor([42.0, 42.0])))

    def test_no_recursion_in_aten_backward_formula(self):
        """Backward of bmm calls bmm again; the fallback must bypass the router.

        If the router weren't bypassed, aten's bmm backward formula would
        re-enter it on every autograd step, either recursing forever or
        silently calling the override impl in the grad path.
        """

        def cond(*a, **k):
            # False so every call hits the fallback — this isolates the
            # "native kernel called via fallback must not re-enter the router"
            # property.
            return False

        def impl(a, b):
            raise AssertionError("impl should not be called when cond=False")

        self.registry.register_op_override("test_dsl", "aten", "bmm", "CPU", cond, impl)
        self._install("bmm", "CPU")

        a = torch.randn(2, 3, 4, requires_grad=True)
        b = torch.randn(2, 4, 5, requires_grad=True)
        expected = a.detach() @ b.detach()
        out = torch.bmm(a, b)
        self.assertEqual(out, expected)

        out.sum().backward()
        self.assertEqual(a.grad.shape, a.shape)
        self.assertEqual(b.grad.shape, b.shape)
        self.assertTrue((a.grad != 0).any())
        self.assertTrue((b.grad != 0).any())

    def test_deregister_reenable_roundtrip(self):
        """deregister must tear down the aten router; reenable must reinstall."""

        def cond(*a, **k):
            return True

        def impl(a, b):
            return torch.full_like(a, 7.0)

        self.registry.register_op_override(
            "test_dsl", "aten", "mul.Tensor", "CPU", cond, impl
        )
        self._install("mul.Tensor", "CPU")

        a = torch.tensor([2.0, 3.0])
        b = torch.tensor([4.0, 5.0])
        mul = torch.ops.aten.mul.Tensor

        # Override active.
        self.assertTrue(torch.equal(mul(a, b), torch.tensor([7.0, 7.0])))
        self.assertIn(("mul.Tensor", "CPU"), self.registry._aten_override_libs)

        # Deregister → native kernel returns.
        self.registry.deregister_op_overrides(disable_dsl_names="test_dsl")
        self.assertTrue(torch.equal(mul(a, b), torch.tensor([8.0, 15.0])))
        self.assertNotIn(("mul.Tensor", "CPU"), self.registry._aten_override_libs)

        # Reenable → override fires again.
        self.registry.reenable_op_overrides(enable_dsl_names="test_dsl")
        self.assertTrue(torch.equal(mul(a, b), torch.tensor([7.0, 7.0])))
        self.assertIn(("mul.Tensor", "CPU"), self.registry._aten_override_libs)

    def test_empty_graph_tears_down_router(self):
        """An empty graph passed to _cleanup_and_reregister_graph must still
        tear down the previously-installed aten router.

        Regression test: before the fix, the `if graph:` guard skipped
        _register_overrides_from_graph entirely, leaving the aten override
        live with a stale closure.
        """

        def cond(*a, **k):
            return True

        def impl(a, b):
            return torch.full_like(a, 99.0)

        self.registry.register_op_override(
            "test_dsl", "aten", "mul.Tensor", "CPU", cond, impl
        )
        self._install("mul.Tensor", "CPU")

        a = torch.tensor([2.0, 3.0])
        b = torch.tensor([4.0, 5.0])
        mul = torch.ops.aten.mul.Tensor
        self.assertTrue(torch.equal(mul(a, b), torch.tensor([99.0, 99.0])))
        self.assertIn(("mul.Tensor", "CPU"), self.registry._aten_override_libs)

        # Simulate a filter-out-everything transformation.
        self.registry._graphs[("mul.Tensor", "CPU")] = []
        self.registry._cleanup_and_reregister_graph(
            "mul.Tensor", "CPU", self.registry._graphs[("mul.Tensor", "CPU")]
        )

        self.assertNotIn(("mul.Tensor", "CPU"), self.registry._aten_override_libs)
        self.assertTrue(torch.equal(mul(a, b), torch.tensor([8.0, 15.0])))

    def test_fake_tensor_shape_inference(self):
        """FakeTensorMode must shape-infer through `_native::<id>` via the
        registered fake kernel (which redispatches to the aten meta).
        """

        def cond(*a, **k):
            return True

        def impl(a, b):
            return torch.full_like(a, 1.0)

        self.registry.register_op_override(
            "test_dsl", "aten", "mul.Tensor", "CPU", cond, impl
        )
        self._install("mul.Tensor", "CPU")

        with FakeTensorMode():
            a = torch.empty(3, 4, dtype=torch.float32)
            b = torch.empty(3, 4, dtype=torch.float32)
            out = torch.ops.aten.mul.Tensor(a, b)

        self.assertEqual(out.shape, torch.Size([3, 4]))
        self.assertEqual(out.dtype, torch.float32)

    def test_unconditional_override_cond_none(self):
        """`cond=None` + `unconditional_override=True` must substitute a
        trivially-true predicate so the impl fires on every call.
        """
        call_count = [0]

        def impl(a, b):
            call_count[0] += 1
            return torch.full_like(a, 5.0)

        self.registry.register_op_override(
            "test_dsl",
            "aten",
            "mul.Tensor",
            "CPU",
            None,
            impl,
            unconditional_override=True,
        )
        self._install("mul.Tensor", "CPU")

        a = torch.tensor([2.0, 3.0])
        b = torch.tensor([4.0, 5.0])
        self.assertTrue(
            torch.equal(torch.ops.aten.mul.Tensor(a, b), torch.tensor([5.0, 5.0]))
        )
        self.assertTrue(
            torch.equal(
                torch.ops.aten.mul.Tensor(torch.tensor([0.0]), torch.tensor([0.0])),
                torch.tensor([5.0]),
            )
        )
        self.assertEqual(call_count[0], 2)


if __name__ == "__main__":
    run_tests()
