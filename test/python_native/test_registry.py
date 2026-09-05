# Owner(s): ["module: dsl-native-ops"]

import contextlib
from unittest.mock import MagicMock, patch

import torch
import torch._dynamo.trace_rules as trace_rules
import torch._native.registry as registry_module
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


class _CowStateWrapperTensor(torch.Tensor):
    elem: torch.Tensor

    __slots__ = ["elem"]

    @staticmethod
    def __new__(cls, elem):
        out = torch.Tensor._make_wrapper_subclass(
            cls,
            elem.size(),
            dtype=elem.dtype,
            layout=elem.layout,
            device=elem.device,
            requires_grad=elem.requires_grad,
            strides=elem.stride(),
            storage_offset=elem.storage_offset(),
        )
        out.elem = elem
        return out

    def __tensor_flatten__(self):
        return ["elem"], None

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        return _CowStateWrapperTensor(inner_tensors["elem"])

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        raise AssertionError("unexpected dispatch")


@skipIfTorchDynamo("Registry tests don't need dynamo compilation")
class TestRegistry(TestCase):
    """Tests for the torch._native.registry module."""

    def setUp(self):
        """Clean up registry state before each test."""
        super().setUp()
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
            "test_dsl", "aten", "add.Tensor", "CPU", cond_fn, impl_fn, "test_node"
        )
        self.assertEqual(node.dsl_name, "test_dsl")
        self.assertEqual(node.lib_symbol, "aten")
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

        key = ("aten", "add.Tensor", "CPU")
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

        key = ("aten", "mul.Tensor", "CPU")
        self.assertTrue(self.registry._graphs[key][0].active)

        # Then deregister
        self.registry.deregister_op_overrides(disable_dsl_names="test_backend")
        self.assertFalse(self.registry._graphs[key][0].active)

    # NEW FUNCTIONALITY TESTS - ONLY THE ESSENTIAL ONES

    def test_reorder_graphs_from_user_function_basic(self):
        """Test basic graph reordering functionality."""
        # Set up test data
        key = ("aten", "test_reorder.Tensor", "CPU")

        def cond_fn(x):
            return True

        def impl_fn(x):
            return x

        # Create nodes in specific order
        nodes = [
            self.registry._OverrideNode(
                "dsl_c",
                "aten",
                "test_reorder.Tensor",
                "CPU",
                cond_fn,
                impl_fn,
                "node_c",
            ),
            self.registry._OverrideNode(
                "dsl_a",
                "aten",
                "test_reorder.Tensor",
                "CPU",
                cond_fn,
                impl_fn,
                "node_a",
            ),
            self.registry._OverrideNode(
                "dsl_b",
                "aten",
                "test_reorder.Tensor",
                "CPU",
                cond_fn,
                impl_fn,
                "node_b",
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
        key = ("aten", "test_error.Tensor", "CPU")

        def cond_fn(x):
            return True

        def impl_fn(x):
            return x

        node = self.registry._OverrideNode(
            "test_dsl",
            "aten",
            "test_error.Tensor",
            "CPU",
            cond_fn,
            impl_fn,
            "test_node",
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

        key = ("aten", "test.Tensor", "CPU")

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

    def test_lib_symbol_off_the_allowlist_rejected(self):
        """Namespace support is opt-in; anything not on the allowlist is
        rejected at registration time.

        `_native` stands in for "not allowlisted" because it can never be
        allowlisted -- it holds the ops carrying the override impls.
        """
        self.assertNotIn("_native", self.registry._ALLOWED_LIB_SYMBOLS)
        with self.assertRaisesRegex(ValueError, "is not overridable"):
            self.registry.register_op_override(
                "test_dsl",
                "_native",
                "some_op",
                "CPU",
                lambda *a, **k: True,
                lambda *a, **k: None,
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
        super().setUp()
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
        # Destroy only what this test installed. On a DSL-equipped machine
        # `import torch` leaves production overrides live in
        # `_aten_override_libs`; destroying those would strip the dispatcher
        # of kernels the registry still lists as installed, for the rest of
        # the process.
        saved_override_libs = self._saved["aten_override_libs"]
        for key, lib in list(self.registry._aten_override_libs.items()):
            if key not in saved_override_libs:
                lib._destroy()
        self.registry._aten_override_libs.clear()
        self.registry._aten_override_libs.update(saved_override_libs)

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

    def _install(self, op_symbol, dispatch_key, lib_symbol="aten"):
        """Build the graph then push it through the real registration path."""
        self.registry._register_overrides_from_graph(
            lib_symbol,
            op_symbol,
            dispatch_key,
            self.registry._graphs[(lib_symbol, op_symbol, dispatch_key)],
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

    def test_compile_session_flag_falls_through_without_recursion(self):
        """The eager router must not redispatch to its own aten override when
        compile-session state is set but Dynamo is not actively tracing it.
        """

        def cond(*a, **k):
            return False

        def impl(a, b):
            raise AssertionError("impl should not be called when cond=False")

        self.registry.register_op_override(
            "test_dsl", "aten", "mul.Tensor", "CPU", cond, impl
        )
        self._install("mul.Tensor", "CPU")

        a = torch.tensor([2.0, 3.0])
        b = torch.tensor([4.0, 5.0])
        with torch.compiler._compile_session_context():
            out = torch.ops.aten.mul.Tensor(a, b)

        self.assertTrue(torch.equal(out, torch.tensor([8.0, 15.0])))

    def test_folded_dynamo_flag_on_real_tensors_does_not_recurse(self):
        """A residual eager frame carrying Dynamo's folded
        is_dynamo_compiling()==True must NOT divert to the aten overload:
        that call re-enters the dispatcher from the top, back into this
        router (RecursionError). Real tensors always take eager dispatch.
        """
        cond_calls = [0]

        def cond(*a, **k):
            # Reached only from the re-entrant call the shortcut triggers, so
            # the guard must be held here: this also fails if the shortcut is
            # removed and cond is consulted directly.
            self.assertTrue(getattr(self.registry._router_active, "on", False))
            cond_calls[0] += 1
            return False

        def impl(a, b):
            raise AssertionError("impl should not be called when cond=False")

        self.registry.register_op_override(
            "test_dsl", "aten", "mul.Tensor", "CPU", cond, impl
        )
        self._install("mul.Tensor", "CPU")

        a = torch.tensor([2.0, 3.0])
        b = torch.tensor([4.0, 5.0])
        # Dynamo folds the call to a True constant while tracing; a frame
        # carrying that constant can still run eagerly. patch() reproduces
        # that state without needing a compiled residual.
        with patch.object(torch.compiler, "is_dynamo_compiling", return_value=True):
            out = torch.ops.aten.mul.Tensor(a, b)

        self.assertTrue(torch.equal(out, torch.tensor([8.0, 15.0])))
        # Eager dispatch ran (cond consulted) exactly once: no runaway re-entry.
        self.assertEqual(cond_calls[0], 1)

    def test_dynamo_shortcut_still_fires_for_outer_call(self):
        """The re-entrancy guard must not disable the shortcut itself: an
        outer call with the folded flag still diverts to aten (that is what
        keeps trace-time graph breaks away)."""
        cond_calls = [0]

        def cond(*a, **k):
            cond_calls[0] += 1
            return False

        def impl(a, b):
            raise AssertionError("impl should not be called when cond=False")

        self.registry.register_op_override(
            "test_dsl", "aten", "mul.Tensor", "CPU", cond, impl
        )
        self._install("mul.Tensor", "CPU")

        a = torch.tensor([2.0, 3.0])
        b = torch.tensor([4.0, 5.0])
        with patch.object(torch.compiler, "is_dynamo_compiling", return_value=True):
            out = torch.ops.aten.mul.Tensor(a, b)

        self.assertTrue(torch.equal(out, torch.tensor([8.0, 15.0])))
        # Outer call diverted to aten; the re-entrant call it triggers takes
        # eager dispatch, consulting cond exactly once (and not recursing).
        self.assertEqual(cond_calls[0], 1)
        self.assertFalse(getattr(self.registry._router_active, "on", False))

    def test_dynamo_shortcut_preserves_cow_fallback(self):
        """COW inputs must keep using the eager fallback path under Dynamo."""
        sentinel_called = [False]

        def cond(*a, **k):
            sentinel_called[0] = True
            return False

        def impl(a, b):
            raise AssertionError("impl should not be called when cond=False")

        self.registry.register_op_override(
            "test_dsl", "aten", "mul.Tensor", "CPU", cond, impl
        )
        self._install("mul.Tensor", "CPU")

        @torch.compile(backend="eager")
        def fn(a, b):
            with torch.device("cpu"):
                return torch.mul(a, b)

        a = torch._lazy_clone(torch.tensor([2.0, 3.0]))
        b = torch._lazy_clone(torch.tensor([4.0, 5.0]))
        self.assertTrue(torch._C._is_cow_tensor(a))
        self.assertTrue(torch._C._is_cow_tensor(b))

        out = fn(a, b)

        self.assertTrue(torch.equal(out, torch.tensor([8.0, 15.0])))
        self.assertTrue(sentinel_called[0])
        self.assertTrue(torch._C._is_cow_tensor(a))
        self.assertTrue(torch._C._is_cow_tensor(b))

    def test_dynamo_guards_on_cow_state(self):
        @torch.compile(backend="eager")
        def fn(a):
            return torch._C._is_cow_tensor(a)

        self.assertFalse(fn(torch.tensor([2.0, 3.0])))
        self.assertTrue(fn(torch._lazy_clone(torch.tensor([2.0, 3.0]))))

    def test_dynamo_graph_breaks_on_sourceless_cow_state(self):
        @torch.compile(backend="eager")
        def fn(a):
            return torch._C._is_cow_tensor(a.view_as(a))

        self.assertFalse(fn(torch.tensor([2.0, 3.0])))
        self.assertTrue(fn(torch._lazy_clone(torch.tensor([2.0, 3.0]))))

    def test_dynamo_graph_breaks_on_mutated_cow_state(self):
        @torch.compile(backend="eager")
        def fn(a):
            a.add_(1)
            return torch._C._is_cow_tensor(a)

        self.assertFalse(fn(torch._lazy_clone(torch.tensor([2.0, 3.0]))))

    def test_dynamo_graph_breaks_after_lazy_clone_changes_cow_state(self):
        @torch.compile(backend="inductor")
        def fn(a):
            clone = a._lazy_clone()
            return torch._C._is_cow_tensor(a), torch._C._is_cow_tensor(clone)

        self.assertEqual(
            fn(torch.tensor([2.0, 3.0])),
            (True, True),
        )

    def test_dynamo_graph_breaks_after_lazy_clone_function_changes_cow_state(self):
        fn_id = id(torch._lazy_clone)
        prior_trace_rule_state = (
            fn_id in trace_rules._allowed_callable_ids,
            fn_id in trace_rules._disallowed_callable_ids,
            fn_id in trace_rules._nonstrict_trace_callable_ids,
        )
        # torch._lazy_clone normally stays out of the FX graph. Force it into
        # the graph here to cover the defensive call_function scan.
        if fn_id in trace_rules._disallowed_callable_ids:
            trace_rules._disallowed_callable_ids.remove(fn_id)
        trace_rules._allowed_callable_ids.add(fn_id)
        try:
            torch._dynamo.reset()

            @torch.compile(backend="eager")
            def fn(a):
                clone = torch._lazy_clone(a)
                return torch._C._is_cow_tensor(a), torch._C._is_cow_tensor(clone)

            self.assertEqual(
                fn(torch.tensor([2.0, 3.0])),
                (True, True),
            )
        finally:
            if fn_id in trace_rules._allowed_callable_ids:
                trace_rules._allowed_callable_ids.remove(fn_id)
            if fn_id in trace_rules._disallowed_callable_ids:
                trace_rules._disallowed_callable_ids.remove(fn_id)
            if fn_id in trace_rules._nonstrict_trace_callable_ids:
                trace_rules._nonstrict_trace_callable_ids.remove(fn_id)
            if prior_trace_rule_state[0]:
                trace_rules._allowed_callable_ids.add(fn_id)
            if prior_trace_rule_state[1]:
                trace_rules._disallowed_callable_ids.add(fn_id)
            if prior_trace_rule_state[2]:
                trace_rules._nonstrict_trace_callable_ids.add(fn_id)
            torch._dynamo.reset()

    def test_dynamo_graph_breaks_after_lazy_clone_view_changes_cow_state(self):
        @torch.compile(backend="eager")
        def fn(a):
            view = a.view(-1)
            clone = view._lazy_clone()
            return (
                torch._C._is_cow_tensor(a),
                torch._C._is_cow_tensor(view),
                torch._C._is_cow_tensor(clone),
            )

        self.assertEqual(
            fn(torch.tensor([2.0, 3.0])),
            (True, True, True),
        )

    def test_cow_guard_misses_on_fake_tensor(self):
        from torch._dynamo.guards import _cow_tensor_matches

        with FakeTensorMode():
            fake = torch.empty(2)

        self.assertFalse(_cow_tensor_matches(fake, False))
        self.assertFalse(_cow_tensor_matches(fake, True))

    def test_is_cow_tensor_rejects_python_tensor_subclasses(self):
        x = _CowStateWrapperTensor(torch.tensor([2.0, 3.0]))
        with self.assertRaisesRegex(
            RuntimeError, "_is_cow_tensor is not defined for Python tensor subclasses"
        ):
            torch._C._is_cow_tensor(x)

    def test_dynamo_graph_breaks_on_subclass_cow_state(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(a):
            return torch._C._is_cow_tensor(a)

        x = _CowStateWrapperTensor(torch.tensor([2.0, 3.0]))
        with self.assertRaisesRegex(
            Exception, "COW tensor check on Python tensor subclass"
        ):
            fn(x)

    def test_dynamo_allows_previously_mutated_cow_state(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(a):
            return torch._C._is_cow_tensor(a)

        x = torch.tensor([2.0, 3.0])
        x.add_(1)
        self.assertFalse(fn(x))

    def test_strict_export_rejects_cow_state(self):
        class Mod(torch.nn.Module):
            def forward(self, a):
                if torch._C._is_cow_tensor(a):
                    return a + 1
                return a + 2

        with self.assertRaisesRegex(Exception, "COW tensor check during export"):
            torch.export.export(
                Mod(),
                (torch._lazy_clone(torch.tensor([2.0, 3.0])),),
                strict=True,
            )

    def test_non_strict_export_rejects_cow_state(self):
        class Mod(torch.nn.Module):
            def forward(self, a):
                if torch._C._is_cow_tensor(a):
                    return a + 1
                return a + 2

        with self.assertRaisesRegex(
            Exception, "_is_cow_tensor is not defined for Python tensor subclasses"
        ):
            torch.export.export(
                Mod(),
                (torch._lazy_clone(torch.tensor([2.0, 3.0])),),
                strict=False,
            )

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
        self.assertIn(("aten", "mul.Tensor", "CPU"), self.registry._aten_override_libs)

        # Deregister → native kernel returns.
        self.registry.deregister_op_overrides(disable_dsl_names="test_dsl")
        self.assertTrue(torch.equal(mul(a, b), torch.tensor([8.0, 15.0])))
        self.assertNotIn(
            ("aten", "mul.Tensor", "CPU"), self.registry._aten_override_libs
        )

        # Reenable → override fires again.
        self.registry.reenable_op_overrides(enable_dsl_names="test_dsl")
        self.assertTrue(torch.equal(mul(a, b), torch.tensor([7.0, 7.0])))
        self.assertIn(("aten", "mul.Tensor", "CPU"), self.registry._aten_override_libs)

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
        self.assertIn(("aten", "mul.Tensor", "CPU"), self.registry._aten_override_libs)

        # Simulate a filter-out-everything transformation.
        self.registry._graphs[("aten", "mul.Tensor", "CPU")] = []
        self.registry._cleanup_and_reregister_graph(
            "aten",
            "mul.Tensor",
            "CPU",
            self.registry._graphs[("aten", "mul.Tensor", "CPU")],
        )

        self.assertNotIn(
            ("aten", "mul.Tensor", "CPU"), self.registry._aten_override_libs
        )
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


@skipIfTorchDynamo("Runtime registry tests exercise the dispatcher directly")
class TestRegistryNonAtenNamespace(TestCase):
    """Overrides on namespaces other than `aten`.

    Mirrors how a Python-defined op looks to the registry: the op's own
    implementation sits at CompositeExplicitAutograd (what
    `torch.library.custom_op` produces), so the override is installed at the
    backend key and the captured fallback resolves to the composite kernel.

    These ops carry no Autograd kernel above the router, so the autograd
    layer -- the shape a real `torch.library.custom_op` takes, and the
    motivation for the widened keys -- is not covered here. That coverage
    lives with the first torch_nn override and its own test.
    """

    NS = "_native_registry_test"
    NS2 = "_native_registry_test2"
    MISSING_NS = "_native_registry_missing_ns"

    def setUp(self):
        super().setUp()
        self.registry = registry_module

        self._saved_graphs = dict(self.registry._graphs)
        self._saved_override_libs = dict(self.registry._aten_override_libs)
        self._saved_maps = {
            name: {k: list(v) for k, v in getattr(self.registry, name).items()}
            for name in (
                "_dsl_name_to_lib_graph",
                "_op_symbol_to_lib_graph",
                "_dispatch_key_to_lib_graph",
            )
        }
        self.registry._graphs.clear()

        # These throwaway namespaces are not on the production allowlist.
        self._stack = contextlib.ExitStack()
        self._stack.enter_context(
            patch.object(
                self.registry,
                "_ALLOWED_LIB_SYMBOLS",
                self.registry._ALLOWED_LIB_SYMBOLS
                | {self.NS, self.NS2, self.MISSING_NS},
            )
        )

        # A throwaway namespace with one op whose only kernel is composite,
        # matching the shape of a `torch.library.custom_op` definition.
        self.lib = self._stack.enter_context(
            torch.library._scoped_library(self.NS, "DEF")
        )
        self.lib.define("twice(Tensor self) -> Tensor")
        self.lib.impl("twice", lambda x: x * 2, "CompositeExplicitAutograd")
        self.op = getattr(torch.ops, self.NS).twice.default

        # A second namespace defining the SAME op symbol, so the tests below
        # exercise the per-op-symbol maps the widened keys share. Its
        # fallback triples rather than doubles, to tell the two apart.
        self.lib2 = self._stack.enter_context(
            torch.library._scoped_library(self.NS2, "DEF")
        )
        self.lib2.define("twice(Tensor self) -> Tensor")
        self.lib2.impl("twice", lambda x: x * 3, "CompositeExplicitAutograd")
        self.op2 = getattr(torch.ops, self.NS2).twice.default

        self._saved_filter_state = (
            set(self.registry._filter_state._dsl_names),
            set(self.registry._filter_state._op_symbols),
            set(self.registry._filter_state._dispatch_keys),
        )

    def tearDown(self):
        # Only what this test installed; production overrides in the snapshot
        # stay live (see TestRegistryRuntime.tearDown).
        for key, lib in list(self.registry._aten_override_libs.items()):
            if key not in self._saved_override_libs:
                lib._destroy()
        self.registry._aten_override_libs.clear()
        self.registry._aten_override_libs.update(self._saved_override_libs)

        self._stack.close()

        dsl_names, op_symbols, dispatch_keys = self._saved_filter_state
        for attr, saved in (
            ("_dsl_names", dsl_names),
            ("_op_symbols", op_symbols),
            ("_dispatch_keys", dispatch_keys),
        ):
            target = getattr(self.registry._filter_state, attr)
            target.clear()
            target.update(saved)

        self.registry._graphs.clear()
        self.registry._graphs.update(self._saved_graphs)
        for name, saved in self._saved_maps.items():
            target = getattr(self.registry, name)
            target.clear()
            for k, v in saved.items():
                target[k] = list(v)
        super().tearDown()

    def _install(self, lib_symbol, op_symbol="twice", dispatch_key="CPU"):
        key = (lib_symbol, op_symbol, dispatch_key)
        self.registry._register_overrides_from_graph(
            lib_symbol, op_symbol, dispatch_key, self.registry._graphs[key]
        )

    def _register(self, cond, impl, lib_symbol=None):
        self.registry.register_op_override(
            "test_dsl", lib_symbol or self.NS, "twice", "CPU", cond, impl
        )

    def test_graph_key_and_node_carry_the_namespace(self):
        self._register(lambda *a, **k: True, lambda x: x)
        key = (self.NS, "twice", "CPU")
        self.assertIn(key, self.registry._graphs)
        self.assertEqual(self.registry._graphs[key][0].lib_symbol, self.NS)

    def test_cond_true_routes_to_impl(self):
        self._register(lambda *a, **k: True, lambda x: torch.full_like(x, 99.0))
        self._install(self.NS)
        self.assertEqual(self.op(torch.tensor([1.0, 2.0])), torch.tensor([99.0, 99.0]))

    def test_cond_false_falls_back_to_composite_kernel(self):
        def impl(x):
            raise AssertionError("impl must not run when cond is False")

        self._register(lambda *a, **k: False, impl)
        self._install(self.NS)
        self.assertEqual(self.op(torch.tensor([1.0, 2.0])), torch.tensor([2.0, 4.0]))

    def test_teardown_restores_the_original_kernel(self):
        self._register(lambda *a, **k: True, lambda x: torch.full_like(x, 99.0))
        self._install(self.NS)
        self.assertEqual(self.op(torch.tensor([1.0])), torch.tensor([99.0]))

        self.registry._graphs[(self.NS, "twice", "CPU")] = []
        self._install(self.NS)
        self.assertNotIn((self.NS, "twice", "CPU"), self.registry._aten_override_libs)
        self.assertEqual(self.op(torch.tensor([1.0])), torch.tensor([2.0]))

    def _register_both(self):
        self._register(lambda *a, **k: True, lambda x: torch.full_like(x, 99.0))
        self._register(
            lambda *a, **k: True,
            lambda x: torch.full_like(x, 7.0),
            lib_symbol=self.NS2,
        )
        self._install(self.NS)
        self._install(self.NS2)

    def test_same_op_symbol_in_two_namespaces_is_independent(self):
        """Two namespaces defining the same op symbol land in one
        per-op-symbol bucket; each must still route to its own override."""
        self._register_both()

        x = torch.tensor([1.0, 2.0])
        self.assertEqual(self.op(x), torch.full_like(x, 99.0))
        self.assertEqual(self.op2(x), torch.full_like(x, 7.0))

        bucket = self.registry._op_symbol_to_lib_graph["twice"]
        self.assertIn((self.NS, "twice", "CPU"), bucket)
        self.assertIn((self.NS2, "twice", "CPU"), bucket)

    def test_op_symbol_filters_are_namespace_blind(self):
        """`disable_op_symbols` matches the bare op symbol in every namespace,
        and a "ns::op"-qualified string matches nothing. Pinned here because
        the widened keys make same-symbol collisions across namespaces
        possible for the first time."""
        self._register_both()
        x = torch.tensor([1.0, 2.0])

        self.registry.deregister_op_overrides(disable_op_symbols=[f"{self.NS}::twice"])
        self.assertEqual(self.op(x), torch.full_like(x, 99.0))
        self.assertEqual(self.op2(x), torch.full_like(x, 7.0))

        self.registry.deregister_op_overrides(disable_op_symbols=["twice"])
        self.assertEqual(self.op(x), x * 2)
        self.assertEqual(self.op2(x), x * 3)

        self.registry.reenable_op_overrides(enable_op_symbols=["twice"])
        self.assertEqual(self.op(x), torch.full_like(x, 99.0))
        self.assertEqual(self.op2(x), torch.full_like(x, 7.0))

    def test_get_dsl_operations_reports_one_namespace(self):
        """Reporting is scoped: the throwaway namespace's op appears only when
        asked for, and never in the default `aten` view."""
        self._register(lambda *a, **k: True, lambda x: x)
        self.assertEqual(
            self.registry.get_dsl_operations("test_dsl", lib_symbol=self.NS),
            ["twice"],
        )
        self.assertNotIn("twice", self.registry.get_dsl_operations("test_dsl"))

    def test_undefined_op_raises_on_install(self):
        """A namespace whose op is not in the dispatcher must fail loudly at
        install time rather than silently registering nothing.

        Loud failure is the deliberate contract: a registration naming an op
        that does not resolve breaks `import torch` rather than degrading
        quietly. Unlike `aten`, whose ops the build guarantees, a non-aten
        op's existence is a runtime property, so drift is caught before it
        ships by the drift-guard test that accompanies each such override.
        """
        self._register(lambda *a, **k: True, lambda x: x, lib_symbol=self.MISSING_NS)
        with self.assertRaisesRegex(AttributeError, "twice op not found"):
            self._install(self.MISSING_NS)


if __name__ == "__main__":
    run_tests()
