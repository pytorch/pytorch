# Owner(s): ["module: dsl-native-ops"]

"""
Tests for native_decomp_table and related compile/export decomposition
machinery exposed by torch._native.registry.

These tests isolate themselves from any overrides that got registered at
import time (e.g. bmm_outer_product) by saving/restoring all relevant
registry-module-level state in setUp / tearDown.
"""

import uuid

import torch
import torch._native.registry as registry_module
from torch._native.registry import native_decomp_table, register_op_override
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


# Common predicates/impls used across several tests.
def _always_true(*a, **kw):
    return True


def _always_false(*a, **kw):
    return False


def _make_fill_impl(value):
    """Build an impl that returns a fresh tensor filled with `value`."""

    def impl(a, *_args, **_kwargs):
        return torch.empty_like(a).fill_(value)

    return impl


@skipIfTorchDynamo("Decomp tests don't need dynamo compilation")
class TestNativeDecompTable(TestCase):
    """Tests for native_decomp_table() and register/deregister syncing."""

    def setUp(self):
        self.registry = registry_module
        # Per-test unique DSL name so generated `_native::<node_id>` ops
        # don't collide across tests. `_defined_native_ops` and the
        # underlying torch dispatcher registration are process-wide and
        # cannot be un-defined; instead, each test uses a fresh name.
        self.dsl_name = f"test_{uuid.uuid4().hex[:8]}"

        # Snapshot registry state that registrations / deregistrations
        # touch. `_graphs` is the source of truth; `_aten_override_libs`
        # and `_libs` are derived (materialized dispatcher registrations)
        # and get rebuilt from `_graphs` on tearDown rather than restored
        # directly -- restoring destroyed Library objects would leave the
        # dispatcher without live kernels. We deliberately do NOT touch
        # `_defined_native_ops` or `_def_libs`; their contents mirror
        # permanent dispatcher state.
        self._snapshot = {
            "graphs": dict(self.registry._graphs),
            "decomp_overrides": dict(self.registry._native_decomp_overrides),
            "dsl_name_to_lib_graph": {
                k: list(v) for k, v in self.registry._dsl_name_to_lib_graph.items()
            },
            "dispatch_key_to_lib_graph": {
                k: list(v) for k, v in self.registry._dispatch_key_to_lib_graph.items()
            },
            "op_symbol_to_lib_graph": {
                k: list(v) for k, v in self.registry._op_symbol_to_lib_graph.items()
            },
            "filter_state": (
                set(self.registry._filter_state._dsl_names),
                set(self.registry._filter_state._op_symbols),
                set(self.registry._filter_state._dispatch_keys),
            ),
        }

        # Destroy live aten overrides and _native IMPL libs so the test's
        # own registrations see pristine aten kernels. These are rebuilt
        # from the snapshotted `_graphs` on tearDown.
        self._destroy_live_libs()

        # Clear dicts that won't corrupt process-wide state.
        self.registry._graphs.clear()
        self.registry._native_decomp_overrides.clear()
        self.registry._dsl_name_to_lib_graph.clear()
        self.registry._dispatch_key_to_lib_graph.clear()
        self.registry._op_symbol_to_lib_graph.clear()
        self.registry._filter_state._dsl_names.clear()
        self.registry._filter_state._op_symbols.clear()
        self.registry._filter_state._dispatch_keys.clear()

    def _destroy_live_libs(self):
        """Tear down every live aten override + _native IMPL library and
        clear their dicts. The C++ dispatcher has no per-kernel removal,
        so Library._destroy() is the only way to unregister."""
        for lib in list(self.registry._aten_override_libs.values()):
            lib._destroy()
        self.registry._aten_override_libs.clear()
        for lib in list(self.registry._libs.values()):
            lib._destroy()
        self.registry._libs.clear()

    def tearDown(self):
        if not hasattr(self, "registry"):
            return
        # Destroy whatever the test installed, then rebuild libs from the
        # snapshotted graph state so other tests (and the rest of the
        # process) see live dispatcher kernels matching the saved graphs.
        self._destroy_live_libs()

        self.registry._graphs.clear()
        self.registry._graphs.update(self._snapshot["graphs"])
        self.registry._native_decomp_overrides.clear()
        self.registry._native_decomp_overrides.update(
            self._snapshot["decomp_overrides"]
        )
        self.registry._dsl_name_to_lib_graph.clear()
        for k, v in self._snapshot["dsl_name_to_lib_graph"].items():
            self.registry._dsl_name_to_lib_graph[k] = list(v)
        self.registry._dispatch_key_to_lib_graph.clear()
        for k, v in self._snapshot["dispatch_key_to_lib_graph"].items():
            self.registry._dispatch_key_to_lib_graph[k] = list(v)
        self.registry._op_symbol_to_lib_graph.clear()
        for k, v in self._snapshot["op_symbol_to_lib_graph"].items():
            self.registry._op_symbol_to_lib_graph[k] = list(v)
        fs = self._snapshot["filter_state"]
        self.registry._filter_state._dsl_names.clear()
        self.registry._filter_state._dsl_names.update(fs[0])
        self.registry._filter_state._op_symbols.clear()
        self.registry._filter_state._op_symbols.update(fs[1])
        self.registry._filter_state._dispatch_keys.clear()
        self.registry._filter_state._dispatch_keys.update(fs[2])

        # Rebuild live libs from the restored graphs.
        self.registry._register_all_overrides()

    # ------------------------------------------------------------------
    # Test helpers.
    # ------------------------------------------------------------------

    def _register(self, op_symbol, cond, impl, dispatch_key="CPU"):
        """Register an aten-namespace override and materialize it."""
        register_op_override(self.dsl_name, "aten", op_symbol, dispatch_key, cond, impl)
        self.registry._register_all_overrides()

    def _export_add(self, cond, impl):
        """Export a module that calls `x + y`; decompose with the native table."""
        self._register("add.Tensor", cond, impl)

        class M(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        ep = torch.export.export(M(), (torch.randn(3), torch.randn(3)))
        return ep.run_decompositions(native_decomp_table())

    def _graph_targets(self, ep):
        """Return the str(target) of every call_function node in an ExportedProgram."""
        return [
            str(node.target)
            for node in ep.graph_module.graph.nodes
            if node.op == "call_function"
        ]

    def assertGraphRoutedToNative(self, ep):
        targets = self._graph_targets(ep)
        self.assertTrue(
            any("_native" in t for t in targets),
            lambda msg: f"{msg}\nExpected a _native call in graph; got targets={targets}\n"
            f"{ep.graph_module.code}",
        )

    def assertGraphNotRoutedToNative(self, ep):
        targets = self._graph_targets(ep)
        for t in targets:
            self.assertNotIn(
                "_native",
                t,
                lambda msg: f"{msg}\nUnexpected _native call; targets={targets}\n{ep.graph_module.code}",
            )

    # ------------------------------------------------------------------
    # Dict-level tests: fast, no export / FakeTensor required.
    # ------------------------------------------------------------------

    def test_table_default_includes_core_aten_and_overrides(self):
        """Default table = core aten decomps + registered overrides."""
        self._register("add.Tensor", _always_true, _make_fill_impl(1))

        table = native_decomp_table()
        self.assertIn(torch.ops.aten.add.Tensor, table)
        # Also contains a lot of default aten decomps (exact count varies by
        # PyTorch version, but it's in the hundreds).
        self.assertGreater(len(table), 100)

    def test_table_override_wins_on_conflict(self):
        """If an op has both a default decomp and a registered override,
        the override wins (merged last)."""
        self._register("add.Tensor", _always_true, _make_fill_impl(99))

        # The stored override is a router closure that wraps our impl -- not
        # our impl directly -- so we verify table identity against the
        # closure we queued.
        table = native_decomp_table()
        self.assertIs(
            table[torch.ops.aten.add.Tensor],
            self.registry._native_decomp_overrides[torch.ops.aten.add.Tensor],
        )

    def test_table_overrides_only(self):
        """overrides_only=True returns just the registered overrides."""
        self._register("add.Tensor", _always_true, _make_fill_impl(1))
        self._register("mul.Tensor", _always_true, _make_fill_impl(2))

        table = native_decomp_table(overrides_only=True)
        self.assertEqual(len(table), 2)
        self.assertIn(torch.ops.aten.add.Tensor, table)
        self.assertIn(torch.ops.aten.mul.Tensor, table)

    def test_deregister_removes_decomp_entry(self):
        """Deregistering an override drops it from the decomp table."""
        self._register("add.Tensor", _always_true, _make_fill_impl(1))
        self.assertIn(
            torch.ops.aten.add.Tensor, native_decomp_table(overrides_only=True)
        )

        self.registry.deregister_op_overrides(disable_dsl_names=self.dsl_name)

        self.assertNotIn(
            torch.ops.aten.add.Tensor, native_decomp_table(overrides_only=True)
        )

    def test_reenable_restores_decomp_entry(self):
        """Reenabling a deregistered override puts it back in the table."""
        self._register("add.Tensor", _always_true, _make_fill_impl(1))
        self.registry.deregister_op_overrides(disable_dsl_names=self.dsl_name)
        self.assertNotIn(
            torch.ops.aten.add.Tensor, native_decomp_table(overrides_only=True)
        )

        self.registry.reenable_op_overrides(enable_dsl_names=self.dsl_name)

        self.assertIn(
            torch.ops.aten.add.Tensor, native_decomp_table(overrides_only=True)
        )

    # ------------------------------------------------------------------
    # End-to-end tests that exercise compile_router's tracing behavior.
    # ------------------------------------------------------------------

    def test_export_with_matching_cond_routes_to_native(self):
        """When cond matches, run_decompositions rewrites aten call to _native."""
        ep = self._export_add(_always_true, _make_fill_impl(7))
        self.assertGraphRoutedToNative(ep)

    def test_export_with_non_matching_cond_preserves_aten(self):
        """When cond returns False, the decomp falls through -- graph keeps aten::<op>."""
        ep = self._export_add(_always_false, _make_fill_impl(-1))
        self.assertGraphNotRoutedToNative(ep)

    def test_export_cond_exception_treated_as_non_match(self):
        """A cond that raises under FakeTensor tracing is swallowed (treated
        as non-match), so the graph falls through to the default lowering."""

        def raising_cond(*a, **kw):
            raise RuntimeError("intentional failure under tracing")

        ep = self._export_add(raising_cond, _make_fill_impl(-1))
        self.assertGraphNotRoutedToNative(ep)


if __name__ == "__main__":
    run_tests()
