# Owner(s): ["module: dynamo"]

import inspect

import torch
import torch._dynamo
import torch._dynamo.testing
import torch.fx.experimental._config as _fx_experimental_config
from torch._dynamo.decorators import mark_unbacked
from torch._dynamo.dynamic_spec import IntVar, ShapeVar, TensorSpec
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.testing import EagerAndRecordGraphs
from torch.fx.experimental.symbolic_shapes import (
    free_unbacked_symbols,
    GuardOnDataDependentSymNode,
)


def _tensor_placeholder_shape(gm):
    """Return the shape of the first tensor-typed placeholder in ``gm``."""
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            ev = node.meta.get("example_value")
            if isinstance(ev, torch.Tensor):
                return ev.shape
    raise AssertionError("no tensor placeholder found")


def _apply_spec_to_tensor(tensor, shape_spec):
    """Apply per-dim spec to a tensor through ``mark_*`` on each dim.
    This is only for testing purposes. Will be removed in next PR.
    """
    if not isinstance(shape_spec, TensorSpec):
        return
    for idx, spec in enumerate(shape_spec):
        if isinstance(spec, IntVar):
            mark_unbacked(tensor, idx)


def _compile_with_dynamic_shapes(fn, dynamic_spec, **compile_kwargs):
    """Compile ``fn`` and apply ``dynamic_spec`` specs on every call."""

    compile_kwargs.setdefault("dynamic", False)
    compiled = torch.compile(fn, **compile_kwargs)
    sig = inspect.signature(fn)

    @torch._dynamo.disable
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        for name, shape_spec in dynamic_spec.items():
            if name in bound.arguments:
                arg = bound.arguments[name]
                if isinstance(arg, torch.Tensor):
                    _apply_spec_to_tensor(arg, shape_spec)
        return compiled(*bound.args, **bound.kwargs)

    return wrapper


class TestShapeVarConstruction(TestCase):
    """Construction of ShapeVar."""

    def test_basic(self):
        s = ShapeVar("batch")
        self.assertEqual(s.name, "batch")
        self.assertEqual(s.min, 0)
        self.assertIsNone(s.max)
        self.assertIsNone(s.optimization_hint)

    def test_with_max(self):
        s = ShapeVar("batch", max=64)
        self.assertEqual(s.max, 64)

    def test_with_optimization_hint(self):
        s = ShapeVar("seq", max=2048, optimization_hint=512)
        self.assertEqual(s.max, 2048)
        self.assertEqual(s.optimization_hint, 512)

    def test_anonymous_name(self):
        s = ShapeVar()
        self.assertTrue(s.name.startswith("_intvar_"))

    def test_anonymous_specs_have_distinct_names(self):
        a = ShapeVar()
        b = ShapeVar()
        self.assertNotEqual(a.name, b.name)

    def test_is_instance_of_intvar(self):
        s = ShapeVar("batch")
        self.assertIsInstance(s, IntVar)

    def test_repr(self):
        s = ShapeVar("batch", max=64, optimization_hint=32)
        self.assertEqual(
            repr(s), "ShapeVar(name='batch', min=0, max=64, optimization_hint=32)"
        )

    def test_repr_minimal(self):
        s = ShapeVar("x")
        self.assertEqual(repr(s), "ShapeVar(name='x', min=0)")

    def test_implicit_min_is_zero(self):
        s = ShapeVar("x")
        self.assertEqual(s.min, 0)

    def test_custom_min(self):
        s = ShapeVar("x", min=1)
        self.assertEqual(s.min, 1)


class TestIntVarConstruction(TestCase):
    """Construction of IntVar."""

    def test_basic(self):
        s = IntVar("offset")
        self.assertEqual(s.name, "offset")
        self.assertIsNone(s.min)
        self.assertIsNone(s.max)
        self.assertIsNone(s.optimization_hint)

    def test_with_range(self):
        s = IntVar("offset", min=-100, max=100)
        self.assertEqual(s.min, -100)
        self.assertEqual(s.max, 100)

    def test_with_optimization_hint(self):
        s = IntVar("size", min=1, max=2048, optimization_hint=512)
        self.assertEqual(s.optimization_hint, 512)

    def test_anonymous_name(self):
        s = IntVar()
        self.assertTrue(s.name.startswith("_intvar_"))

    def test_anonymous_specs_have_distinct_names(self):
        a = IntVar()
        b = IntVar()
        self.assertNotEqual(a.name, b.name)

    def test_allows_negative_range(self):
        s = IntVar("offset", min=-50, max=-10)
        self.assertEqual(s.min, -50)
        self.assertEqual(s.max, -10)

    def test_repr(self):
        s = IntVar("offset", min=-100, max=100, optimization_hint=0)
        self.assertEqual(
            repr(s), "IntVar(name='offset', min=-100, max=100, optimization_hint=0)"
        )

    def test_repr_minimal(self):
        s = IntVar("x")
        self.assertEqual(repr(s), "IntVar(name='x')")


class TestTensorSpecConstruction(TestCase):
    """Construction and list-like interface."""

    def test_list_construction(self):
        ts = TensorSpec([ShapeVar("batch"), None, 10])
        self.assertIsInstance(ts[0], ShapeVar)
        self.assertIsNone(ts[1])
        self.assertEqual(ts[2], 10)

    def test_tuple_construction(self):
        ts = TensorSpec((ShapeVar("batch"), None))
        self.assertEqual(len(ts), 2)
        self.assertIsInstance(ts[0], ShapeVar)
        self.assertIsNone(ts[1])

    def test_len(self):
        ts = TensorSpec([None, None, None])
        self.assertEqual(len(ts), 3)

    def test_iter(self):
        sv = ShapeVar("batch")
        ts = TensorSpec([sv, None])
        items = list(ts)
        self.assertEqual(len(items), 2)
        self.assertIs(items[0], sv)
        self.assertIsNone(items[1])

    def test_index_out_of_range(self):
        ts = TensorSpec([None, None])
        with self.assertRaises(IndexError):
            ts[5]

    def test_repr(self):
        ts = TensorSpec([ShapeVar("batch"), None, 10])
        r = repr(ts)
        self.assertIn("ShapeVar(", r)
        self.assertIn("None", r)
        self.assertIn("10", r)


class TestShapeVarCompile(TestCase):
    """ShapeVar + torch.compile integration."""

    def test_unbacked_graph_has_unbacked_symbol(self):
        """ShapeVar dim appears as an unbacked SymInt; single compile covers all shapes."""
        backend = EagerAndRecordGraphs()
        fn = _compile_with_dynamic_shapes(
            lambda x: x.sum(0),
            {"x": TensorSpec([ShapeVar("batch"), None])},
            backend=backend,
        )
        for n in [4, 8, 16, 32]:
            fn(torch.randn(n, 3))

        self.assertEqual(len(backend.graphs), 1)
        shape = _tensor_placeholder_shape(backend.graphs[0])
        self.assertIsInstance(shape[0], torch.SymInt)
        self.assertGreater(len(free_unbacked_symbols(shape[0])), 0)

    @_fx_experimental_config.patch(no_data_dependent_graph_break=True)
    def test_unbacked_raises_dde_on_branching(self):
        """A function that branches on size(0) must raise a data-dependent
        error when that dim is marked with ShapeVar (unbacked)."""

        def fn(x):
            if x.size(0) > 5:
                return x + 1
            return x - 1

        compiled = _compile_with_dynamic_shapes(
            fn,
            {"x": TensorSpec([ShapeVar(), None])},
            backend="eager",
            fullgraph=True,
        )
        with self.assertRaises(GuardOnDataDependentSymNode) as cm:
            compiled(torch.randn(10, 3))
        free_syms = cm.exception.cond.free_symbols
        self.assertEqual(len(free_syms), 1)
        (sym,) = free_syms
        self.assertTrue(
            str(sym).startswith("u"),
            msg=f"expected unbacked symbol (u-prefix), got {sym!r}",
        )

    @torch._dynamo.config.patch(automatic_dynamic_shapes=False)
    def test_none_entry_is_static(self):
        """A ``None`` entry doesn't mark the dim — it stays static.
        Each distinct shape at dim 1 triggers a recompile."""
        backend = EagerAndRecordGraphs()
        ts = TensorSpec([ShapeVar("batch"), None])
        fn = _compile_with_dynamic_shapes(
            lambda x: x + 1,
            {"x": ts},
            backend=backend,
        )

        fn(torch.randn(4, 3))
        self.assertEqual(len(backend.graphs), 1)

        fn(torch.randn(8, 3))  # dim 0 unbacked → no recompile; dim 1 same
        self.assertEqual(len(backend.graphs), 1)

        fn(torch.randn(8, 5))  # dim 1 changed → recompile (static)
        self.assertEqual(len(backend.graphs), 2)

        fn(torch.randn(16, 5))  # dim 0 unbacked → no recompile; dim 1 same
        self.assertEqual(len(backend.graphs), 2)


if __name__ == "__main__":
    run_tests()
