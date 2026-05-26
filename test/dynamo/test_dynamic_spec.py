# Owner(s): ["module: dynamo"]

import itertools

import torch
import torch._dynamo
import torch._dynamo.testing
import torch.fx.experimental._config as _fx_experimental_config
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.testing import EagerAndRecordGraphs
from torch.fx.experimental.dynamic_spec import (
    DictSpec,
    IntVar,
    ObjectSpec,
    ParamsSpec,
    ShapesSpec,
    ShapeVar,
    TensorSpec,
)
from torch.fx.experimental.symbolic_shapes import (
    free_unbacked_symbols,
    GuardOnDataDependentSymNode,
)


def _reset_uid_counter():
    """Reset the global IntVar uid counter so uids start at 0 per test."""
    IntVar._uid_counter = itertools.count()


def _tensor_placeholder_shape(gm):
    """Return the shape of the first tensor-typed placeholder in ``gm``."""
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            ev = node.meta.get("example_value")
            if isinstance(ev, torch.Tensor):
                return ev.shape
    raise AssertionError("no tensor placeholder found")


def _tensor_placeholders(gm):
    out = []
    for n in gm.graph.nodes:
        if n.op == "placeholder" and isinstance(
            n.meta.get("example_value"), torch.Tensor
        ):
            out.append(n)
    return out


class TestShapeVarConstruction(TestCase):
    """Construction of ShapeVar."""

    def setUp(self):
        super().setUp()
        _reset_uid_counter()

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
        self.assertEqual(s.name, "anon")

    def test_anonymous_repr_uses_id(self):
        s = ShapeVar()
        self.assertEqual(repr(s), "ShapeVar(anon#0, min=0)")

    def test_anonymous_specs_have_distinct_reprs(self):
        a = ShapeVar()
        b = ShapeVar()
        # Both have name=None but distinct ids -> distinct reprs.
        self.assertNotEqual(repr(a), repr(b))

    def test_same_named_specs_have_distinct_reprs(self):
        a = ShapeVar("batch")
        b = ShapeVar("batch")
        # Same name, distinct ids -> distinct reprs.
        self.assertNotEqual(repr(a), repr(b))

    def test_is_instance_of_intvar(self):
        s = ShapeVar("batch")
        self.assertIsInstance(s, IntVar)

    def test_repr(self):
        s = ShapeVar("batch", max=64, optimization_hint=32)
        self.assertEqual(
            repr(s),
            "ShapeVar(batch#0, min=0, max=64, optimization_hint=32)",
        )

    def test_repr_minimal(self):
        s = ShapeVar("x")
        self.assertEqual(repr(s), "ShapeVar(x#0, min=0)")

    def test_implicit_min_is_zero(self):
        s = ShapeVar("x")
        self.assertEqual(s.min, 0)

    def test_custom_min(self):
        s = ShapeVar("x", min=1)
        self.assertEqual(s.min, 1)


class TestIntVarConstruction(TestCase):
    """Construction of IntVar."""

    def setUp(self):
        super().setUp()
        _reset_uid_counter()

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
        self.assertEqual(s.name, "anon")

    def test_anonymous_repr_uses_id(self):
        s = IntVar()
        self.assertEqual(repr(s), "IntVar(anon#0)")

    def test_anonymous_specs_have_distinct_reprs(self):
        a = IntVar()
        b = IntVar()
        self.assertNotEqual(repr(a), repr(b))

    def test_allows_negative_range(self):
        s = IntVar("offset", min=-50, max=-10)
        self.assertEqual(s.min, -50)
        self.assertEqual(s.max, -10)

    def test_repr(self):
        s = IntVar("offset", min=-100, max=100, optimization_hint=0)
        self.assertEqual(
            repr(s),
            "IntVar(offset#0, min=-100, max=100, optimization_hint=0)",
        )

    def test_repr_minimal(self):
        s = IntVar("x")
        self.assertEqual(repr(s), "IntVar(x#0)")


class TestTensorSpecConstruction(TestCase):
    """Construction and list-like interface."""

    def setUp(self):
        super().setUp()
        _reset_uid_counter()

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
        sv = ShapeVar("batch")
        ts = TensorSpec([sv, None, 10])
        self.assertEqual(
            repr(ts),
            """\
Tensor:
  0: ShapeVar(batch#0, min=0)
  1: None
  2: 10""",
        )


class TestParamsSpecConstruction(TestCase):
    """Construction of ParamsSpec and ShapesSpec."""

    def setUp(self):
        super().setUp()
        _reset_uid_counter()

    def test_shapes_spec_repr(self):
        sv = ShapeVar("batch")
        ss = ShapesSpec(params=ParamsSpec({"x": TensorSpec([sv, None])}))
        self.assertEqual(
            repr(ss),
            """\
shapes_spec:
  params:
    x:
      Tensor:
        0: ShapeVar(batch#0, min=0)
        1: None""",
        )

    def test_params_spec_repr_with_varargs_and_varkw(self):
        sv = ShapeVar("batch")
        ps = ParamsSpec(
            {
                "x": TensorSpec([sv]),
                "*args": [TensorSpec([sv]), None],
                "**kwargs": {"foo": IntVar("k")},
            }
        )
        self.assertEqual(
            repr(ps),
            """\
x:
  Tensor:
    0: ShapeVar(batch#0, min=0)
*args:
  0:
    Tensor:
      0: ShapeVar(batch#0, min=0)
  1: None
**kwargs:
  foo: IntVar(k#1)""",
        )

    def test_params_spec_to_jsonable_with_varargs_and_varkw(self):
        ps = ParamsSpec(
            {
                "x": TensorSpec([ShapeVar("batch")]),
                "*args": [TensorSpec([ShapeVar("a")]), None],
                "**kwargs": {"foo": IntVar("k")},
            }
        )
        self.assertEqual(
            ps.to_jsonable(),
            {
                "type": "ParamsSpec",
                "params": {
                    "x": {
                        "type": "TensorSpec",
                        "dims": [
                            {
                                "type": "ShapeVar",
                                "name": "batch",
                                "min": 0,
                                "max": None,
                                "optimization_hint": None,
                            },
                        ],
                    },
                    "*args": [
                        {
                            "type": "TensorSpec",
                            "dims": [
                                {
                                    "type": "ShapeVar",
                                    "name": "a",
                                    "min": 0,
                                    "max": None,
                                    "optimization_hint": None,
                                }
                            ],
                        },
                        None,
                    ],
                    "**kwargs": {
                        "foo": {
                            "type": "IntVar",
                            "name": "k",
                            "min": None,
                            "max": None,
                            "optimization_hint": None,
                        }
                    },
                },
            },
        )

    def test_params_spec_rejects_unknown_sentinel(self):
        with self.assertRaisesRegex(ValueError, r"Unknown sentinel key"):
            ParamsSpec({"*unknown": TensorSpec([ShapeVar("a")])})

    def test_params_spec_rejects_bad_varargs_value(self):
        with self.assertRaisesRegex(ValueError, r"\*args.*must be a list"):
            ParamsSpec({"*args": TensorSpec([ShapeVar("a")])})  # not a list

    def test_params_spec_rejects_bad_varkw_value(self):
        with self.assertRaisesRegex(ValueError, r"\*\*kwargs.*must be a dict"):
            ParamsSpec({"**kwargs": [TensorSpec([ShapeVar("a")])]})  # not a dict


class TestShapeVarCompile(TestCase):
    """ShapeVar + torch.compile integration."""

    def test_static_int_spec_mismatch_raises(self):
        """If shapes_spec declares a scalar int as static=10, passing 42 should error."""

        def fn(x, n):
            return x + n

        compiled = torch.compile(
            fn,
            backend="eager",
            shapes_spec={"n": 10},
        )
        with self.assertRaisesRegex(
            torch._dynamo.exc.InternalTorchDynamoError,
            r"shapes_spec declares L\['n'\] as static with value 10, but got 42",
        ):
            compiled(torch.randn(4), 42)

    def test_static_tensor_dim_mismatch_raises(self):
        """If shapes_spec declares dim 1 as static=3, passing a tensor with dim 1=5 should error."""

        def fn(x):
            return x + 1

        compiled = torch.compile(
            fn,
            backend="eager",
            shapes_spec={"x": TensorSpec([ShapeVar("batch"), 3])},
        )
        with self.assertRaisesRegex(
            torch._dynamo.exc.InternalTorchDynamoError,
            r"shapes_spec declares dim 1 as static with value 3, but got 5",
        ):
            compiled(torch.randn(4, 5))

    def test_unbacked_graph_has_unbacked_symbol(self):
        """ShapeVar dim appears as an unbacked SymInt; single compile covers all shapes."""
        backend = EagerAndRecordGraphs()
        fn = torch.compile(
            lambda x: x.sum(0),
            backend=backend,
            shapes_spec={"x": TensorSpec([ShapeVar("batch"), None])},
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

        compiled = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            shapes_spec={"x": TensorSpec([ShapeVar(), None])},
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

    def test_none_entry_is_static(self):
        """A ``None`` entry is implicit static.
        Each distinct shape at dim 1 triggers a recompile."""
        backend = EagerAndRecordGraphs()
        ts = TensorSpec([ShapeVar("batch"), None])
        fn = torch.compile(
            lambda x: x + 1,
            fullgraph=True,
            backend=backend,
            shapes_spec={"x": ts},
        )

        fn(torch.randn(4, 3))
        fn(torch.randn(4, 5))
        fn(torch.randn(4, 7))
        self.assertEqual(len(backend.graphs), 3)
        # dim 0 is ShapeVar -> SymInt; dim 1 is None -> static int
        shape = _tensor_placeholder_shape(backend.graphs[0])
        self.assertIsInstance(shape[0], torch.SymInt)
        self.assertIsInstance(shape[1], int)

    def test_none_scalar_int_is_static(self):
        """A scalar int arg not mentioned in shapes_spec stays static (specialized).
        Each distinct value triggers a recompile."""
        backend = EagerAndRecordGraphs()

        def fn(x, n):
            return x + n

        compiled = torch.compile(
            fn,
            backend=backend,
            shapes_spec={"x": TensorSpec([ShapeVar("batch")])},
        )

        compiled(torch.randn(2), 10)
        compiled(torch.randn(3), 10)
        compiled(torch.randn(4), 20)
        compiled(torch.randn(5), 20)
        compiled(torch.randn(6), 30)
        self.assertEqual(len(backend.graphs), 3)
        # n is specialized -> not a placeholder; x dim 0 is SymInt (ShapeVar)
        phs = _tensor_placeholders(backend.graphs[0])
        self.assertEqual(len(phs), 1)
        self.assertIsInstance(phs[0].meta["example_value"].shape[0], torch.SymInt)

    def test_unspecified_tensor_is_all_static(self):
        """A tensor not mentioned in shapes_spec has all dims static when
        shapes_spec is provided."""
        backend = EagerAndRecordGraphs()

        def fn(x, y):
            return x * 19, y * 10

        compiled = torch.compile(
            fn,
            backend=backend,
            shapes_spec={"x": TensorSpec([ShapeVar("batch"), None])},
        )
        x = torch.randn(12, 3)
        compiled(x, torch.randn(4, 3))
        # y not in spec -> all static; changing y's shape recompiles each time
        compiled(x, torch.randn(8, 3))
        compiled(x, torch.randn(12, 3))
        compiled(x, torch.randn(16, 3))
        self.assertEqual(len(backend.graphs), 4)

        # x dim 0 is SymInt (ShapeVar), x dim 1 is static; y is fully static
        phs = _tensor_placeholders(backend.graphs[0])
        self.assertEqual(len(phs), 2)
        x_shape = phs[0].meta["example_value"].shape
        y_shape = phs[1].meta["example_value"].shape
        self.assertIsInstance(x_shape[0], torch.SymInt)
        self.assertIsInstance(x_shape[1], int)
        self.assertIsInstance(y_shape[0], int)
        self.assertIsInstance(y_shape[1], int)

    def test_params_spec_shorthand(self):
        """shapes_spec=ParamsSpec(...) is auto-wrapped into ShapesSpec."""
        backend = EagerAndRecordGraphs()
        fn = torch.compile(
            lambda x: x.sum(0),
            backend=backend,
            shapes_spec=ParamsSpec({"x": TensorSpec([ShapeVar("batch"), None])}),
        )
        for n in [4, 8, 16]:
            fn(torch.randn(n, 3))
        # Unbacked dim 0 -> single compile
        self.assertEqual(len(backend.graphs), 1)

    def test_dict_shorthand(self):
        """shapes_spec={...} (bare dict) is auto-wrapped into
        ShapesSpec(params=ParamsSpec(dict))."""
        backend = EagerAndRecordGraphs()
        fn = torch.compile(
            lambda x: x.sum(0),
            backend=backend,
            shapes_spec={"x": TensorSpec([ShapeVar("batch"), None])},
        )
        for n in [4, 8, 16]:
            fn(torch.randn(n, 3))
        self.assertEqual(len(backend.graphs), 1)

    def test_normalize_rejects_bad_type(self):
        """Passing something that's not dict/ParamsSpec/ShapesSpec/None
        should raise TypeError at compile entry."""
        with self.assertRaisesRegex(TypeError, "shapes_spec must be"):
            torch.compile(lambda x: x, shapes_spec="not a spec")

    @_fx_experimental_config.patch(no_data_dependent_graph_break=True)
    def test_min_max_bypasses_dde_on_branching(self):
        """Mirror of test_unbacked_raises_dde_on_branching: setting min/max on
        ShapeVar constrains the symbol so the same branch is statically
        evaluated and does NOT raise a data-dependent error."""

        def fn(x):
            if x.size(0) > 5:
                return x + 1
            return x - 1

        compiled = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            shapes_spec={"x": TensorSpec([ShapeVar("batch", min=10, max=100), None])},
        )
        # min=10 > 5 → branch resolves statically, no DDE.
        compiled(torch.randn(20, 3))

    def test_shapes_spec_with_dynamic_raises(self):
        """Setting both shapes_spec and dynamic should raise ValueError."""
        ts = TensorSpec([ShapeVar("batch"), None])
        for dynamic in (True, False):
            with self.assertRaisesRegex(
                ValueError,
                r"`dynamic` and `shapes_spec` cannot both be set",
            ):
                torch.compile(
                    lambda x: x + 1,
                    backend="eager",
                    dynamic=dynamic,
                    shapes_spec={"x": ts},
                )

    def test_tensor_dim_optimization_hint_in_shape_env(self):
        """ShapeVar's optimization_hint propagates to shape_env.var_to_hint_override."""
        backend = EagerAndRecordGraphs()
        fn = torch.compile(
            lambda x: x.sum(0),
            backend=backend,
            shapes_spec={
                "x": TensorSpec([ShapeVar("batch", optimization_hint=32), None])
            },
        )
        fn(torch.randn(8, 3))
        shape = _tensor_placeholder_shape(backend.graphs[0])
        sym = shape[0]
        self.assertIsInstance(sym, torch.SymInt)
        expr = sym.node.expr
        self.assertEqual(sym.node.shape_env.var_to_hint_override.get(expr), 32)

    def test_scalar_int_optimization_hint_in_shape_env(self):
        """IntVar's optimization_hint propagates to shape_env.var_to_hint_override."""
        backend = EagerAndRecordGraphs()

        def fn(x, n):
            return x + n

        compiled = torch.compile(
            fn,
            backend=backend,
            shapes_spec={"n": IntVar("size", optimization_hint=128)},
        )
        compiled(torch.randn(4), 100)
        sym = None
        for node in backend.graphs[0].graph.nodes:
            if node.op == "placeholder":
                ev = node.meta.get("example_value")
                if isinstance(ev, torch.SymInt):
                    sym = ev
                    break
        self.assertIsNotNone(sym)
        expr = sym.node.expr
        self.assertEqual(sym.node.shape_env.var_to_hint_override.get(expr), 128)


class TestShapeVarDedup(TestCase):
    """Sharing the same ShapeVar / IntVar across spec positions emits a
    runtime equality check between the resulting unbacked SymInts, so
    comparisons inside the compiled fn don't raise DDE.
    """

    def setUp(self):
        super().setUp()
        _reset_uid_counter()

    def test_tensor_dim_vs_tensor_dim(self):
        """Two tensor dims sharing one ShapeVar: x.shape[0] == y.shape[0]
        must not DDE inside the compiled fn."""
        B = ShapeVar("batch")

        def fn(x, y):
            if x.shape[0] == y.shape[0]:
                return x.sum() + y.sum()
            return x.sum() - y.sum()

        compiled = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            shapes_spec={
                "x": TensorSpec([B, None]),
                "y": TensorSpec([B, None]),
            },
        )
        out = compiled(torch.randn(8, 3), torch.randn(8, 4))
        self.assertTrue(torch.is_tensor(out))

    def test_tensor_dim_vs_int(self):
        """A tensor dim and a scalar int sharing one ShapeVar: comparing
        them inside the compiled fn must not DDE."""
        B = ShapeVar("batch")

        def fn(x, n):
            if x.shape[0] == n:
                return x.sum() + n
            return x.sum() - n

        compiled = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            shapes_spec={
                "x": TensorSpec([B, None]),
                "n": B,
            },
        )
        out = compiled(torch.randn(8, 3), 8)
        self.assertTrue(torch.is_tensor(out))

    def test_int_vs_int(self):
        """Two scalar int args sharing one IntVar: a == b must not DDE."""
        S = IntVar("size")

        def fn(a, b):
            if a == b:
                return a + b
            return a - b

        compiled = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            shapes_spec={"a": S, "b": S},
        )
        self.assertEqual(compiled(4, 4), 8)

    def test_distinct_shape_vars_still_dde(self):
        """Sanity check: distinct ShapeVars do not share a symbol, so
        comparing them still DDEs (no dedup contamination across vars)."""
        A = ShapeVar("a")
        B = ShapeVar("b")

        def fn(x, y):
            if x.shape[0] == y.shape[0]:
                return x.sum() + y.sum()
            return x.sum() - y.sum()

        compiled = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            shapes_spec={
                "x": TensorSpec([A, None]),
                "y": TensorSpec([B, None]),
            },
        )
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            r"Could not guard on data-dependent expression",
        ):
            compiled(torch.randn(8, 3), torch.randn(8, 4))


class TestDerivedDimSpec(TestCase):
    def setUp(self):
        super().setUp()
        _reset_uid_counter()

    def test_derived_dim(self):
        """y's dim 0 = 2 * x's dim 0: correct shape runs; mismatched shape
        raises with the failed guard expression. The derived spec also lets
        ``y.size()[0] == 2 * x.size()[0]`` inside the compiled fn resolve
        without DDE."""
        B = ShapeVar("batch")

        def fn(x, y):
            if y.size()[0] == 2 * x.size()[0]:
                return x.sum() + y.sum()
            return x.sum() - y.sum()

        compiled = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            shapes_spec={
                "x": TensorSpec([B, None]),
                "y": TensorSpec([B * 2, None]),
            },
        )
        # Correct: y.shape[0] = 8 = 2 * x.shape[0]
        out = compiled(torch.randn(4, 3), torch.randn(8, 5))
        self.assertTrue(torch.is_tensor(out))

        # Violation: 7 != 2 * 4 → guard fails
        torch._dynamo.reset()
        with self.assertRaisesRegex(AssertionError, "Guard fail"):
            compiled(torch.randn(4, 3), torch.randn(7, 5))

        # Sanity: WITHOUT the derived spec the same conditional DDEs.
        torch._dynamo.reset()
        Y = ShapeVar("y_dim")
        compiled_no_derived = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            shapes_spec={
                "x": TensorSpec([B, None]),
                "y": TensorSpec([Y, None]),
            },
        )
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            r"Could not guard on data-dependent expression",
        ):
            compiled_no_derived(torch.randn(4, 3), torch.randn(8, 5))

    def test_multi_var_derived(self):
        """Composite expression over multiple IntVars: z.shape[0] = A * B + 1.
        Correct shape runs; mismatched shape raises. The derived spec also
        lets ``z.size()[0] == x.size()[0] * y.size()[0] + 1`` resolve
        without DDE."""
        A = ShapeVar("a")
        B = ShapeVar("b")

        def fn(x, y, z):
            if z.size()[0] == x.size()[0] * y.size()[0] + 1:
                return x.sum() + y.sum() + z.sum()
            return x.sum() - y.sum() - z.sum()

        compiled = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            shapes_spec={
                "x": TensorSpec([A, None]),
                "y": TensorSpec([B, None]),
                "z": TensorSpec([A * B + 1, None]),
            },
        )
        # Correct: z.shape[0] = 3 * 4 + 1 = 13
        out = compiled(torch.randn(3, 2), torch.randn(4, 2), torch.randn(13, 2))
        self.assertTrue(torch.is_tensor(out))

        # Violation: 99 != 3 * 4 + 1
        torch._dynamo.reset()
        with self.assertRaisesRegex(AssertionError, "Guard fail"):
            compiled(torch.randn(3, 2), torch.randn(4, 2), torch.randn(99, 2))

        # Sanity: WITHOUT the derived spec the same conditional DDEs.
        torch._dynamo.reset()
        Z = ShapeVar("z_dim")
        compiled_no_derived = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            shapes_spec={
                "x": TensorSpec([A, None]),
                "y": TensorSpec([B, None]),
                "z": TensorSpec([Z, None]),
            },
        )
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            r"Could not guard on data-dependent expression",
        ):
            compiled_no_derived(
                torch.randn(3, 2), torch.randn(4, 2), torch.randn(13, 2)
            )

    def test_orphan_intvar_raises(self):
        """B is used in derived expression but never as a bare slot → finalize raises."""
        A = ShapeVar("a")
        B = ShapeVar("b")

        def fn(x):
            return x.sum()

        # A * B in slot but neither A nor B has a bare-IntVar binding via inputs
        compiled = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            shapes_spec={"x": TensorSpec([A * B, None])},
        )
        with self.assertRaises(torch._dynamo.exc.InternalTorchDynamoError) as cm:
            compiled(torch.randn(4, 3))
        self.assertIn(
            "ValueError: shapes_spec: 1 pending check(s) reference unbound "
            "IntVar(s) ['a', 'b']. Every IntVar used in a derived "
            "expression or assumption must also appear as a bare-IntVar slot "
            "somewhere in the spec.",
            str(cm.exception),
        )

    def test_derived_scalar_arg(self):
        """Scalar arg slot can be a derived expression: n must equal 2 * x.shape[0].
        Correct value runs; mismatched value raises. The derived spec also
        lets ``n == 2 * x.size()[0]`` resolve without DDE."""
        B = ShapeVar("batch")

        def fn(x, n):
            if n == 2 * x.size()[0]:
                return x.sum() + n
            return x.sum() - n

        compiled = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            shapes_spec={"x": TensorSpec([B, None]), "n": B * 2},
        )
        # Correct: n = 8 = 2 * x.size()[0]
        out = compiled(torch.randn(4, 3), 8)
        self.assertTrue(torch.is_tensor(out))

        # Violation: n = 7 != 2 * 4
        torch._dynamo.reset()
        with self.assertRaisesRegex(AssertionError, "Guard fail"):
            compiled(torch.randn(4, 3), 7)

        # Sanity: WITHOUT the derived spec the same conditional DDEs.
        torch._dynamo.reset()
        N = IntVar("n_var")
        compiled_no_derived = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            shapes_spec={"x": TensorSpec([B, None]), "n": N},
        )
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            r"Could not guard on data-dependent expression",
        ):
            compiled_no_derived(torch.randn(4, 3), 8)

    def test_foreign_symint_rejected_at_construction(self):
        """A SymInt backed by a different ShapeEnv must be rejected at
        TensorSpec construction."""
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        real_env = ShapeEnv()
        foreign_symint = real_env.create_unbacked_symint()
        with self.assertRaisesRegex(
            TypeError,
            r"TensorSpec dim 0: SymInt spec values must originate from spec "
            r"IntVar / ShapeVar; got u0 backed by a different ShapeEnv\.",
        ):
            TensorSpec([foreign_symint, None])

    def test_misuse_in_python_conditional_raises(self):
        """Using a spec IntVar in a Python bool context raises (don't allow
        accidental guards on spec-time values)."""
        A = ShapeVar("a")
        with self.assertRaises(Exception):
            if A > 1:
                pass

    def test_misuse_torch_check_outside_assumptions_raises(self):
        """torch._check on a spec IntVar outside the assumptions context raises."""
        A = ShapeVar("a")
        with self.assertRaises(Exception):
            torch._check(A > 1)

    def test_order_independence(self):
        """Composite slot (z = A * B) wired BEFORE bare-IntVar slots: the
        derived check is deferred, then emitted once both A and B are bound.
        Correct shape runs; mismatched shape raises with the failed guard."""
        A = ShapeVar("a")
        B = ShapeVar("b")

        def fn(z, x, y):
            # z processed first (dim references unbound A and B);
            # x binds A, y binds B; pending check then fires.
            if z.size()[0] == x.size()[0] * y.size()[0]:
                return z.sum() + x.sum() + y.sum()
            return z.sum() - x.sum() - y.sum()

        compiled = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            shapes_spec={
                "z": TensorSpec([A * B, None]),
                "x": TensorSpec([A, None]),
                "y": TensorSpec([B, None]),
            },
        )
        # Correct: z.shape[0] = 3 * 4 = 12
        out = compiled(torch.randn(12, 2), torch.randn(3, 2), torch.randn(4, 2))
        self.assertTrue(torch.is_tensor(out))

        # Violation: 99 != 3 * 4
        torch._dynamo.reset()
        with self.assertRaisesRegex(AssertionError, "Guard fail"):
            compiled(torch.randn(99, 2), torch.randn(3, 2), torch.randn(4, 2))

        # Sanity: WITHOUT the derived spec the same conditional DDEs.
        torch._dynamo.reset()
        Z = ShapeVar("z_dim")
        compiled_no_derived = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            shapes_spec={
                "z": TensorSpec([Z, None]),
                "x": TensorSpec([A, None]),
                "y": TensorSpec([B, None]),
            },
        )
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            r"Could not guard on data-dependent expression",
        ):
            compiled_no_derived(
                torch.randn(12, 2), torch.randn(3, 2), torch.randn(4, 2)
            )

    def test_same_derived_expr_in_two_slots(self):
        """Two tensor dims both spec'd as the same derived expression (B * 2)
        must both equal each other at runtime.

        Note: the equality is enforced at runtime (each slot deferred-asserts
        ``u_i == 2 * u_x``) but ShapeEnv doesn't transitively conclude
        ``u_y == u_z`` at compile time, so ``if y.size()[0] == z.size()[0]``
        would DDE."""
        B = ShapeVar("batch")

        def fn(x, y, z):
            return x.sum() + y.sum() + z.sum()

        compiled = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            shapes_spec={
                "x": TensorSpec([B, None]),
                "y": TensorSpec([B * 2, None]),
                "z": TensorSpec([B * 2, None]),
            },
        )
        # y.shape[0] and z.shape[0] both = 8 (= 2 * x.shape[0])
        out = compiled(torch.randn(4, 3), torch.randn(8, 5), torch.randn(8, 5))
        self.assertTrue(torch.is_tensor(out))


class TestObjectSpec(TestCase):
    """``ObjectSpec`` data class — construction, access, repr."""

    def setUp(self):
        super().setUp()
        _reset_uid_counter()

    def test_dict_construction(self):
        """Fields supplied via the constructor preserve identity and order."""
        spec = TensorSpec([ShapeVar("h"), None])
        os = ObjectSpec({"weight": spec, "bias": None})
        self.assertEqual(len(os), 2)
        self.assertIn("weight", os)
        self.assertIs(os._fields["weight"], spec)
        self.assertIsNone(os._fields["bias"])

    def test_iter_and_items(self):
        """Iteration walks fields in insertion order; items() yields pairs."""
        sv_w = ShapeVar("h")
        sv_b = ShapeVar("h")
        ts_w = TensorSpec([sv_w, None])
        ts_b = TensorSpec([sv_b])
        os = ObjectSpec({"weight": ts_w, "bias": ts_b})
        self.assertEqual(list(os), ["weight", "bias"])
        self.assertEqual(list(os.items()), [("weight", ts_w), ("bias", ts_b)])

    def test_recursive_nesting(self):
        """A field's value may itself be an ``ObjectSpec``."""
        inner_spec = TensorSpec([ShapeVar("h"), None])
        inner = ObjectSpec({"weight": inner_spec})
        outer = ObjectSpec({"inner": inner})
        self.assertIs(outer._fields["inner"], inner)
        self.assertIs(inner._fields["weight"], inner_spec)

    def test_repr_with_none_leaf(self):
        """Single-field repr with a ``None`` leaf renders inline."""
        os = ObjectSpec({"weight": None})
        self.assertEqual(
            repr(os),
            """\
object_spec:
  .weight: None""",
        )

    def test_repr_with_tensorspec(self):
        """Multi-line leaf repr is indented under its field name."""
        os = ObjectSpec({"weight": TensorSpec([ShapeVar("batch"), None])})
        self.assertEqual(
            repr(os),
            """\
object_spec:
  .weight:
    Tensor:
      0: ShapeVar(batch#0, min=0)
      1: None""",
        )

    def test_repr_nested(self):
        """Nested ``ObjectSpec`` repr indents recursively."""
        inner = ObjectSpec({"weight": None})
        outer = ObjectSpec({"inner": inner})
        self.assertEqual(
            repr(outer),
            """\
object_spec:
  .inner:
    object_spec:
      .weight: None""",
        )

    def test_to_jsonable(self):
        """``to_jsonable`` recurses into spec children; raw leaves pass through."""
        os = ObjectSpec(
            {
                "weight": TensorSpec([ShapeVar("h"), None]),
                "bias": None,
            }
        )
        out = os.to_jsonable()
        self.assertEqual(out["type"], "ObjectSpec")
        self.assertIsInstance(out["fields"]["weight"], dict)
        self.assertEqual(out["fields"]["weight"]["type"], "TensorSpec")
        self.assertIsNone(out["fields"]["bias"])


class TestObjectSpecCompile(TestCase):
    """End-to-end: ``ObjectSpec`` routes through to a tensor reached
    via attribute access on a function arg (``obj.w``)."""

    def setUp(self):
        super().setUp()
        _reset_uid_counter()

    def test_attr_tensor_dim_dynamic(self):
        """A tensor reached via attribute access (``obj.w``) honors the
        ``ShapeVar`` in its ``ObjectSpec`` field; varying that dim does
        not recompile."""

        class Container:
            def __init__(self, w):
                self.w = w

        def fn(obj):
            return obj.w + 1

        backend = EagerAndRecordGraphs()
        compiled = torch.compile(
            fn,
            backend=backend,
            shapes_spec={"obj": ObjectSpec({"w": TensorSpec([ShapeVar("h"), None])})},
        )

        compiled(Container(torch.randn(4, 3)))
        self.assertEqual(len(backend.graphs), 1)

        # Different dim 0 size — dynamic absorbs it, no recompile.
        compiled(Container(torch.randn(8, 3)))
        self.assertEqual(len(backend.graphs), 1)

        # Captured weight placeholder has a SymInt at dim 0.
        shape = _tensor_placeholder_shape(backend.graphs[-1])
        self.assertIsInstance(shape[0], torch.SymInt)

    def test_nn_module_parameter_dim_dynamic(self):
        """An ``nn.Parameter`` reached via ``self.weight`` honors the
        spec. Parameters are normally force-marked static and routed
        as graph attributes, bypassing ``_automatic_dynamic``; the
        spec-aware specialization fixes in ``wrap_module`` /
        ``wrap_tensor`` skip those fast paths when a spec applies, so
        the Parameter flows through the dynamic-shape path."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(4, 3))

            def forward(self):
                return self.weight + 1

        m = M()
        backend = EagerAndRecordGraphs()
        compiled = torch.compile(
            m,
            backend=backend,
            shapes_spec={
                "self": ObjectSpec({"weight": TensorSpec([ShapeVar("h"), None])})
            },
        )

        compiled()
        self.assertEqual(len(backend.graphs), 1)
        m.weight = torch.nn.Parameter(torch.randn(8, 3))
        compiled()
        self.assertEqual(len(backend.graphs), 1)

        # Captured weight placeholder has a SymInt at dim 0.
        shape = _tensor_placeholder_shape(backend.graphs[-1])
        self.assertIsInstance(shape[0], torch.SymInt)


class TestDictSpecCompile(TestCase):
    def setUp(self):
        super().setUp()
        _reset_uid_counter()

    def test_dict_spec_value_marked_dynamic(self):
        """A tensor reached via ``cfg["x"]`` honors the ``ShapeVar`` in
        its ``DictSpec`` entry; varying that dim does not recompile."""

        def fn(cfg):
            return cfg["x"] + 1

        backend = EagerAndRecordGraphs()
        compiled = torch.compile(
            fn,
            backend=backend,
            shapes_spec={"cfg": DictSpec({"x": TensorSpec([ShapeVar("h"), None])})},
        )

        compiled({"x": torch.randn(4, 3)})
        compiled({"x": torch.randn(8, 3)})

        self.assertEqual(len(backend.graphs), 1)

        # Captured placeholder has a SymInt at dim 0.
        shape = _tensor_placeholder_shape(backend.graphs[-1])
        self.assertIsInstance(shape[0], torch.SymInt)

    def test_dict_spec_missing_key_stays_static(self):
        """A dict entry not present in ``DictSpec`` is all-static; each
        new shape recompiles."""

        def fn(cfg):
            return cfg["x"].sum() + cfg["y"].sum()

        backend = EagerAndRecordGraphs()
        compiled = torch.compile(
            fn,
            backend=backend,
            shapes_spec={"cfg": DictSpec({"x": TensorSpec([ShapeVar("h"), None])})},
        )

        x = torch.randn(4, 3)
        compiled({"x": x, "y": torch.randn(4, 3)})
        # y not in spec -> all static; changing y's shape recompiles each time
        compiled({"x": x, "y": torch.randn(4, 5)})
        compiled({"x": x, "y": torch.randn(4, 7)})
        self.assertEqual(len(backend.graphs), 3)

        # x dim 0 is SymInt (ShapeVar via DictSpec); y is fully static.
        phs = _tensor_placeholders(backend.graphs[0])
        self.assertEqual(len(phs), 2)
        x_shape = phs[0].meta["example_value"].shape
        y_shape = phs[1].meta["example_value"].shape
        self.assertIsInstance(x_shape[0], torch.SymInt)
        self.assertIsInstance(x_shape[1], int)
        self.assertIsInstance(y_shape[0], int)
        self.assertIsInstance(y_shape[1], int)

    def test_dict_spec_int_key_marked_dynamic(self):
        """``DictSpec`` accepts int keys; ``cfg[0]`` honors the entry."""

        def fn(cfg):
            return cfg[0] + 1

        backend = EagerAndRecordGraphs()
        compiled = torch.compile(
            fn,
            backend=backend,
            shapes_spec={"cfg": DictSpec({0: TensorSpec([ShapeVar("h"), None])})},
        )

        compiled({0: torch.randn(4, 3)})
        compiled({0: torch.randn(8, 3)})
        self.assertEqual(len(backend.graphs), 1)

        shape = _tensor_placeholder_shape(backend.graphs[-1])
        self.assertIsInstance(shape[0], torch.SymInt)

    def test_walk_terminal_container_raises(self):
        """The spec declares ``x`` as a ``DictSpec`` (a container), but the
        compiled function is called with a plain tensor for ``x``. Because the
        spec walk terminates on the container instead of a ``TensorSpec`` leaf,
        compilation must raise ``RuntimeError`` with ``"shapes_spec walk ended
        on a container"``."""

        def fn(x):
            return x + 1

        # Spec says ``x`` is a DictSpec, but the user passes a tensor.
        # The spec walk terminates on the DictSpec container at the
        # arg root, which is not a leaf.
        compiled = torch.compile(
            fn,
            backend="eager",
            shapes_spec={"x": DictSpec({"k": TensorSpec([ShapeVar("h"), None])})},
        )

        with self.assertRaisesRegex(
            (torch._dynamo.exc.InternalTorchDynamoError, RuntimeError),
            r"shapes_spec walk ended on a container",
        ):
            compiled(torch.randn(4, 3))


class TestVarargsCompile(TestCase):
    def setUp(self):
        super().setUp()
        _reset_uid_counter()

    def test_named_and_varargs_and_varkw(self):
        """Combined: named tensor + ``*args`` tensor + ``**kwargs`` tensor.
        ``ShapeVar`` dims become SymInts
        """

        def f(x, *args, **kwargs):
            return x + args[0] + kwargs["foo"]

        backend = EagerAndRecordGraphs()
        compiled = torch.compile(
            f,
            backend=backend,
            shapes_spec={
                "x": TensorSpec([ShapeVar("a"), None]),
                "*args": [TensorSpec([ShapeVar("b"), None])],
                "**kwargs": {"foo": TensorSpec([ShapeVar("c"), None])},
            },
        )

        compiled(torch.randn(4, 3), torch.randn(4, 3), foo=torch.randn(4, 3))
        self.assertEqual(len(backend.graphs), 1)

        # Vary only the ShapeVar-marked dims -> no recompile.
        compiled(torch.randn(8, 3), torch.randn(8, 3), foo=torch.randn(8, 3))
        compiled(torch.randn(16, 3), torch.randn(16, 3), foo=torch.randn(16, 3))
        self.assertEqual(len(backend.graphs), 1)

        # All three tensor placeholders have SymInt at dim 0 and static int at dim 1.
        phs = _tensor_placeholders(backend.graphs[0])
        self.assertEqual(len(phs), 3)
        for ph in phs:
            shape = ph.meta["example_value"].shape
            self.assertIsInstance(shape[0], torch.SymInt)
            self.assertIsInstance(shape[1], int)

    def test_pure_varargs(self):
        """A ``*args``-only spec marks each positional tensor's dim 0 dynamic."""

        def f(*args):
            return args[0] + args[1]

        backend = EagerAndRecordGraphs()
        compiled = torch.compile(
            f,
            backend=backend,
            shapes_spec={
                "*args": [
                    TensorSpec([ShapeVar("a"), None]),
                    TensorSpec([ShapeVar("b"), None]),
                ],
            },
        )

        compiled(torch.randn(4, 3), torch.randn(4, 3))
        compiled(torch.randn(8, 3), torch.randn(8, 3))
        compiled(torch.randn(16, 3), torch.randn(16, 3))
        self.assertEqual(len(backend.graphs), 1)

        phs = _tensor_placeholders(backend.graphs[0])
        self.assertEqual(len(phs), 2)
        for ph in phs:
            shape = ph.meta["example_value"].shape
            self.assertIsInstance(shape[0], torch.SymInt)
            self.assertIsInstance(shape[1], int)

    def test_varargs_extra_positions_are_static(self):
        """Positional args past the end of the ``*args`` spec list are
        treated as fully static — changing their shape triggers a recompile."""

        def f(*args):
            return args[0].sum() + args[1].sum() + args[2].sum()

        backend = EagerAndRecordGraphs()
        compiled = torch.compile(
            f,
            backend=backend,
            shapes_spec={
                # Spec covers only the first 2 *args entries.
                "*args": [
                    TensorSpec([ShapeVar("a"), None]),
                    TensorSpec([ShapeVar("b"), None]),
                ],
            },
        )

        # Vary the third (unspecified) arg's shape — each new shape recompiles
        # because it is treated as static.
        compiled(torch.randn(4, 3), torch.randn(4, 3), torch.randn(4, 3))
        compiled(torch.randn(8, 3), torch.randn(8, 3), torch.randn(4, 5))
        compiled(torch.randn(16, 3), torch.randn(16, 3), torch.randn(4, 7))
        self.assertEqual(len(backend.graphs), 3)

        # First two *args have SymInt dim 0 (ShapeVar); third is fully static.
        phs = _tensor_placeholders(backend.graphs[0])
        self.assertEqual(len(phs), 3)
        self.assertIsInstance(phs[0].meta["example_value"].shape[0], torch.SymInt)
        self.assertIsInstance(phs[1].meta["example_value"].shape[0], torch.SymInt)
        self.assertIsInstance(phs[2].meta["example_value"].shape[0], int)
        self.assertIsInstance(phs[2].meta["example_value"].shape[1], int)

    def test_pure_varkw(self):
        """A ``**kwargs``-only spec marks each keyword tensor's dim 0 dynamic."""

        def f(**kwargs):
            return kwargs["a"] + kwargs["b"]

        backend = EagerAndRecordGraphs()
        compiled = torch.compile(
            f,
            backend=backend,
            shapes_spec={
                "**kwargs": {
                    "a": TensorSpec([ShapeVar("a"), None]),
                    "b": TensorSpec([ShapeVar("b"), None]),
                },
            },
        )

        compiled(a=torch.randn(4, 3), b=torch.randn(4, 3))
        compiled(a=torch.randn(8, 3), b=torch.randn(8, 3))
        compiled(a=torch.randn(16, 3), b=torch.randn(16, 3))
        self.assertEqual(len(backend.graphs), 1)

        phs = _tensor_placeholders(backend.graphs[0])
        self.assertEqual(len(phs), 2)
        for ph in phs:
            shape = ph.meta["example_value"].shape
            self.assertIsInstance(shape[0], torch.SymInt)
            self.assertIsInstance(shape[1], int)


class TestWalkSpecRaises(TestCase):
    """Unit-style tests for ``_walk_spec`` raising on type-mismatched
    access paths (vs. silently returning None)."""

    def setUp(self):
        super().setUp()
        _reset_uid_counter()

    def test_walk_object_spec_with_subscript_raises(self):
        """``ObjectSpec`` paired with a subscript token must raise."""
        from torch._dynamo.variables.builder import (
            _AttrToken,
            _SubscriptToken,
            _walk_spec,
        )

        root = ObjectSpec({"x": TensorSpec([ShapeVar("h"), None])})
        with self.assertRaisesRegex(
            RuntimeError, r"ObjectSpec.*expects an attribute access"
        ):
            _walk_spec(root, [_SubscriptToken("x")])
        # sanity: the matching ``.x`` (AttrToken) form still resolves.
        ts = _walk_spec(root, [_AttrToken("x")])
        self.assertIsInstance(ts, TensorSpec)

    def test_walk_dict_spec_with_attr_raises(self):
        """``DictSpec`` paired with an attribute token must raise."""
        from torch._dynamo.variables.builder import (
            _AttrToken,
            _SubscriptToken,
            _walk_spec,
        )

        root = DictSpec({"x": TensorSpec([ShapeVar("h"), None])})
        with self.assertRaisesRegex(RuntimeError, r"DictSpec.*expects.*subscript"):
            _walk_spec(root, [_AttrToken("x")])
        # sanity: the matching ``["x"]`` (SubscriptToken) form still resolves.
        ts = _walk_spec(root, [_SubscriptToken("x")])
        self.assertIsInstance(ts, TensorSpec)

    def test_walk_leaf_with_remaining_token_raises(self):
        """A leaf spec with remaining tokens in the path must raise."""
        from torch._dynamo.variables.builder import _AttrToken, _walk_spec

        root = TensorSpec([ShapeVar("h"), None])
        with self.assertRaisesRegex(
            RuntimeError, r"leaf spec.*cannot consume further token"
        ):
            _walk_spec(root, [_AttrToken("something")])


if __name__ == "__main__":
    run_tests()
