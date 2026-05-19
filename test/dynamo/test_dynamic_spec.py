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
