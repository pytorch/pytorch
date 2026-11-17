# Owner(s): ["oncall: export"]

import itertools

import torch
from torch.fx.experimental.dynamic_spec import (
    DictSpec as DICT,
    IntVar,
    ObjectSpec as OBJ,
    ParamsSpec as PARAMS,
    SeqSpec as L,
    ShapesSpec,
    ShapeVar as VAR,
    STATIC,
    TensorSpec as T,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import (
    free_unbacked_symbols,
    GuardOnDataDependentSymNode,
)
from torch.testing._internal.common_utils import run_tests, TestCase


def _reset_uid_counter():
    """Reset the global IntVar uid counter so uids start at 0 per test."""
    IntVar._uid_counter = itertools.count()


def _user_input_placeholder_vals(gm):
    """Return the per-placeholder ``meta['val']`` (tensor or SymInt) for
    user inputs of ``gm``."""
    out = []
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        val = node.meta.get("val", node.meta.get("example_value"))
        if isinstance(val, (torch.Tensor, torch.SymInt)):
            out.append(val)
    return out


def _first_tensor_placeholder_shape(gm):
    for val in _user_input_placeholder_vals(gm):
        if isinstance(val, torch.Tensor):
            return val.shape
    raise AssertionError("no tensor placeholder found")


class _ModX(torch.nn.Module):
    def forward(self, x):
        return x.sum(0)


class _ModXYIndep(torch.nn.Module):
    """Two-input model whose ops don't unify the input shapes (no broadcast)."""

    def forward(self, x, y):
        return x.sum(0), y.sum(0)


class _ModXN(torch.nn.Module):
    def forward(self, x, n):
        return x + n


class TestMakeFxDynamicSpec(TestCase):
    """make_fx(dynamic_shapes=...) support for the ShapesSpec / ParamsSpec API."""

    def setUp(self):
        super().setUp()
        _reset_uid_counter()

    def test_function_tensorspec_shape_var_produces_symint(self):
        def f(x):
            return x.sum(0)

        gm = make_fx(
            f,
            tracing_mode="fake",
            dynamic_shapes={"x": T([VAR("batch"), STATIC])},
        )(torch.randn(8, 3))

        shape = _first_tensor_placeholder_shape(gm)
        self.assertIsInstance(shape[0], torch.SymInt)
        self.assertEqual(len(free_unbacked_symbols(shape[0])), 1)
        # Static dim stays an int.
        self.assertEqual(int(shape[1]), 3)

        self.assertExpectedInline(
            gm.print_readable(print_output=False).strip(),
            """\
class f(torch.nn.Module):
    def forward(self, x_1: "f32[u0, 3]"):
        sym_size_int: "Sym(u0)" = torch.ops.aten.sym_size.int(x_1, 0)
        ge: "Sym(u0 >= 0)" = sym_size_int >= 0;  sym_size_int = None
        _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
        sum_1: "f32[3]" = torch.ops.aten.sum.dim_IntList(x_1, [0]);  x_1 = None
        return sum_1""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )

    def test_module_paramsspec_shape_var_produces_symint(self):
        gm = make_fx(
            _ModX(),
            tracing_mode="fake",
            dynamic_shapes={"x": T([VAR("batch"), STATIC])},
        )(torch.randn(8, 3))

        shape = _first_tensor_placeholder_shape(gm)
        self.assertIsInstance(shape[0], torch.SymInt)
        self.assertTrue(free_unbacked_symbols(shape[0]))

    def test_all_static_spec_keeps_dims_static(self):
        def f(x):
            return x + 1

        gm = make_fx(
            f,
            tracing_mode="fake",
            dynamic_shapes={"x": T([STATIC, STATIC])},
        )(torch.randn(4, 5))

        shape = _first_tensor_placeholder_shape(gm)
        self.assertEqual(int(shape[0]), 4)
        self.assertEqual(int(shape[1]), 5)

    def test_static_dim_mismatch_raises(self):
        def f(x):
            return x + 1

        with self.assertRaisesRegex(
            ValueError,
            r"shapes_spec declares dim 1 as static with value 3, but got 5",
        ):
            make_fx(
                f,
                tracing_mode="fake",
                dynamic_shapes={"x": T([VAR("batch"), 3])},
            )(torch.randn(4, 5))

    def test_no_params_spec_short_circuits(self):
        def f(x):
            return x + 1

        gm = make_fx(
            f,
            tracing_mode="fake",
            dynamic_shapes=ShapesSpec(),
        )(torch.randn(4, 5))

        # No ShapeVars declared -> dims remain static.
        shape = _first_tensor_placeholder_shape(gm)
        self.assertEqual(int(shape[0]), 4)
        self.assertEqual(int(shape[1]), 5)

    def test_two_tensors_independent_shape_vars(self):
        """Distinct ShapeVars across two inputs each produce their own SymInt
        (use a model whose ops don't unify the two shapes, so the symbols
        remain distinct in the printed graph)."""
        gm = make_fx(
            _ModXYIndep(),
            tracing_mode="fake",
            dynamic_shapes={
                "x": T([VAR("a"), STATIC]),
                "y": T([VAR("b"), STATIC]),
            },
        )(torch.randn(4, 3), torch.randn(7, 3))

        self.assertExpectedInline(
            gm.print_readable(print_output=False).strip(),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[u0, 3]", arg1_1: "f32[u1, 3]"):
        sym_size_int: "Sym(u0)" = torch.ops.aten.sym_size.int(arg0_1, 0)
        ge: "Sym(u0 >= 0)" = sym_size_int >= 0;  sym_size_int = None
        _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
        sym_size_int_1: "Sym(u1)" = torch.ops.aten.sym_size.int(arg1_1, 0)
        ge_1: "Sym(u1 >= 0)" = sym_size_int_1 >= 0;  sym_size_int_1 = None
        _assert_scalar_default_1 = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u1 >= 0 on node 'ge_1'");  ge_1 = _assert_scalar_default_1 = None
        sum_1: "f32[3]" = torch.ops.aten.sum.dim_IntList(arg0_1, [0]);  arg0_1 = None
        sum_2: "f32[3]" = torch.ops.aten.sum.dim_IntList(arg1_1, [0]);  arg1_1 = None
        return (sum_1, sum_2)""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )

    def test_same_shape_var_object_links_dims(self):
        """Passing the SAME ``ShapeVar`` instance for both inputs makes
        them share one symbol in the traced graph (``u0`` for both),
        even with the independent-ops model."""
        a = VAR("a")
        gm = make_fx(
            _ModXYIndep(),
            tracing_mode="fake",
            dynamic_shapes={"x": T([a, STATIC]), "y": T([a, STATIC])},
        )(torch.randn(5, 3), torch.randn(5, 3))

        self.assertExpectedInline(
            gm.print_readable(print_output=False).strip(),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[u0, 3]", arg1_1: "f32[u0, 3]"):
        sym_size_int: "Sym(u0)" = torch.ops.aten.sym_size.int(arg0_1, 0)
        ge: "Sym(u0 >= 0)" = sym_size_int >= 0
        _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
        sym_size_int_1: "Sym(u0)" = torch.ops.aten.sym_size.int(arg1_1, 0)
        ge_1: "Sym(u0 >= 0)" = sym_size_int_1 >= 0
        _assert_scalar_default_1 = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u1 >= 0 on node 'ge_1'");  ge_1 = _assert_scalar_default_1 = None
        eq: "Sym(True)" = sym_size_int_1 == sym_size_int;  sym_size_int_1 = sym_size_int = None
        _assert_scalar_default_2 = torch.ops.aten._assert_scalar.default(eq, "Runtime assertion failed for expression Eq(u1, u0) on node 'eq'");  eq = _assert_scalar_default_2 = None
        sum_1: "f32[3]" = torch.ops.aten.sum.dim_IntList(arg0_1, [0]);  arg0_1 = None
        sum_2: "f32[3]" = torch.ops.aten.sum.dim_IntList(arg1_1, [0]);  arg1_1 = None
        return (sum_1, sum_2)""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )

    def test_int_input_with_intvar_spec(self):
        """An int input declared via ``IntVar`` becomes an *unbacked* SymInt
        (``u0``), not a backed one (``s0``) — the ShapesSpec path is
        unbacked-only for soundness."""
        gm = make_fx(
            _ModXN(),
            tracing_mode="fake",
            dynamic_shapes={"x": T([STATIC]), "n": IntVar("n")},
        )(torch.randn(4), 7)

        sym_placeholders = [
            val
            for val in _user_input_placeholder_vals(gm)
            if isinstance(val, torch.SymInt)
        ]
        self.assertEqual(len(sym_placeholders), 1)
        sym = sym_placeholders[0]
        self.assertTrue(free_unbacked_symbols(sym))
        self.assertEqual(str(sym.node.expr), "u0")

    def test_dict_shorthand_accepted(self):
        """A bare ``dict`` is accepted as shorthand for
        ``ShapesSpec(PARAMS(dict))``."""

        def f(x):
            return x.sum(0)

        gm = make_fx(
            f,
            tracing_mode="fake",
            dynamic_shapes={"x": T([VAR("batch"), STATIC])},
        )(torch.randn(8, 3))

        shape = _first_tensor_placeholder_shape(gm)
        self.assertIsInstance(shape[0], torch.SymInt)
        self.assertTrue(free_unbacked_symbols(shape[0]))
        self.assertEqual(int(shape[1]), 3)

    def test_requires_tracing_mode_fake(self):
        def f(x):
            return x + 1

        for bad_mode in ("real", "symbolic"):
            with self.assertRaisesRegex(
                ValueError,
                r"make_fx\(dynamic_shapes=\.\.\.\) requires tracing_mode='fake'",
            ):
                make_fx(
                    f,
                    tracing_mode=bad_mode,
                    dynamic_shapes={"x": T([VAR("batch"), STATIC])},
                )(torch.randn(4, 5))

    def test_invalid_spec_type_raises(self):
        def f(x):
            return x

        with self.assertRaisesRegex(
            TypeError,
            r"dynamic spec expects a dict, ShapesSpec, or ParamsSpec",
        ):
            make_fx(
                f,
                tracing_mode="fake",
                dynamic_shapes=42,  # neither dict nor Spec object
            )(torch.randn(2))

    def test_symbolic_dim_flows_through_ops(self):
        """The ShapeVar symbol propagates into op outputs in the traced graph,
        and the resulting GraphModule runs with varied sizes."""

        def f(x):
            return x.sum(1)

        gm = make_fx(
            f,
            tracing_mode="fake",
            dynamic_shapes={"x": T([VAR("batch"), STATIC])},
        )(torch.randn(6, 4))

        self.assertExpectedInline(
            gm.print_readable(print_output=False).strip(),
            """\
class f(torch.nn.Module):
    def forward(self, x_1: "f32[u0, 4]"):
        sym_size_int: "Sym(u0)" = torch.ops.aten.sym_size.int(x_1, 0)
        ge: "Sym(u0 >= 0)" = sym_size_int >= 0;  sym_size_int = None
        _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
        sum_1: "f32[u0]" = torch.ops.aten.sum.dim_IntList(x_1, [1]);  x_1 = None
        return sum_1""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )

        gm(torch.randn(8, 4))
        gm(torch.randn(13, 4))

    def test_shape_var_min_bound_constrains_range(self):
        def f(x):
            return x + 1

        gm = make_fx(
            f,
            tracing_mode="fake",
            dynamic_shapes={"x": T([VAR("batch", min=4, max=128), STATIC])},
        )(torch.randn(10, 3))

        shape = _first_tensor_placeholder_shape(gm)
        sym = shape[0]
        self.assertIsInstance(sym, torch.SymInt)
        self.assertTrue(free_unbacked_symbols(sym))
        # ``gm.shape_env`` is only set in ``tracing_mode='symbolic'``,
        # so reach into the SymInt's own shape env instead.
        shape_env = sym.node.shape_env
        vr = shape_env.var_to_range[sym.node.expr]
        self.assertEqual(int(vr.lower), 4)
        self.assertEqual(int(vr.upper), 128)

    def test_branching_on_shape_var_dim_raises_dde(self):
        """Branching on a ShapeVar dim with no bounds raises a
        data-dependent error (mirrors ``test_unbacked_raises_dde_on_branching``
        in the strict-export suite)."""

        def f(x):
            if x.size(0) > 5:
                return x + 1
            return x - 1

        with self.assertRaisesRegex(
            GuardOnDataDependentSymNode,
            r"Could not guard on data-dependent expression",
        ):
            make_fx(
                f,
                tracing_mode="fake",
                dynamic_shapes={"x": T([VAR("batch"), STATIC])},
            )(torch.randn(8, 3))

    def test_derived_dim_in_traced_graph(self):
        """Derived expressions (``B * 2``) materialize as derived dim
        expressions in the traced graph. The ``add`` (not ``sub``) proves the
        relation ``2*u0 == y.size(0)`` held at trace time."""
        B = VAR("batch")

        class M(torch.nn.Module):
            def forward(self, x, y):
                # Branching on the derived relation proves it is known at
                # trace time (otherwise this would be data-dependent).
                if x.size(0) * 2 == y.size(0):
                    return x.sum() + y.sum()
                return x.sum() - y.sum()

        gm = make_fx(
            M(),
            tracing_mode="fake",
            dynamic_shapes={
                "x": T([B, STATIC]),
                "y": T([B * 2, STATIC]),
            },
        )(torch.randn(4, 3), torch.randn(8, 5))

        self.assertExpectedInline(
            gm.print_readable(print_output=False).strip(),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[u0, 3]", arg1_1: "f32[2*u0, 5]"):
        sym_size_int_2: "Sym(u0)" = torch.ops.aten.sym_size.int(arg0_1, 0)
        ge: "Sym(u0 >= 0)" = sym_size_int_2 >= 0
        _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
        sym_size_int_3: "Sym(2*u0)" = torch.ops.aten.sym_size.int(arg1_1, 0)
        ge_1: "Sym(2*u0 >= 0)" = sym_size_int_3 >= 0
        _assert_scalar_default_1 = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u1 >= 0 on node 'ge_1'");  ge_1 = _assert_scalar_default_1 = None
        mul_1: "Sym(2*u0)" = 2 * sym_size_int_2;  sym_size_int_2 = None
        eq_1: "Sym(True)" = sym_size_int_3 == mul_1;  sym_size_int_3 = mul_1 = None
        _assert_scalar_default_2 = torch.ops.aten._assert_scalar.default(eq_1, "Runtime assertion failed for expression Eq(u1, 2*u0) on node 'eq_1'");  eq_1 = _assert_scalar_default_2 = None
        sum_1: "f32[]" = torch.ops.aten.sum.default(arg0_1);  arg0_1 = None
        sum_2: "f32[]" = torch.ops.aten.sum.default(arg1_1);  arg1_1 = None
        add: "f32[]" = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
        return add""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )

    def test_unbound_derived_dim_raises(self):
        """A derived dim (``A * 2``) whose ``IntVar`` never appears as a bare
        slot leaves a pending check that can never drain. ``_finalize_spec_wiring``
        must run for any spec (not just ones with assumptions) so this surfaces
        as an error instead of silently dropping the relation."""
        A = VAR("a")

        def f(x):
            return x.sum(0)

        with self.assertRaisesRegex(
            ValueError,
            r"pending check\(s\) reference unbound IntVar\(s\) \['a'\]",
        ):
            make_fx(
                f,
                tracing_mode="fake",
                dynamic_shapes={"x": T([A * 2, STATIC])},
            )(torch.randn(8, 3))

    def test_assumption(self):
        """A relational ``assumptions=[A > B]`` is wired into the shape env:
        the assumed-true branch is taken at trace time and the relation
        is materialized as a runtime assertion in the graph."""
        A = VAR("a")
        B = VAR("b")

        class M(torch.nn.Module):
            def forward(self, x, y):
                # Branching on the assumed relation proves it is known at
                # trace time.
                if x.size(0) > y.size(0):
                    return x.sum() + y.sum()
                return x.sum() - y.sum()

        gm = make_fx(
            M(),
            tracing_mode="fake",
            dynamic_shapes=ShapesSpec(
                params=PARAMS({"x": T([A, STATIC]), "y": T([B, STATIC])}),
                assumptions=[A > B],
            ),
        )(torch.randn(5, 2), torch.randn(3, 2))

        # The assumption resolves the branch at trace time: ``A > B`` is
        # assumed true, so the ``+`` branch is taken.
        self.assertTrue(
            any(
                node.op == "call_function" and "aten.add" in str(node.target)
                for node in gm.graph.nodes
            )
        )
        # Sanity: the other branch must NOT have been taken.
        self.assertFalse(
            any(
                node.op == "call_function" and "aten.sub" in str(node.target)
                for node in gm.graph.nodes
            )
        )
        # The assumption is materialized as a runtime assertion node in the
        # graph and enforced when the graph is executed.
        self.assertTrue(
            any(
                node.op == "call_function" and "_assert_scalar" in str(node.target)
                for node in gm.graph.nodes
            )
        )
        # Correct: a=5 > b=3.
        gm(torch.randn(5, 2), torch.randn(3, 2))
        # Violation: a=2 not > b=3.
        with self.assertRaisesRegex(RuntimeError, "Runtime assertion failed"):
            gm(torch.randn(2, 2), torch.randn(3, 2))

    def test_tensor_dim_optimization_hint(self):
        """A ShapeVar's ``optimization_hint`` lands in the shape env's
        ``var_to_hint_override`` for the tensor dim's symbol."""
        b = VAR("batch", optimization_hint=32)
        gm = make_fx(
            _ModX(),
            tracing_mode="fake",
            dynamic_shapes={"x": T([b, STATIC])},
        )(torch.randn(8, 3))

        sym = _first_tensor_placeholder_shape(gm)[0]
        self.assertIsInstance(sym, torch.SymInt)
        self.assertEqual(sym.node.shape_env.var_to_hint_override.get(sym.node.expr), 32)

    def test_scalar_input_optimization_hint(self):
        """An ``IntVar``'s ``optimization_hint`` lands in the shape env's
        ``var_to_hint_override`` for a scalar input's symbol."""
        gm = make_fx(
            _ModXN(),
            tracing_mode="fake",
            dynamic_shapes={"x": T([STATIC]), "n": IntVar("n", optimization_hint=512)},
        )(torch.randn(4), 7)

        sym = next(
            v for v in _user_input_placeholder_vals(gm) if isinstance(v, torch.SymInt)
        )
        self.assertEqual(
            sym.node.shape_env.var_to_hint_override.get(sym.node.expr), 512
        )

    def test_container_specs_dict_seq_object(self):
        """DictSpec / SeqSpec / ObjectSpec each mark a nested tensor dim
        dynamic (``u0``/``u1``/``u2``); unspecified entries stay static."""
        from collections import namedtuple

        Pair = namedtuple("Pair", ["first", "second"])

        class M(torch.nn.Module):
            def forward(self, d, xs, p):
                return d["b"].sum(0) + xs[1].sum(0) + p.second.sum(0)

        gm = make_fx(
            M(),
            tracing_mode="fake",
            dynamic_shapes={
                "d": DICT({"b": T([VAR("B"), STATIC])}),  # "a" omitted -> static
                "xs": L([STATIC, T([VAR("X"), STATIC])]),  # pos 0 static
                "p": OBJ({"second": T([VAR("P"), STATIC])}),  # "first" -> static
            },
        )(
            {"a": torch.randn(7, 3), "b": torch.randn(8, 3)},
            [torch.randn(4, 3), torch.randn(5, 3)],
            Pair(torch.randn(6, 3), torch.randn(9, 3)),
        )

        # d["b"], xs[1], p.second -> unbacked (u0/u1/u2); the rest stay static.
        self.assertExpectedInline(
            gm.print_readable(print_output=False).strip(),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0, arg1, arg2):
        arg0_1: "f32[7, 3]"; arg0_2: "f32[u0, 3]"; arg1_1: "f32[4, 3]"; arg1_2: "f32[u1, 3]"; arg2_1: "f32[6, 3]"; arg2_2: "f32[u2, 3]";
        arg0_1, arg0_2, arg1_1, arg1_2, arg2_1, arg2_2, = fx_pytree.tree_flatten_spec([arg0, arg1, arg2], self._in_spec)
        sym_size_int: "Sym(u0)" = torch.ops.aten.sym_size.int(arg0_2, 0)
        ge: "Sym(u0 >= 0)" = sym_size_int >= 0;  sym_size_int = None
        _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
        sym_size_int_1: "Sym(u1)" = torch.ops.aten.sym_size.int(arg1_2, 0)
        ge_1: "Sym(u1 >= 0)" = sym_size_int_1 >= 0;  sym_size_int_1 = None
        _assert_scalar_default_1 = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u1 >= 0 on node 'ge_1'");  ge_1 = _assert_scalar_default_1 = None
        sym_size_int_2: "Sym(u2)" = torch.ops.aten.sym_size.int(arg2_2, 0)
        ge_2: "Sym(u2 >= 0)" = sym_size_int_2 >= 0;  sym_size_int_2 = None
        _assert_scalar_default_2 = torch.ops.aten._assert_scalar.default(ge_2, "Runtime assertion failed for expression u2 >= 0 on node 'ge_2'");  ge_2 = _assert_scalar_default_2 = None
        sum_1: "f32[3]" = torch.ops.aten.sum.dim_IntList(arg0_2, [0]);  arg0_2 = None
        sum_2: "f32[3]" = torch.ops.aten.sum.dim_IntList(arg1_2, [0]);  arg1_2 = None
        add: "f32[3]" = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
        sum_3: "f32[3]" = torch.ops.aten.sum.dim_IntList(arg2_2, [0]);  arg2_2 = None
        add_1: "f32[3]" = torch.ops.aten.add.Tensor(add, sum_3);  add = sum_3 = None
        return pytree.tree_unflatten([add_1], self._out_spec)""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )

    # ---- @dynamic_spec(...) decorator (auto-attach) ----

    def test_dynamic_spec_decorator_dict_form(self):
        """``@dynamic_spec({"x": ...})`` on a function is auto-applied by make_fx."""
        from torch.fx.experimental.dynamic_spec import dynamic_spec

        @dynamic_spec({"x": T([VAR("B"), STATIC])})
        def fn(x):
            return x.sum(0)

        gm = make_fx(fn, tracing_mode="fake")(torch.randn(8, 3))
        (val,) = _user_input_placeholder_vals(gm)
        self.assertIsInstance(val, torch.Tensor)
        self.assertEqual(len(free_unbacked_symbols(val.shape[0])), 1)

    def test_dynamic_spec_decorator_with_assumptions(self):
        """``@dynamic_spec(ShapesSpec(params=..., assumptions=...))`` is auto-applied
        and the assumption is wired into the shape env: branching on the
        assumed relation resolves statically (no DDE) and the assertion
        appears in the traced graph."""
        from torch.fx.experimental.dynamic_spec import dynamic_spec

        B = VAR("batch")

        @dynamic_spec(
            ShapesSpec(
                PARAMS({"x": T([B, STATIC])}),
                assumptions=[B % 2 == 0],
            )
        )
        def fn(x):
            # Branching on the assumption: without it, this would raise a DDE.
            if x.shape[0] % 2 == 0:
                return x.sum(0)
            return x.sum(0) * -1

        gm = make_fx(fn, tracing_mode="fake")(torch.randn(8, 3))
        # The assumption (``Eq(Mod(u0, 2), 0)``) is also materialized as a
        # runtime assertion in the traced graph.
        graph_repr = gm.print_readable(print_output=False)
        self.assertIn("Mod(u0, 2)", graph_repr)
        self.assertIn("_assert_scalar", graph_repr)

    def test_dynamic_spec_decorator_and_explicit_kwarg_raises(self):
        """Mixing the decorator with ``make_fx(..., dynamic_shapes=...)``
        is ambiguous and must raise."""
        from torch.fx.experimental.dynamic_spec import dynamic_spec

        @dynamic_spec({"x": T([VAR("B"), STATIC])})
        def fn(x):
            return x.sum(0)

        with self.assertRaisesRegex(
            ValueError,
            r"`@dynamic_spec\(\.\.\.\)` is attached.*AND a `dynamic_shapes=`",
        ):
            make_fx(
                fn,
                tracing_mode="fake",
                dynamic_shapes=PARAMS({"x": T([VAR("Z"), STATIC])}),
            )(torch.randn(8, 3))


if __name__ == "__main__":
    run_tests()
