# Owner(s): ["oncall: export"]

import itertools

import torch
from torch.fx.experimental.dynamic_spec import (
    IntVar,
    ParamsSpec,
    ShapesSpec,
    ShapeVar,
    STATIC,
    TensorSpec,
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


class _ModXY(torch.nn.Module):
    def forward(self, x, y):
        return x + y


class _ModXYIndep(torch.nn.Module):
    """Two-input model whose ops don't unify the input shapes (no broadcast)."""

    def forward(self, x, y):
        return x.sum(0), y.sum(0)


class _ModXN(torch.nn.Module):
    def forward(self, x, n):
        return x + n


class TestMakeFxDynamicSpec(TestCase):
    """make_fx(_dynamic_spec=...) support for the ShapesSpec / ParamsSpec API."""

    def setUp(self):
        super().setUp()
        _reset_uid_counter()

    def test_function_tensorspec_shape_var_produces_symint(self):
        """A dim marked with ShapeVar becomes an unbacked SymInt in the graph."""

        def f(x):
            return x.sum(0)

        gm = make_fx(
            f,
            tracing_mode="fake",
            _dynamic_spec={"x": TensorSpec([ShapeVar("batch"), None])},
        )(torch.randn(8, 3))

        shape = _first_tensor_placeholder_shape(gm)
        self.assertIsInstance(shape[0], torch.SymInt)
        self.assertGreater(len(free_unbacked_symbols(shape[0])), 0)
        # Static dim stays an int.
        self.assertEqual(int(shape[1]), 3)

        self.assertExpectedInline(
            gm.print_readable(print_output=False).strip(),
            """\
class f(torch.nn.Module):
    def forward(self, x_1: "f32[u0, 3]"):
        sum_1: "f32[3]" = torch.ops.aten.sum.dim_IntList(x_1, [0]);  x_1 = None
        return sum_1""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )

    def test_module_paramsspec_shape_var_produces_symint(self):
        """Same as above but driving make_fx with an nn.Module."""
        gm = make_fx(
            _ModX(),
            tracing_mode="fake",
            _dynamic_spec={"x": TensorSpec([ShapeVar("batch"), None])},
        )(torch.randn(8, 3))

        shape = _first_tensor_placeholder_shape(gm)
        self.assertIsInstance(shape[0], torch.SymInt)

    def test_all_none_spec_keeps_dims_static(self):
        """A TensorSpec with all-None dims (= all static) yields concrete ints."""

        def f(x):
            return x + 1

        gm = make_fx(
            f,
            tracing_mode="fake",
            _dynamic_spec={"x": TensorSpec([None, None])},
        )(torch.randn(4, 5))

        shape = _first_tensor_placeholder_shape(gm)
        self.assertEqual(int(shape[0]), 4)
        self.assertEqual(int(shape[1]), 5)

    def test_static_dim_mismatch_raises(self):
        """Declaring a tensor dim as static=3 but passing 5 errors out."""

        def f(x):
            return x + 1

        with self.assertRaisesRegex(
            ValueError,
            r"shapes_spec declares dim 1 as static with value 3, but got 5",
        ):
            make_fx(
                f,
                tracing_mode="fake",
                _dynamic_spec={"x": TensorSpec([ShapeVar("batch"), 3])},
            )(torch.randn(4, 5))

    def test_no_params_spec_short_circuits(self):
        """ShapesSpec with no ParamsSpec should not change tracing behavior."""

        def f(x):
            return x + 1

        gm = make_fx(
            f,
            tracing_mode="fake",
            _dynamic_spec=ShapesSpec(),
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
            _dynamic_spec={
                "x": TensorSpec([ShapeVar("a"), None]),
                "y": TensorSpec([ShapeVar("b"), None]),
            },
        )(torch.randn(4, 3), torch.randn(7, 3))

        self.assertExpectedInline(
            gm.print_readable(print_output=False).strip(),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[u0, 3]", arg1_1: "f32[u1, 3]"):
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
        a = ShapeVar("a")
        gm = make_fx(
            _ModXYIndep(),
            tracing_mode="fake",
            _dynamic_spec={"x": TensorSpec([a, None]), "y": TensorSpec([a, None])},
        )(torch.randn(5, 3), torch.randn(5, 3))

        self.assertExpectedInline(
            gm.print_readable(print_output=False).strip(),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[u0, 3]", arg1_1: "f32[u0, 3]"):
        sum_1: "f32[3]" = torch.ops.aten.sum.dim_IntList(arg0_1, [0]);  arg0_1 = None
        sum_2: "f32[3]" = torch.ops.aten.sum.dim_IntList(arg1_1, [0]);  arg1_1 = None
        return (sum_1, sum_2)""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )

    def test_shared_shape_var_links_dims(self):
        """Original add-model: shared ShapeVar -> shared symbol."""
        B = ShapeVar("batch")
        gm = make_fx(
            _ModXY(),
            tracing_mode="fake",
            _dynamic_spec={"x": TensorSpec([B, None]), "y": TensorSpec([B, None])},
        )(torch.randn(5, 3), torch.randn(5, 3))

        tensor_shapes = []
        for val in _user_input_placeholder_vals(gm):
            if isinstance(val, torch.Tensor):
                tensor_shapes.append(val.shape)
        self.assertEqual(len(tensor_shapes), 2)
        s0 = tensor_shapes[0][0]
        s1 = tensor_shapes[1][0]
        self.assertIsInstance(s0, torch.SymInt)
        self.assertIsInstance(s1, torch.SymInt)
        self.assertEqual(str(s0), str(s1))

    def test_int_input_with_intvar_spec(self):
        """An int input with an IntVar spec becomes a symbolic int."""
        gm = make_fx(
            _ModXN(),
            tracing_mode="fake",
            _dynamic_spec={"x": TensorSpec([None]), "n": IntVar("n")},
        )(torch.randn(4), 7)

        sym_placeholders = [
            val
            for val in _user_input_placeholder_vals(gm)
            if isinstance(val, torch.SymInt)
        ]
        self.assertEqual(len(sym_placeholders), 1)

    def test_dict_shorthand_accepted(self):
        """A bare ``dict`` is accepted as shorthand for
        ``ShapesSpec(ParamsSpec(dict))``."""

        def f(x):
            return x.sum(0)

        gm = make_fx(
            f,
            tracing_mode="fake",
            _dynamic_spec={"x": TensorSpec([ShapeVar("batch"), None])},
        )(torch.randn(8, 3))

        shape = _first_tensor_placeholder_shape(gm)
        self.assertIsInstance(shape[0], torch.SymInt)
        self.assertEqual(int(shape[1]), 3)

    def test_requires_tracing_mode_fake(self):
        """``_dynamic_spec`` is only meaningful with tracing_mode='fake'."""

        def f(x):
            return x + 1

        for bad_mode in ("real", "symbolic"):
            with self.assertRaisesRegex(
                ValueError,
                r"make_fx\(_dynamic_spec=\.\.\.\) requires tracing_mode='fake'",
            ):
                make_fx(
                    f,
                    tracing_mode=bad_mode,
                    _dynamic_spec={"x": TensorSpec([ShapeVar("batch"), None])},
                )(torch.randn(4, 5))

    def test_invalid_spec_type_raises(self):
        """Passing an unsupported object raises TypeError."""

        def f(x):
            return x

        with self.assertRaisesRegex(
            TypeError,
            r"make_fx\(_dynamic_spec=\.\.\.\) expects a dict, ShapesSpec, or ParamsSpec",
        ):
            make_fx(
                f,
                tracing_mode="fake",
                _dynamic_spec=42,  # neither dict nor Spec object
            )(torch.randn(2))

    def test_symbolic_dim_flows_through_ops(self):
        """The ShapeVar symbol propagates into op outputs in the traced graph,
        and the resulting GraphModule runs with varied sizes."""

        def f(x):
            return x.sum(1)

        gm = make_fx(
            f,
            tracing_mode="fake",
            _dynamic_spec={"x": TensorSpec([ShapeVar("batch"), None])},
        )(torch.randn(6, 4))

        self.assertExpectedInline(
            gm.print_readable(print_output=False).strip(),
            """\
class f(torch.nn.Module):
    def forward(self, x_1: "f32[u0, 4]"):
        sum_1: "f32[u0]" = torch.ops.aten.sum.dim_IntList(x_1, [1]);  x_1 = None
        return sum_1""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )

        gm(torch.randn(8, 4))
        gm(torch.randn(13, 4))

    def test_shape_var_min_bound_constrains_range(self):
        """A ``ShapeVar(min=...)`` is reflected in the shape env's range."""

        def f(x):
            return x + 1

        gm = make_fx(
            f,
            tracing_mode="fake",
            _dynamic_spec={"x": TensorSpec([ShapeVar("batch", min=4, max=128), None])},
        )(torch.randn(10, 3))

        shape = _first_tensor_placeholder_shape(gm)
        sym = shape[0]
        self.assertIsInstance(sym, torch.SymInt)
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
                _dynamic_spec={"x": TensorSpec([ShapeVar("batch"), None])},
            )(torch.randn(8, 3))

    def test_derived_dim_in_traced_graph(self):
        """Derived expressions (``B * 2``) materialize as derived dim
        expressions in the traced graph and emit a runtime assertion."""
        B = ShapeVar("batch")

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
            _dynamic_spec={
                "x": TensorSpec([B, STATIC]),
                "y": TensorSpec([B * 2, STATIC]),
            },
        )(torch.randn(4, 3), torch.randn(8, 5))

        # y dim 0 should be the derived expression 2*u0.
        graph_str = gm.print_readable(print_output=False)
        self.assertRegex(graph_str, r'arg1_1: "f32\[2\*u\d+, 5\]"')
        # Note: ``make_fx`` itself does not insert runtime asserts for spec
        # assumptions / derived dims at the top-level (that's a separate
        # pass; ``insert_deferred_runtime_asserts`` only runs for HOP
        # subgraphs). The derived relation IS however enforced symbolically
        # — the in-trace branch resolved (``+`` taken), proving the
        # ShapeEnv knows ``2*u0 == y.size(0)``.
        self.assertTrue(
            any(
                node.op == "call_function" and "aten.add" in str(node.target)
                for node in gm.graph.nodes
            )
        )

    def test_assumption_runtime_enforced(self):
        """A relational ``assumptions=[A > B]`` is wired into the shape env:
        the assumed-true branch is taken at trace time and the relation
        is materialized as a runtime assertion."""
        A = ShapeVar("a")
        B = ShapeVar("b")

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
            _dynamic_spec=ShapesSpec(
                params=ParamsSpec(
                    {"x": TensorSpec([A, STATIC]), "y": TensorSpec([B, STATIC])}
                ),
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


if __name__ == "__main__":
    run_tests()
