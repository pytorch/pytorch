# Owner(s): ["oncall: export"]

import itertools
import re
from unittest import mock

import torch
import torch._dynamo
import torch.fx.experimental._config as _fx_experimental_config
import torch.utils._pytree as pytree
from torch.export import export
from torch.export._trace import (
    _export_to_aten_ir,
    _export_to_aten_ir_make_fx,
    _export_to_torch_ir,
    _non_strict_export,
    _strict_export,
)
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
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.testing._internal.common_utils import run_tests, TestCase


def _reset_uid_counter():
    """Reset the global IntVar uid counter so uids start at 0 per test."""
    IntVar._uid_counter = itertools.count()


def _user_input_placeholders(gm):
    """Return all user-input placeholders that have a tensor/symint val."""
    out = []
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        val = node.meta.get("val", node.meta.get("example_value"))
        if isinstance(val, (torch.Tensor, torch.SymInt)):
            out.append((node, val))
    return out


def _first_tensor_placeholder_shape(gm):
    for _node, val in _user_input_placeholders(gm):
        if isinstance(val, torch.Tensor):
            return val.shape
    raise AssertionError("no tensor placeholder found")


def _placeholder_dim0_sym_names(gm):
    """Return the stringified dim-0 SymInt name for each tensor placeholder.

    Used to keep ``assertExpectedInline`` snapshots stable against
    ShapeEnv's process-wide fresh-symbol counter (which ``setUp`` does
    not reset).
    """
    names = []
    for _node, val in _user_input_placeholders(gm):
        if isinstance(val, torch.Tensor):
            sym = val.shape[0]
            if isinstance(sym, torch.SymInt):
                names.append(str(sym))
    return names


def _has_assert_scalar(gm):
    """True if the graph contains an aten._assert_scalar runtime-assert node."""
    return any(
        node.op == "call_function" and "_assert_scalar" in str(node.target)
        for node in gm.graph.nodes
    )


# Modules with explicit signatures so ParamsSpec name lookup works.
class _ModX(torch.nn.Module):
    def forward(self, x):
        return x.sum(0)


class _ModXPlus(torch.nn.Module):
    def forward(self, x):
        return x + 1


class _ModXN(torch.nn.Module):
    def forward(self, x, n):
        return x + n


class _ModBranch(torch.nn.Module):
    def forward(self, x):
        if x.size(0) > 5:
            return x + 1
        return x - 1


class _TestExportDynamicSpecBase(TestCase):
    """torch.export.export support for the new ShapesSpec/ParamsSpec API."""

    def setUp(self):
        super().setUp()
        _reset_uid_counter()

    def test_unbacked_graph_has_unbacked_symbol(self):
        B = VAR("batch")
        x_spec = T([B, STATIC])
        ep = export(
            _ModX(),
            (torch.randn(8, 3),),
            dynamic_shapes=PARAMS({"x": x_spec}),
            strict=self.strict,
        )
        shape = _first_tensor_placeholder_shape(ep.graph_module)
        self.assertIsInstance(shape[0], torch.SymInt)
        self.assertEqual(len(free_unbacked_symbols(shape[0])), 1)
        self.assertExpectedInline(
            str(ep).strip(),
            """\
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, x: "f32[u0, 3]"):
            sym_size_int: "Sym(u0)" = torch.ops.aten.sym_size.int(x, 0)
            ge: "Sym(u0 >= 0)" = sym_size_int >= 0;  sym_size_int = None
            _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
            sum_1: "f32[3]" = torch.ops.aten.sum.dim_IntList(x, [0]);  x = None
            return (sum_1,)
Graph signature:
    x: USER_INPUT
    sum_1: USER_OUTPUT
Range constraints: {u0: VR[0, int_oo]}""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )

        # Sanity-check: the exported program runs with different shapes.
        ep.module()(torch.randn(16, 3))
        ep.module()(torch.randn(32, 3))

    def test_static_int_spec_mismatch_raises(self):
        # Source-name format differs: strict uses dynamo's flat-args path,
        # non-strict uses pytree keypath sources.
        if self.strict:
            regex = (
                r"shapes_spec declared L\['flat_args'\]\[1\] as static with "
                r"value 10, but got 42 at trace time"
            )
        else:
            regex = (
                r"shapes_spec declared L\['n'\] as static with value 10, "
                r"but got 42 at trace time"
            )
        with self.assertRaisesRegex(ValueError, regex):
            export(
                _ModXN(),
                (torch.randn(4), 42),
                dynamic_shapes=PARAMS({"n": 10}),
                strict=self.strict,
            )

    def test_static_tensor_dim_mismatch_raises(self):
        with self.assertRaisesRegex(
            ValueError,
            r"shapes_spec declares dim 1 as static with value 3, but got 5",
        ):
            export(
                _ModXPlus(),
                (torch.randn(4, 5),),
                dynamic_shapes=PARAMS({"x": T([VAR("batch"), 3])}),
                strict=self.strict,
            )

    def test_params_spec_shorthand(self):
        ep = export(
            _ModX(),
            (torch.randn(8, 3),),
            dynamic_shapes=PARAMS({"x": T([VAR("batch"), STATIC])}),
            strict=self.strict,
        )
        shape = _first_tensor_placeholder_shape(ep.graph_module)
        self.assertIsInstance(shape[0], torch.SymInt)

    @_fx_experimental_config.patch(no_data_dependent_graph_break=True)
    def test_min_max_bypasses_dde_on_branching(self):
        # min=10 > 5 → branch resolves statically, no DDE.
        export(
            _ModBranch(),
            (torch.randn(20, 3),),
            dynamic_shapes=PARAMS({"x": T([VAR("batch", min=10, max=100), STATIC])}),
            strict=self.strict,
        )

    @_fx_experimental_config.patch(no_data_dependent_graph_break=True)
    def test_unbacked_raises_dde_on_branching(self):
        """Without min/max, branching on a ShapeVar dim raises a DDE.
        Strict wraps it as UserError; non-strict raises the raw DDE.
        """
        exc = (
            torch._dynamo.exc.UserError
            if self.strict
            else torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode
        )
        with self.assertRaisesRegex(
            exc, r"Could not guard on data-dependent expression"
        ):
            export(
                _ModBranch(),
                (torch.randn(10, 3),),
                dynamic_shapes=PARAMS({"x": T([VAR(), STATIC])}),
                strict=self.strict,
            )

    def test_tensor_dim_optimization_hint_in_shape_env(self):
        b = VAR("batch", optimization_hint=32)
        ep = export(
            _ModX(),
            (torch.randn(8, 3),),
            dynamic_shapes=PARAMS({"x": T([b, STATIC])}),
            strict=self.strict,
        )
        shape = _first_tensor_placeholder_shape(ep.graph_module)
        sym = shape[0]
        self.assertIsInstance(sym, torch.SymInt)
        expr = sym.node.expr
        self.assertEqual(sym.node.shape_env.var_to_hint_override.get(expr), 32)

    def test_params_spec_matched_by_name_across_positional_kwargs_and_call_order(
        self,
    ):
        """Spec entries bind by forward-param *name* — for positional and
        kwarg inputs alike, and regardless of kwarg order (placeholders
        follow call order). Inputs not in the spec stay static."""

        class M(torch.nn.Module):
            def forward(self, x, y, z, n):
                return x.sum(0) + y.sum(0) + z.sum(0) * n

        ep = export(
            M(),
            args=(torch.randn(8, 3),),  # x positional
            # kwargs in non-signature order (z, n before y):
            kwargs={"z": torch.randn(7, 3), "n": 2, "y": torch.randn(5, 3)},
            dynamic_shapes=PARAMS(
                {
                    "x": T([VAR("X"), STATIC]),
                    "y": T([VAR("Y"), STATIC]),
                }
            ),
            strict=self.strict,
        )
        # Placeholders follow call order (x, z, n, y), not signature order:
        # x and y are spec'd by name → unbacked (u0, u1); z (kwarg, no spec)
        # keeps its literal shape; n (scalar, no spec) stays a plain input
        # with its value 2 baked into the math (mul by 2).
        self.assertExpectedInline(
            str(ep).strip(),
            """\
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, x: "f32[u0, 3]", z: "f32[7, 3]", n, y: "f32[u1, 3]"):
            sym_size_int: "Sym(u0)" = torch.ops.aten.sym_size.int(x, 0)
            ge: "Sym(u0 >= 0)" = sym_size_int >= 0;  sym_size_int = None
            _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
            sym_size_int_1: "Sym(u1)" = torch.ops.aten.sym_size.int(y, 0)
            ge_1: "Sym(u1 >= 0)" = sym_size_int_1 >= 0;  sym_size_int_1 = None
            _assert_scalar_default_1 = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u1 >= 0 on node 'ge_1'");  ge_1 = _assert_scalar_default_1 = None
            sum_1: "f32[3]" = torch.ops.aten.sum.dim_IntList(x, [0]);  x = None
            sum_2: "f32[3]" = torch.ops.aten.sum.dim_IntList(y, [0]);  y = None
            add: "f32[3]" = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
            sum_3: "f32[3]" = torch.ops.aten.sum.dim_IntList(z, [0]);  z = None
            mul: "f32[3]" = torch.ops.aten.mul.Tensor(sum_3, 2);  sum_3 = None
            add_1: "f32[3]" = torch.ops.aten.add.Tensor(add, mul);  add = mul = None
            return (add_1,)
Graph signature:
    x: USER_INPUT
    z: USER_INPUT
    n: USER_INPUT
    y: USER_INPUT
    add_1: USER_OUTPUT
Range constraints: {u0: VR[0, int_oo], u1: VR[0, int_oo]}""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )
        ep.module()(torch.randn(20, 3), z=torch.randn(7, 3), n=2, y=torch.randn(99, 3))

    def test_scalar_int_input_via_int_var(self):
        class M(torch.nn.Module):
            def forward(self, x, n):
                return x.sum(0) * n

        ep = export(
            M(),
            (torch.randn(8, 3), 5),
            dynamic_shapes=PARAMS(
                {
                    "x": T([VAR("B"), STATIC]),
                    "n": IntVar("n_size"),
                }
            ),
            strict=self.strict,
        )
        ep_str = str(ep)
        self.assertRegex(ep_str, r'n: "Sym\(u\d+\)"')
        ep.module()(torch.randn(20, 3), 42)
        ep.module()(torch.randn(50, 3), 99)

    def test_multi_leaf_arg_with_leaf_spec_raises(self):
        class M(torch.nn.Module):
            def forward(self, xs):
                return xs[0] + xs[1]

        with self.assertRaisesRegex(
            ValueError,
            r"shapes_spec\['xs'\]: spec is TensorSpec but the actual arg is list, not a Tensor\.",
        ):
            export(
                M(),
                args=([torch.randn(8, 3), torch.randn(8, 3)],),
                dynamic_shapes=PARAMS({"xs": T([VAR("B"), STATIC])}),
                strict=self.strict,
            )

    def test_multi_leaf_arg_no_spec_stays_static(self):
        class M(torch.nn.Module):
            def forward(self, xs):
                return xs[0] + xs[1]

        ep = export(
            M(),
            args=([torch.randn(8, 3), torch.randn(8, 3)],),
            dynamic_shapes=ShapesSpec(),  # no params → all static
            strict=self.strict,
        )
        ep_str = str(ep)
        # Both list elements come through as static placeholders.
        self.assertIn('xs_0: "f32[8, 3]"', ep_str)
        self.assertIn('xs_1: "f32[8, 3]"', ep_str)
        out = ep.module()([torch.randn(8, 3), torch.randn(8, 3)])
        self.assertEqual(out.shape, torch.Size([8, 3]))

    def test_spec_entry_for_omitted_default_raises(self):
        class M(torch.nn.Module):
            def forward(self, x, y=None):
                if y is None:
                    return x.sum(0)
                return x.sum(0) + y.sum(0)

        with self.assertRaisesRegex(
            ValueError,
            r"ParamsSpec has entries \['y'\] that do not match",
        ):
            export(
                M(),
                args=(torch.randn(8, 3),),  # y not passed
                dynamic_shapes=PARAMS(
                    {
                        "x": T([VAR("B"), STATIC]),
                        "y": T([VAR("Y"), STATIC]),  # no such arg
                    }
                ),
                strict=self.strict,
            )

    def test_explicit_none_spec_for_passed_arg_does_not_raise(self):
        """Explicit ``None`` in ParamsSpec for a passed arg means "static"
        and must not be flagged as an unmatched spec key (regression: a
        prior `.get()` check conflated key-absent with value-is-None).
        """
        ep = export(
            _ModXPlus(),
            args=(torch.randn(8, 3),),
            dynamic_shapes=PARAMS({"x": None}),
            strict=self.strict,
        )
        shape = _first_tensor_placeholder_shape(ep.graph_module)
        self.assertEqual(tuple(shape), (8, 3))

    def test_named_dims_vs_shapes_spec(self):
        from torch.export import Dim

        class M(torch.nn.Module):
            def forward(self, x):
                return x.sum(0)

        ep_legacy = export(
            M(),
            (torch.randn(8, 3),),
            dynamic_shapes={"x": (Dim("B"), None)},
            strict=self.strict,
        )
        ep_new = export(
            M(),
            (torch.randn(8, 3),),
            dynamic_shapes=PARAMS({"x": T([VAR("B"), STATIC])}),
            strict=self.strict,
        )

        # Snapshot legacy graph (backed s* symbol, range constraint).
        # Symbol name is whatever ShapeEnv allocated (counter is process-wide).
        (legacy_sym,) = _placeholder_dim0_sym_names(ep_legacy.graph_module)
        self.assertExpectedInline(
            str(ep_legacy).strip(),
            f"""\
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, x: "f32[{legacy_sym}, 3]"):
            sum_1: "f32[3]" = torch.ops.aten.sum.dim_IntList(x, [0]);  x = None
            return (sum_1,)
Graph signature:
    x: USER_INPUT
    sum_1: USER_OUTPUT
Range constraints: {{{legacy_sym}: VR[0, int_oo]}}""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )
        # Snapshot new-API graph (unbacked u* symbol, range constraint).
        self.assertExpectedInline(
            str(ep_new).strip(),
            """\
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, x: "f32[u0, 3]"):
            sym_size_int: "Sym(u0)" = torch.ops.aten.sym_size.int(x, 0)
            ge: "Sym(u0 >= 0)" = sym_size_int >= 0;  sym_size_int = None
            _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
            sum_1: "f32[3]" = torch.ops.aten.sum.dim_IntList(x, [0]);  x = None
            return (sum_1,)
Graph signature:
    x: USER_INPUT
    sum_1: USER_OUTPUT
Range constraints: {u0: VR[0, int_oo]}""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )

        for ep in (ep_legacy, ep_new):
            shape = _first_tensor_placeholder_shape(ep.graph_module)
            self.assertIsInstance(shape[0], torch.SymInt)
            self.assertIsInstance(shape[1], int)
            self.assertEqual(int(shape[1]), 3)
            ep.module()(torch.randn(20, 3))
            ep.module()(torch.randn(64, 3))

    def test_named_dims_vs_shapes_spec2(self):
        """Parity: forward(x, y, z=None) with x positional, y as kwarg,
        z omitted. Both APIs mark x and y dim 0 dynamic; z stays out
        of the graph."""
        from torch.export import Dim

        class M(torch.nn.Module):
            def forward(self, x, y, z=None):
                return x.sum(0) + y.sum(0) * (z if z is not None else 1)

        ep_legacy = export(
            M(),
            args=(torch.randn(8, 3),),
            kwargs={"y": torch.randn(5, 3)},
            dynamic_shapes={"x": (Dim("X"), None), "y": (Dim("Y"), None)},
            strict=self.strict,
        )
        ep_new = export(
            M(),
            args=(torch.randn(8, 3),),
            kwargs={"y": torch.randn(5, 3)},
            dynamic_shapes=PARAMS(
                {
                    "x": T([VAR("X"), STATIC]),
                    "y": T([VAR("Y"), STATIC]),
                }
            ),
            strict=self.strict,
        )

        # Snapshot legacy: placeholders are (x, y) in signature order.
        # Symbol names are whatever ShapeEnv allocated (counter is process-wide).
        sx, sy = _placeholder_dim0_sym_names(ep_legacy.graph_module)
        self.assertExpectedInline(
            str(ep_legacy).strip(),
            f"""\
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, x: "f32[{sx}, 3]", y: "f32[{sy}, 3]"):
            sum_1: "f32[3]" = torch.ops.aten.sum.dim_IntList(x, [0]);  x = None
            sum_2: "f32[3]" = torch.ops.aten.sum.dim_IntList(y, [0]);  y = None
            mul: "f32[3]" = torch.ops.aten.mul.Tensor(sum_2, 1);  sum_2 = None
            add: "f32[3]" = torch.ops.aten.add.Tensor(sum_1, mul);  sum_1 = mul = None
            return (add,)
Graph signature:
    x: USER_INPUT
    y: USER_INPUT
    add: USER_OUTPUT
Range constraints: {{{sx}: VR[0, int_oo], {sy}: VR[0, int_oo]}}""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )
        # Snapshot new-API: same placeholder names in same order — proves the
        # translator preserved (x, y) ordering through the wrapper.
        self.assertExpectedInline(
            str(ep_new).strip(),
            """\
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, x: "f32[u0, 3]", y: "f32[u1, 3]"):
            sym_size_int: "Sym(u0)" = torch.ops.aten.sym_size.int(x, 0)
            ge: "Sym(u0 >= 0)" = sym_size_int >= 0;  sym_size_int = None
            _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
            sym_size_int_1: "Sym(u1)" = torch.ops.aten.sym_size.int(y, 0)
            ge_1: "Sym(u1 >= 0)" = sym_size_int_1 >= 0;  sym_size_int_1 = None
            _assert_scalar_default_1 = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u1 >= 0 on node 'ge_1'");  ge_1 = _assert_scalar_default_1 = None
            sum_1: "f32[3]" = torch.ops.aten.sum.dim_IntList(x, [0]);  x = None
            sum_2: "f32[3]" = torch.ops.aten.sum.dim_IntList(y, [0]);  y = None
            mul: "f32[3]" = torch.ops.aten.mul.Tensor(sum_2, 1);  sum_2 = None
            add: "f32[3]" = torch.ops.aten.add.Tensor(sum_1, mul);  sum_1 = mul = None
            return (add,)
Graph signature:
    x: USER_INPUT
    y: USER_INPUT
    add: USER_OUTPUT
Range constraints: {u0: VR[0, int_oo], u1: VR[0, int_oo]}""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )

        # Graphs are pinned above; just verify both run at varying shapes.
        for ep in (ep_legacy, ep_new):
            ep.module()(torch.randn(20, 3), y=torch.randn(99, 3))

    def test_legacy_parity_default_kwarg_omitted(self):
        from torch.export import Dim

        class M(torch.nn.Module):
            def forward(self, x, y=None):
                if y is None:
                    return x.sum(0)
                return x.sum(0) + y.sum(0)

        ep_legacy = export(
            M(),
            args=(torch.randn(8, 3),),
            dynamic_shapes={"x": (Dim("X"), None)},
            strict=self.strict,
        )
        ep_new = export(
            M(),
            args=(torch.randn(8, 3),),
            dynamic_shapes=PARAMS({"x": T([VAR("X"), STATIC])}),
            strict=self.strict,
        )

        # Legacy: only x in graph signature, no y placeholder.
        # Symbol name is whatever ShapeEnv allocated (counter is process-wide).
        (legacy_sym,) = _placeholder_dim0_sym_names(ep_legacy.graph_module)
        self.assertExpectedInline(
            str(ep_legacy).strip(),
            f"""\
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, x: "f32[{legacy_sym}, 3]"):
            sum_1: "f32[3]" = torch.ops.aten.sum.dim_IntList(x, [0]);  x = None
            return (sum_1,)
Graph signature:
    x: USER_INPUT
    sum_1: USER_OUTPUT
Range constraints: {{{legacy_sym}: VR[0, int_oo]}}""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )
        # New API: same — y dropped from graph signature.
        self.assertExpectedInline(
            str(ep_new).strip(),
            """\
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, x: "f32[u0, 3]"):
            sym_size_int: "Sym(u0)" = torch.ops.aten.sym_size.int(x, 0)
            ge: "Sym(u0 >= 0)" = sym_size_int >= 0;  sym_size_int = None
            _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
            sum_1: "f32[3]" = torch.ops.aten.sum.dim_IntList(x, [0]);  x = None
            return (sum_1,)
Graph signature:
    x: USER_INPUT
    sum_1: USER_OUTPUT
Range constraints: {u0: VR[0, int_oo]}""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )

    def test_user_varargs_in_forward_marked_dynamic_via_varargs_spec(self):
        class M(torch.nn.Module):
            def forward(self, *args):
                return args[0].sum() + args[1].sum()

        ep = export(
            M(),
            args=(torch.randn(8, 3), torch.randn(5, 3)),
            dynamic_shapes=PARAMS(
                {
                    "*args": [
                        T([VAR("A"), STATIC]),
                        T([VAR("B"), STATIC]),
                    ]
                }
            ),
            strict=self.strict,
        )
        ep_str = str(ep)
        # Both args dim 0 are unbacked (different symbols).
        self.assertRegex(ep_str, r'args_0: "f32\[u\d+, 3\]"')
        self.assertRegex(ep_str, r'args_1: "f32\[u\d+, 3\]"')
        ep.module()(torch.randn(20, 3), torch.randn(99, 3))

    def test_user_varargs_with_named_arg_before(self):
        class M(torch.nn.Module):
            def forward(self, x, *args):
                return x.sum() + args[0].sum() + args[1].sum()

        ep = export(
            M(),
            args=(torch.randn(4, 3), torch.randn(8, 3), torch.randn(5, 3)),
            dynamic_shapes=PARAMS(
                {
                    "x": T([VAR("X"), STATIC]),
                    "*args": [
                        T([VAR("A"), STATIC]),
                        T([VAR("B"), STATIC]),
                    ],
                },
            ),
            strict=self.strict,
        )
        ep_str = str(ep)
        self.assertRegex(ep_str, r'x: "f32\[u\d+, 3\]"')
        self.assertRegex(ep_str, r'args_0: "f32\[u\d+, 3\]"')
        self.assertRegex(ep_str, r'args_1: "f32\[u\d+, 3\]"')
        ep.module()(torch.randn(7, 3), torch.randn(20, 3), torch.randn(99, 3))

    def test_user_varargs_partial_spec_leaves_remainder_static(self):
        class M(torch.nn.Module):
            def forward(self, *args):
                return args[0].sum() + args[1].sum() + args[2].sum()

        ep = export(
            M(),
            args=(torch.randn(8, 3), torch.randn(5, 3), torch.randn(6, 3)),
            dynamic_shapes=PARAMS(
                {
                    "*args": [
                        T([VAR("A"), STATIC]),
                    ]
                }
            ),
            strict=self.strict,
        )
        ep_str = str(ep)
        # First arg dynamic, others literally sized.
        self.assertRegex(ep_str, r'args_0: "f32\[u\d+, 3\]"')
        self.assertIn('args_1: "f32[5, 3]"', ep_str)
        self.assertIn('args_2: "f32[6, 3]"', ep_str)

    def test_user_varkw_in_forward_marked_dynamic_via_varkw_spec(self):
        class M(torch.nn.Module):
            def forward(self, **kwargs):
                return kwargs["foo"].sum() + kwargs["bar"].sum()

        ep = export(
            M(),
            args=(),
            kwargs={"foo": torch.randn(8, 3), "bar": torch.randn(5, 3)},
            dynamic_shapes=PARAMS(
                {
                    "**kwargs": {
                        "foo": T([VAR("F"), STATIC]),
                        "bar": T([VAR("B"), STATIC]),
                    }
                }
            ),
            strict=self.strict,
        )
        ep_str = str(ep)
        self.assertRegex(ep_str, r'foo: "f32\[u\d+, 3\]"')
        self.assertRegex(ep_str, r'bar: "f32\[u\d+, 3\]"')
        ep.module()(foo=torch.randn(20, 3), bar=torch.randn(99, 3))

    def test_prefer_deferred_runtime_asserts_raises_with_shapes_spec(self):
        """`prefer_deferred_runtime_asserts_over_guards` is meaningful only
        for backed shapes; combining it with the unbacked-only ShapesSpec API
        raise."""
        with self.assertRaisesRegex(
            ValueError,
            r"`prefer_deferred_runtime_asserts_over_guards=True` cannot be "
            r"combined with `dynamic_shapes=ShapesSpec",
        ):
            export(
                _ModX(),
                (torch.randn(8, 3),),
                dynamic_shapes=PARAMS({"x": T([VAR("batch"), STATIC])}),
                strict=self.strict,
                prefer_deferred_runtime_asserts_over_guards=True,
            )

    def test_derived_dim_runtime_enforced(self):
        """Derived dim ``y dim0 = B * 2`` is enforced at runtime: the graph
        placeholder shows ``2*u0``, a correct input runs, and a violating
        input raises a runtime assertion (mirrors dynamo
        ``TestDerivedDimSpec.test_derived_dim`` but with export semantics)."""
        B = VAR("batch")

        class M(torch.nn.Module):
            def forward(self, x, y):
                # Branching on the derived relation proves it is known at
                # trace time (otherwise this would be a data-dependent error).
                if x.size(0) * 2 == y.size(0):
                    return x.sum() + y.sum()
                return x.sum() - y.sum()

        ep = export(
            M(),
            (torch.randn(4, 3), torch.randn(8, 5)),
            dynamic_shapes=PARAMS(
                {
                    "x": T([B, STATIC]),
                    "y": T([B * 2, STATIC]),
                }
            ),
            strict=self.strict,
        )
        ep_str = str(ep)
        # y dim 0 is the derived expression 2*u0.
        self.assertRegex(ep_str, r'y: "f32\[2\*u\d+, 5\]"')
        # The derived constraint is materialized as a runtime assert.
        self.assertTrue(_has_assert_scalar(ep.graph_module))
        # Correct input: y.shape[0] == 2 * x.shape[0].
        ep.module()(torch.randn(4, 3), torch.randn(8, 5))
        # Violation: 7 != 2 * 4 -> runtime assertion fires.
        with self.assertRaisesRegex(RuntimeError, "Runtime assertion failed"):
            ep.module()(torch.randn(4, 3), torch.randn(7, 5))

    def test_multi_var_derived_runtime_enforced(self):
        A = VAR("a")
        B = VAR("b")

        class M(torch.nn.Module):
            def forward(self, x, y, z):
                return x.sum() + y.sum() + z.sum()

        ep = export(
            M(),
            (torch.randn(3, 2), torch.randn(4, 2), torch.randn(13, 2)),
            dynamic_shapes=PARAMS(
                {
                    "x": T([A, STATIC]),
                    "y": T([B, STATIC]),
                    "z": T([A * B + 1, STATIC]),
                }
            ),
            strict=self.strict,
        )
        self.assertTrue(_has_assert_scalar(ep.graph_module))
        # Correct: z.shape[0] == 3 * 4 + 1 == 13.
        ep.module()(torch.randn(3, 2), torch.randn(4, 2), torch.randn(13, 2))
        # Violation: 99 != 3 * 4 + 1.
        with self.assertRaisesRegex(RuntimeError, "Runtime assertion failed"):
            ep.module()(torch.randn(3, 2), torch.randn(4, 2), torch.randn(99, 2))

    def test_assumption_runtime_enforced(self):
        A = VAR("a")
        B = VAR("b")

        class M(torch.nn.Module):
            def forward(self, x, y):
                # Branching on the assumed relation proves it is known at
                # trace time (otherwise this would be a data-dependent error).
                if x.size(0) > y.size(0):
                    return x.sum() + y.sum()
                return x.sum() - y.sum()

        ep = export(
            M(),
            (torch.randn(5, 2), torch.randn(3, 2)),
            dynamic_shapes=ShapesSpec(
                params=PARAMS(
                    {
                        "x": T([A, STATIC]),
                        "y": T([B, STATIC]),
                    }
                ),
                assumptions=[A > B],
            ),
            strict=self.strict,
        )
        self.assertTrue(_has_assert_scalar(ep.graph_module))
        # Correct: a=5 > b=3.
        ep.module()(torch.randn(5, 2), torch.randn(3, 2))
        # Violation: a=2 not > b=3.
        with self.assertRaisesRegex(RuntimeError, "Runtime assertion failed"):
            ep.module()(torch.randn(2, 2), torch.randn(3, 2))

    def test_min_max_in_range_constraints(self):
        ep = export(
            _ModX(),
            (torch.randn(20, 3),),
            dynamic_shapes=PARAMS({"x": T([VAR("b", min=10, max=100), STATIC])}),
            strict=self.strict,
        )
        rcs = ep.range_constraints
        self.assertEqual(len(rcs), 1)
        (vr,) = rcs.values()
        self.assertEqual(int(vr.lower), 10)
        self.assertEqual(int(vr.upper), 100)

    # ---- @dynamic_spec(...) decorator (auto-attach) ----

    def test_spec_decorator_per_param_form(self):
        """``@dynamic_spec({"x": ...})`` attached to ``forward`` is auto-applied."""
        from torch.fx.experimental.dynamic_spec import dynamic_spec

        class M(torch.nn.Module):
            @dynamic_spec({"x": T([VAR("B"), STATIC])})
            def forward(self, x):
                return x.sum(0)

        ep = export(M(), (torch.randn(8, 3),), strict=self.strict)
        shape = _first_tensor_placeholder_shape(ep.graph_module)
        self.assertIsInstance(shape[0], torch.SymInt)
        self.assertEqual(len(free_unbacked_symbols(shape[0])), 1)

    def test_spec_decorator_full_form_with_assumptions(self):
        """``@dynamic_spec(ShapesSpec(params=..., assumptions=...))`` form is auto-applied,
        and the assumption is enforced as a runtime assertion."""
        from torch.fx.experimental.dynamic_spec import dynamic_spec

        B = VAR("batch")

        class M(torch.nn.Module):
            @dynamic_spec(
                ShapesSpec(
                    PARAMS({"x": T([B, STATIC])}),
                    assumptions=[B % 2 == 0],
                )
            )
            def forward(self, x):
                return x.sum(0)

        ep = export(M(), (torch.randn(8, 3),), strict=self.strict)
        # Valid input (dim 0 is even) -- runs cleanly.
        ep.module()(torch.randn(8, 3))
        ep.module()(torch.randn(20, 3))
        # Violation (dim 0 is odd) -- runtime assertion fires, proving the
        # assumption from the decorator was actually wired into the graph.
        with self.assertRaisesRegex(RuntimeError, "Runtime assertion failed"):
            ep.module()(torch.randn(7, 3))

    def test_spec_decorator_and_explicit_dynamic_shapes_raises(self):
        """Mixing ``@dynamic_spec(...)`` with an explicit ``dynamic_shapes=`` kwarg
        is ambiguous and must raise."""
        from torch.fx.experimental.dynamic_spec import dynamic_spec

        class M(torch.nn.Module):
            @dynamic_spec({"x": T([VAR("B"), STATIC])})
            def forward(self, x):
                return x.sum(0)

        with self.assertRaisesRegex(
            ValueError,
            r"`@dynamic_spec\(\.\.\.\)` is attached.*AND a `dynamic_shapes=`",
        ):
            export(
                M(),
                (torch.randn(8, 3),),
                dynamic_shapes=PARAMS({"x": T([VAR("B"), STATIC])}),
                strict=self.strict,
            )

    def test_export_to_torch_ir_shapes_spec_direct(self):
        # Strict-only internal-API test; skip in non-strict mode.
        if not self.strict:
            self.skipTest("strict-only internal-API test")
        gm = _export_to_torch_ir(
            _ModX(),
            (torch.randn(8, 3),),
            {},
            dynamic_shapes=PARAMS({"x": T([VAR("batch"), STATIC])}),
        )
        self.assertIsInstance(gm, torch.fx.GraphModule)
        # The user placeholder carries an unbacked dim (u0).
        self.assertExpectedInline(
            gm.print_readable(print_output=False),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, x):
        arg_0: "f32[u0, 3]";
        arg_0, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
        l_flat_args_0_ = arg_0
        res: "f32[3]" = l_flat_args_0_.sum(0);  l_flat_args_0_ = None
        return pytree.tree_unflatten((res,), self._out_spec)""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )

    def test_export_to_torch_ir_legacy_v1_shapes_spec_raises(self):
        # Strict-only internal-API test; skip in non-strict mode.
        if not self.strict:
            self.skipTest("strict-only internal-API test")
        with mock.patch.object(
            torch._export.config, "use_new_tracer_experimental", False
        ):
            with self.assertRaisesRegex(
                NotImplementedError,
                r"ShapesSpec is not supported on the legacy v1",
            ):
                _export_to_torch_ir(
                    _ModX(),
                    (torch.randn(8, 3),),
                    {},
                    dynamic_shapes=PARAMS({"x": T([VAR("batch"), STATIC])}),
                )

    def test_strict_export_shapes_spec_direct(self):
        # Strict-only internal-API test; skip in non-strict mode.
        if not self.strict:
            self.skipTest("strict-only internal-API test")
        args = (torch.randn(8, 3),)
        _, in_spec = pytree.tree_flatten((args, {}))
        artifact = _strict_export(
            mod=_ModX(),
            args=args,
            kwargs={},
            dynamic_shapes=PARAMS({"x": T([VAR("batch"), STATIC])}),
            preserve_module_call_signature=(),
            orig_in_spec=in_spec,
            prefer_deferred_runtime_asserts_over_guards=False,
            _to_aten_func=_export_to_aten_ir_make_fx,
        )
        self.assertExpectedInline(
            artifact.aten.gm.print_readable(print_output=False),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, x: "f32[u0, 3]"):
        sym_size_int: "Sym(u0)" = torch.ops.aten.sym_size.int(x, 0)
        ge: "Sym(u0 >= 0)" = sym_size_int >= 0;  sym_size_int = None
        _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
        sum_1: "f32[3]" = torch.ops.aten.sum.dim_IntList(x, [0]);  x = None
        return (sum_1,)""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )

    def test_non_strict_export_shapes_spec_direct(self):
        # Non-strict-only internal-API test; skip in strict mode.
        # Verifies ``_non_strict_export`` with ``_to_aten_func=_export_to_aten_ir``
        # accepts ShapesSpec (replaces the old NotImplementedError gate test).
        if self.strict:
            self.skipTest("non-strict-only internal-API test")
        args = (torch.randn(8, 3),)
        _, in_spec = pytree.tree_flatten((args, {}))
        artifact = _non_strict_export(
            mod=_ModX(),
            args=args,
            kwargs={},
            dynamic_shapes=PARAMS({"x": T([VAR("batch"), STATIC])}),
            preserve_module_call_signature=(),
            orig_in_spec=in_spec,
            prefer_deferred_runtime_asserts_over_guards=False,
            _to_aten_func=_export_to_aten_ir,
        )
        self.assertExpectedInline(
            artifact.aten.gm.print_readable(print_output=False),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, x: "f32[u0, 3]"):
        sym_size_int: "Sym(u0)" = torch.ops.aten.sym_size.int(x, 0)
        ge: "Sym(u0 >= 0)" = sym_size_int >= 0;  sym_size_int = None
        _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
        sum_1: "f32[3]" = torch.ops.aten.sum.dim_IntList(x, [0]);  x = None
        return (sum_1,)""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )


class TestExportDynamicSpecStrict(_TestExportDynamicSpecBase):
    strict = True


class TestExportDynamicSpecNonStrict(_TestExportDynamicSpecBase):
    strict = False


del _TestExportDynamicSpecBase


class _TestContainerSpecBase(TestCase):
    def setUp(self):
        super().setUp()
        _reset_uid_counter()

    # ---- Positive cases ----

    def test_seq_spec_on_list_of_tensors(self):
        """SeqSpec marks each list position dynamic independently."""

        class M(torch.nn.Module):
            def forward(self, xs):
                return xs[0].sum(0) + xs[1].sum(0)

        ep = export(
            M(),
            args=([torch.randn(8, 3), torch.randn(5, 3)],),
            dynamic_shapes=PARAMS(
                {
                    "xs": L(
                        [
                            T([VAR("A"), STATIC]),
                            T([VAR("B"), STATIC]),
                        ]
                    )
                }
            ),
            strict=self.strict,
        )
        ep_str = str(ep)
        # Both list elements have unbacked dim 0.
        self.assertRegex(ep_str, r'xs_0: "f32\[u\d+, 3\]"')
        self.assertRegex(ep_str, r'xs_1: "f32\[u\d+, 3\]"')
        ep.module()([torch.randn(20, 3), torch.randn(99, 3)])

    def test_dict_spec_on_dict_of_tensors_partial(self):
        class M(torch.nn.Module):
            def forward(self, d):
                return d["a"].sum(0) + d["b"].sum(0)

        ep = export(
            M(),
            args=({"a": torch.randn(7, 3), "b": torch.randn(8, 3)},),
            dynamic_shapes=PARAMS({"d": DICT({"b": T([VAR("B"), STATIC])})}),
            strict=self.strict,
        )
        ep_str = str(ep)
        # a is omitted from the spec → static; b is dynamic.
        self.assertIn('d_a: "f32[7, 3]"', ep_str)
        self.assertRegex(ep_str, r'd_b: "f32\[u\d+, 3\]"')

    def test_object_spec_on_namedtuple(self):
        from collections import namedtuple

        Pair = namedtuple("Pair", ["first", "second"])

        class M(torch.nn.Module):
            def forward(self, p):
                return p.first.sum(0) + p.second.sum(0)

        ep = export(
            M(),
            args=(Pair(torch.randn(5, 3), torch.randn(8, 3)),),
            dynamic_shapes=PARAMS({"p": OBJ({"second": T([VAR("S"), STATIC])})}),
            strict=self.strict,
        )
        ep_str = str(ep)
        # first is omitted from the spec → static; second is dynamic.
        self.assertIn('p_first: "f32[5, 3]"', ep_str)
        self.assertRegex(ep_str, r'p_second: "f32\[u\d+, 3\]"')

    def test_seq_spec_on_namedtuple(self):
        from collections import namedtuple

        Pair = namedtuple("Pair", ["first", "second"])

        class M(torch.nn.Module):
            def forward(self, p):
                return p.first.sum(0) + p.second.sum(0)

        ep = export(
            M(),
            args=(Pair(torch.randn(5, 3), torch.randn(8, 3)),),
            dynamic_shapes=PARAMS({"p": L([None, T([VAR("B"), STATIC])])}),
            strict=self.strict,
        )
        ep_str = str(ep)
        # Position 0 → static (None entry); position 1 → dynamic.
        self.assertIn('p_first: "f32[5, 3]"', ep_str)
        self.assertRegex(ep_str, r'p_second: "f32\[u\d+, 3\]"')

    def test_object_spec_on_custom_pytree_node(self):
        """ObjectSpec on a class registered via
        ``pytree.register_pytree_node`` with a caller-supplied
        ``flatten_with_keys_fn`` that yields ``GetAttrKey`` entries."""

        class MyContainer:
            def __init__(self, a, b):
                self.a = a
                self.b = b

        # Caller supplies flatten / unflatten / flatten_with_keys_fn —
        # the KeyPath shape is whatever the caller chooses to return.
        # We use GetAttrKey so ObjectSpec can address fields by name.
        def _flatten(c):
            return [c.a, c.b], None

        def _unflatten(values, _context):
            a, b = values
            return MyContainer(a, b)

        def _flatten_with_keys(c):
            children = [
                (pytree.GetAttrKey("a"), c.a),
                (pytree.GetAttrKey("b"), c.b),
            ]
            return children, None

        pytree.register_pytree_node(
            MyContainer,
            _flatten,
            _unflatten,
            serialized_type_name="test_dynamic_spec_export.MyContainer",
            flatten_with_keys_fn=_flatten_with_keys,
        )

        class M(torch.nn.Module):
            def forward(self, c):
                return c.a.sum(0) + c.b.sum(0)

        ep = export(
            M(),
            args=(MyContainer(torch.randn(5, 3), torch.randn(8, 3)),),
            dynamic_shapes=PARAMS({"c": OBJ({"b": T([VAR("B"), STATIC])})}),
            strict=self.strict,
        )
        ep_str = str(ep)
        # a is omitted from the spec → static; b is dynamic.
        self.assertIn('c_a: "f32[5, 3]"', ep_str)
        self.assertRegex(ep_str, r'c_b: "f32\[u\d+, 3\]"')

    def test_object_spec_via_torch_export_register_dataclass(self):
        import dataclasses

        import torch.export as te

        @dataclasses.dataclass
        class ExportedBox:
            x: torch.Tensor
            y: torch.Tensor

        te.register_dataclass(ExportedBox)

        class M(torch.nn.Module):
            def forward(self, box):
                return box.x.sum(0) + box.y.sum(0)

        ep = export(
            M(),
            args=(ExportedBox(torch.randn(5, 3), torch.randn(8, 3)),),
            dynamic_shapes=PARAMS({"box": OBJ({"y": T([VAR("Y"), STATIC])})}),
            strict=self.strict,
        )
        ep_str = str(ep)
        # x is omitted from the spec → static; y is dynamic.
        self.assertIn('box_x: "f32[5, 3]"', ep_str)
        self.assertRegex(ep_str, r'box_y: "f32\[u\d+, 3\]"')

    def test_object_spec_on_dataclass(self):
        import dataclasses

        @dataclasses.dataclass
        class Box:
            x: torch.Tensor
            y: torch.Tensor

        pytree.register_dataclass(Box)

        class M(torch.nn.Module):
            def forward(self, box):
                return box.x.sum(0) + box.y.sum(0)

        ep = export(
            M(),
            args=(Box(torch.randn(5, 3), torch.randn(8, 3)),),
            dynamic_shapes=PARAMS({"box": OBJ({"y": T([VAR("Y"), STATIC])})}),
            strict=self.strict,
        )
        ep_str = str(ep)
        # x is omitted from the spec → static; y is dynamic.
        self.assertIn('box_x: "f32[5, 3]"', ep_str)
        self.assertRegex(ep_str, r'box_y: "f32\[u\d+, 3\]"')

    def test_nested_dict_of_seq_spec(self):
        """Nested DictSpec({"foo": SeqSpec([TensorSpec, None])}) — first
        list elem dynamic, second left to default (static)."""

        class M(torch.nn.Module):
            def forward(self, d):
                return d["foo"][0].sum(0) + d["foo"][1].sum(0)

        ep = export(
            M(),
            args=({"foo": [torch.randn(8, 3), torch.randn(5, 3)]},),
            dynamic_shapes=PARAMS(
                {
                    "d": DICT(
                        {
                            "foo": L(
                                [
                                    T([VAR("A"), STATIC]),
                                    T([5, 3]),
                                ]
                            )
                        }
                    )
                }
            ),
            strict=self.strict,
        )
        ep_str = str(ep)
        self.assertRegex(ep_str, r'd_foo_0: "f32\[u\d+, 3\]"')
        self.assertIn('d_foo_1: "f32[5, 3]"', ep_str)

    def test_mixed_leaf_and_container_under_params_spec(self):
        class M(torch.nn.Module):
            def forward(self, a, b):
                return a.sum(0) + b[0].sum(0) + b[1].sum(0)

        ep = export(
            M(),
            args=(
                torch.randn(8, 3),
                [torch.randn(5, 3), torch.randn(6, 3)],
            ),
            dynamic_shapes=PARAMS(
                {
                    "a": T([VAR("A"), STATIC]),
                    "b": L(
                        [
                            T([VAR("B0"), STATIC]),
                            T([VAR("B1"), STATIC]),
                        ]
                    ),
                }
            ),
            strict=self.strict,
        )
        ep_str = str(ep)
        self.assertRegex(ep_str, r'a: "f32\[u\d+, 3\]"')
        self.assertRegex(ep_str, r'b_0: "f32\[u\d+, 3\]"')
        self.assertRegex(ep_str, r'b_1: "f32\[u\d+, 3\]"')

    # ---- Partial-spec cases ----

    def test_seq_spec_shorter_than_runtime_list_tail_static(self):
        class M(torch.nn.Module):
            def forward(self, xs):
                return xs[0].sum() + xs[1].sum() + xs[2].sum()

        ep = export(
            M(),
            args=([torch.randn(8, 3), torch.randn(5, 3), torch.randn(6, 3)],),
            dynamic_shapes=PARAMS({"xs": L([T([VAR("A"), STATIC])])}),
            strict=self.strict,
        )
        ep_str = str(ep)
        self.assertRegex(ep_str, r'xs_0: "f32\[u\d+, 3\]"')
        self.assertIn('xs_1: "f32[5, 3]"', ep_str)
        self.assertIn('xs_2: "f32[6, 3]"', ep_str)

    # ---- Negative / structural cases ----

    def test_seq_spec_on_dict_arg_raises(self):
        class M(torch.nn.Module):
            def forward(self, d):
                return d["a"]

        with self.assertRaisesRegex(
            ValueError,
            r"SeqSpec expected list/tuple, got dict",
        ):
            export(
                M(),
                args=({"a": torch.randn(4)},),
                dynamic_shapes=PARAMS({"d": L([T([VAR("A")])])}),
                strict=self.strict,
            )

    def test_seq_spec_longer_than_runtime_list_raises(self):
        class M(torch.nn.Module):
            def forward(self, xs):
                return xs[0]

        with self.assertRaisesRegex(
            ValueError,
            r"SeqSpec has 2 entries beyond runtime sequence length 1",
        ):
            export(
                M(),
                args=([torch.randn(4)],),
                dynamic_shapes=PARAMS(
                    {
                        "xs": L(
                            [
                                T([VAR("A")]),
                                T([VAR("B")]),
                            ]
                        )
                    }
                ),
                strict=self.strict,
            )

    def test_dict_spec_key_not_in_runtime_dict_raises(self):
        class M(torch.nn.Module):
            def forward(self, d):
                return d["a"]

        with self.assertRaisesRegex(
            ValueError,
            r"shapes_spec\['d'\]: DictSpec has entries \['missing'\] that do not match any key in the runtime dict\. Runtime keys: \['a'\]",
        ):
            export(
                M(),
                args=({"a": torch.randn(4)},),
                dynamic_shapes=PARAMS({"d": DICT({"missing": T([VAR("A")])})}),
                strict=self.strict,
            )

    def test_object_spec_attr_not_on_runtime_object_raises(self):
        import dataclasses

        @dataclasses.dataclass
        class Box2:
            x: torch.Tensor

        pytree.register_dataclass(Box2)

        class M(torch.nn.Module):
            def forward(self, box):
                return box.x

        with self.assertRaisesRegex(
            ValueError,
            r"ObjectSpec has entries .*'nope'.* that do not match any attribute",
        ):
            export(
                M(),
                args=(Box2(torch.randn(4)),),
                dynamic_shapes=PARAMS({"box": OBJ({"nope": T([VAR("A")])})}),
                strict=self.strict,
            )

    def test_where_path_accumulates_in_nested_error(self):
        """Errors from nested specs should report the full path to the
        offending spot, exercising all three ``where``-formatting
        variants: ``[str]`` (DictSpec), ``.attr`` (ObjectSpec), and
        ``[int]`` (SeqSpec)."""
        import dataclasses

        @dataclasses.dataclass
        class Box:
            items: list

        pytree.register_dataclass(Box)

        class M(torch.nn.Module):
            def forward(self, batch):
                return batch["b"].items[1]["data"]

        with self.assertRaisesRegex(
            ValueError,
            re.escape(
                "shapes_spec['batch']['b'].items[1]: "
                "DictSpec has entries ['missing'] that do not match "
                "any key in the runtime dict. Runtime keys: ['data']"
            ),
        ):
            export(
                M(),
                args=({"b": Box(items=[torch.randn(3), {"data": torch.randn(3)}])},),
                dynamic_shapes=PARAMS(
                    {
                        "batch": DICT(
                            {
                                "b": OBJ(
                                    {
                                        "items": L(
                                            [
                                                None,
                                                DICT({"missing": T([VAR("A")])}),
                                            ]
                                        )
                                    }
                                )
                            }
                        )
                    }
                ),
                strict=self.strict,
            )

    def test_object_spec_on_pytree_node_without_keys_fn_raises(self):
        class KeyslessContainer:
            def __init__(self, x):
                self.x = x

        pytree.register_pytree_node(
            KeyslessContainer,
            lambda c: ([c.x], None),
            lambda values, _context: KeyslessContainer(next(iter(values))),
            serialized_type_name="test_dynamic_spec_export.KeyslessContainer",
        )

        class M(torch.nn.Module):
            def forward(self, c):
                return c.x

        with self.assertRaisesRegex(
            ValueError,
            r"export requires `flatten_with_keys_fn` to be registered for "
            r"type KeyslessContainer",
        ):
            export(
                M(),
                args=(KeyslessContainer(torch.randn(4)),),
                dynamic_shapes=PARAMS({"c": OBJ({"x": T([VAR("A")])})}),
                strict=self.strict,
            )

    # ---- Alignment invariant: walker order == pytree.tree_flatten order ----

    def test_walker_alignment_with_pytree_flatten(self):
        """For each structured input value, ``_walk_spec`` must
        return leaf specs in exactly the same order as
        ``pytree.tree_flatten(value)``.

        Proof technique: build a "complete" container spec where every
        leaf is a unique TensorSpec whose ShapeVar name encodes the
        ``id()`` of the tensor it targets. After flattening the user
        spec, assert that the i-th returned spec carries the name
        corresponding to the i-th tensor from
        ``pytree.tree_leaves(value)``. Any drift in iteration order
        would cause a name mismatch.
        """
        import dataclasses

        from torch.fx.experimental._spec_binding import _walk_spec

        @dataclasses.dataclass
        class _Box:
            u: torch.Tensor
            v: torch.Tensor

        pytree.register_dataclass(_Box)

        def _build_complete_spec(v):
            """Build a container spec that targets every tensor in v
            with a uniquely-named ShapeVar."""
            if isinstance(v, torch.Tensor):
                return T([VAR(f"t{id(v)}")])
            if isinstance(v, (list, tuple)):
                return L([_build_complete_spec(e) for e in v])
            if isinstance(v, dict):
                return DICT({k: _build_complete_spec(val) for k, val in v.items()})
            if dataclasses.is_dataclass(v) and not isinstance(v, type):
                return OBJ(
                    {
                        f.name: _build_complete_spec(getattr(v, f.name))
                        for f in dataclasses.fields(v)
                    }
                )
            # Leaf type with no spec needed (None / int / etc.)
            return None

        cases = [
            # Bare tensor — trivial leaf case.
            torch.randn(3),
            # Plain containers.
            [torch.randn(3), torch.randn(4)],
            (torch.randn(3), torch.randn(4)),
            {"a": torch.randn(3), "b": torch.randn(4)},
            # Different dict insertion order — must still align.
            {"z": torch.randn(3), "a": torch.randn(4), "m": torch.randn(5)},
            # Dataclass.
            _Box(torch.randn(3), torch.randn(4)),
            # Nested dict-of-list-of-tensors.
            {"x": [torch.randn(3), torch.randn(4)], "y": torch.randn(5)},
            # Nested list-of-dataclass-and-dict.
            [
                _Box(torch.randn(3), torch.randn(4)),
                {"k": torch.randn(5), "j": torch.randn(6)},
            ],
            # Deeply nested.
            {
                "a": [
                    _Box(torch.randn(2), torch.randn(3)),
                    {"inner": [torch.randn(4), torch.randn(5)]},
                ],
                "b": torch.randn(6),
            },
        ]

        for arg_value in cases:
            spec = _build_complete_spec(arg_value)
            leaves = pytree.tree_leaves(arg_value)
            out = _walk_spec(spec, arg_value, where="<root>")
            self.assertEqual(
                len(out),
                len(leaves),
                lambda msg: f"{msg}\nleaf-count drift for case {arg_value!r}",
            )
            # Per-slot check: each slot's spec name must match the
            # tensor at that flat position from pytree.tree_flatten.
            for i, (slot_spec, leaf) in enumerate(zip(out, leaves, strict=True)):
                self.assertIsInstance(
                    slot_spec,
                    T,
                    msg=lambda msg: f"{msg}\nslot {i} is {slot_spec!r} (expected TensorSpec) "
                    f"for case {arg_value!r}",
                )
                expected = f"t{id(leaf)}"
                actual = slot_spec._specs[0].name
                self.assertEqual(
                    actual,
                    expected,
                    msg=(
                        lambda msg: f"{msg}\nalignment drift at slot {i}: spec name "
                        f"{actual!r} does not match pytree-flatten leaf "
                        f"name {expected!r} for case {arg_value!r}"
                    ),
                )

    def test_walker_alignment_no_spec_advances_correctly(self):
        """When user_spec=None, the walker must return exactly
        len(pytree.tree_leaves(value)) — i.e. no-spec subtrees stay in
        lockstep with pytree's flatten."""
        from torch.fx.experimental._spec_binding import _walk_spec

        cases = [
            torch.randn(3),
            [torch.randn(3), torch.randn(4), torch.randn(5)],
            {"a": [torch.randn(3), torch.randn(4)], "b": torch.randn(5)},
            (torch.randn(3), {"k": torch.randn(4)}, [torch.randn(5)]),
        ]
        for arg_value in cases:
            expected = len(pytree.tree_leaves(arg_value))
            out = _walk_spec(None, arg_value, where="<root>")
            self.assertEqual(
                len(out),
                expected,
                lambda msg: f"{msg}\nno-spec leaf-count drift for {arg_value!r}",
            )


class TestContainerSpecStrict(_TestContainerSpecBase):
    strict = True


class TestContainerSpecNonStrict(_TestContainerSpecBase):
    strict = False


del _TestContainerSpecBase
if __name__ == "__main__":
    run_tests()
