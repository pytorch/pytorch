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
    DictSpec,
    IntVar,
    ObjectSpec,
    ParamsSpec,
    SeqSpec,
    ShapesSpec,
    ShapeVar,
    STATIC,
    TensorSpec,
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


class TestExportDynamicSpec(TestCase):
    """torch.export.export support for the new ShapesSpec/ParamsSpec API."""

    def setUp(self):
        super().setUp()
        _reset_uid_counter()

    def test_unbacked_graph_has_unbacked_symbol(self):
        """Tensor dim marked with ShapeVar shows up as an unbacked SymInt
        in the exported graph, and the export accepts varying inputs."""
        B = ShapeVar("batch")
        x_spec = TensorSpec([B, None])
        ep = export(
            _ModX(),
            (torch.randn(8, 3),),
            dynamic_shapes=ParamsSpec({"x": x_spec}),
            strict=True,
        )
        shape = _first_tensor_placeholder_shape(ep.graph_module)
        self.assertIsInstance(shape[0], torch.SymInt)
        self.assertGreater(len(free_unbacked_symbols(shape[0])), 0)
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
        """Declaring a scalar int as static=10 but passing 42 should error."""
        with self.assertRaisesRegex(
            ValueError,
            r"shapes_spec declares L\['flat_args'\]\[1\] as static with value 10, but got 42",
        ):
            export(
                _ModXN(),
                (torch.randn(4), 42),
                dynamic_shapes=ShapesSpec(params=ParamsSpec({"n": 10})),
                strict=True,
            )

    def test_static_tensor_dim_mismatch_raises(self):
        """Declaring dim 1 as static=3 but passing dim 1=5 should error."""
        with self.assertRaisesRegex(
            ValueError,
            r"shapes_spec declares dim 1 as static with value 3, but got 5",
        ):
            export(
                _ModXPlus(),
                (torch.randn(4, 5),),
                dynamic_shapes=ShapesSpec(
                    params=ParamsSpec({"x": TensorSpec([ShapeVar("batch"), 3])})
                ),
                strict=True,
            )

    def test_params_spec_shorthand(self):
        """dynamic_shapes=ParamsSpec(...) is auto-wrapped into ShapesSpec."""
        ep = export(
            _ModX(),
            (torch.randn(8, 3),),
            dynamic_shapes=ParamsSpec({"x": TensorSpec([ShapeVar("batch"), None])}),
            strict=True,
        )
        shape = _first_tensor_placeholder_shape(ep.graph_module)
        self.assertIsInstance(shape[0], torch.SymInt)

    @_fx_experimental_config.patch(no_data_dependent_graph_break=True)
    def test_min_max_bypasses_dde_on_branching(self):
        """Setting min/max on ShapeVar lets a branch resolve statically."""
        # min=10 > 5 → branch resolves statically, no DDE.
        export(
            _ModBranch(),
            (torch.randn(20, 3),),
            dynamic_shapes=ShapesSpec(
                params=ParamsSpec(
                    {"x": TensorSpec([ShapeVar("batch", min=10, max=100), None])}
                )
            ),
            strict=True,
        )

    @_fx_experimental_config.patch(no_data_dependent_graph_break=True)
    def test_unbacked_raises_dde_on_branching(self):
        """Without min/max, branching on a ShapeVar dim raises a DDE
        (export wraps it as a UserError)."""
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            r"Could not guard on data-dependent expression",
        ):
            export(
                _ModBranch(),
                (torch.randn(10, 3),),
                dynamic_shapes=ShapesSpec(
                    params=ParamsSpec({"x": TensorSpec([ShapeVar(), None])})
                ),
                strict=True,
            )

    def test_tensor_dim_optimization_hint_in_shape_env(self):
        """ShapeVar's optimization_hint propagates to var_to_hint_override."""
        b = ShapeVar("batch", optimization_hint=32)
        ep = export(
            _ModX(),
            (torch.randn(8, 3),),
            dynamic_shapes=ShapesSpec(params=ParamsSpec({"x": TensorSpec([b, None])})),
            strict=True,
        )
        shape = _first_tensor_placeholder_shape(ep.graph_module)
        sym = shape[0]
        self.assertIsInstance(sym, torch.SymInt)
        expr = sym.node.expr
        self.assertEqual(sym.node.shape_env.var_to_hint_override.get(expr), 32)

    def test_params_spec_keys_match_forward_args_for_mixed_positional_and_kwargs(
        self,
    ):
        """A spec entry is matched to an input by name (the forward parameter
        name), regardless of whether the value is passed positionally (in
        ``args``) or as a keyword (in ``kwargs``). Inputs not in the spec stay
        static (a scalar arg stays specialized to its literal value)."""

        class M(torch.nn.Module):
            def forward(self, x, y, z=None):
                return x.sum(0) + y.sum(0) * (z if z is not None else 1)

        ep = export(
            M(),
            args=(torch.randn(8, 3),),
            kwargs={"y": torch.randn(5, 3), "z": 7},
            dynamic_shapes=ShapesSpec(
                params=ParamsSpec(
                    {
                        "x": TensorSpec([ShapeVar("A"), None]),
                        "y": TensorSpec([ShapeVar("B"), None]),
                    }
                )
            ),
            strict=True,
        )
        ep_str = str(ep)
        self.assertRegex(ep_str, r'x: "f32\[u\d+, 3\]"')
        self.assertRegex(ep_str, r'y: "f32\[u\d+, 3\]"')
        # z (scalar int kwarg, not in spec) gets specialized to literal 7.
        self.assertIn("mul.Tensor", ep_str)
        ep.module()(torch.randn(20, 3), y=torch.randn(99, 3), z=7)

    def test_scalar_int_input_via_int_var(self):
        """A positional scalar int input marked dynamic via ``IntVar``
        becomes an unbacked SymInt in the exported program."""

        class M(torch.nn.Module):
            def forward(self, x, n):
                return x.sum(0) * n

        ep = export(
            M(),
            (torch.randn(8, 3), 5),
            dynamic_shapes=ShapesSpec(
                params=ParamsSpec(
                    {
                        "x": TensorSpec([ShapeVar("B"), None]),
                        "n": IntVar("n_size"),
                    }
                )
            ),
            strict=True,
        )
        ep_str = str(ep)
        self.assertRegex(ep_str, r'n: "Sym\(u\d+\)"')
        ep.module()(torch.randn(20, 3), 42)
        ep.module()(torch.randn(50, 3), 99)

    def test_kwarg_in_non_signature_order_uses_call_order(self):
        """A spec entry is matched to an input by name, not by position. So
        even when kwargs are passed in a different order than the forward
        signature, the spec'd dim still lands on the correct input."""

        class M(torch.nn.Module):
            def forward(self, x, y, z):
                return x.sum(0) + y.sum(0) + z.sum(0)

        ep = export(
            M(),
            args=(torch.randn(8, 3),),
            kwargs={"z": torch.randn(7, 3), "y": torch.randn(5, 3)},
            dynamic_shapes=ShapesSpec(
                params=ParamsSpec({"y": TensorSpec([ShapeVar("Y"), None])})
            ),
            strict=True,
        )
        # y dim 0 is unbacked (u0); x and z keep their static literal sizes.
        self.assertExpectedInline(
            str(ep).strip(),
            """\
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, x: "f32[8, 3]", z: "f32[7, 3]", y: "f32[u0, 3]"):
            sym_size_int: "Sym(u0)" = torch.ops.aten.sym_size.int(y, 0)
            ge: "Sym(u0 >= 0)" = sym_size_int >= 0;  sym_size_int = None
            _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
            sum_1: "f32[3]" = torch.ops.aten.sum.dim_IntList(x, [0]);  x = None
            sum_2: "f32[3]" = torch.ops.aten.sum.dim_IntList(y, [0]);  y = None
            add: "f32[3]" = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
            sum_3: "f32[3]" = torch.ops.aten.sum.dim_IntList(z, [0]);  z = None
            add_1: "f32[3]" = torch.ops.aten.add.Tensor(add, sum_3);  add = sum_3 = None
            return (add_1,)
Graph signature:
    x: USER_INPUT
    z: USER_INPUT
    y: USER_INPUT
    add_1: USER_OUTPUT
Range constraints: {u0: VR[0, int_oo]}""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )

    def test_multi_leaf_arg_with_leaf_spec_raises(self):
        """Multi-leaf arg (e.g. ``list[Tensor]``) with a leaf spec
        (``TensorSpec``) is structurally incompatible.
        """

        class M(torch.nn.Module):
            def forward(self, xs):
                return xs[0] + xs[1]

        with self.assertRaisesRegex(
            ValueError,
            r"shapes_spec\['xs'\].*TensorSpec.*not a Tensor",
        ):
            export(
                M(),
                args=([torch.randn(8, 3), torch.randn(8, 3)],),
                dynamic_shapes=ShapesSpec(
                    params=ParamsSpec({"xs": TensorSpec([ShapeVar("B"), None])})
                ),
                strict=True,
            )

    def test_multi_leaf_arg_no_spec_stays_static(self):
        """No spec entry for a multi-leaf arg → all flat slots stay static
        (per 'anything not expressible in ParamsSpec is static').
        """

        class M(torch.nn.Module):
            def forward(self, xs):
                return xs[0] + xs[1]

        ep = export(
            M(),
            args=([torch.randn(8, 3), torch.randn(8, 3)],),
            dynamic_shapes=ShapesSpec(),  # no params → all static
            strict=True,
        )
        ep_str = str(ep)
        # Both list elements come through as static placeholders.
        self.assertIn('xs_0: "f32[8, 3]"', ep_str)
        self.assertIn('xs_1: "f32[8, 3]"', ep_str)
        out = ep.module()([torch.randn(8, 3), torch.randn(8, 3)])
        self.assertEqual(out.shape, torch.Size([8, 3]))

    def test_spec_entry_for_omitted_default_raises(self):
        """A spec entry for a defaulted param the caller omits matches no
        passed argument, so it raises rather than being silently ignored."""

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
                dynamic_shapes=ShapesSpec(
                    params=ParamsSpec(
                        {
                            "x": TensorSpec([ShapeVar("B"), None]),
                            "y": TensorSpec([ShapeVar("Y"), None]),  # no such arg
                        }
                    )
                ),
                strict=True,
            )

    def test_explicit_none_spec_for_passed_arg_does_not_raise(self):
        """Explicit ``None`` in ParamsSpec for a passed arg means "static"
        and must not be flagged as an unmatched spec key (regression: a
        prior `.get()` check conflated key-absent with value-is-None).
        """
        ep = export(
            _ModXPlus(),
            args=(torch.randn(8, 3),),
            dynamic_shapes=ShapesSpec(
                params=ParamsSpec({"x": None}),  # explicit static spec for x
            ),
            strict=True,
        )
        shape = _first_tensor_placeholder_shape(ep.graph_module)
        self.assertEqual(tuple(shape), (8, 3))

    def test_non_strict_raises_not_implemented(self):
        """Non-strict export does not yet support ShapesSpec/ParamsSpec."""
        with self.assertRaisesRegex(
            NotImplementedError,
            r"ShapesSpec/ParamsSpec in dynamic_shapes is not yet supported "
            r"in non-strict export",
        ):
            export(
                _ModX(),
                (torch.randn(8, 3),),
                dynamic_shapes=ShapesSpec(
                    params=ParamsSpec({"x": TensorSpec([ShapeVar("batch"), None])})
                ),
                strict=False,
            )

    def test_named_dims_vs_shapes_spec(self):
        from torch.export import Dim

        class M(torch.nn.Module):
            def forward(self, x):
                return x.sum(0)

        ep_legacy = export(
            M(),
            (torch.randn(8, 3),),
            dynamic_shapes={"x": (Dim("B"), None)},
            strict=True,
        )
        ep_new = export(
            M(),
            (torch.randn(8, 3),),
            dynamic_shapes=ShapesSpec(
                params=ParamsSpec({"x": TensorSpec([ShapeVar("B"), None])})
            ),
            strict=True,
        )

        # Snapshot legacy graph (backed s* symbol, range constraint).
        self.assertExpectedInline(
            str(ep_legacy).strip(),
            """\
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, x: "f32[s6, 3]"):
            sum_1: "f32[3]" = torch.ops.aten.sum.dim_IntList(x, [0]);  x = None
            return (sum_1,)
Graph signature:
    x: USER_INPUT
    sum_1: USER_OUTPUT
Range constraints: {s6: VR[0, int_oo]}""",
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
            strict=True,
        )
        ep_new = export(
            M(),
            args=(torch.randn(8, 3),),
            kwargs={"y": torch.randn(5, 3)},
            dynamic_shapes=ParamsSpec(
                {
                    "x": TensorSpec([ShapeVar("X"), None]),
                    "y": TensorSpec([ShapeVar("Y"), None]),
                }
            ),
            strict=True,
        )

        # Snapshot legacy: placeholders are (x, y) in signature order.
        self.assertExpectedInline(
            str(ep_legacy).strip(),
            """\
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, x: "f32[s6, 3]", y: "f32[s27, 3]"):
            sum_1: "f32[3]" = torch.ops.aten.sum.dim_IntList(x, [0]);  x = None
            sum_2: "f32[3]" = torch.ops.aten.sum.dim_IntList(y, [0]);  y = None
            mul: "f32[3]" = torch.ops.aten.mul.Tensor(sum_2, 1);  sum_2 = None
            add: "f32[3]" = torch.ops.aten.add.Tensor(sum_1, mul);  sum_1 = mul = None
            return (add,)
Graph signature:
    x: USER_INPUT
    y: USER_INPUT
    add: USER_OUTPUT
Range constraints: {s6: VR[0, int_oo], s27: VR[0, int_oo]}""",
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
        """Parity: optional kwarg `y` not passed → not in graph either way."""
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
            strict=True,
        )
        ep_new = export(
            M(),
            args=(torch.randn(8, 3),),
            dynamic_shapes=ShapesSpec(
                params=ParamsSpec({"x": TensorSpec([ShapeVar("X"), None])})
            ),
            strict=True,
        )

        # Legacy: only x in graph signature, no y placeholder.
        self.assertExpectedInline(
            str(ep_legacy).strip(),
            """\
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, x: "f32[s6, 3]"):
            sum_1: "f32[3]" = torch.ops.aten.sum.dim_IntList(x, [0]);  x = None
            return (sum_1,)
Graph signature:
    x: USER_INPUT
    sum_1: USER_OUTPUT
Range constraints: {s6: VR[0, int_oo]}""",
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
        """User's `def forward(*args)` with ``ParamsSpec({"*args": [...]})`` —
        each varargs position can be marked dynamic independently. This
        exercises the translator's user-`_varargs` plumbing."""

        class M(torch.nn.Module):
            def forward(self, *args):
                return args[0].sum() + args[1].sum()

        ep = export(
            M(),
            args=(torch.randn(8, 3), torch.randn(5, 3)),
            dynamic_shapes=ShapesSpec(
                params=ParamsSpec(
                    {
                        "*args": [
                            TensorSpec([ShapeVar("A"), None]),
                            TensorSpec([ShapeVar("B"), None]),
                        ]
                    }
                )
            ),
            strict=True,
        )
        ep_str = str(ep)
        # Both args dim 0 are unbacked (different symbols).
        self.assertRegex(ep_str, r'args_0: "f32\[u\d+, 3\]"')
        self.assertRegex(ep_str, r'args_1: "f32\[u\d+, 3\]"')
        ep.module()(torch.randn(20, 3), torch.randn(99, 3))

    def test_user_varargs_with_named_arg_before(self):
        """Mixed: `def forward(self, x, *args)` — `x` named, the rest
        captured by `*args`. User passes spec for both."""

        class M(torch.nn.Module):
            def forward(self, x, *args):
                return x.sum() + args[0].sum() + args[1].sum()

        ep = export(
            M(),
            args=(torch.randn(4, 3), torch.randn(8, 3), torch.randn(5, 3)),
            dynamic_shapes=ShapesSpec(
                params=ParamsSpec(
                    {
                        "x": TensorSpec([ShapeVar("X"), None]),
                        "*args": [
                            TensorSpec([ShapeVar("A"), None]),
                            TensorSpec([ShapeVar("B"), None]),
                        ],
                    },
                )
            ),
            strict=True,
        )
        ep_str = str(ep)
        self.assertRegex(ep_str, r'x: "f32\[u\d+, 3\]"')
        self.assertRegex(ep_str, r'args_0: "f32\[u\d+, 3\]"')
        self.assertRegex(ep_str, r'args_1: "f32\[u\d+, 3\]"')
        ep.module()(torch.randn(7, 3), torch.randn(20, 3), torch.randn(99, 3))

    def test_user_varargs_partial_spec_leaves_remainder_static(self):
        """User specifies only the first ``*args`` slot; remaining slots
        stay static (per "anything not expressible in ParamsSpec is
        static")."""

        class M(torch.nn.Module):
            def forward(self, *args):
                return args[0].sum() + args[1].sum() + args[2].sum()

        ep = export(
            M(),
            args=(torch.randn(8, 3), torch.randn(5, 3), torch.randn(6, 3)),
            dynamic_shapes=ShapesSpec(
                params=ParamsSpec(
                    {
                        "*args": [
                            TensorSpec([ShapeVar("A"), None]),
                        ]
                    }
                )
            ),
            strict=True,
        )
        ep_str = str(ep)
        # First arg dynamic, others literally sized.
        self.assertRegex(ep_str, r'args_0: "f32\[u\d+, 3\]"')
        self.assertIn('args_1: "f32[5, 3]"', ep_str)
        self.assertIn('args_2: "f32[6, 3]"', ep_str)

    def test_user_varkw_in_forward_marked_dynamic_via_varkw_spec(self):
        """When ``forward`` accepts ``**kwargs``, a kwarg can be marked
        dynamic by putting it under the reserved ``"**kwargs"`` key, e.g.
        ``ParamsSpec({"**kwargs": {"foo": TensorSpec(...)}})``."""

        class M(torch.nn.Module):
            def forward(self, **kwargs):
                return kwargs["foo"].sum() + kwargs["bar"].sum()

        ep = export(
            M(),
            args=(),
            kwargs={"foo": torch.randn(8, 3), "bar": torch.randn(5, 3)},
            dynamic_shapes=ShapesSpec(
                params=ParamsSpec(
                    {
                        "**kwargs": {
                            "foo": TensorSpec([ShapeVar("F"), None]),
                            "bar": TensorSpec([ShapeVar("B"), None]),
                        }
                    }
                )
            ),
            strict=True,
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
                dynamic_shapes=ShapesSpec(
                    params=ParamsSpec({"x": TensorSpec([ShapeVar("batch"), None])})
                ),
                strict=True,
                prefer_deferred_runtime_asserts_over_guards=True,
            )

    def test_derived_dim_runtime_enforced(self):
        """Derived dim ``y dim0 = B * 2`` is enforced at runtime: the graph
        placeholder shows ``2*u0``, a correct input runs, and a violating
        input raises a runtime assertion (mirrors dynamo
        ``TestDerivedDimSpec.test_derived_dim`` but with export semantics)."""
        B = ShapeVar("batch")

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
            dynamic_shapes=ShapesSpec(
                params=ParamsSpec(
                    {
                        "x": TensorSpec([B, STATIC]),
                        "y": TensorSpec([B * 2, STATIC]),
                    }
                )
            ),
            strict=True,
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
        """Composite derived dim ``z dim0 = A * B + 1`` is enforced at
        runtime: correct input runs, violating input raises."""
        A = ShapeVar("a")
        B = ShapeVar("b")

        class M(torch.nn.Module):
            def forward(self, x, y, z):
                return x.sum() + y.sum() + z.sum()

        ep = export(
            M(),
            (torch.randn(3, 2), torch.randn(4, 2), torch.randn(13, 2)),
            dynamic_shapes=ShapesSpec(
                params=ParamsSpec(
                    {
                        "x": TensorSpec([A, STATIC]),
                        "y": TensorSpec([B, STATIC]),
                        "z": TensorSpec([A * B + 1, STATIC]),
                    }
                )
            ),
            strict=True,
        )
        self.assertTrue(_has_assert_scalar(ep.graph_module))
        # Correct: z.shape[0] == 3 * 4 + 1 == 13.
        ep.module()(torch.randn(3, 2), torch.randn(4, 2), torch.randn(13, 2))
        # Violation: 99 != 3 * 4 + 1.
        with self.assertRaisesRegex(RuntimeError, "Runtime assertion failed"):
            ep.module()(torch.randn(3, 2), torch.randn(4, 2), torch.randn(99, 2))

    def test_assumption_runtime_enforced(self):
        """Relational assumption ``A > B`` is enforced at runtime: correct
        input (``a > b``) runs, violating input (``a <= b``) raises"""
        A = ShapeVar("a")
        B = ShapeVar("b")

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
                params=ParamsSpec(
                    {
                        "x": TensorSpec([A, STATIC]),
                        "y": TensorSpec([B, STATIC]),
                    }
                ),
                assumptions=[A > B],
            ),
            strict=True,
        )
        self.assertTrue(_has_assert_scalar(ep.graph_module))
        # Correct: a=5 > b=3.
        ep.module()(torch.randn(5, 2), torch.randn(3, 2))
        # Violation: a=2 not > b=3.
        with self.assertRaisesRegex(RuntimeError, "Runtime assertion failed"):
            ep.module()(torch.randn(2, 2), torch.randn(3, 2))

    def test_min_max_in_range_constraints(self):
        """``ShapeVar(min=10, max=100)`` propagates to ``ep.range_constraints``
        via the inline-constraints channel."""
        ep = export(
            _ModX(),
            (torch.randn(20, 3),),
            dynamic_shapes=ShapesSpec(
                params=ParamsSpec(
                    {"x": TensorSpec([ShapeVar("b", min=10, max=100), None])}
                )
            ),
            strict=True,
        )
        rcs = ep.range_constraints
        self.assertEqual(len(rcs), 1)
        (vr,) = rcs.values()
        self.assertEqual(int(vr.lower), 10)
        self.assertEqual(int(vr.upper), 100)


class TestExportDynamicSpecInternalAPIs(TestCase):
    """Direct unit tests for the internal ``torch.export._trace`` entrypoints
    with ShapesSpec. The public ``TestExportDynamicSpec`` only exercises these
    transitively through ``torch.export.export``; these call them in isolation.
    """

    def setUp(self):
        super().setUp()
        _reset_uid_counter()

    def _spec(self):
        return ShapesSpec(
            params=ParamsSpec({"x": TensorSpec([ShapeVar("batch"), None])})
        )

    def test_export_to_torch_ir_shapes_spec_direct(self):
        """`_export_to_torch_ir` accepts a ShapesSpec (new tracer) and produces
        a torch-IR GraphModule whose user placeholder has an unbacked dim."""
        gm = _export_to_torch_ir(
            _ModX(), (torch.randn(8, 3),), {}, dynamic_shapes=self._spec()
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
        """The legacy v1 dynamo.export path (use_new_tracer_experimental=False)
        does not support ShapesSpec and must raise NotImplementedError."""
        with mock.patch.object(
            torch._export.config, "use_new_tracer_experimental", False
        ):
            with self.assertRaisesRegex(
                NotImplementedError,
                r"ShapesSpec is not supported on the legacy v1",
            ):
                _export_to_torch_ir(
                    _ModX(), (torch.randn(8, 3),), {}, dynamic_shapes=self._spec()
                )

    def test_strict_export_shapes_spec_direct(self):
        """`_strict_export` in isolation produces an aten artifact whose graph
        carries the unbacked-input runtime assert."""
        args = (torch.randn(8, 3),)
        _, in_spec = pytree.tree_flatten((args, {}))
        artifact = _strict_export(
            mod=_ModX(),
            args=args,
            kwargs={},
            dynamic_shapes=self._spec(),
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

    def test_non_strict_export_shapes_spec_raises_direct(self):
        """`_non_strict_export` rejects ShapesSpec up front."""
        args = (torch.randn(8, 3),)
        _, in_spec = pytree.tree_flatten((args, {}))
        with self.assertRaisesRegex(
            NotImplementedError,
            r"not yet supported .*in non-strict export",
        ):
            _non_strict_export(
                mod=_ModX(),
                args=args,
                kwargs={},
                dynamic_shapes=self._spec(),
                preserve_module_call_signature=(),
                orig_in_spec=in_spec,
                prefer_deferred_runtime_asserts_over_guards=False,
                _to_aten_func=_export_to_aten_ir,
            )


class TestContainerSpec(TestCase):
    """Container-spec walker (`_walk_spec` in
    `torch/_dynamo/functional_export.py`): SeqSpec / DictSpec /
    ObjectSpec on per-arg slots, partial specs, structural mismatches,
    and pytree-walk-order alignment."""

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
            dynamic_shapes=ShapesSpec(
                params=ParamsSpec(
                    {
                        "xs": SeqSpec(
                            [
                                TensorSpec([ShapeVar("A"), STATIC]),
                                TensorSpec([ShapeVar("B"), STATIC]),
                            ]
                        )
                    }
                )
            ),
            strict=True,
        )
        ep_str = str(ep)
        # Both list elements have unbacked dim 0.
        self.assertRegex(ep_str, r'xs_0: "f32\[u\d+, 3\]"')
        self.assertRegex(ep_str, r'xs_1: "f32\[u\d+, 3\]"')
        ep.module()([torch.randn(20, 3), torch.randn(99, 3)])

    def test_dict_spec_on_dict_of_tensors_partial(self):
        """DictSpec on dict: only "b" is dynamic; missing "a" key means
        static. "a" comes first in iteration order, so the walker must
        skip past its leaves before reaching the dynamic "b"."""

        class M(torch.nn.Module):
            def forward(self, d):
                return d["a"].sum(0) + d["b"].sum(0)

        ep = export(
            M(),
            args=({"a": torch.randn(7, 3), "b": torch.randn(8, 3)},),
            dynamic_shapes=ShapesSpec(
                params=ParamsSpec(
                    {"d": DictSpec({"b": TensorSpec([ShapeVar("B"), STATIC])})}
                )
            ),
            strict=True,
        )
        ep_str = str(ep)
        # a is omitted from the spec → static; b is dynamic.
        self.assertIn('d_a: "f32[7, 3]"', ep_str)
        self.assertRegex(ep_str, r'd_b: "f32\[u\d+, 3\]"')

    def test_object_spec_on_namedtuple(self):
        """ObjectSpec on a namedtuple subclass. Namedtuples are
        auto-registered by pytree with ``GetAttrKey(field_name)``
        entries (via ``_namedtuple_flatten_with_keys``), so ObjectSpec
        works on them without any extra registration. Mark `second`
        dynamic (not `first`) so the walker has to advance past a
        static leaf before hitting the dynamic one."""
        from collections import namedtuple

        Pair = namedtuple("Pair", ["first", "second"])

        class M(torch.nn.Module):
            def forward(self, p):
                return p.first.sum(0) + p.second.sum(0)

        ep = export(
            M(),
            args=(Pair(torch.randn(5, 3), torch.randn(8, 3)),),
            dynamic_shapes=ShapesSpec(
                params=ParamsSpec(
                    {"p": ObjectSpec({"second": TensorSpec([ShapeVar("S"), STATIC])})}
                )
            ),
            strict=True,
        )
        ep_str = str(ep)
        # first is omitted from the spec → static; second is dynamic.
        self.assertIn('p_first: "f32[5, 3]"', ep_str)
        self.assertRegex(ep_str, r'p_second: "f32\[u\d+, 3\]"')

    def test_seq_spec_on_namedtuple(self):
        """SeqSpec on a namedtuple subclass. Namedtuple is a tuple
        subclass, so the walker's ``isinstance(arg_value, (list, tuple))``
        check passes and treats it positionally — addressable by index
        instead of by field name. Position 0 is None (static), position
        1 is dynamic, so the walker has to skip past a static leaf
        before hitting the dynamic entry."""
        from collections import namedtuple

        Pair = namedtuple("Pair", ["first", "second"])

        class M(torch.nn.Module):
            def forward(self, p):
                return p.first.sum(0) + p.second.sum(0)

        ep = export(
            M(),
            args=(Pair(torch.randn(5, 3), torch.randn(8, 3)),),
            dynamic_shapes=ShapesSpec(
                params=ParamsSpec(
                    {"p": SeqSpec([None, TensorSpec([ShapeVar("B"), STATIC])])}
                )
            ),
            strict=True,
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
            dynamic_shapes=ShapesSpec(
                params=ParamsSpec(
                    {"c": ObjectSpec({"b": TensorSpec([ShapeVar("B"), STATIC])})}
                )
            ),
            strict=True,
        )
        ep_str = str(ep)
        # a is omitted from the spec → static; b is dynamic.
        self.assertIn('c_a: "f32[5, 3]"', ep_str)
        self.assertRegex(ep_str, r'c_b: "f32\[u\d+, 3\]"')

    def test_object_spec_via_torch_export_register_dataclass(self):
        """ObjectSpec works when the dataclass is registered through the
        ``torch.export.register_dataclass`` public API (thin wrapper
        around ``pytree.register_dataclass``)."""
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
            dynamic_shapes=ShapesSpec(
                params=ParamsSpec(
                    {"box": ObjectSpec({"y": TensorSpec([ShapeVar("Y"), STATIC])})}
                )
            ),
            strict=True,
        )
        ep_str = str(ep)
        # x is omitted from the spec → static; y is dynamic.
        self.assertIn('box_x: "f32[5, 3]"', ep_str)
        self.assertRegex(ep_str, r'box_y: "f32\[u\d+, 3\]"')

    def test_object_spec_on_dataclass(self):
        """ObjectSpec on a pytree-registered dataclass marks one field
        dynamic; the other field stays static (missing-attr semantics).
        Mark `y` (not `x`) so the walker has to skip past `x`'s static
        leaves first."""
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
            dynamic_shapes=ShapesSpec(
                params=ParamsSpec(
                    {"box": ObjectSpec({"y": TensorSpec([ShapeVar("Y"), STATIC])})}
                )
            ),
            strict=True,
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
            dynamic_shapes=ShapesSpec(
                params=ParamsSpec(
                    {
                        "d": DictSpec(
                            {
                                "foo": SeqSpec(
                                    [TensorSpec([ShapeVar("A"), STATIC]), None]
                                )
                            }
                        )
                    }
                )
            ),
            strict=True,
        )
        ep_str = str(ep)
        self.assertRegex(ep_str, r'd_foo_0: "f32\[u\d+, 3\]"')
        self.assertIn('d_foo_1: "f32[5, 3]"', ep_str)

    def test_mixed_leaf_and_container_under_params_spec(self):
        """ParamsSpec({"a": TensorSpec(...), "b": SeqSpec(...)}) — first
        arg is leaf-spec'd, second is container-spec'd."""

        class M(torch.nn.Module):
            def forward(self, a, b):
                return a.sum(0) + b[0].sum(0) + b[1].sum(0)

        ep = export(
            M(),
            args=(
                torch.randn(8, 3),
                [torch.randn(5, 3), torch.randn(6, 3)],
            ),
            dynamic_shapes=ShapesSpec(
                params=ParamsSpec(
                    {
                        "a": TensorSpec([ShapeVar("A"), STATIC]),
                        "b": SeqSpec(
                            [
                                TensorSpec([ShapeVar("B0"), STATIC]),
                                TensorSpec([ShapeVar("B1"), STATIC]),
                            ]
                        ),
                    }
                )
            ),
            strict=True,
        )
        ep_str = str(ep)
        self.assertRegex(ep_str, r'a: "f32\[u\d+, 3\]"')
        self.assertRegex(ep_str, r'b_0: "f32\[u\d+, 3\]"')
        self.assertRegex(ep_str, r'b_1: "f32\[u\d+, 3\]"')

    # ---- Partial-spec cases ----

    def test_seq_spec_shorter_than_runtime_list_tail_static(self):
        """SeqSpec with fewer entries than the runtime list: tail
        positions are static."""

        class M(torch.nn.Module):
            def forward(self, xs):
                return xs[0].sum() + xs[1].sum() + xs[2].sum()

        ep = export(
            M(),
            args=([torch.randn(8, 3), torch.randn(5, 3), torch.randn(6, 3)],),
            dynamic_shapes=ShapesSpec(
                params=ParamsSpec(
                    {"xs": SeqSpec([TensorSpec([ShapeVar("A"), STATIC])])}
                )
            ),
            strict=True,
        )
        ep_str = str(ep)
        self.assertRegex(ep_str, r'xs_0: "f32\[u\d+, 3\]"')
        self.assertIn('xs_1: "f32[5, 3]"', ep_str)
        self.assertIn('xs_2: "f32[6, 3]"', ep_str)

    # ---- Negative / structural cases ----

    def test_seq_spec_on_dict_arg_raises(self):
        """SeqSpec wrapped around a dict-typed runtime arg is a structural
        mismatch."""

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
                dynamic_shapes=ShapesSpec(
                    params=ParamsSpec({"d": SeqSpec([TensorSpec([ShapeVar("A")])])})
                ),
                strict=True,
            )

    def test_seq_spec_longer_than_runtime_list_raises(self):
        """SeqSpec with more entries than the runtime list (the spec
        references positions the runtime value doesn't have)."""

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
                dynamic_shapes=ShapesSpec(
                    params=ParamsSpec(
                        {
                            "xs": SeqSpec(
                                [
                                    TensorSpec([ShapeVar("A")]),
                                    TensorSpec([ShapeVar("B")]),
                                ]
                            )
                        }
                    )
                ),
                strict=True,
            )

    def test_dict_spec_key_not_in_runtime_dict_raises(self):
        """DictSpec referencing a key that's not present in the runtime
        dict is a user mistake."""

        class M(torch.nn.Module):
            def forward(self, d):
                return d["a"]

        with self.assertRaisesRegex(
            ValueError,
            r"DictSpec has entries .*'missing'.* that do not match any key",
        ):
            export(
                M(),
                args=({"a": torch.randn(4)},),
                dynamic_shapes=ShapesSpec(
                    params=ParamsSpec(
                        {"d": DictSpec({"missing": TensorSpec([ShapeVar("A")])})}
                    )
                ),
                strict=True,
            )

    def test_object_spec_attr_not_on_runtime_object_raises(self):
        """ObjectSpec referencing an attribute the runtime dataclass
        doesn't have raises."""
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
                dynamic_shapes=ShapesSpec(
                    params=ParamsSpec(
                        {"box": ObjectSpec({"nope": TensorSpec([ShapeVar("A")])})}
                    )
                ),
                strict=True,
            )

    def test_where_path_accumulates_in_nested_error(self):
        """The ``where`` path string passed through ``_walk_spec`` must
        accumulate as the walker descends, so errors point to the exact
        spot in the structure. Provoked here via a nested
        ``DictSpec({"inputs": SeqSpec([..., DictSpec({"missing": ...})])})``
        where the innermost DictSpec has a key not in the runtime dict.
        The full path should appear in the error message."""

        class M(torch.nn.Module):
            def forward(self, batch):
                return batch["inputs"][1]["data"]

        with self.assertRaisesRegex(
            ValueError,
            # Full exact-match (re.escape'd): verifies the path
            # accumulates as: top-level param 'batch' → DictSpec key
            # 'inputs' → SeqSpec index [1] → nested DictSpec
            # unmatched-key error with full path prefix.
            re.escape(
                "shapes_spec['batch']['inputs'][1]: "
                "DictSpec has entries ['missing'] that do not match "
                "any key in the runtime dict. Runtime keys: ['data']"
            ),
        ):
            export(
                M(),
                args=(
                    {
                        "inputs": [
                            torch.randn(3),
                            {"data": torch.randn(3)},
                        ]
                    },
                ),
                dynamic_shapes=ShapesSpec(
                    params=ParamsSpec(
                        {
                            "batch": DictSpec(
                                {
                                    "inputs": SeqSpec(
                                        [
                                            None,
                                            DictSpec(
                                                {
                                                    "missing": TensorSpec(
                                                        [ShapeVar("A")]
                                                    )
                                                }
                                            ),
                                        ]
                                    )
                                }
                            )
                        }
                    )
                ),
                strict=True,
            )

    def test_where_path_accumulates_through_object_spec(self):
        """Like ``test_where_path_accumulates_in_nested_error`` but
        exercises ``.attr`` descent through an ``ObjectSpec`` (dataclass
        traversal) in the middle of the path."""
        import dataclasses

        @dataclasses.dataclass
        class Box:
            items: dict[str, torch.Tensor]

        pytree.register_dataclass(Box)

        class M(torch.nn.Module):
            def forward(self, batch):
                return batch.items["foo"]

        with self.assertRaisesRegex(
            ValueError,
            # Path: shapes_spec['batch'] (named-positional) → .items
            # (ObjectSpec attribute) → nested DictSpec unmatched-key
            # error.
            re.escape(
                "shapes_spec['batch'].items: DictSpec has entries "
                "['missing'] that do not match any key in the runtime "
                "dict. Runtime keys: ['foo']"
            ),
        ):
            export(
                M(),
                args=(Box(items={"foo": torch.randn(3)}),),
                dynamic_shapes=ShapesSpec(
                    params=ParamsSpec(
                        {
                            "batch": ObjectSpec(
                                {
                                    "items": DictSpec(
                                        {
                                            "missing": TensorSpec(
                                                [ShapeVar("A")]
                                            )
                                        }
                                    )
                                }
                            )
                        }
                    )
                ),
                strict=True,
            )

    def test_object_spec_on_pytree_node_without_keys_fn_raises(self):
        """ObjectSpec on a type registered via
        ``pytree.register_pytree_node`` *without* a ``flatten_with_keys_fn``.
        Export accepts the value (its input check uses plain
        ``tree_flatten``, which doesn't need keys), but the walker
        cannot address children by attribute name, so it raises with a
        message guiding the user to provide a key fn."""

        class KeyslessContainer:
            def __init__(self, x):
                self.x = x

        # NOTE: no flatten_with_keys_fn passed — that's the whole point.
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
                dynamic_shapes=ShapesSpec(
                    params=ParamsSpec(
                        {"c": ObjectSpec({"x": TensorSpec([ShapeVar("A")])})}
                    )
                ),
                strict=True,
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

        from torch._dynamo.functional_export import _walk_spec

        @dataclasses.dataclass
        class _Box:
            u: torch.Tensor
            v: torch.Tensor

        pytree.register_dataclass(_Box)

        def _build_complete_spec(v):
            """Build a container spec that targets every tensor in v
            with a uniquely-named ShapeVar."""
            if isinstance(v, torch.Tensor):
                return TensorSpec([ShapeVar(f"t{id(v)}")])
            if isinstance(v, (list, tuple)):
                return SeqSpec([_build_complete_spec(e) for e in v])
            if isinstance(v, dict):
                return DictSpec({k: _build_complete_spec(val) for k, val in v.items()})
            if dataclasses.is_dataclass(v) and not isinstance(v, type):
                return ObjectSpec(
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
            out: list = [None] * len(leaves)
            consumed = _walk_spec(spec, arg_value, out, 0, where="<root>")
            self.assertEqual(
                consumed, len(leaves), f"leaf-count drift for case {arg_value!r}"
            )
            # Per-slot check: each slot's spec name must match the
            # tensor at that flat position from pytree.tree_flatten.
            for i, (slot_spec, leaf) in enumerate(zip(out, leaves, strict=True)):
                self.assertIsInstance(
                    slot_spec,
                    TensorSpec,
                    msg=f"slot {i} is {slot_spec!r} (expected TensorSpec) "
                    f"for case {arg_value!r}",
                )
                expected = f"t{id(leaf)}"
                actual = slot_spec._specs[0].name
                self.assertEqual(
                    actual,
                    expected,
                    msg=(
                        f"alignment drift at slot {i}: spec name "
                        f"{actual!r} does not match pytree-flatten leaf "
                        f"name {expected!r} for case {arg_value!r}"
                    ),
                )

    def test_walker_alignment_no_spec_advances_correctly(self):
        """When user_spec=None, the walker must return exactly
        len(pytree.tree_leaves(value)) — i.e. no-spec subtrees stay in
        lockstep with pytree's flatten."""
        from torch._dynamo.functional_export import _walk_spec

        cases = [
            torch.randn(3),
            [torch.randn(3), torch.randn(4), torch.randn(5)],
            {"a": [torch.randn(3), torch.randn(4)], "b": torch.randn(5)},
            (torch.randn(3), {"k": torch.randn(4)}, [torch.randn(5)]),
        ]
        for arg_value in cases:
            expected = len(pytree.tree_leaves(arg_value))
            consumed = _walk_spec(
                None, arg_value, [None] * expected, 0, where="<root>"
            )
            self.assertEqual(
                consumed, expected, f"no-spec leaf-count drift for {arg_value!r}"
            )


if __name__ == "__main__":
    run_tests()
