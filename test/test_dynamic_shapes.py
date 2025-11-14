# Owner(s): ["oncall: jit"]
# ruff: noqa: F841
import contextlib
import copy
import itertools
import math
import operator
import unittest

import numpy as np
import pytest
import sympy

import torch
import torch.fx
import torch.nn.functional as F
from torch import sym_int, SymBool, SymFloat, SymInt
from torch._C import _disabled_torch_function_impl
from torch._dynamo.testing import CompileCounter, CompileCounterWithBackend
from torch._inductor.utils import fresh_cache
from torch.fx.experimental import sym_node
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.sym_node import method_to_operator, SymNode, to_node
from torch.fx.experimental.symbolic_shapes import (
    _constrain_range_for_size,
    DimConstraints,
    DimDynamic,
    expect_true,
    guard_bool,
    guard_float,
    guard_int,
    GuardOnDataDependentSymNode,
    has_free_symbols,
    hint_int,
    is_symbolic,
    ShapeEnv,
    StatelessSymbolicContext,
    statically_known_false,
    statically_known_true,
)
from torch.testing._internal.common_dtype import all_types_and
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    TEST_XPU,
    TestCase,
)
from torch.testing._internal.logging_utils import logs_to_string
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._sympy.functions import (
    CleanDiv,
    FloorDiv,
    IsNonOverlappingAndDenseIndicator,
    Mod,
)


aten = torch.ops.aten

meta_funcs = {}


def register_meta(op):
    def decorator(f):
        def add_func(op):
            meta_funcs[op] = f

        pytree.tree_map_(add_func, op)
        return f

    return decorator


@register_meta([aten.add.Tensor, aten.sub.Tensor])
def binary_meta(a, b):
    return a.new_empty(a.shape)


@register_meta(aten.cat.default)
def cat_meta(tensors, dim=0):
    concat_length = 0
    shape = tensors[0].shape
    for tensor in tensors:
        for idx, (common_length, length) in enumerate(zip(shape, tensor.shape)):
            if idx == dim:
                concat_length = concat_length + length
            else:
                assert length == common_length
    new_shape = list(shape)
    new_shape[dim] = concat_length
    return tensors[0].new_empty(new_shape)


@register_meta([aten.narrow_copy.default])
def narrow_copy_symint_meta(a, dim, start, length, **kwargs):
    shape = []
    for i, x in enumerate(a.shape):
        if i == dim:
            shape.append(length)
        else:
            shape.append(x)
    return a.new_empty(tuple(shape))


@register_meta([aten.expand.default])
def expand_symint_meta(a, size, implicit=False):
    return a.new_empty(size)


def create_contiguous(shape):
    strides = [1]
    for dim in reversed(shape[:-1]):
        strides.append(dim * strides[-1])
    return list(reversed(strides))


class FakeSymbolicTensor(torch.Tensor):
    @staticmethod
    def __new__(
        cls,
        sym_shape,
        sym_strides,
        dtype,
        layout,
        requires_grad,
        device,
        storage_offset=0,
    ):
        # TODO: this is wrong in general
        sym_stride = create_contiguous(sym_shape)
        r = torch.Tensor._make_wrapper_subclass(
            cls,
            sym_shape,
            sym_stride,
            storage_offset,
            dtype=dtype,
            layout=layout,
            requires_grad=requires_grad,
            device=device,
        )
        return r

    __torch_function__ = _disabled_torch_function_impl

    def new_empty(self, shape):
        return FakeSymbolicTensor(
            shape, None, self.dtype, self.layout, self.requires_grad, self.device
        )

    @classmethod
    def __torch_dispatch__(cls, func_overload, types, args=(), kwargs=None):
        if func_overload in meta_funcs:
            return meta_funcs[func_overload](*args, **kwargs)

        if func_overload == torch.ops.aten.new_empty.default:
            self = args[0]
            shape = args[1]
            return FakeSymbolicTensor(
                shape,
                self.stride(),
                self.dtype,
                self.layout,
                self.requires_grad,
                self.device,
            )

        raise RuntimeError(f"operator {func_overload} not supported")


def create_symbolic_tensor(name, arg, shape_env, source=None, dynamic_dims=None):
    from torch._dynamo.source import ConstantSource

    if source is None:
        source = ConstantSource(name)
    constraint_dims = [None] * arg.dim()
    if dynamic_dims is None:
        dynamic_dims = [DimDynamic.DUCK] * arg.dim()
    (
        sym_shapes,
        sym_strides,
        sym_storage_offset,
    ) = shape_env.create_symbolic_sizes_strides_storage_offset(
        arg,
        source=source,
        symbolic_context=StatelessSymbolicContext(
            dynamic_sizes=dynamic_dims, constraint_sizes=constraint_dims
        ),
    )
    return FakeSymbolicTensor(
        sym_shapes,
        sym_strides,
        arg.dtype,
        arg.layout,
        arg.requires_grad,
        arg.device,
        sym_storage_offset,
    )


def create_fake_tensor_with_dynamic_size(x, shape_env, dynamic_sizes, dynamic_strides):
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode(shape_env=shape_env) as fake_mode:
        return fake_mode.from_tensor(
            x,
            symbolic_context=StatelessSymbolicContext(
                dynamic_sizes=dynamic_sizes,
                dynamic_strides=dynamic_strides,
            ),
        )


def create_symtype(cls, pytype, shape_env, val, duck=True, **kwargs):
    from torch._dynamo.source import ConstantSource

    symbol = shape_env.create_symbol(
        val,
        source=ConstantSource(f"__testing_only{len(shape_env.var_to_val)}"),
        dynamic_dim=DimDynamic.DUCK if duck else DimDynamic.DYNAMIC,
        constraint_dim=None,
        **kwargs,
    )
    return cls(SymNode(symbol, shape_env, pytype, hint=val))


# TODO: default duck to False
def create_symint(shape_env, i: int, duck=True, **kwargs) -> SymInt:
    return create_symtype(SymInt, int, shape_env, i, duck=duck, **kwargs)


def create_symbool(shape_env, b: bool) -> SymBool:
    return create_symtype(SymBool, bool, shape_env, b)


def create_symfloat(shape_env, f: float) -> SymFloat:
    return create_symtype(SymFloat, float, shape_env, f)


@skipIfTorchDynamo(
    "Creating ShapeEnv fails for confusing reasons (also we never expect dynamo to see code like this)"
)
class TestPySymInt(TestCase):
    def test_arith_ops(self):
        shape_env = ShapeEnv()
        symints = []
        for i in range(2, 5):
            symints.append((i, create_symint(shape_env, i)))

        ops = [
            operator.add,
            operator.sub,
            operator.floordiv,
            operator.mul,
            operator.mod,
        ]

        for op in ops:
            for args in itertools.permutations(symints, 2):
                if not isinstance(args[0][1], int) and (
                    (op != operator.mod or op != operator.floordiv) and args[1][0] != 0
                ):
                    self.assertTrue(
                        op(args[0][1], args[1][1]) == op(args[0][0], args[1][0])
                    )

    def test_reverse_arith_ops(self):
        shape_env = ShapeEnv()

        a = create_symint(shape_env, 2)
        self.assertTrue(5 // a == 5 // 2)

        a = create_symint(shape_env, 2)
        self.assertTrue(5 * a == 5 * 2)

    def test_sympify_symint(self):
        shape_env = ShapeEnv()
        a = create_symint(shape_env, 2)
        self.assertIs(sympy.sympify(a), a.node.expr)
        b = create_symfloat(shape_env, 3.0)
        self.assertIs(sympy.sympify(b), b.node.expr)
        c = create_symbool(shape_env, True)
        self.assertIs(sympy.sympify(c), c.node.expr)

    def test_roundtrip(self):
        shape_env = ShapeEnv()
        x = create_symbolic_tensor("x", torch.randn(5, 4, 3), shape_env)

        self.assertTrue(not isinstance(x.shape[0], SymNode))
        self.assertTrue(isinstance(x.shape[0], SymInt))

        self.assertTrue(x.shape[0] == 5)
        self.assertTrue(x.shape[1] == 4)
        self.assertTrue(x.shape[2], 3)

        self.assertTrue(x.size()[0], 5)
        self.assertTrue(x.size()[1], 4)
        # Should be simplifiable to an integer.
        # Ref: https://github.com/pytorch/pytorch/pull/107492
        self.assertTrue(isinstance(x.size()[1], SymInt))
        self.assertTrue(
            isinstance(x.size()[1].node.maybe_as_int(), int)
        )  # due to guard above
        self.assertTrue(x.size()[2] == 3)

        self.assertTrue(x.size(0) == 5)
        self.assertTrue(x.size(1) == 4)
        self.assertTrue(x.size(2) == 3)
        self.assertTrue(isinstance(x.size(2), SymInt))
        self.assertTrue(isinstance(x.size(2).node.maybe_as_int(), int))

        y = create_symbolic_tensor("y", torch.randn(5, 4, 3)[1:], shape_env)
        self.assertTrue(isinstance(y.storage_offset(), SymInt))
        self.assertTrue(y.storage_offset() == 12)

    def test_binary(self):
        shape_env = ShapeEnv()
        x = create_symbolic_tensor("x", torch.randn(5, 4, 3), shape_env)
        y = create_symbolic_tensor("y", torch.randn(5, 4, 3), shape_env)

        z = x + y
        self.assertTrue(z.shape[0] == 5)
        self.assertTrue(z.shape[1] == 4)
        self.assertTrue(z.shape[2] == 3)

        # broadcasting
        y = create_symbolic_tensor("y2", torch.randn(1, 4, 1), shape_env)
        z = x + y
        self.assertTrue(z.shape[0] == 5)
        self.assertTrue(z.shape[1] == 4)
        self.assertTrue(z.shape[2] == 3)

    def test_symint_args(self):
        shape_env = ShapeEnv()
        x = create_symbolic_tensor("x", torch.randn(5, 4, 3), shape_env)
        y = create_symbolic_tensor("y", torch.randn(5, 4, 1), shape_env)
        LAST_DIM = 2
        z = x.narrow_copy(LAST_DIM, 0, y.shape[LAST_DIM])
        self.assertTrue(z.shape[2] == y.shape[2])

        # arithmetic expr with two symints
        z = x.narrow_copy(LAST_DIM, 0, x.shape[LAST_DIM] - y.shape[LAST_DIM])
        self.assertTrue(z.shape[2] == 2)

        # arithmetic expr with a symint and python int
        z = x.narrow_copy(LAST_DIM, 0, x.shape[LAST_DIM] - 1)
        self.assertTrue(z.shape[2] == 2)

    def test_symint_vargs(self):
        shape_env = ShapeEnv()
        x = create_symbolic_tensor("x", torch.randn(5, 4, 3), shape_env)
        y = create_symbolic_tensor("y", torch.randn(1, 4, 1), shape_env)

        # varargs
        z = y.expand(x.shape[0], y.shape[1], x.shape[2])
        self.assertTrue(z.shape[0] == 5)
        self.assertTrue(z.shape[1] == 4)
        self.assertTrue(z.shape[2] == 3)

        # shape list
        z = y.expand((x.shape[0], y.shape[1], x.shape[2]))
        self.assertTrue(z.shape[0] == 5)
        self.assertTrue(z.shape[1] == 4)
        self.assertTrue(z.shape[2] == 3)

        # mixed python symints and ints
        z = y.expand(x.shape[0], y.shape[1], 3)
        self.assertTrue(z.shape[0] == 5)
        self.assertTrue(z.shape[1] == 4)
        self.assertTrue(z.shape[2] == 3)

        # mixed python symints and ints in a list
        z = y.expand((x.shape[0], y.shape[1], 3))
        self.assertTrue(z.shape[0] == 5)
        self.assertTrue(z.shape[1] == 4)
        self.assertTrue(z.shape[2] == 3)

        # mixed python symints and ints
        z = y.expand(5, y.shape[1], x.shape[2])
        self.assertTrue(z.shape[0] == 5)
        self.assertTrue(z.shape[1] == 4)
        self.assertTrue(z.shape[2] == 3)

        # mixed python ints and symints in a list
        z = y.expand((5, y.shape[1], x.shape[2]))
        self.assertTrue(z.shape[0] == 5)
        self.assertTrue(z.shape[1] == 4)
        self.assertTrue(z.shape[2] == 3)

        z = y.expand((y.shape[1],))
        z = y.expand(y.shape[1])

    def test_symint_bitwise_and(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 0b1100)
        b0 = create_symint(shape_env, 0b1010)
        res_and = a0 & b0
        self.assertEqual(res_and, 0b1000)
        self.assertIsInstance(res_and, torch.SymInt, msg=type(res_and))
        self.assertExpectedInline(
            str(shape_env.guards[0][0]), """Eq(BitwiseFn_bitwise_and(s97, s26), 8)"""
        )

        a1 = create_symint(shape_env, 3)
        b1 = create_symbool(shape_env, True)
        self.assertEqual(a1 & b1, 1)

        a2 = create_symint(shape_env, 0b1100)
        self.assertEqual(a2 & 0b1010, 0b1000)

        a3 = create_symbool(shape_env, True)
        b3 = create_symbool(shape_env, True)
        self.assertEqual(a3 & b3, True)

    def test_symint_bitwise_or(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 0b1100)
        b0 = create_symint(shape_env, 0b1010)
        res_or = a0 | b0
        self.assertEqual(res_or, 0b1110)
        self.assertIsInstance(res_or, torch.SymInt, msg=type(res_or))
        self.assertExpectedInline(
            str(shape_env.guards[0][0]), """Eq(BitwiseFn_bitwise_or(s97, s26), 14)"""
        )

    def test_symint_bitwise_xor(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 0b1100)
        b0 = create_symint(shape_env, 0b1010)
        res_xor = a0 ^ b0
        self.assertEqual(res_xor, 0b0110)
        self.assertIsInstance(res_xor, torch.SymInt, msg=type(res_xor))
        self.assertExpectedInline(
            str(shape_env.guards[0][0]), """Eq(BitwiseFn_bitwise_xor(s97, s26), 6)"""
        )

    def test_stride(self):
        shape_env = ShapeEnv()
        x = create_symbolic_tensor("x", torch.randn(5, 5), shape_env)
        self.assertIsInstance(x.stride()[0], SymInt)

    def test_size_expressions(self):
        shape_env = ShapeEnv()
        x = create_symbolic_tensor("x", torch.randn(5), shape_env)
        expand_x = x.expand(x.shape[0], x.shape[0])
        if expand_x.shape[0] > 3:
            result = expand_x + expand_x
        else:
            result = expand_x + expand_x

        gt_op, _bt, is_size_obv = shape_env.guards[-1]
        self.assertTrue(isinstance(gt_op, sympy.core.relational.StrictGreaterThan))
        self.assertTrue(str(x.shape[0]), str(gt_op.args[0]))
        self.assertTrue(str(expand_x.shape[1]), str(x.shape[0]))
        self.assertTrue(str(expand_x.shape[1]), str(result.shape[0]))
        self.assertFalse(is_size_obv)

    def test_floordiv_static(self):
        shape_env = ShapeEnv()
        s0 = create_symint(shape_env, 8)
        # This was extracted from
        # python test/inductor/test_cuda_cpp_wrapper.py -k
        # DynamicShapesCudaWrapperCudaTests.test_insignificant_strides_cuda_dynamic_shapes_cuda_wrapper
        bool(s0 % 2 == 0)
        bool(s0 % (s0 // 2) == 0)
        bool(2 * (s0 // 2) == s0)
        self.assertTrue(statically_known_true(s0 // (s0 // 2) == 2))

    def test_numel(self):
        shape_env = ShapeEnv()
        x = create_symbolic_tensor("x", torch.randn(5), shape_env)
        self.assertIsInstance(x.numel(), torch.SymInt)
        self.assertIsInstance(torch.numel(x), torch.SymInt)

        x = torch.rand(3, 3)
        self.assertIsInstance(x.numel(), int)
        self.assertIsInstance(torch.numel(x), int)

    def test_int_to_float(self):
        shape_env = ShapeEnv()
        x = create_symbolic_tensor("x", torch.randn(5), shape_env)
        r = torch.sym_float(x.shape[0])
        self.assertIsInstance(r, torch.SymFloat, msg=type(r))

    def test_aten_ops(self):
        shape_env = ShapeEnv()
        x = create_symbolic_tensor("x", torch.randn(5), shape_env)
        torch.ops.aten.narrow_copy.default(x, 0, 0, x.shape[0])

        shape_env = ShapeEnv()
        x = create_symbolic_tensor("x2", torch.randn(5, 4, 3), shape_env)
        torch.ops.aten.expand.default(x, [x.shape[0], x.shape[1], x.shape[2]])

    def test_fx_trace_intlist(self):
        class CustomModule(torch.nn.Module):
            def forward(self, x):
                bs, c, h, w = x.shape
                return F.pad(x, (0, w % 2, 0, h % 2, 0, 0))

        m = CustomModule()
        x = torch.rand(1, 3, 4, 4)
        # should not TypeError: pad(): argument 'pad' (position 2) must be
        # tuple of ints, not tuple
        torch.fx.symbolic_trace(m)

    def test_meta_symint(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 2)
        r = torch.empty(a0, device="meta")
        self.assertIsInstance(r.shape[0], SymInt)

    def test_guard_int(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 2)
        self.assertEqual(guard_int(a0), 2)
        self.assertExpectedInline(str(shape_env.guards[0][0]), """Eq(s97, 2)""")

    def test_sym_sum(self):
        shape_env = ShapeEnv()
        s0 = create_symint(shape_env, 2)
        s1 = create_symint(shape_env, 3)
        s2 = create_symint(shape_env, 4)
        self.assertEqual(
            (s0 + s1 + s2).node.expr, torch.sym_sum([s0, s1, s2]).node.expr
        )

    def test_prefer_deferred_runtime_assertions_over_guards(self):
        shape_env = ShapeEnv(prefer_deferred_runtime_asserts_over_guards=True)
        s0 = create_symint(shape_env, 2)
        self.assertEqual(guard_int(s0), 2)
        self.assertExpectedInline(str(shape_env.guards[0][0]), """Eq(s97, 2)""")

        shape_env = ShapeEnv(prefer_deferred_runtime_asserts_over_guards=True)
        s0 = create_symint(shape_env, 2)
        self.assertTrue(expect_true(s0 == 2))
        self.assertEqual(len(shape_env.guards), 0)
        self.assertExpectedInline(
            str([ra.expr for ra in shape_env.deferred_runtime_asserts[None]]),
            """[Eq(s97, 2)]""",
        )

    def test_sym_int(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 5)
        r = sym_int(a0)
        self.assertEqual(r, 5)
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        self.assertExpectedInline(str(shape_env.guards[0][0]), """Eq(s97, 5)""")

        a1 = create_symint(shape_env, 7)
        r = sym_int(a1 / 2)
        self.assertEqual(guard_int(r), 3)
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        self.assertExpectedInline(
            str(shape_env.guards[1][0]), """Eq(TruncToInt(IntTrueDiv(s26, 2)), 3)"""
        )

        a3 = create_symint(shape_env, 3)
        r = sym_int(2.0 * torch.sym_float(a3))
        self.assertEqual(guard_int(r), 6)
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        self.assertExpectedInline(
            str(shape_env.guards[2][0]), """Eq(TruncToInt(2.0*ToFloat(s57)), 6)"""
        )

    def test_sym_log2(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 4)
        r = torch._sym_log2(a0)
        self.assertEqual(r, 2.0)
        self.assertIsInstance(r, torch.SymFloat, msg=type(r))
        self.assertExpectedInline(
            str(shape_env.guards[0][0]), """Eq(OpaqueUnaryFn_log2(ToFloat(s97)), 2.0)"""
        )

    def test_sym_sqrt(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 4)
        r = torch._sym_sqrt(a0)
        self.assertEqual(r, 2)
        self.assertIsInstance(r, torch.SymFloat, msg=type(r))
        self.assertExpectedInline(
            str(shape_env.guards[0][0]), """Eq(OpaqueUnaryFn_sqrt(ToFloat(s97)), 2.0)"""
        )

    def test_sym_floor(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 5)
        r = math.floor(a0 / 2)
        self.assertEqual(r, 2)
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        self.assertExpectedInline(
            str(shape_env.guards[0][0]),
            """Eq(FloorToInt(IntTrueDiv(s97, 2)), 2)""",
        )
        r = math.floor(3.0 * a0)
        self.assertEqual(r, 15)
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        self.assertExpectedInline(
            str(shape_env.guards[1][0]),
            """Eq(FloorToInt(3.0*ToFloat(s97)), 15)""",
        )

    def test_sym_trunc(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 5)
        r = math.trunc(a0 / 2)
        self.assertEqual(r, 2)
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        self.assertExpectedInline(
            str(shape_env.guards[0][0]), """Eq(TruncToInt(IntTrueDiv(s97, 2)), 2)"""
        )
        r = torch.sym_int(torch.sym_sqrt(a0))
        self.assertEqual(r, 2)
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        self.assertExpectedInline(
            str(shape_env.guards[1][0]),
            """Eq(TruncToInt(OpaqueUnaryFn_sqrt(ToFloat(s97))), 2)""",
        )

    def test_sym_ceil(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 5)
        r = math.ceil(a0 / 2)
        self.assertEqual(r, 3)
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        self.assertExpectedInline(
            str(shape_env.guards[0][0]),
            """Eq(CeilToInt(IntTrueDiv(s97, 2)), 3)""",
        )
        r1 = 3.0 * a0
        r = math.floor(r1)
        self.assertEqual(r, 15)
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        self.assertExpectedInline(
            str(shape_env.guards[1][0]),
            """Eq(FloorToInt(3.0*ToFloat(s97)), 15)""",
        )

    def test_sym_ite(self):
        shape_env = ShapeEnv()
        t = create_symint(shape_env, 5)
        f = create_symint(shape_env, 4)
        b1 = True
        r1 = torch.sym_ite(b1, t, f)
        self.assertTrue(r1 is t)
        b2 = False
        r2 = torch.sym_ite(b2, t, f)
        self.assertTrue(r2 is f)
        b3 = t == 5
        r3 = torch.sym_ite(b3, t, f)
        self.assertEqual(len(shape_env.guards), 0)
        self.assertEqual(r3, 5)
        self.assertEqual(type(t), type(r3))
        self.assertExpectedInline(
            str(shape_env.guards[0][0]),
            """Eq(Piecewise((s97, Eq(s97, 5)), (s26, True)), 5)""",
        )
        b4 = f == 5
        r4 = torch.sym_ite(b4, t, f)
        self.assertEqual(len(shape_env.guards), 1)
        self.assertEqual(r4, 4)
        self.assertEqual(type(f), type(r4))
        self.assertExpectedInline(
            str(shape_env.guards[1][0]),
            """Eq(Piecewise((s97, Eq(s26, 5)), (s26, True)), 4)""",
        )

    def test_tracing_sym_ite(self):
        def f(x):
            b = x.shape[0] == 5
            ret = torch.sym_ite(b, x.shape[0], x.shape[1])
            return ret

        gm = make_fx(f, tracing_mode="symbolic")(torch.ones(4, 5))
        self.assertEqual(len(gm.shape_env.guards), 0)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x_1):
    sym_size_int = torch.ops.aten.sym_size.int(x_1, 0)
    eq = sym_size_int == 5
    sym_size_int_1 = torch.ops.aten.sym_size.int(x_1, 1);  x_1 = None
    sym_ite = torch.sym_ite(eq, sym_size_int, sym_size_int_1);  eq = sym_size_int = sym_size_int_1 = None
    return sym_ite""",
        )
        r1 = gm(torch.ones(4, 5))
        self.assertIsInstance(r1, int)
        self.assertEqual(r1, 5)
        r2 = gm(torch.ones(5, 4))
        self.assertIsInstance(r2, int)
        self.assertEqual(r2, 5)

    def test_int_conversion(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 2)
        int(a0)
        self.assertExpectedInline(str(shape_env.guards[0][0]), """Eq(s97, 2)""")

    def test_data_dependent_guard(self):
        shape_env = ShapeEnv()
        s0 = shape_env.create_unbacked_symint()
        self.assertRaises(GuardOnDataDependentSymNode, lambda: bool(s0 == 0))

    def test_data_dependent_guard_propagate_real_tensors(self):
        shape_env = ShapeEnv()
        s0 = shape_env.create_unbacked_symint()
        shape_env.set_unbacked_var_to_val(s0.node.expr, 0)
        self.assertEqual(bool(s0 == 0), True)

    def test_expect_true_basic(self):
        shape_env = ShapeEnv()
        i0 = shape_env.create_unbacked_symint()
        i0_sym = i0.node.expr
        # This doesn't error
        self.assertTrue(expect_true(i0 == 0))
        # This generates a deferred runtime assert via replacement
        self.assertEqual(shape_env.replacements[i0_sym], 0)
        # After expecting true, guards now resolve given the runtime assert
        bool(i0 == 0)

    def test_expect_true_with_s0(self):
        shape_env = ShapeEnv()
        s0 = create_symint(shape_env, 5)
        i0 = shape_env.create_unbacked_symint()
        self.assertTrue(expect_true(i0 < s0))
        self.assertExpectedInline(
            str([ra.expr for ra in shape_env.deferred_runtime_asserts[i0.node.expr]]),
            """[u0 < s97]""",
        )
        self.assertTrue(i0 < s0)
        self.assertTrue(i0 != s0)
        self.assertFalse(i0 > s0)
        self.assertFalse(i0 >= s0)

    def test_expect_true_prefer_later(self):
        shape_env = ShapeEnv()
        i0 = shape_env.create_unbacked_symint()
        i1 = shape_env.create_unbacked_symint()
        i1_sym = i1.node.expr
        self.assertTrue(expect_true(i0 + i1 == 10))
        # Importantly, this is put in i1, not i0!
        self.assertExpectedInline(
            str([ra.expr for ra in shape_env.deferred_runtime_asserts[i1_sym]]),
            """[Eq(u0 + u1, 10)]""",
        )
        self.assertTrue(i0 + i1 == 10)
        # NB: We currently don't support deriving that we can substitute
        # i0 + i1 with 10; maybe we should, but this means our rewriting
        # system is no longer confluent (it's probably OK though, because
        # you're unlikely to get other equalities like this on the
        # unbacked SymInts.)

    def test_unbacked_substitution(self):
        shape_env = ShapeEnv()
        i0 = shape_env.create_unbacked_symint()
        i1 = shape_env.create_unbacked_symint()
        _constrain_range_for_size(i0)
        _constrain_range_for_size(i1)
        self.assertTrue(expect_true(i0 == i1 * 4))
        self.assertExpectedInline(str(i0), """u0""")

        i2 = shape_env.create_unbacked_symint()
        i3 = shape_env.create_unbacked_symint()
        _constrain_range_for_size(i2)
        _constrain_range_for_size(i3)
        self.assertTrue(expect_true(i2 * 4 == i3))
        self.assertExpectedInline(str(i3), """u3""")

    def test_avoid_unbacked_substitution(self):
        shape_env = ShapeEnv()
        i0 = shape_env.create_unbacked_symint()
        _constrain_range_for_size(i0)
        i1 = shape_env.create_unbacked_symint()
        _constrain_range_for_size(i1)
        self.assertTrue(expect_true(i0 == 10 - i1))
        self.assertExpectedInline(str(i0), """u0""")

    def test_expect_true_double_digits(self):
        shape_env = ShapeEnv()
        ia = [shape_env.create_unbacked_symint() for _ in range(11)]  # allocate 10
        self.assertEqual(str(ia[-1]), "u10")
        self.assertTrue(expect_true(sum(ia) == 20))
        self.assertEqual(len(shape_env.deferred_runtime_asserts[ia[-1].node.expr]), 1)

    def test_expect_true_refine_range(self):
        shape_env = ShapeEnv()
        for i, rel in enumerate(
            [lambda x: x > 4, lambda x: 4 < x, lambda x: x >= 5, lambda x: 5 <= x]
        ):
            with self.subTest(f"i = {i}"):
                i0 = shape_env.create_unbacked_symint()
                self.assertTrue(expect_true(rel(i0)))
                self.assertTrue(statically_known_true(i0 != 3))
                self.assertTrue(statically_known_true(i0 != 4))
                self.assertFalse(statically_known_true(i0 != 5))
                self.assertFalse(statically_known_true(i0 != 6))
                self.assertTrue(statically_known_true(i0 > 4))
                self.assertTrue(statically_known_true(i0 >= 5))

        for i, rel in enumerate(
            [lambda x: x < 4, lambda x: 4 > x, lambda x: x <= 3, lambda x: 3 >= x]
        ):
            with self.subTest(f"i = {i}"):
                i0 = shape_env.create_unbacked_symint()
                self.assertTrue(expect_true(rel(i0)))
                self.assertFalse(statically_known_true(i0 != 2))
                self.assertFalse(statically_known_true(i0 != 3))
                self.assertTrue(statically_known_true(i0 != 4))
                self.assertTrue(statically_known_true(i0 != 5))
                self.assertTrue(statically_known_true(i0 < 4))
                self.assertTrue(statically_known_true(i0 <= 5))

    def test_guard_refine_range(self):
        shape_env = ShapeEnv()
        for i, rel in enumerate(
            [lambda x: x > 4, lambda x: 4 < x, lambda x: x >= 5, lambda x: 5 <= x]
        ):
            with self.subTest(f"i = {i}"):
                i0 = create_symint(shape_env, 10, duck=False)
                self.assertTrue(bool(rel(i0)))
                self.assertTrue(statically_known_true(i0 != 3))
                self.assertTrue(statically_known_true(i0 != 4))
                self.assertFalse(statically_known_true(i0 != 5))
                self.assertFalse(statically_known_true(i0 != 6))
                self.assertTrue(statically_known_true(i0 > 4))
                self.assertTrue(statically_known_true(i0 >= 5))

        for i, rel in enumerate(
            [lambda x: x > 4, lambda x: 4 < x, lambda x: x >= 5, lambda x: 5 <= x]
        ):
            with self.subTest(f"i = {i}"):
                i0 = create_symint(shape_env, 2, duck=False)
                self.assertFalse(bool(rel(i0)))
                self.assertFalse(statically_known_true(i0 != 3))
                self.assertFalse(statically_known_true(i0 != 4))
                self.assertTrue(statically_known_true(i0 != 5))
                self.assertTrue(statically_known_true(i0 != 6))
                self.assertTrue(statically_known_true(i0 <= 4))
                self.assertTrue(statically_known_true(i0 < 5))

        for i, rel in enumerate(
            [lambda x: x < 4, lambda x: 4 > x, lambda x: x <= 3, lambda x: 3 >= x]
        ):
            with self.subTest(f"i = {i}"):
                i0 = create_symint(shape_env, 2, duck=False)
                self.assertTrue(bool(rel(i0)))
                self.assertFalse(statically_known_true(i0 != 2))
                self.assertFalse(statically_known_true(i0 != 3))
                self.assertTrue(statically_known_true(i0 != 4))
                self.assertTrue(statically_known_true(i0 != 5))
                self.assertTrue(statically_known_true(i0 < 4))
                self.assertTrue(statically_known_true(i0 <= 3))

        for i, rel in enumerate(
            [lambda x: x < 4, lambda x: 4 > x, lambda x: x <= 3, lambda x: 3 >= x]
        ):
            with self.subTest(f"i = {i}"):
                i0 = create_symint(shape_env, 10, duck=False)
                self.assertFalse(bool(rel(i0)))
                self.assertTrue(statically_known_true(i0 != 2))
                self.assertTrue(statically_known_true(i0 != 3))
                self.assertFalse(statically_known_true(i0 != 4))
                self.assertFalse(statically_known_true(i0 != 5))
                self.assertTrue(statically_known_true(i0 >= 4))
                self.assertTrue(statically_known_true(i0 > 3))

    def test_mul_int_oo_nan(self):
        shape_env = ShapeEnv()
        s0 = create_symint(shape_env, 5, duck=False)
        s1 = create_symint(shape_env, 6, duck=False)
        s2 = create_symint(shape_env, 5, duck=False)
        bool(s0 * (s1 // s0) == s2)

    def test_non_overlapping_and_dense_backed(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 5)
        r = torch.empty_strided((a0, 7), (1, a0), device="meta")
        self.assertTrue(torch.ops.aten.is_non_overlapping_and_dense.default(r))

    def test_non_overlapping_and_dense_unbacked(self):
        shape_env = ShapeEnv()
        u0 = shape_env.create_unbacked_symint()
        cf = torch.ops.aten.is_non_overlapping_and_dense.default

        self.assertEqual(IsNonOverlappingAndDenseIndicator(u0.node.expr, 2, 2, 1), 1)
        self.assertEqual(IsNonOverlappingAndDenseIndicator(2, u0.node.expr, 1, 2), 1)
        self.assertTrue(cf(torch.empty_strided((u0, 2), (2, 1), device="meta")))
        self.assertTrue(cf(torch.empty_strided((2, u0), (1, 2), device="meta")))

        self.assertEqual(IsNonOverlappingAndDenseIndicator(u0.node.expr, 1), 1)
        self.assertEqual(IsNonOverlappingAndDenseIndicator(1, u0.node.expr), 1)
        self.assertTrue(cf(torch.empty_strided((u0,), (1,), device="meta")))
        self.assertTrue(cf(torch.empty_strided((1,), (u0,), device="meta")))

        Max = torch.sym_max
        # NB: This only works because we're able to determine this tensor is
        # contiguous. transpose(0, 1) makes it stop working
        self.assertTrue(
            cf(
                torch.empty_strided(
                    (2, 3, 1, u0),
                    (3 * Max(1, u0), Max(1, u0), Max(1, u0), 1),
                    device="meta",
                )
            )
        )

    def test_prims_non_overlapping_and_dense(self):
        shape_env = ShapeEnv()
        cf = torch._prims_common.is_non_overlapping_and_dense

        # backed case
        a0 = create_symint(shape_env, 5)
        self.assertTrue(cf(torch.empty_strided((a0, 7), (1, a0), device="meta")))

        # unbacked
        u0 = shape_env.create_unbacked_symint()
        self.assertTrue(cf(torch.empty_strided((u0, 2), (2, 1), device="meta")))
        self.assertTrue(cf(torch.empty_strided((2, u0), (1, 2), device="meta")))
        self.assertTrue(cf(torch.empty_strided((u0,), (1,), device="meta")))
        self.assertTrue(cf(torch.empty_strided((1,), (u0,), device="meta")))

        Max = torch.sym_max
        self.assertTrue(
            cf(
                torch.empty_strided(
                    (2, 3, 1, u0),
                    (3 * Max(1, u0), Max(1, u0), Max(1, u0), 1),
                    device="meta",
                )
            )
        )
        self.assertFalse(
            cf(
                torch.empty_strided(
                    (2, 3, 1, u0),
                    (Max(1, u0), Max(1, u0), 1, 3 * Max(1, u0)),
                    device="meta",
                )
            )
        )

        # return False on arbitrary strides
        u1 = shape_env.create_unbacked_symint()
        self.assertFalse(
            cf(
                torch.empty_strided(
                    (2 * u0, u0, 1),
                    (u1, u0, u0 + u1),
                    device="meta",
                )
            )
        )
        self.assertFalse(
            cf(
                torch.empty_strided(
                    (2, 3, u0),
                    (u1, 3, 1),
                    device="meta",
                )
            )
        )

    def test_sympy_optimized_add_binary_search(self):
        import sympy

        from torch.fx.experimental.sym_node import _binary_search_insert_arg

        a = sympy.Symbol("a")
        b = sympy.Symbol("b")
        c = sympy.Symbol("c")

        args = []
        args = _binary_search_insert_arg([], b)
        self.assertEqual(args, [b])

        self.assertEqual(_binary_search_insert_arg(args, b), None)

        args = _binary_search_insert_arg(args, a)
        self.assertEqual(args, [a, b])

        self.assertEqual(_binary_search_insert_arg(args, b), None)
        self.assertEqual(_binary_search_insert_arg(args, a), None)

        args = _binary_search_insert_arg(args, c)
        self.assertEqual(args, [a, b, c])

        self.assertEqual(_binary_search_insert_arg(args, a), None)
        self.assertEqual(_binary_search_insert_arg(args, b), None)
        self.assertEqual(_binary_search_insert_arg(args, c), None)

        a1 = sympy.Symbol("a1")
        a2 = sympy.Symbol("a2")

        args = _binary_search_insert_arg(args, a1)
        self.assertEqual(args, [a, a1, b, c])

        args = _binary_search_insert_arg(args, a2)
        self.assertEqual(args, [a, a1, a2, b, c])

        c1 = sympy.Symbol("c1")
        args = _binary_search_insert_arg(args, c1)
        self.assertEqual(args, [a, a1, a2, b, c, c1])

        # insert to front
        _a = sympy.Symbol("_a")
        args = _binary_search_insert_arg(args, _a)
        self.assertEqual(args, [_a, a, a1, a2, b, c, c1])

    def test_floor_clean_div_axioms(self):
        # Test that if we add an axiom that have FloorDiv, after which the
        # shapeEnv changed such that it can be simplified it to CleanDiv, then
        # We still correctly replace CleanDiv with the axiom value of FloorDiv.
        shape_env = ShapeEnv()
        a = shape_env.create_unbacked_symint()

        shape_env.guard_or_defer_runtime_assert((a // 3 == 1).node.expr, " test")

        from sympy import Eq

        test1 = Eq(FloorDiv(a.node.expr, 3), 1)
        test2 = Eq(CleanDiv(a.node.expr, 3), 1)

        self.assertTrue(shape_env.evaluate_expr(test1))
        self.assertEqual(shape_env._maybe_evaluate_static(test2), None)

        # After this FloorDiv(a, 3) is simplified to CleanDiv(a, 3)
        shape_env.guard_or_defer_runtime_assert(Eq(Mod(a, 3), 0), " test")
        self.assertEqual(test2, shape_env.simplify(test1))

        self.assertTrue(shape_env.evaluate_expr(test1))
        self.assertTrue(shape_env.evaluate_expr(test2))

    def test_sympy_optimized_add(self):
        shape_env = ShapeEnv()
        s0 = create_symint(shape_env, 2)
        s1 = create_symint(shape_env, 3)
        s2 = create_symint(shape_env, 4)
        sum = s0 + s1

        self.assertTrue(sum.node._optimized_summation)

        def assert_optimized(sym):
            self.assertTrue(sym.node._optimized_summation)

        def assert_not_optimized(sym):
            self.assertFalse(getattr(sym.node, "_optimized_summation", False))

        assert_optimized(sum)

        # add duplicate symbol
        assert_not_optimized(sum + s0)

        # add constant.
        assert_not_optimized(sum + 1)

        # add new unique symbol, should maintain _optimized_summation property.
        assert_optimized(sum + s2)

        assert_optimized(s2 + sum)

        # add x + (a+b) with no  _optimized_summation on the rhs sum.
        a = create_symint(shape_env, 10)
        b = create_symint(shape_env, 11)
        two_sum = torch.sym_sum([a, b])
        assert_not_optimized(two_sum)
        assert_optimized(sum + two_sum)

        # adding two expressions of length >2 that are _optimized_summation.
        a = s0 + s1 + s2
        s3 = create_symint(shape_env, 10)
        s4 = create_symint(shape_env, 20)
        s5 = create_symint(shape_env, 30)
        b = s3 + s4 + s5
        assert_optimized(a)
        assert_optimized(b)
        assert_not_optimized(a + b)
        assert_not_optimized(b + a)
        assert_not_optimized(b + a + b)

    def test_max_of_unique_summation_opt(self):
        shape_env = ShapeEnv()
        s0 = shape_env.create_unbacked_symint()
        s1 = shape_env.create_unbacked_symint()
        s2 = shape_env.create_unbacked_symint()
        s3 = shape_env.create_unbacked_symint()
        s4 = shape_env.create_unbacked_symint()
        s5 = shape_env.create_unbacked_symint()
        s7 = shape_env.create_unbacked_symint()

        def assert_optimized(sym):
            self.assertTrue(sym.node.expr.unique_summations_symbols is not None)

        def assert_not_optimized(sym):
            getattr(sym.node.expr, "unique_summations_symbols", None)

        mx1 = torch.sym_max(s0, s1)
        assert_not_optimized(mx1)

        mx2 = torch.sym_max(s0 + s1, s2 + s3)
        assert_optimized(mx2)

        mx3 = torch.sym_max(mx2, s4 + s5)
        assert_optimized(mx3)
        assert_optimized(torch.sym_max(s4 + s5, mx2))

        assert_not_optimized(torch.sym_max(mx3, s7))
        assert_not_optimized(torch.sym_max(mx3, 10))
        assert_not_optimized(torch.sym_max(mx3, s3 + s7))
        assert_not_optimized(torch.sym_max(mx3, s7 * 2))

    def test_sym_max_multi_max_simplify(self):
        shape_env = ShapeEnv()
        u0 = shape_env.create_unbacked_symint()
        self.assertTrue(
            statically_known_true(
                torch.sym_max(1, torch.sym_max(257, u0)) == torch.sym_max(257, u0)
            )
        )

    def test_numpy_sym_max(self):
        self.assertEqual(torch.sym_max(np.int64(10), 12), 12)
        self.assertEqual(torch.sym_max(np.int64(12), 10), 12)
        self.assertEqual(torch.sym_max(np.int64(10), 12.5), 12.5)
        self.assertEqual(torch.sym_max(np.int64(14), 12.5), 14.0)
        self.assertEqual(torch.sym_max(np.float64(14.0), 12), 14.0)
        self.assertEqual(torch.sym_max(np.float64(14.0), 16), 16.0)

    def test_numpy_sym_min(self):
        self.assertEqual(torch.sym_min(np.int64(10), 12), 10)
        self.assertEqual(torch.sym_min(np.int64(12), 10), 10)
        self.assertEqual(torch.sym_min(np.int64(10), 12.5), 10.0)
        self.assertEqual(torch.sym_min(np.int64(14), 12.5), 12.5)
        self.assertEqual(torch.sym_min(np.float64(14.0), 12), 12.0)
        self.assertEqual(torch.sym_min(np.float64(14.0), 16), 14.0)

    def test_debug_has_internal_overlap_unbacked(self):
        shape_env = ShapeEnv()
        u0 = shape_env.create_unbacked_symint()
        cf = torch._debug_has_internal_overlap
        self.assertEqual(cf(torch.empty_strided((u0, 2), (2, 1), device="meta")), 0)
        self.assertEqual(cf(torch.empty_strided((2, u0), (1, 2), device="meta")), 0)
        self.assertEqual(cf(torch.empty_strided((u0,), (1,), device="meta")), 0)
        self.assertEqual(cf(torch.empty_strided((1,), (u0,), device="meta")), 2)
        Max = torch.sym_max
        self.assertEqual(
            cf(
                torch.empty_strided(
                    (2, 3, 1, u0),
                    (3 * Max(1, u0), Max(1, u0), Max(1, u0), 1),
                    device="meta",
                )
            ),
            2,
        )

        # Wobbling these to zero is OK too
        self.assertEqual(cf(torch.empty_strided((u0, 2), (3, 1), device="meta")), 2)
        self.assertEqual(cf(torch.empty_strided((2, u0), (1, 3), device="meta")), 2)

    def test_specialize_zero_one(self):
        shape_env = ShapeEnv(specialize_zero_one=True)
        a0 = create_symint(shape_env, 5)
        assert a0 != 1
        self.assertEqual(len(shape_env.guards), 0)

        shape_env = ShapeEnv(specialize_zero_one=False)
        a0 = create_symint(shape_env, 5)
        assert a0 != 1
        self.assertEqual(len(shape_env.guards), 1)

    def test_duck_shape(self):
        shape_env = ShapeEnv(duck_shape=True)
        a0 = create_symint(shape_env, 5)
        a1 = create_symint(shape_env, 5)
        assert a0 == a1
        self.assertEqual(len(shape_env.guards), 0)

        shape_env = ShapeEnv(duck_shape=False)
        a0 = create_symint(shape_env, 5)
        a1 = create_symint(shape_env, 5)
        assert a0 == a1
        self.assertEqual(len(shape_env.guards), 1)

    def test_int_bool(self):
        # See https://github.com/pytorch/pytorch/issues/95981
        shape_env = ShapeEnv(duck_shape=True)
        a0 = create_symint(shape_env, 5)
        assert a0
        self.assertEqual(len(shape_env.guards), 0)

    def test_symint_as_scalar(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 2)

        sym_int_encountered = False

        class TestSymInt(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                assert func == torch.ops.aten.add.Tensor

                nonlocal sym_int_encountered
                # WARNING: do not do identity tests on the outer
                # SymInt/SymFloat, they are NOT STABLE
                sym_int_encountered = kwargs["alpha"].node is a0.node
                kwargs["alpha"] = 0
                return func(*args)

        x = torch.rand([4, 4])
        with TestSymInt():
            y = torch.add(x, x, alpha=a0)

        self.assertTrue(sym_int_encountered)

    def test_deepcopy(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 2)
        assert a0 < 4
        new_shape_env = copy.deepcopy(shape_env)
        self.assertEqual(len(new_shape_env.guards), 1)

    def test_print_readable_with_symints(self):
        def f(a, b):
            dim0 = a.shape[0] + b.shape[0]
            dim1 = a.shape[1] + b.shape[1]
            d = a.new_empty(dim0, dim1)
            d = torch.ops.aten.native_dropout(d, 0.5, train=True)
            return d

        fx_g = make_fx(f, tracing_mode="symbolic")(torch.randn(5, 3), torch.randn(4, 3))
        out = fx_g.print_readable(print_output=False)

        self.assertExpectedInline(
            out.strip(),
            """\
class f(torch.nn.Module):
    def forward(self, a_1: "f32[s75, s96]", b_1: "f32[s57, s96]"):
        # No stacktrace found for following nodes
        sym_size_int: "Sym(s75)" = torch.ops.aten.sym_size.int(a_1, 0)
        sym_size_int_1: "Sym(s57)" = torch.ops.aten.sym_size.int(b_1, 0)
        add: "Sym(s57 + s75)" = sym_size_int + sym_size_int_1;  sym_size_int = sym_size_int_1 = None
        sym_size_int_2: "Sym(s96)" = torch.ops.aten.sym_size.int(a_1, 1)
        sym_size_int_3: "Sym(s96)" = torch.ops.aten.sym_size.int(b_1, 1);  b_1 = None
        add_1: "Sym(2*s96)" = sym_size_int_2 + sym_size_int_3;  sym_size_int_2 = sym_size_int_3 = None
        new_empty: "f32[s57 + s75, 2*s96]" = torch.ops.aten.new_empty.default(a_1, [add, add_1], pin_memory = False);  a_1 = add = add_1 = None
        native_dropout = torch.ops.aten.native_dropout.default(new_empty, 0.5, True);  new_empty = None
        getitem: "f32[s57 + s75, 2*s96]" = native_dropout[0]
        getitem_1: "b8[s57 + s75, 2*s96]" = native_dropout[1];  native_dropout = None
        return (getitem, getitem_1)""",  # noqa: B950
        )

    def test_statically_known_true(self):
        shape_env = ShapeEnv()
        s2, s3, s4 = (create_symint(shape_env, i) for i in range(2, 5))

        # Statically known true
        self.assertTrue(statically_known_true(True))
        self.assertTrue(statically_known_true(s2 == s2))
        self.assertTrue(statically_known_true(s2 * s3 > s3))
        self.assertTrue(statically_known_true(s3 * s4 > s4))
        self.assertTrue(statically_known_true((s3 + s3) % 2 == 0))

        # Statically known false
        self.assertFalse(statically_known_true(False))
        self.assertFalse(statically_known_true(s3 * s4 <= s4))
        self.assertFalse(statically_known_true((s3 + s3) % 2 == 1))

        # True for hints, but not known statically
        self.assertFalse(statically_known_true(s2 + s2 == s4))
        self.assertFalse(statically_known_true(s4 % s2 == 0))
        self.assertFalse(statically_known_true(s2 != s3))
        self.assertFalse(statically_known_true(s3 * s4 > s2))

        # False for hints, but not known statically
        self.assertFalse(statically_known_true(s2 == s3))
        self.assertFalse(statically_known_true(s2 > s3))
        self.assertFalse(statically_known_true(s3 + s3 == s4))

        # No guards should be generated
        self.assertEqual(len(shape_env.guards), 0)

    def test_statically_known_false(self):
        shape_env = ShapeEnv()
        s2, s3, s4 = (create_symint(shape_env, i) for i in range(2, 5))

        # Statically known true
        self.assertFalse(statically_known_false(True))
        self.assertFalse(statically_known_false(s2 == s2))
        self.assertFalse(statically_known_false(s2 * s3 > s3))
        self.assertFalse(statically_known_false(s3 * s4 > s4))
        self.assertFalse(statically_known_false((s3 + s3) % 2 == 0))

        # Statically known false
        self.assertTrue(statically_known_false(False))
        self.assertTrue(statically_known_false(s3 * s4 <= s4))
        self.assertTrue(statically_known_false((s3 + s3) % 2 == 1))

        # True for hints, but not known statically
        self.assertFalse(statically_known_false(s2 + s2 == s4))
        self.assertFalse(statically_known_false(s4 % s2 == 0))
        self.assertFalse(statically_known_false(s2 != s3))
        self.assertFalse(statically_known_false(s3 * s4 > s2))

        # False for hints, but not known statically
        self.assertFalse(statically_known_false(s2 == s3))
        self.assertFalse(statically_known_false(s2 > s3))
        self.assertFalse(statically_known_false(s3 + s3 == s4))

        # No guards should be generated
        self.assertEqual(len(shape_env.guards), 0)

    def test_ephemeral_source_simplification(self):
        from torch._dynamo.source import EphemeralSource

        # For full robustness, ensure the ephemeral source symbols are simplified out regardless
        # of construction order or check order.
        for construct_ephemeral_first, x_first_in_check in itertools.product(
            [False, True], [False, True]
        ):
            shape_env = ShapeEnv()
            shape = (5, 10)
            dynamic_dims = [DimDynamic.DYNAMIC for _ in shape]
            x = create_symbolic_tensor(
                "x",
                torch.randn(*shape),
                shape_env,
                source=(EphemeralSource() if construct_ephemeral_first else None),
                dynamic_dims=dynamic_dims,
            )
            y = create_symbolic_tensor(
                "y",
                torch.randn(*shape),
                shape_env,
                source=(EphemeralSource() if not construct_ephemeral_first else None),
                dynamic_dims=dynamic_dims,
            )
            t_with_ephemeral = x if construct_ephemeral_first else y

            def _get_ephemeral_source_symbols(t):
                return [
                    s.node.expr
                    for s in itertools.chain(t.shape, t.stride(), (t.storage_offset(),))
                    if isinstance(s, torch.SymInt)
                    and s.node.expr in shape_env.var_to_sources
                    and any(
                        source.is_ephemeral()
                        for source in shape_env.var_to_sources[s.node.expr]
                    )
                ]

            # these checks should simplify out the ephemeral symbols, regardless of the
            # ordering x == y or y == x
            self.assertTrue(len(_get_ephemeral_source_symbols(t_with_ephemeral)) > 0)
            if x_first_in_check:
                torch._check(x.size() == y.size())
                torch._check(x.stride() == y.stride())
                torch._check(x.storage_offset() == y.storage_offset())
            else:
                torch._check(y.size() == x.size())
                torch._check(y.stride() == x.stride())
                torch._check(y.storage_offset() == x.storage_offset())
            self.assertEqual(len(_get_ephemeral_source_symbols(t_with_ephemeral)), 0)

    def test_ephemeral_source_unified_with_non_ephemeral_source(self):
        from torch._dynamo.source import EphemeralSource

        for construct_ephemeral_first in (False, True):
            shape_env = ShapeEnv()
            shape = (5, 10)
            # use duck sizing here to ensure symbol reuse across x and y
            duck_dims = [DimDynamic.DUCK for _ in shape]
            x = create_symbolic_tensor(
                "x",
                torch.randn(*shape),
                shape_env,
                source=(EphemeralSource() if construct_ephemeral_first else None),
                dynamic_dims=duck_dims,
            )
            y = create_symbolic_tensor(
                "y",
                torch.randn(*shape),
                shape_env,
                source=(EphemeralSource() if not construct_ephemeral_first else None),
                dynamic_dims=duck_dims,
            )

            # regardless of construction order, non-ephemeral sources should be preferred
            # first in the var_to_sources list for potential guarding later on
            for source_list in shape_env.var_to_sources.values():
                self.assertFalse(source_list[0].is_ephemeral())

            self.assertEqual(x.size(), y.size())
            self.assertEqual(x.stride(), y.stride())
            self.assertEqual(x.storage_offset(), y.storage_offset())

    def test_tensor_factory_with_symint(self):
        args = list(range(3))
        expected = torch.tensor(args)

        shape_env = ShapeEnv()
        sym_args = [create_symint(shape_env, i) for i in args]

        # test tensor factories
        for dt in all_types_and(torch.half, torch.bfloat16):
            res = torch.tensor(sym_args, dtype=dt)
            self.assertEqual(res, expected, exact_dtype=False)

        # test legacy tensor factories
        legacy_ctors = [
            torch.Tensor,
            torch.LongTensor,
            torch.DoubleTensor,
            torch.FloatTensor,
            torch.IntTensor,
            torch.ShortTensor,
            torch.HalfTensor,
            torch.ByteTensor,
        ]
        for Tensor in legacy_ctors:
            res = Tensor(sym_args)
            self.assertEqual(res, expected, exact_dtype=False)

    def test_backed_size_oblivious_01_spec(self):
        from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

        @torch.compile(dynamic=True, fullgraph=True)
        def f(a, b):
            if guard_size_oblivious(a.size(0) == 1):
                return b * 10
            else:
                return b * 20

        with torch.fx.experimental._config.patch(backed_size_oblivious=True):
            # always go to the >= 2 branch.
            self.assertEqual(
                f(torch.tensor([1]), torch.tensor([1])), torch.tensor([20])
            )

    @fresh_cache()
    def test_slice_backed_size_oblivious(self):
        @torch.compile(backend="inductor", fullgraph=True, dynamic=True)
        def f(x):
            return x[:5]

        with torch.fx.experimental._config.patch(backed_size_oblivious=True):
            f(torch.randn(10, 10))

    def test_baddbmm_symint(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)

        B, M, K, N = [shape_env.create_unbacked_symint() for _ in range(4)]

        with fake_mode:
            A = torch.empty((B, M, K), device="meta")
            Bmat = torch.empty((B, K, N), device="meta")
            bias3 = torch.empty((B, M, N), device="meta")

            _ = torch.baddbmm(bias3, A, Bmat)


@skipIfTorchDynamo(
    "Creating ShapeEnv fails for confusing reasons (also we never expect dynamo to see code like this)"
)
class TestSymNumberMagicMethods(TestCase):
    def _do_test(self, fn, inp1, inp2, shape_env, is_unary_fn):
        with self.subTest(fn=fn, inp1=inp1, inp2=inp2, is_unary_fn=is_unary_fn):
            return self._do_test2(fn, inp1, inp2, shape_env, is_unary_fn)

    def _do_test2(self, fn, inp1, inp2, shape_env, is_unary_fn):
        # Helper function
        # NB: don't use one as that will get specialized
        # TODO: We don't have to circuitously create the float, can just
        # create a symfloat directly
        seed_node = (create_symint(shape_env, 2) / 2.0).node
        bool_seed_node = (create_symint(shape_env, 2) == 2).node

        def get_sym_inp(inp):
            # NB: this must come before int
            if isinstance(inp, bool):
                return torch.SymBool(to_node(bool_seed_node, inp))
            elif isinstance(inp, int):
                return torch.SymInt(to_node(seed_node, inp))
            else:
                return torch.SymFloat(to_node(seed_node, inp))

        if fn == "float_pow":
            if inp1 < 0:
                return

        if fn == "pow_by_natural":
            if isinstance(inp1, float) or isinstance(inp2, float):
                return
            if inp2 < 0:
                return

        def maybe_xfail(inp1, inp2):
            if fn == "sym_sqrt" and inp1 < 0:
                # ValueError: math domain error
                return self.assertRaises((ValueError,))
            elif (
                fn in ("float_truediv", "int_truediv", "int_floordiv", "mod")
                and inp2 == 0
            ):
                # ZeroDivisionError: division by zero
                return self.assertRaises((ZeroDivisionError,))
            elif fn in ["float_pow", "pow_by_natural"] and inp1 == 0 and inp2 < 0:
                # ZeroDivisionError: 0.0 cannot be raised to a negative power
                return self.assertRaises((ZeroDivisionError,))
            elif (
                # TODO: dear catastrophe waitress,
                # this doesn't work
                fn in ["float_pow", "pow_by_natural"]
                and inp1 < 0
                and (
                    type(inp1) is (SymInt, SymFloat) or type(inp2) is (SymInt, SymFloat)
                )
                and (type(inp1) is (SymFloat, float) or type(inp2) is (SymFloat, float))
            ):
                # Complex result, which we do not support:
                # TypeError: Cannot convert complex to float
                return self.assertRaises((RuntimeError,))
            elif fn in ("lshift", "rshift") and not (
                isinstance(inp1, (SymInt, int)) and isinstance(inp2, (SymInt, int))
            ):
                # TypeError: unsupported operand type(s)
                return self.assertRaises((TypeError,))
            elif fn in ("lshift", "rshift") and inp2 < 0:
                # ValueError: math domain error
                return self.assertRaises((ValueError,))
            else:
                return contextlib.nullcontext()

        lambda_apply = method_to_operator(fn)

        def guard_fn(v):
            if type(v) in (SymBool, bool):
                return guard_bool(v)
            elif type(v) in (SymFloat, float):
                return guard_float(v)
            else:  # SymInt, int
                return guard_int(v)

        # Get reference result
        with maybe_xfail(inp1, inp2):
            if is_unary_fn:
                ref_out = lambda_apply(inp1)
            else:
                ref_out = lambda_apply(inp1, inp2)

        # Symified first arg
        sym_inp1 = get_sym_inp(inp1)
        with maybe_xfail(sym_inp1, inp2):
            if is_unary_fn:
                out = lambda_apply(sym_inp1)
            else:
                out = lambda_apply(sym_inp1, inp2)
            self.assertTrue(isinstance(out, (SymInt, SymFloat, SymBool)))
            out = guard_fn(out)
            self.assertEqual(out, ref_out)

        if is_unary_fn:
            return

        # Symified second arg
        sym_inp2 = get_sym_inp(inp2)
        with maybe_xfail(inp1, sym_inp2):
            out = lambda_apply(inp1, sym_inp2)
            self.assertTrue(isinstance(out, (SymInt, SymFloat, SymBool)))
            out = guard_fn(out)
            self.assertEqual(out, ref_out)

        # Symified both args
        with maybe_xfail(sym_inp1, sym_inp2):
            out = lambda_apply(sym_inp1, sym_inp2)
            self.assertTrue(isinstance(out, (SymInt, SymFloat, SymBool)))
            out = guard_fn(out)
            self.assertEqual(out, ref_out)

    @parametrize("fn", list(sym_node.magic_methods.keys()))
    def test_bool_method(self, fn):
        # sym_ite has its own tests
        if fn not in sym_node.bool_magic_methods or fn == "sym_ite":
            self.skipTest(f"{fn} is non-bool")

        is_unary_fn = fn in sym_node.unary_methods
        shape_env = ShapeEnv()
        self._do_test(fn, True, False, shape_env, is_unary_fn)

    @parametrize("fn", list(sym_node.magic_methods.keys()))
    @parametrize("first_type", ["int", "float"])
    @parametrize("second_type", ["int", "float"])
    def test_method(self, fn, first_type, second_type):
        if first_type == "float":
            # TODO: Hmm, this looks like we skip all floats
            self.skipTest(f"{fn} is not a float magic method")

        if (
            first_type == "int" or second_type == "int"
        ) and fn in sym_node.only_float_magic_methods:
            self.skipTest(f"{fn} is not an int method")

        if second_type == "float" and fn in ["mod"]:
            self.skipTest(f"{fn} only handles int")

        if fn in sym_node.bitwise_ops and (first_type != "int" or second_type != "int"):
            self.skipTest(f"{fn} is a bitwise op, only handles int")

        is_unary_fn = fn in sym_node.unary_methods or fn == "round"
        # Second argument is ignored for unary function. So only run for one type
        if is_unary_fn and second_type == "float":
            self.skipTest(f"{fn} is unary and already tested")

        if fn in sym_node.bool_magic_methods:
            self.skipTest(f"{fn} is bool")

        # Only floats here since these will be converted to int if necessary.
        # We also ignore complex and bool.
        values = (
            0.0,
            1.0,
            0.5 if fn in ("sym_acos", "sym_asin") else 2.5,  # avoid math domain error
        )

        neg_values = tuple(-x for x in values)

        for inp1, inp2 in itertools.chain(
            itertools.product(values, values),
            itertools.product(values, neg_values),
            itertools.product(neg_values, values),
            itertools.product(neg_values, neg_values),
        ):
            if first_type == "int":
                inp1 = int(inp1)
            if second_type == "int":
                inp2 = int(inp2)

            shape_env = ShapeEnv()

            self._do_test(fn, inp1, inp2, shape_env, is_unary_fn)

    def get_constant_bool(self, val):
        return SymBool(torch._C._get_constant_bool_symnode(val))

    @unittest.expectedFailure
    def test_symint_hashing(self):
        shape_env = ShapeEnv()
        hash(create_symint(shape_env, 3))

    def test_symnode_hashing(self):
        shape_env = ShapeEnv()

        # These all trigger specialization when hashed
        hash(create_symbool(shape_env, True))
        # We should be passing in float here, but create_symbol currently
        # only supports int
        hash(create_symfloat(shape_env, 3.0))

        # NestedInt (SymInt), constant SymBool, SymNode are hashable
        j1 = torch._C._get_nested_int(1, 1)
        j1_copy = torch._C._get_nested_int(1, 1)
        j2 = torch._C._get_nested_int(2, 1)
        t = self.get_constant_bool(True)
        t_copy = self.get_constant_bool(True)
        f = self.get_constant_bool(False)
        n = create_symint(shape_env, 3).node
        m = self.get_constant_bool(True).node

        self.assertIs(j1 == j1_copy, True)
        self.assertEqual(hash(j1), hash(j1_copy))
        self.assertIs(j1 == j2, False)
        self.assertNotEqual(hash(j1), hash(j2))
        self.assertIs(t == t_copy, True)
        self.assertEqual(hash(t), hash(t_copy))
        self.assertIs(t == f, False)
        self.assertNotEqual(hash(t), hash(f))

        hash(n)
        hash(m)

    def test_symint_deepcopy(self):
        shape_env = ShapeEnv()

        symnodes = (torch._C._get_nested_int(1, 1),)
        deepcopied_symnodes = copy.deepcopy(symnodes)
        self.assertEqual(symnodes, deepcopied_symnodes)

    def test_non_symbolic_symnode(self):
        j1 = torch._C._get_nested_int(1, 1)
        j2 = torch._C._get_nested_int(1, 1)
        j3 = torch._C._get_nested_int(3, 1)

        self.assertIsInstance(j1, torch.SymInt)
        self.assertNotIsInstance(j1, int)

        with self.assertRaisesRegex(
            RuntimeError, "add not supported by NestedIntSymNode"
        ):
            j1 + 3

        self.assertFalse(j1 == 3)
        with self.assertRaisesRegex(RuntimeError, "indeterminate"):
            self.assertFalse(3 >= j2)

        self.assertIs(j1 == j1, True)
        self.assertIs(j1 == j2, True)
        self.assertIs(j1 == j3, False)
        self.assertIs(j1 != j3, True)
        self.assertIs(j1 != j2, False)

        x = self.get_constant_bool(True)
        #
        # Unary
        #
        # op(constant SymBool)
        self.assertIs(x.__sym_not__(), False)

        #
        # Binary
        #
        # op(constant SymBool, bool)
        # op(constant SymBool, constant SymBool)
        # op(bool, constant SymBool)
        self.assertIs(operator.and_(x, True), True)
        self.assertIs(operator.and_(x, x), True)
        self.assertIs(operator.and_(True, x), True)

        # op(symbolic SymBool, constant Symbool)
        # op(constant SymBool, symbolic Symbool)
        shape_env = ShapeEnv()
        a = create_symint(shape_env, 2)
        b = create_symint(shape_env, 2)
        c = a == b  # symbolic SymBool
        d = self.get_constant_bool(True)
        e = operator.and_(c, d)
        f = operator.and_(d, c)
        self.assertTrue(is_symbolic(e))
        self.assertTrue(is_symbolic(f))
        self.assertIs(e.node.guard_bool("", 0), True)
        self.assertIs(f.node.guard_bool("", 0), True)

        # Comparing sizes
        sz1 = torch.Size([j1, j1, j1])
        sz2 = torch.Size([j1, j1, j1])
        self.assertIs(sz1 == sz2, True)

        sz1 = torch.Size([3, j1, 4])
        sz2 = torch.Size([3, j2, 4])
        self.assertIs(sz1 == sz2, True)
        self.assertIs(sz1 != sz2, False)

    def test_stride_symnode(self):
        shape_env = ShapeEnv()

        # check everything static
        t = create_fake_tensor_with_dynamic_size(
            torch.ones(3, 6),
            shape_env,
            dynamic_sizes=[
                DimDynamic.STATIC,
                DimDynamic.STATIC,
            ],
            dynamic_strides=[
                DimDynamic.INFER_STRIDE,
                DimDynamic.INFER_STRIDE,
            ],
        )
        self.assertTrue(all(isinstance(size, int) for size in t.size()))
        self.assertTrue(all(isinstance(stride, int) for stride in t.stride()))

        # check dynamic size but static dims
        t = create_fake_tensor_with_dynamic_size(
            torch.ones(3, 6),
            shape_env,
            dynamic_sizes=[
                DimDynamic.DYNAMIC,
                DimDynamic.DYNAMIC,
            ],
            dynamic_strides=[
                DimDynamic.INFER_STRIDE,
                DimDynamic.INFER_STRIDE,
            ],
        )
        # Expect stride to be inferred
        s0, s1 = t.size()
        s2, s3 = t.stride()
        self.assertTrue(isinstance(s0, torch.SymInt))
        self.assertTrue(isinstance(s1, torch.SymInt))
        self.assertTrue(isinstance(s2, torch.SymInt))
        self.assertTrue(s1 == s2)
        self.assertEqual(s3, 1)

        # Check dynamic stride but static dims
        t = create_fake_tensor_with_dynamic_size(
            torch.ones(3, 6),
            shape_env,
            dynamic_sizes=[
                DimDynamic.STATIC,
                DimDynamic.STATIC,
            ],
            dynamic_strides=[
                DimDynamic.DYNAMIC,
                DimDynamic.INFER_STRIDE,
            ],
        )
        s0, s1 = t.size()
        s2, s3 = t.stride()
        self.assertTrue(isinstance(s0, int))
        self.assertTrue(isinstance(s1, int))
        self.assertTrue(isinstance(s2, torch.SymInt))
        self.assertTrue(isinstance(s3, int))

        # Check dynamic sizes and dims, and ensure different symbol
        t = create_fake_tensor_with_dynamic_size(
            torch.ones(3, 6),
            shape_env,
            dynamic_sizes=[
                DimDynamic.DYNAMIC,
                DimDynamic.DYNAMIC,
            ],
            dynamic_strides=[
                DimDynamic.DYNAMIC,
                DimDynamic.INFER_STRIDE,
            ],
        )
        s0, s1 = t.size()
        s2, s3 = t.stride()
        self.assertTrue(isinstance(s0, torch.SymInt))
        self.assertTrue(isinstance(s1, torch.SymInt))
        self.assertTrue(isinstance(s2, torch.SymInt))
        self.assertTrue(isinstance(s3, int))
        self.assertTrue(str(s1.node.expr) != str(s2.node.expr))

    @fresh_cache()
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @parametrize("backend", ["inductor", "eager"])
    def test_dynamic_int_basic_compile(self, backend):
        from torch.fx.experimental.sym_node import DynamicInt

        cnt = CompileCounterWithBackend(backend)

        # test scalar inputs to function
        def f(x, y, z):
            out = torch.tensor([x + y + z])
            out = out + torch.zeros(abs(x) + 2).sum()  # test out tensor construction
            return out

        fn = torch.compile(f, fullgraph=True, backend=cnt)
        x = DynamicInt(1)
        z = DynamicInt(3)
        self.assertEqual(fn(x, x, z), f(1, 1, 3))  # guard: x == y
        self.assertEqual(fn(2, 2, 0), f(2, 2, 0))
        self.assertEqual(fn(-1, -1, 2), f(-1, -1, 2))
        self.assertEqual(cnt.frame_count, 1)  # no recompiles

        self.assertEqual(fn(3, 4, 5), f(3, 4, 5))  # now we recompile
        self.assertEqual(cnt.frame_count, 2)

        # test nn module property
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.i = DynamicInt(1)

            def forward(self, x):
                return torch.tensor([x + self.i])

        cnt.clear()
        m = Foo()
        mc = torch.compile(m, backend=cnt, fullgraph=True)

        self.assertEqual(mc(DynamicInt(0)), m(0))
        mc.i = -2  # override attribute
        self.assertEqual(mc(-1), m(-1))
        self.assertEqual(cnt.frame_count, 1)

    def test_dynamic_int_eager_usage(self):
        from torch.fx.experimental.sym_node import DynamicInt

        w = DynamicInt(-1)
        x = DynamicInt(0)
        y = DynamicInt(1)
        z = DynamicInt(2)

        def check(l, r):
            self.assertTrue(isinstance(l, DynamicInt))
            self.assertEqual(l, r)

        # test arithmetic
        check(2 * y + z, 4)
        check((10 - z) // 2, 4)
        check(1 // z, 0)
        check(-w + w**2, 2)
        check(x % z, 0)
        check(1 << z, 4)
        check(z | y, 3)
        check(min(y, z), 1)
        self.assertTrue(z > -2)
        with self.assertRaises(ZeroDivisionError):
            y % x

        # math, numpy
        self.assertEqual(math.cos(x), y)
        self.assertEqual(math.prod([z, z], start=z), 8)
        self.assertEqual(np.arange(z)[y], 1)
        self.assertTrue(np.allclose(np.ones([y, z]).sum(axis=x), np.ones(z)))

        # test conversions
        self.assertTrue(isinstance(x + 2, int))
        self.assertTrue(isinstance(x + 2, DynamicInt))
        self.assertEqual(y / 2.0, 0.5)  # this could return DynamicFloat in future
        self.assertEqual(float(z), 2.0)
        self.assertFalse(bool(x))
        self.assertEqual(DynamicInt(x).real, x.real)

        # torch functions, scalar inputs
        self.assertEqual(torch.arange(z)[:w][x], 0)
        self.assertEqual(torch.add(torch.tensor(w), torch.tensor(w), alpha=z), -3)
        self.assertEqual(
            list(torch.nn.Linear(z, y)(torch.randn(z * 2, z)).shape), [4, 1]
        )
        self.assertEqual(z * torch.ones(z).sum(dim=x), 4)


instantiate_parametrized_tests(TestSymNumberMagicMethods)


class TestFloorDiv(TestCase):
    @staticmethod
    def python_floordiv(x, y):
        return x // y

    @staticmethod
    def torch_floordiv(x, y):
        # Note: we fully evaluate here since FloorDiv might not always do
        # that.
        shape_env = ShapeEnv()
        return shape_env.evaluate_expr(FloorDiv(x, y))

    @staticmethod
    def yield_test_cases(values, negate=True):
        for x, y in values:
            yield (x, y)
            if negate:
                yield (-x, y)
                yield (x, -y)
                yield (-x, -y)

    def test_floordiv_float_int(self):
        values = ((7, 2),)

        for x, y in TestFloorDiv.yield_test_cases(values):
            self.assertEqual(
                TestFloorDiv.python_floordiv(x, y), TestFloorDiv.torch_floordiv(x, y)
            )

    def test_floordiv_div_by_one(self):
        values = ((2, 1),)

        for x, y in TestFloorDiv.yield_test_cases(values):
            self.assertEqual(
                TestFloorDiv.python_floordiv(x, y), TestFloorDiv.torch_floordiv(x, y)
            )

    def test_floordiv_div_does_not_generate_non_int_rational(self):
        s14 = sympy.Symbol("s14", integer=True, positive=True)
        s37 = sympy.Symbol("s37", integer=True, positive=True)

        inner_expr = FloorDiv(s14, 2016)
        middle_expr = (24 * s37 + 672) * inner_expr
        numerator = middle_expr + 21
        denominator = 22
        result = FloorDiv(numerator, denominator)
        rationals = result.atoms(sympy.Rational)
        all_rationals_ints = all(r.q == 1 for r in rationals)
        self.assertTrue(all_rationals_ints)

    def test_floordiv_simplify(self):
        # Tests how we simplify or evaluate FloorDiv without free variables
        shape_env = ShapeEnv()
        result = 21
        exprs = (7 * FloorDiv(6, 2),)

        for expr in exprs:
            self.assertEqual(expr, result)
            self.assertEqual(expr.doit(deep=False), result)
            self.assertEqual(expr.doit(deep=True), result)
            self.assertEqual(sympy.simplify(expr), result)
            self.assertEqual(shape_env.simplify(expr), result)
            self.assertEqual(shape_env.evaluate_expr(expr), result)

    def test_floordiv_assumptions(self):
        cases = (
            sympy.Symbol("i1", integer=True),
            sympy.Symbol("i2", integer=True),
        )

        for base, divisor in itertools.product(cases, repeat=2):

            def op():
                return FloorDiv(base, divisor)

            def is_complex(x):
                return x.is_integer is False and x.is_real is False and x.is_complex

            if is_complex(base) or is_complex(divisor):
                self.assertRaisesRegex(
                    TypeError,
                    (
                        r"unsupported operand type\(s\) for //: 'Symbol' and 'Symbol',"
                        r" expected integer or real"
                    ),
                    op,
                )
                continue

            op = op()

            # In regular Python, x//x == 1.0 if x is a float, but FloorDiv
            # always returns an integer 1 when both args are the same object.
            # This even works for Symbols with no assumptions specified.
            if base is divisor:
                self.assertTrue(op.is_integer)
                self.assertTrue(op.is_real)
            elif base.is_integer and divisor.is_integer:
                self.assertTrue(op.is_integer)
                self.assertTrue(op.is_real)
            else:
                self.assertEqual(op.is_integer, None)
                self.assertTrue(op.is_real)


class TestDimConstraints(TestCase):
    @skipIfTorchDynamo("mark_dynamic not supported")
    def test_simplify_max_1_0(self):
        x = torch.rand(10)
        torch._dynamo.mark_dynamic(x, 0, max=20, min=5)

        @torch.compile(fullgraph=True)
        def func(x, v):
            # test that statically_known_true
            if (v == 0 or v == 1) and not statically_known_true(
                max(v, (-1 + x.size()[0] // 2)) == (-1 + x.size()[0] // 2)
            ):
                raise AssertionError("error")

            if max(v, (-1 + x.size()[0] // 2)) == (-1 + x.size()[0] // 2):
                return x * 400
            else:
                return (x * 10) * 100

        # testing that this does not throw constraint violation error.
        self.assertEqual(func(x, 1), x * 400)
        self.assertEqual(func(x, 0), x * 400)

    def test_dim_constraints_reduce_congruences_simple(self):
        from sympy import Symbol

        s = Symbol("s", positive=True, integer=True)
        dim_constraints = DimConstraints({}, {}, set(), {})
        dim_constraints._congruences[s] = {
            (s / 2) % 2,
            (s / 2) % 8,
            (s / 2) % 4,
            s % 2,
            ((s / 16) + 2) % 4,
        }
        congruences = dim_constraints._reduce_congruences()
        self.assertEqual(congruences[s], {(s + 32) % 64})

    def test_dim_constraints_reduce_inequalities_simple(self):
        from sympy import Eq, Interval, Ne, Symbol
        from sympy.solvers.inequalities import reduce_inequalities

        s = Symbol("s", positive=True, integer=True)
        exprs = {
            s >= 2,
            Ne(8 * s, 16),
            Ne(s / 2, 1),
            Ne(16 * s, 32),
            s < 16,
            Ne(s, 2),
            s / 2 < 16,
            s / 2 > 1,
            s / 2 >= 2,
            Ne(3 * s / 2, 3),
        }
        solution = reduce_inequalities(exprs, s).as_set()
        self.assertEqual(solution, Interval.Ropen(4, 16))

        exprs.add(Eq(s / 2, 4))
        solution = reduce_inequalities(exprs, s).as_set()
        self.assertEqual(solution, {8})

    def test_dim_constraints_reduce_inequalities_error(self):
        from collections import defaultdict

        from sympy import Symbol
        from sympy.solvers.inequalities import reduce_inequalities

        from torch._dynamo.source import (
            LocalSource,
            TensorProperty,
            TensorPropertySource,
        )
        from torch.fx.experimental.symbolic_shapes import DynamicDimConstraintPrinter

        s0 = Symbol("s0", positive=True, integer=True)
        exprs = {
            4 * s0**3 - 4 * s0**2 + s0 <= 2147483647,
            s0 >= 2,
            s0**3 <= 2147483647,
            s0 <= 2147483647,
        }
        answer = reduce_inequalities(exprs, s0)

        symbol_to_source = defaultdict(list)
        symbol_to_source[s0].append(
            TensorPropertySource(
                base=LocalSource(local_name="a"), prop=TensorProperty.SIZE, idx=0
            )
        )
        dcp = DynamicDimConstraintPrinter(symbol_to_source, {})
        with self.assertRaisesRegex(
            AssertionError,
            "Unknown symbol.*created by constraints solver",
        ):
            dcp.doprint(answer)

    def test_dim_constraints_solve_full(self):
        from sympy import Eq, Integer, Ne, Symbol

        from torch._dynamo.source import (
            LocalSource,
            TensorProperty,
            TensorPropertySource,
        )

        src0 = TensorPropertySource(
            base=LocalSource(local_name="a"), prop=TensorProperty.SIZE, idx=0
        )
        src2 = TensorPropertySource(
            base=LocalSource(local_name="b"), prop=TensorProperty.SIZE, idx=0
        )
        src3 = TensorPropertySource(
            base=LocalSource(local_name="c"), prop=TensorProperty.SIZE, idx=0
        )
        src4 = TensorPropertySource(
            base=LocalSource(local_name="d"), prop=TensorProperty.SIZE, idx=0
        )

        src1 = TensorPropertySource(
            base=LocalSource(local_name="a"), prop=TensorProperty.SIZE, idx=2
        )
        src7 = TensorPropertySource(
            base=LocalSource(local_name="a"), prop=TensorProperty.SIZE, idx=3
        )

        src5 = TensorPropertySource(
            base=LocalSource(local_name="a"), prop=TensorProperty.SIZE, idx=1
        )
        src8 = TensorPropertySource(
            base=LocalSource(local_name="b"), prop=TensorProperty.SIZE, idx=1
        )

        src6 = TensorPropertySource(
            base=LocalSource(local_name="c"), prop=TensorProperty.SIZE, idx=1
        )
        src9 = TensorPropertySource(
            base=LocalSource(local_name="d"), prop=TensorProperty.SIZE, idx=1
        )
        src10 = TensorPropertySource(
            base=LocalSource(local_name="e"), prop=TensorProperty.SIZE, idx=1
        )

        src11 = TensorPropertySource(
            base=LocalSource(local_name="f"), prop=TensorProperty.SIZE, idx=1
        )
        src12 = TensorPropertySource(
            base=LocalSource(local_name="b"), prop=TensorProperty.SIZE, idx=2
        )

        s0 = Symbol("s0", positive=True, integer=True)
        s1 = Symbol("s1", positive=True, integer=True)
        s5 = Symbol("s5", positive=True, integer=True)
        s6 = Symbol("s6", positive=True, integer=True)
        symbol_to_source = {
            s0: [src0, src2, src3, src4],
            s1: [src1, src7],
            s5: [src5, src8],
            s6: [src6, src9, src10],
        }
        var_to_val = {s0: 8, s1: 96, s5: 22, s6: 21}
        marked_dynamic = {s0, s1, s5, s6}
        dim_constraints = DimConstraints(
            symbol_to_source, var_to_val, marked_dynamic, {}
        )
        dim_constraints.add_equality(src2, s0)
        dim_constraints.add_equality(src3, s0)
        dim_constraints.add_equality(src4, s0)
        dim_constraints.add_equality(src7, s1)
        dim_constraints.add_equality(src8, s5)
        dim_constraints.add_equality(src9, s6)
        dim_constraints.add_equality(src10, s6)
        dim_constraints.add_equality(src11, Integer(1))
        dim_constraints.add_equality(src12, Integer(3))

        dim_constraints.add(s1**2 <= 2147483647)
        dim_constraints.add(32 * s1**2 <= 2147483647)
        dim_constraints.add(s0 < 16)
        dim_constraints.add(Eq(Mod(s1, 2), 0))
        dim_constraints.add(Ne(FloorDiv(s1, 2), 1))
        dim_constraints.add(Ne((FloorDiv(s1, 2)) ** 2, 1))
        dim_constraints.add(32 * (FloorDiv(s1, 2)) ** 2 <= 2147483647)
        dim_constraints.add((FloorDiv(s1, 2)) ** 2 > 1)
        dim_constraints.add(Ne(FloorDiv(s1, 2), 1))
        dim_constraints.add(
            64 * (FloorDiv((FloorDiv(s1, 2) - 1), 2)) ** 2
            + 128 * (FloorDiv((FloorDiv(s1, 2) - 1), 2))
            + 64
            <= 2147483647
        )
        dim_constraints.add(Ne(FloorDiv((FloorDiv(s1, 2) - 1), 2) + 1, 1))
        dim_constraints.add(
            Ne(
                (FloorDiv((FloorDiv(s1, 2) - 1), 2)) ** 2
                + 2 * (FloorDiv((FloorDiv(s1, 2) - 1), 2))
                + 1,
                1,
            )
        )
        dim_constraints.add(Ne(FloorDiv((FloorDiv(s1, 2) - 1), 2) + 1, 1))
        dim_constraints.add(
            (FloorDiv((FloorDiv(s1, 2) - 1), 2)) ** 2
            + 2 * (FloorDiv((FloorDiv(s1, 2) - 1), 2))
            + 1
            > 1
        )
        dim_constraints.add(
            128 * (FloorDiv((FloorDiv(s1, 2) - 1), 4)) ** 2
            + 256 * (FloorDiv((FloorDiv(s1, 2) - 1), 4))
            + 128
            <= 2147483647
        )
        dim_constraints.add(Ne(FloorDiv((FloorDiv(s1, 2) - 1), 4) + 1, 1))
        dim_constraints.add(
            Ne(
                (FloorDiv((FloorDiv(s1, 2) - 1), 4)) ** 2
                + 2 * (FloorDiv((FloorDiv(s1, 2) - 1), 4))
                + 1,
                1,
            )
        )
        dim_constraints.add(Ne(FloorDiv((FloorDiv(s1, 2) - 1), 4) + 1, 1))
        dim_constraints.add(
            (FloorDiv((FloorDiv(s1, 2) - 1), 4)) ** 2
            + 2 * (FloorDiv((FloorDiv(s1, 2) - 1), 4))
            + 1
            > 1
        )
        dim_constraints.add(
            256 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            + 512 * (FloorDiv((FloorDiv(s1, 2) - 1), 8))
            + 256
            <= 2147483647
        )
        dim_constraints.add(Ne(FloorDiv((FloorDiv(s1, 2) - 1), 8) + 1, 1))
        dim_constraints.add(
            Ne(
                (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                + 2 * (FloorDiv((FloorDiv(s1, 2) - 1), 8))
                + 1,
                1,
            )
        )
        dim_constraints.add(Ne(FloorDiv((FloorDiv(s1, 2) - 1), 8) + 1, 1))
        dim_constraints.add(
            (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            + 2 * (FloorDiv((FloorDiv(s1, 2) - 1), 8))
            + 1
            > 1
        )
        dim_constraints.add(FloorDiv((FloorDiv(s1, 2) - 1), 8) + 1 >= 3)
        dim_constraints.add(
            60 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 120 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 60
            <= 2147483647
        )
        dim_constraints.add(FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1 >= 0)
        dim_constraints.add(FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1 >= 1)
        dim_constraints.add(
            Ne(
                60 * s0 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 120 * s0 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 60 * s0,
                0,
            )
        )
        dim_constraints.add(Ne(FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1, 1))
        dim_constraints.add(
            Ne(
                (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 2 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 1,
                1,
            )
        )
        dim_constraints.add(
            Ne(
                (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 2 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 1,
                0,
            )
        )
        dim_constraints.add(
            (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 2 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 1
            >= 0
        )
        dim_constraints.add(Ne(FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1, 0))
        dim_constraints.add(
            1
            < 60 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 120 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 60
        )
        dim_constraints.add(Ne(FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1, -1))
        dim_constraints.add(
            Ne(
                60 * s0 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 120 * s0 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 60 * s0,
                120 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 240 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 120,
            )
        )
        dim_constraints.add(
            120 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 240 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 120
            > 0
        )
        dim_constraints.add(
            Eq(
                60 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2 * (Mod(s0, 2))
                - 120 * FloorDiv((FloorDiv(s1, 2) - 1), 8) * Mod(s0, 2)
                + 60 * (Mod(s0, 2)),
                0,
            )
        )
        dim_constraints.add(
            Ne(
                120 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 240 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 120,
                0,
            )
        )
        dim_constraints.add(
            Ne(
                60
                * (FloorDiv(s0, 2))
                * (FloorDiv(s0, (FloorDiv(s0, 2))))
                * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 120
                * FloorDiv(s0, 2)
                * FloorDiv(s0, (FloorDiv(s0, 2)))
                * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 60 * (FloorDiv(s0, 2)) * (FloorDiv(s0, (FloorDiv(s0, 2)))),
                0,
            )
        )
        dim_constraints.add(Ne(FloorDiv(s0, 2), 1))
        dim_constraints.add(
            Ne(
                60 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 120 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 60,
                0,
            )
        )
        dim_constraints.add(
            60 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 120 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 60
            >= 0
        )
        dim_constraints.add(
            1
            < 60
            * (FloorDiv(s0, (FloorDiv(s0, 2))))
            * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 120 * FloorDiv(s0, (FloorDiv(s0, 2))) * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 60 * (FloorDiv(s0, (FloorDiv(s0, 2))))
        )
        dim_constraints.add(Ne(16 * s0, 32))
        dim_constraints.add(Eq(16 * (Mod(s0, 2)), 0))
        dim_constraints.add(Ne(16 * s0, 32))
        dim_constraints.add(Eq(16 * (Mod(s0, 2)), 0))
        dim_constraints.add(FloorDiv(s0, 2) >= 2)
        dim_constraints.add(Ne(FloorDiv(s0, 2), 1))
        dim_constraints.add(1 < FloorDiv(s0, 2))
        dim_constraints.add(Ne(s0, 2))
        dim_constraints.add(
            60
            * (FloorDiv(s0, (FloorDiv(s0, 2))))
            * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 120 * FloorDiv(s0, (FloorDiv(s0, 2))) * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 60 * (FloorDiv(s0, (FloorDiv(s0, 2))))
            >= 60 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 120 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 60
        )
        dim_constraints.add(
            60
            * (FloorDiv(s0, 2))
            * (FloorDiv(s0, (FloorDiv(s0, 2))))
            * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 120
            * FloorDiv(s0, 2)
            * FloorDiv(s0, (FloorDiv(s0, 2)))
            * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 60 * (FloorDiv(s0, 2)) * (FloorDiv(s0, (FloorDiv(s0, 2))))
            > 0
        )
        dim_constraints.add(
            Ne(
                60
                * (FloorDiv(s0, 2))
                * (FloorDiv(s0, (FloorDiv(s0, 2))))
                * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 120
                * FloorDiv(s0, 2)
                * FloorDiv(s0, (FloorDiv(s0, 2)))
                * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 60 * (FloorDiv(s0, 2)) * (FloorDiv(s0, (FloorDiv(s0, 2)))),
                3 * (FloorDiv(s0, 2)) * (FloorDiv(s0, (FloorDiv(s0, 2)))),
            )
        )
        dim_constraints.add(
            Ne(
                20 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 40 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 20,
                0,
            )
        )
        dim_constraints.add(
            20 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 40 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 20
            >= 0
        )
        dim_constraints.add(
            Ne(
                20 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 40 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 20,
                20,
            )
        )
        dim_constraints.add(
            Ne(
                20
                * (
                    Mod(
                        1,
                        (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                        - 2 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                        + 1,
                    )
                ),
                0,
            )
        )
        dim_constraints.add(
            Ne(
                20
                * (FloorDiv((FloorDiv(s1, 2) - 1), 8))
                * (
                    Mod(
                        1,
                        (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                        / (FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1)
                        - 2
                        * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                        / (FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1)
                        + 1 / (FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1),
                    )
                )
                - 20
                * Mod(
                    1,
                    (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                    / (FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1)
                    - 2
                    * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                    / (FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1)
                    + 1 / (FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1),
                ),
                0,
            )
        )
        dim_constraints.add(Ne(FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1, 1))
        dim_constraints.add(
            (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 2 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 1
            >= 1
        )
        dim_constraints.add(
            20 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 40 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 20
            >= 0
        )
        dim_constraints.add(
            20 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 40 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 20
            >= 1
        )
        dim_constraints.add(
            20 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 40 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 20
            >= 2
        )
        dim_constraints.add(
            20 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 40 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 20
            > 1
        )
        dim_constraints.add(
            20 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 40 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 20
            < 60 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 120 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 60
        )
        dim_constraints.add(
            Ne(
                60 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 120 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 60,
                60,
            )
        )
        dim_constraints.add(
            Ne(
                FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1,
                (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 2 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 1,
            )
        )
        dim_constraints.add(
            Eq(
                (FloorDiv((FloorDiv(s1, 2) - 1), 8))
                * (
                    Mod(
                        (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                        / (FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1)
                        - 2
                        * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                        / (FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1)
                        + 1 / (FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1),
                        1,
                    )
                )
                - Mod(
                    (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                    / (FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1)
                    - 2
                    * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                    / (FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1)
                    + 1 / (FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1),
                    1,
                ),
                0,
            )
        )
        dim_constraints.add(
            Ne(
                (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 2 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 1,
                FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1,
            )
        )
        dim_constraints.add(Ne(8 * s0, 16))
        dim_constraints.add(
            60 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 120 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 60
            >= (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 2 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 1
        )
        dim_constraints.add(
            60
            * (FloorDiv(s0, (FloorDiv(s0, 2))))
            * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 120 * FloorDiv(s0, (FloorDiv(s0, 2))) * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 60 * (FloorDiv(s0, (FloorDiv(s0, 2))))
            <= 2147483647
        )
        dim_constraints.add(
            90 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 180 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 90
            <= 2147483647
        )
        dim_constraints.add(FloorDiv(s0, 2) < 16)
        dim_constraints.add(FloorDiv(s0, 2) > 1)
        dim_constraints.add(
            Ne(
                90 * (FloorDiv(s0, 2)) * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 180 * FloorDiv(s0, 2) * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 90 * (FloorDiv(s0, 2)),
                0,
            )
        )
        dim_constraints.add(
            1
            < 90 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 180 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 90
        )
        dim_constraints.add(
            (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 2 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 1
            > 1
        )
        dim_constraints.add(
            60
            * (FloorDiv(s0, (FloorDiv(s0, 2))))
            * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 120 * FloorDiv(s0, (FloorDiv(s0, 2))) * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 60 * (FloorDiv(s0, (FloorDiv(s0, 2))))
            > 1
        )
        dim_constraints.add(
            Ne(
                60 * (FloorDiv(s0, 2)) * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 120 * FloorDiv(s0, 2) * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 60 * (FloorDiv(s0, 2)),
                0,
            )
        )
        dim_constraints.add(
            90 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 180 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 90
            > 1
        )
        dim_constraints.add(
            60 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 120 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 60
            > 1
        )
        dim_constraints.add(
            Ne(
                60 * (FloorDiv(s0, 2)) * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 120 * FloorDiv(s0, 2) * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 60 * (FloorDiv(s0, 2)),
                3 * (FloorDiv(s0, 2)),
            )
        )
        dim_constraints.add(
            60 * (FloorDiv(s0, 2)) * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 120 * FloorDiv(s0, 2) * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 60 * (FloorDiv(s0, 2))
            > 0
        )
        dim_constraints.add(
            60 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 120 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 60
            > 0
        )
        dim_constraints.add(
            Ne(
                120 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 240 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 120,
                0,
            )
        )
        dim_constraints.add(
            1
            < 120 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 240 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 120
        )
        dim_constraints.add(
            Ne(
                120 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 240 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 120,
                6,
            )
        )
        dim_constraints.add(
            120 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 240 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 120
            > 0
        )
        dim_constraints.add(
            Ne(
                120 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 240 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 120,
                0,
            )
        )
        dim_constraints.add(
            120 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 240 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 120
            <= 2147483647
        )
        dim_constraints.add(
            120 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 240 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 120
            <= 20480
        )
        dim_constraints.add(
            Ne(
                90 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 180 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 90,
                0,
            )
        )
        dim_constraints.add(
            120 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 240 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 120
            > 1
        )
        dim_constraints.add(
            90 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 180 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 90
            <= 20480
        )
        dim_constraints.add(
            60 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 120 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 60
            <= 20480
        )
        dim_constraints.add(
            Ne(
                240 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 480 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 240,
                0,
            )
        )
        dim_constraints.add(Eq(6 * s5, 132))
        dim_constraints.add(Eq(4, FloorDiv(s0, 2)))
        dim_constraints.add(Eq(FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1, 4))
        dim_constraints.add(
            Ne(
                64 * (FloorDiv(s0, 2)) * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 128 * FloorDiv(s0, 2) * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 64 * (FloorDiv(s0, 2)),
                0,
            )
        )
        dim_constraints.add(
            1
            < 64 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 128 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 64
        )
        dim_constraints.add(
            64 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 128 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 64
            <= 2147483647
        )
        dim_constraints.add(
            64 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 128 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 64
            > 1
        )
        dim_constraints.add(
            62 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 124 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 62
            <= 2147483647
        )
        dim_constraints.add(
            Ne(
                62 * (FloorDiv(s0, 2)) * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 124 * FloorDiv(s0, 2) * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 62 * (FloorDiv(s0, 2)),
                0,
            )
        )
        dim_constraints.add(
            1
            < 62 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 124 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 62
        )
        dim_constraints.add(Ne(3 * (FloorDiv(s0, 2)), 3))
        dim_constraints.add(Ne(3 * (FloorDiv(s0, 2)), 3))
        dim_constraints.add(Eq(FloorDiv(s0, 2), 4))
        dim_constraints.add(Eq(4, FloorDiv(s0, 2)))
        dim_constraints.add(Eq(FloorDiv(s0, 2), 4))
        dim_constraints.add(FloorDiv((FloorDiv(s1, 2) - 1), 8) - 1 >= 3)
        dim_constraints.add(
            64 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 384 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 576
            <= 2147483647
        )
        dim_constraints.add(FloorDiv((FloorDiv(s1, 2) - 1), 8) - 3 >= 0)
        dim_constraints.add(FloorDiv((FloorDiv(s1, 2) - 1), 8) - 3 >= 1)
        dim_constraints.add(
            Ne(
                64 * (FloorDiv(s0, 2)) * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 384 * FloorDiv(s0, 2) * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 576 * (FloorDiv(s0, 2)),
                0,
            )
        )
        dim_constraints.add(Ne(FloorDiv((FloorDiv(s1, 2) - 1), 8) - 3, 1))
        dim_constraints.add(
            Ne(
                (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 6 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 9,
                1,
            )
        )
        dim_constraints.add(
            Ne(
                (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 6 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 9,
                0,
            )
        )
        dim_constraints.add(
            (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 6 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 9
            >= 0
        )
        dim_constraints.add(Ne(FloorDiv((FloorDiv(s1, 2) - 1), 8) - 3, 0))
        dim_constraints.add(
            1
            < 64 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 384 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 576
        )
        dim_constraints.add(Ne(FloorDiv((FloorDiv(s1, 2) - 1), 8) - 3, 1))
        dim_constraints.add(
            Ne(
                64 * (FloorDiv(s0, 2)) * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 384 * FloorDiv(s0, 2) * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 576 * (FloorDiv(s0, 2)),
                256,
            )
        )
        dim_constraints.add(
            Eq(
                64
                * (
                    Mod(
                        (FloorDiv(s0, 2)) * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                        - 6 * FloorDiv(s0, 2) * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                        + 9 * (FloorDiv(s0, 2)),
                        4,
                    )
                ),
                0,
            )
        )
        dim_constraints.add(
            Eq(
                FloorDiv(s0, 2),
                FloorDiv(
                    (
                        (FloorDiv(s0, 2)) * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                        - 6 * FloorDiv(s0, 2) * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                        + 9 * (FloorDiv(s0, 2))
                    ),
                    4,
                ),
            )
        )
        dim_constraints.add(
            Eq(
                FloorDiv(
                    (
                        (FloorDiv(s0, 2)) * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                        - 6 * FloorDiv(s0, 2) * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                        + 9 * (FloorDiv(s0, 2))
                    ),
                    4,
                ),
                FloorDiv(s0, 2),
            )
        )
        dim_constraints.add(
            Ne(64 * (Mod(FloorDiv((FloorDiv(s1, 2) - 1), 8) + 1, 4)), 0)
        )
        dim_constraints.add(
            Eq(
                64
                * (
                    Mod(
                        (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                        - 6 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                        + 1,
                        4,
                    )
                ),
                0,
            )
        )
        dim_constraints.add(
            64 * (FloorDiv(s0, 2)) * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 384 * FloorDiv(s0, 2) * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 576 * (FloorDiv(s0, 2))
            > 0
        )
        dim_constraints.add(
            (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 6 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 9
            >= 1
        )
        dim_constraints.add(
            Eq(
                64 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 384 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 576,
                256,
            )
        )
        dim_constraints.add(
            60 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 360 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 540
            <= 2147483647
        )
        dim_constraints.add(
            Ne(
                60 * (FloorDiv(s0, 2)) * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 360 * FloorDiv(s0, 2) * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 540 * (FloorDiv(s0, 2)),
                0,
            )
        )
        dim_constraints.add(
            1
            < 60 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 360 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 540
        )
        dim_constraints.add(
            (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 6 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 9
            <= 2147483647
        )
        dim_constraints.add(
            Ne(
                (FloorDiv(s0, 2)) * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
                - 6 * FloorDiv(s0, 2) * FloorDiv((FloorDiv(s1, 2) - 1), 8)
                + 9 * (FloorDiv(s0, 2)),
                0,
            )
        )
        dim_constraints.add(
            1
            < (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 6 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 9
        )
        dim_constraints.add(
            (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 6 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 9
            > 1
        )
        dim_constraints.add(
            60 * (FloorDiv((FloorDiv(s1, 2) - 1), 8)) ** 2
            - 360 * FloorDiv((FloorDiv(s1, 2) - 1), 8)
            + 540
            > 1
        )
        dim_constraints.add(s0 >= 2)
        dim_constraints.add(s1 >= 2)
        dim_constraints.add(s6 >= 2)
        dim_constraints.add(s5 >= 2)

        dim_constraints.solve()
        self.assertEqual(
            dim_constraints._static_results,
            {
                "L['c'].size()[0] == 8",
                "L['d'].size()[0] == 8",
                "L['a'].size()[2] == 96",
                "L['f'].size()[1] == 1",
                "L['a'].size()[3] == 96",
                "L['b'].size()[2] == 3",
                "L['b'].size()[1] == 22",
                "L['b'].size()[0] == 8",
                "L['a'].size()[1] == 22",
                "L['a'].size()[0] == 8",
            },
        )
        self.assertEqual(
            dim_constraints._dynamic_results,
            {
                "2 <= L['c'].size()[1]",
                "L['d'].size()[1] == L['c'].size()[1]",
                "L['e'].size()[1] == L['c'].size()[1]",
            },
        )


class TestGuardsExpressions(TestCase):
    """
    Tests the guards-related methods used by the inductor FX graph cache.
    """

    def test_guards_gt_lt(self):
        shape_env = ShapeEnv()
        s0 = create_symint(shape_env, 6)
        s1 = create_symint(shape_env, 7)
        s2 = create_symint(shape_env, 5)

        guard_int(sym_int(s0 > 5))
        guard_int(sym_int(s0 < 7))

        guards = shape_env.produce_guards_expression([s0])

        self.assertTrue(shape_env.evaluate_guards_expression(guards, [hint_int(s0)]))
        self.assertFalse(shape_env.evaluate_guards_expression(guards, [hint_int(s1)]))
        self.assertFalse(shape_env.evaluate_guards_expression(guards, [hint_int(s2)]))

    def test_guards_float_print(self):
        shape_env = ShapeEnv()
        s0 = create_symint(shape_env, 3)
        guard_bool(2 / s0 == 2 / 3)
        guards = shape_env.produce_guards_expression([s0])
        self.assertTrue(shape_env.evaluate_guards_expression(guards, [hint_int(s0)]))

    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_guard_or_true(self):
        from torch.fx.experimental.symbolic_shapes import guard_or_true

        def func(a, b):
            x = a.item()
            if guard_or_true(x == 1):
                return b * 10
            else:
                return b * 20

        # eager.
        self.assertEqual(func(torch.tensor([1]), torch.tensor([1])), torch.tensor([10]))
        self.assertEqual(func(torch.tensor([2]), torch.tensor([1])), torch.tensor([20]))

        # compile with unbacked.
        unbacked_func = torch.compile(func, dynamic=True, fullgraph=True)
        a = torch.tensor([1])
        b = torch.tensor([1])
        unbacked_func(a, b)

        # always return b*10
        self.assertEqual(
            unbacked_func(torch.tensor([1]), torch.tensor([1])), torch.tensor([10])
        )
        self.assertEqual(
            unbacked_func(torch.tensor([2]), torch.tensor([1])), torch.tensor([10])
        )

        # Test that statically known true works.
        def func2(a, b):
            x = a.item()
            if guard_or_true(x != x):
                return b * 10
            else:
                return b * 20

        unbacked_func2 = torch.compile(func2, dynamic=True, fullgraph=True)
        a = torch.tensor([1])
        b = torch.tensor([1])
        unbacked_func2(a, b)
        # always return b*20
        self.assertEqual(
            unbacked_func2(torch.tensor([1]), torch.tensor([1])), torch.tensor([20])
        )
        self.assertEqual(
            unbacked_func2(torch.tensor([2]), torch.tensor([1])), torch.tensor([20])
        )

        # Test backed_size_oblivious
        with torch.fx.experimental._config.patch("backed_size_oblivious", True):

            def func3(a, b):
                if guard_or_true(a.size()[0] != 9):
                    return b * 10
                else:
                    return b * 20

            compiled = torch.compile(func3, dynamic=True, fullgraph=True)
            a = torch.rand(9, 2)
            b = torch.rand(3, 4)

            self.assertEqual(func3(a, b), b * 20)
            self.assertEqual(compiled(a, b), b * 10)

    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_guard_or_false(self):
        from torch.fx.experimental.symbolic_shapes import guard_or_false

        def func(a, b):
            x = a.item()
            if guard_or_false(x == 1):
                return b * 10
            else:
                return b * 20

        # eager.
        self.assertEqual(func(torch.tensor([1]), torch.tensor([1])), torch.tensor([10]))
        self.assertEqual(func(torch.tensor([2]), torch.tensor([1])), torch.tensor([20]))

        # compile with unbacked.
        unbacked_func = torch.compile(func, dynamic=True, fullgraph=True)
        a = torch.tensor([1])
        b = torch.tensor([1])
        unbacked_func(a, b)

        # always return b*20
        self.assertEqual(
            unbacked_func(torch.tensor([1]), torch.tensor([1])), torch.tensor([20])
        )
        self.assertEqual(
            unbacked_func(torch.tensor([2]), torch.tensor([1])), torch.tensor([20])
        )

        # Test that statically known true works.
        def func2(a, b):
            x = a.item()
            if guard_or_false(x == x):
                return b * 10
            else:
                return b * 20

        unbacked_func2 = torch.compile(func2, dynamic=True, fullgraph=True)
        a = torch.tensor([1])
        b = torch.tensor([1])
        unbacked_func2(a, b)
        # always return b*10
        self.assertEqual(
            unbacked_func2(torch.tensor([1]), torch.tensor([1])), torch.tensor([10])
        )
        self.assertEqual(
            unbacked_func2(torch.tensor([2]), torch.tensor([1])), torch.tensor([10])
        )

        # Test backed_size_oblivious
        with torch.fx.experimental._config.patch("backed_size_oblivious", True):

            def func3(a, b):
                if guard_or_false(a.size()[0] == 9):
                    return b * 10
                else:
                    return b * 20

            compiled = torch.compile(func3, dynamic=True, fullgraph=True)
            a = torch.rand(9, 2)
            b = torch.rand(3, 4)

            self.assertEqual(func3(a, b), b * 10)
            self.assertEqual(compiled(a, b), b * 20)

    def test_guards_float_div(self):
        shape_env = ShapeEnv()
        s0 = create_symint(shape_env, 8)
        s1 = create_symint(shape_env, 7)

        guard_int(sym_int(s0 / 2.0))
        guards = shape_env.produce_guards_expression([s0])

        self.assertIn("math.trunc(", guards)
        self.assertIn("float(", guards)
        self.assertTrue(shape_env.evaluate_guards_expression(guards, [hint_int(s0)]))
        self.assertFalse(shape_env.evaluate_guards_expression(guards, [hint_int(s1)]))

    @unittest.skipIf(
        TEST_XPU, "Skipped on XPU"
    )  # https://github.com/intel/torch-xpu-ops/issues/2169"
    @skipIfTorchDynamo("Attempt to trace generator")
    @torch.fx.experimental._config.patch("use_duck_shape", False)
    def test_size_comparison_no_recompile(self):
        """
        Test that size comparisons don't cause recompilation.
        When comparing x.size() == b.size() with different sizes,
        the compiled function should only compile once.
        We should not guard in sizes of the inner elements.
        """
        cnt = CompileCounter()

        @torch.compile(fullgraph=True, dynamic=True, backend=cnt)
        def f(x, b):
            if x.size() == b.size():
                return x
            return x * 2

        # First call: shapes differ (1, 2) vs (2, 4, 9), so if branch is False
        f(torch.rand(10, 2), torch.rand(20, 4, 9))

        # Second call: shapes differ again (1, 2) vs (1, 4, 9), so if branch is False
        f(torch.rand(10, 2), torch.rand(10, 4, 9))

        # Should only compile once despite different input shapes
        self.assertEqual(
            cnt.frame_count,
            1,
            f"Expected 1 compilation, got {cnt.frame_count}. "
            f"Size comparison should not cause recompilation.",
        )

    def test_remove_symbols_without_guarding(self):
        from torch._functorch.partitioners import _remove_symbols_without_guarding

        shape_env = ShapeEnv()

        x = create_fake_tensor_with_dynamic_size(
            torch.randn(5, 8),
            shape_env,
            dynamic_sizes=[
                DimDynamic.DYNAMIC,
                DimDynamic.DYNAMIC,
            ],
            dynamic_strides=[
                DimDynamic.INFER_STRIDE,
                DimDynamic.INFER_STRIDE,
            ],
        )

        self.assertEqual(f"{x.stride()}", "(s49, 1)")
        self.assertEqual(f"{x.shape}", "torch.Size([s26, s49])")

        x_clean = _remove_symbols_without_guarding(x, 4096)

        self.assertEqual(f"{x_clean.stride()}", "(8, 1)")
        self.assertEqual(f"{x_clean.shape}", "torch.Size([5, 8])")


def custom_pass(graph: torch.fx.Graph) -> torch.fx.Graph:
    for node in graph.nodes:
        if node.name == "arg3_1":
            assert node.meta["val"].size()[0] == 2
    return graph


class TestUnbacked(TestCase):
    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/156135")
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @parametrize("backend", ["inductor", "eager"])
    def test_deferred_neq_assert(self, backend):
        @torch.compile(fullgraph=True, backend=backend)
        def func(a):
            torch._check(a.item() != 5)
            return a.item() * 10

        func(torch.tensor([100]))

        with self.assertRaises(RuntimeError):
            func(torch.tensor([5]))

    # Test a situation where we generate a runtime assert i.e: u1==s1, then we specialize s1
    # later on to a constant.
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @parametrize("backend", ["inductor", "eager"])
    def test_post_specialize_runtime_assert1(self, backend):
        @torch.compile(dynamic=True, backend=backend)
        def func(x, y):
            u0 = y.item()
            s0 = x.size()[0]
            s1 = x.size()[1]
            torch._check(u0 + s0 + s1 == 102)
            assert s0 == 2
            return x * 10

        func(torch.rand(2, 50), torch.tensor([50]))
        with self.assertRaises(RuntimeError):
            func(torch.rand(2, 50), torch.tensor([51]))

    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @torch._inductor.config.patch(post_grad_custom_pre_pass=custom_pass)
    @parametrize("backend", ["inductor", "eager"])
    def test_post_specialize_runtime_assert2(self, backend):
        @torch.compile(dynamic=True, backend=backend)
        def func(x, y):
            u0 = y.item()
            s0 = x.size()[0]
            s1 = x.size()[1]
            torch._check(u0 + s0 + s1 == 102)
            return x * 10

        func(torch.rand(2, 50), torch.tensor([50]))
        with self.assertRaises(RuntimeError):
            func(torch.rand(2, 50), torch.tensor([51]))

    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/156135")
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @parametrize("backend", ["inductor", "eager"])
    def test_deferred_sym_or_assert(self, backend):
        @torch.compile(fullgraph=True, backend=backend)
        def func(a, b):
            torch._check(operator.or_(a.item() == 5, b.item() == 5))
            return a.item() * 10

        func(torch.tensor([5]), torch.tensor([100]))
        func(torch.tensor([100]), torch.tensor([5]))

    def test_has_free_symbols(self):
        self.assertFalse(has_free_symbols(sympy.S.true))
        self.assertFalse(has_free_symbols(sympy.Max(1, 10, evaluate=False)))

        self.assertFalse(has_free_symbols(sympy.sympify("1")))
        self.assertFalse(has_free_symbols(sympy.sympify("1.1")))
        self.assertTrue(has_free_symbols(sympy.sympify("a")))
        self.assertTrue(has_free_symbols(sympy.sympify("a*2")))
        self.assertTrue(has_free_symbols(sympy.sympify("a+b")))

    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/156135")
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @parametrize("backend", ["inductor", "eager"])
    def test_deferred_sym_eq_assert(self, backend):
        @torch.compile(fullgraph=True, backend=backend)
        def func(a, b):
            torch._check(b.item() == 5)
            return a * 10

        func(torch.tensor([5]), torch.tensor([5]))
        with self.assertRaises(RuntimeError):
            func(torch.tensor([100]), torch.tensor([1]))

    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @parametrize("backend", ["inductor", "eager"])
    @skipIfTorchDynamo("mark_unbacked is not traceable")
    def test_deferred_with_unbacked_input(self, backend):
        @torch.compile(fullgraph=True, dynamic=True, backend=backend)
        def func(a, b):
            torch._check(a.size()[0] == b.size()[0])
            return a * 10

        a = torch.rand(1, 1)
        b = torch.rand(1, 1)
        torch._dynamo.decorators.mark_unbacked(a, 0)
        torch._dynamo.decorators.mark_unbacked(b, 0)
        func(a, b)

        # inductor adds the check sometimes itself so it will be reflected
        # as AssertionError.
        with self.assertRaises((AssertionError, RuntimeError)):
            func(a, torch.rand(2, 1))

    @pytest.mark.xfail(reason="https://github.com/pytorch/pytorch/issues/163785")
    @skipIfTorchDynamo("mark_unbacked is not traceable")
    def test_do_not_guard_unbacked_inputs(self):
        @torch.compile(fullgraph=True, dynamic=True, backend="inductor")
        def func(a, b):
            a.expand(b.shape)
            return a * 10

        a = torch.rand(1, 1)
        b = torch.rand(1, 1)

        torch._dynamo.decorators.mark_unbacked(a, 0)
        torch._dynamo.decorators.mark_unbacked(a, 1)
        torch._dynamo.decorators.mark_unbacked(b, 0)
        torch._dynamo.decorators.mark_unbacked(b, 1)

        log_stream, ctx = logs_to_string("torch._dynamo.guards", "guards")
        with ctx():
            func(a, b)
            func(torch.rand(4, 5), torch.rand(4, 5))

        guards = "\n".join(log_stream.getvalue().strip().split("\n")[4:]).strip()
        self.assertFalse("SYMBOLIC_SHAPE_GUARD" in guards)

    @skipIfTorchDynamo("mark_unbacked is not traceable")
    def test_div_unbacked_eq_input_tensors(self):
        @torch.compile(fullgraph=True)
        def func(a, b):
            x = a.size()[0]
            y = b.size()[0]
            torch._check(x == y)
            if x // y == 1:
                a = a * 10
            if 2 * x // y == 2:
                a = a * 20
            return a

        a = torch.randn(10, 10)
        b = torch.randn(10, 20)

        torch._dynamo.decorators.mark_unbacked(a, 0)
        torch._dynamo.decorators.mark_unbacked(b, 0)
        func(a, b)

    @torch.compiler.config.patch(unbacked_sources="L['x'],L['y']")
    def test_div_unbacked_eq_input_ints(self):
        @torch.compile(fullgraph=True)
        def func(x, y):
            a = torch.rand(1)
            torch._check(x == y)
            if x // y == 1:
                a = a * 10
            if 2 * x // y == 2:
                a = a * 20
            return a

        func(10, 10)

    @skipIfTorchDynamo("mark_unbacked is not traceable")
    @torch.compiler.config.patch(unbacked_sources="L['y']")
    def test_div_unbacked_eq_globals(self):
        tensor = torch.rand(10, 44)
        y = 10

        @torch.compile(fullgraph=True, dynamic=True)
        def func():
            a = torch.rand(1)
            x = tensor.size()[0]
            torch._check(x == y)
            if x // y == 1:
                a = a * 10
            if 2 * x // y == 2:
                a = a * 20
            return a

        torch._dynamo.decorators.mark_unbacked(tensor, 0)
        func()

    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_div_unbacked_eq_item(self):
        @torch.compile(fullgraph=True)
        def func(a, b):
            x = a.item()
            y = b.item()
            torch._check(x == y)
            # TODO we should not need those torch checks.
            torch._check(x // y == 1)
            torch._check(2 * x // y == 2)
            if x // y == 1:
                a = a * 10
            if 2 * x // y == 2:
                a = a * 20
            return a

        a = torch.tensor([1])
        b = torch.tensor([1])
        func(a, b)


class TestUbackedOps(TestCase):
    @fresh_cache()
    @skipIfTorchDynamo("not allowed to trace mark_unbacked")
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_unbacked_reshape1(self):
        cnt = CompileCounterWithBackend("inductor")

        # Reshape happens in place reshape (no-clone)
        # reshape u1 -> (u0*u0)
        def func(x, y):
            f = y.item()
            t1 = x.view((f, f))
            t2 = x.reshape((f, f))
            t3 = torch._ops.ops.aten.view_copy(x, (f, f))
            return t1 * 10, t2 * 10, t3

        compiled_func = torch.compile(
            fullgraph=True,
            backend=cnt,
            dynamic=True,
        )(func)

        # create a non-contiguous with data being even numbers in [0:cnt-1]
        # and reshape it into sqrt(cnt)*sqrt(cnt)
        def make_non_contiguous_tensor_and_test(cnt):
            # create a non-contiguous tensor x that is skipping odd indices.
            x = torch.arange(cnt * 2)
            x = x.as_strided((x.size()[0] // 2,), (2,))

            torch._dynamo.decorators.mark_unbacked(x, 0)
            sz = torch.tensor([int(math.sqrt(cnt))])
            compiled_result = compiled_func(x, sz)
            eager_result = func(x, sz)
            self.assertEqual(compiled_result, eager_result)

        log_stream, ctx = logs_to_string(
            "torch._functorch._aot_autograd.graph_capture", "aot_graphs"
        )
        with ctx():
            make_non_contiguous_tensor_and_test(4)
        aot_graphs = "\n".join(log_stream.getvalue().strip().split("\n")[4:]).strip()
        self.assertExpectedInline(
            aot_graphs,
            """\
def forward(self, arg0_1: "i64[1][1]cpu", arg1_1: "Sym(u1)", arg2_1: "Sym(s7)", arg3_1: "i64[u1][s7]cpu"):
        ge_1: "Sym(u1 >= 0)" = arg1_1 >= 0
        _assert_scalar = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u1 >= 0 on node 'ge'");  ge_1 = _assert_scalar = None
        _local_scalar_dense: "Sym(u0)" = torch.ops.aten._local_scalar_dense.default(arg0_1);  arg0_1 = None
        ge_2: "Sym(u0 >= 0)" = _local_scalar_dense >= 0
        _assert_scalar_1 = torch.ops.aten._assert_scalar.default(ge_2, "Runtime assertion failed for expression u0 >= 0 on node 'ge_1'");  ge_2 = _assert_scalar_1 = None
        pow_1: "Sym(u0**2)" = _local_scalar_dense ** 2
        eq: "Sym(Eq(u1, u0**2))" = arg1_1 == pow_1;  arg1_1 = pow_1 = None
        _assert_scalar_2 = torch.ops.aten._assert_scalar.default(eq, "Runtime assertion failed for expression Eq(u1, u0**2) on node 'eq'");  eq = _assert_scalar_2 = None
        view: "i64[u0, u0][s7*u0, s7]cpu" = torch.ops.aten.view.default(arg3_1, [_local_scalar_dense, _local_scalar_dense])
        view_1: "i64[u0, u0][s7*u0, s7]cpu" = torch.ops.aten.view.default(arg3_1, [_local_scalar_dense, _local_scalar_dense])
        view_2: "i64[u0, u0][s7*u0, s7]cpu" = torch.ops.aten.view.default(arg3_1, [_local_scalar_dense, _local_scalar_dense]);  arg3_1 = _local_scalar_dense = None
        clone: "i64[u0, u0][Max(1, u0), 1]cpu" = torch.ops.aten.clone.default(view_2);  view_2 = None
        mul_11: "i64[u0, u0][Max(1, u0), 1]cpu" = torch.ops.aten.mul.Tensor(view, 10);  view = None
        mul_14: "i64[u0, u0][Max(1, u0), 1]cpu" = torch.ops.aten.mul.Tensor(view_1, 10);  view_1 = None
        return (mul_11, mul_14, clone)""",  # noqa: B950
            ignore_comments=True,
            ignore_empty_lines=True,
        )

        make_non_contiguous_tensor_and_test(49)
        self.assertEqual(cnt.frame_count, 1)

        # Pass in a contiguous tensor, it will recompile due to stride being 1 (0/1 specialization).
        # marking strides unbacked would have avoided the recompilation here.
        x = torch.arange(100)
        torch._dynamo.decorators.mark_unbacked(x, 0)

        log_stream, ctx = logs_to_string(
            "torch._functorch._aot_autograd.graph_capture", "aot_graphs"
        )
        with ctx():
            compiled_result = compiled_func(x, torch.tensor([10]))
            eager_result = func(x, torch.tensor([10]))
            self.assertEqual(compiled_result, eager_result)
            self.assertEqual(cnt.frame_count, 2)

        aot_graphs = "\n".join(log_stream.getvalue().strip().split("\n")[4:]).strip()
        self.assertExpectedInline(
            aot_graphs,
            """\
def forward(self, arg0_1: "i64[1][1]cpu", arg1_1: "Sym(u1)", arg2_1: "i64[u1][1]cpu"):
        ge_1: "Sym(u1 >= 0)" = arg1_1 >= 0
        _assert_scalar = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u1 >= 0 on node 'ge'");  ge_1 = _assert_scalar = None
        _local_scalar_dense: "Sym(u0)" = torch.ops.aten._local_scalar_dense.default(arg0_1);  arg0_1 = None
        ge_2: "Sym(u0 >= 0)" = _local_scalar_dense >= 0
        _assert_scalar_1 = torch.ops.aten._assert_scalar.default(ge_2, "Runtime assertion failed for expression u0 >= 0 on node 'ge_1'");  ge_2 = _assert_scalar_1 = None
        pow_1: "Sym(u0**2)" = _local_scalar_dense ** 2
        eq: "Sym(Eq(u1, u0**2))" = arg1_1 == pow_1;  arg1_1 = pow_1 = None
        _assert_scalar_2 = torch.ops.aten._assert_scalar.default(eq, "Runtime assertion failed for expression Eq(u1, u0**2) on node 'eq'");  eq = _assert_scalar_2 = None
        view: "i64[u0, u0][Max(1, u0), 1]cpu" = torch.ops.aten.view.default(arg2_1, [_local_scalar_dense, _local_scalar_dense])
        view_1: "i64[u0, u0][Max(1, u0), 1]cpu" = torch.ops.aten.view.default(arg2_1, [_local_scalar_dense, _local_scalar_dense])
        view_2: "i64[u0, u0][Max(1, u0), 1]cpu" = torch.ops.aten.view.default(arg2_1, [_local_scalar_dense, _local_scalar_dense]);  arg2_1 = _local_scalar_dense = None
        clone: "i64[u0, u0][Max(1, u0), 1]cpu" = torch.ops.aten.clone.default(view_2);  view_2 = None
        mul_6: "i64[u0, u0][Max(1, u0), 1]cpu" = torch.ops.aten.mul.Tensor(view, 10);  view = None
        mul_9: "i64[u0, u0][Max(1, u0), 1]cpu" = torch.ops.aten.mul.Tensor(view_1, 10);  view_1 = None
        return (mul_6, mul_9, clone)""",  # noqa: B950
            ignore_comments=True,
            ignore_empty_lines=True,
        )

        x = torch.arange(25)
        compiled_result = compiled_func(x, torch.tensor([5]))
        eager_result = func(x, torch.tensor([5]))
        self.assertEqual(cnt.frame_count, 2)

    @skipIfTorchDynamo("not allowed to trace mark_unbacked")
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_unbacked_reshape2(self):
        cnt = CompileCounterWithBackend("inductor")

        # This reshape requires a clone when the input is not contiguous and we can't compute strides.
        # reshape (u2, u3) -> (u0, u1)
        def func(x, y):
            u0, u1 = y.tolist()

            result1 = torch.reshape(x, (u0, u1))
            return result1 * 10

        compiled_func = torch.compile(fullgraph=True, backend=cnt, dynamic=True)(func)

        x = torch.randn(10, 10)
        # make x not contiguous.
        x = x.t_()
        torch._dynamo.decorators.mark_unbacked(x, 0)
        torch._dynamo.decorators.mark_unbacked(x, 1)

        log_stream, ctx = logs_to_string(
            "torch._functorch._aot_autograd.graph_capture", "aot_graphs"
        )
        with ctx():
            result_eager = func(x, torch.tensor([5, 20]))
            result_compiled = compiled_func(x, torch.tensor([5, 20]))
            self.assertEqual(result_compiled, result_eager)
            self.assertEqual(cnt.frame_count, 1)

        aot_graphs = "\n".join(log_stream.getvalue().strip().split("\n")[4:]).strip()
        self.assertExpectedInline(
            aot_graphs,
            """\
def forward(self, arg0_1: "i64[2][1]cpu", arg1_1: "Sym(u2)", arg2_1: "Sym(u3)", arg3_1: "f32[u2, u3][1, u2]cpu"):
        ge_1: "Sym(u2 >= 0)" = arg1_1 >= 0
        _assert_scalar = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u2 >= 0 on node 'ge'");  ge_1 = _assert_scalar = None
        ge_3: "Sym(u3 >= 0)" = arg2_1 >= 0
        _assert_scalar_1 = torch.ops.aten._assert_scalar.default(ge_3, "Runtime assertion failed for expression u3 >= 0 on node 'ge_1'");  ge_3 = _assert_scalar_1 = None
        select: "i64[][]cpu" = torch.ops.aten.select.int(arg0_1, 0, 0)
        _local_scalar_dense: "Sym(u0)" = torch.ops.aten._local_scalar_dense.default(select);  select = None
        ge_4: "Sym(u0 >= 0)" = _local_scalar_dense >= 0
        _assert_scalar_2 = torch.ops.aten._assert_scalar.default(ge_4, "Runtime assertion failed for expression u0 >= 0 on node 'ge_2'");  ge_4 = _assert_scalar_2 = None
        sym_sum: "Sym(u0 + 1)" = torch.sym_sum((1, _local_scalar_dense))
        gt: "Sym(u0 + 1 > 0)" = sym_sum > 0;  sym_sum = None
        _assert_scalar_3 = torch.ops.aten._assert_scalar.default(gt, "Runtime assertion failed for expression 0 < u0 + 1 on node 'gt'");  gt = _assert_scalar_3 = None
        select_1: "i64[][]cpu" = torch.ops.aten.select.int(arg0_1, 0, 1);  arg0_1 = None
        _local_scalar_dense_1: "Sym(u1)" = torch.ops.aten._local_scalar_dense.default(select_1);  select_1 = None
        ge_5: "Sym(u1 >= 0)" = _local_scalar_dense_1 >= 0
        _assert_scalar_4 = torch.ops.aten._assert_scalar.default(ge_5, "Runtime assertion failed for expression u1 >= 0 on node 'ge_3'");  ge_5 = _assert_scalar_4 = None
        sym_sum_1: "Sym(u1 + 1)" = torch.sym_sum((1, _local_scalar_dense_1))
        gt_1: "Sym(u1 + 1 > 0)" = sym_sum_1 > 0;  sym_sum_1 = None
        _assert_scalar_5 = torch.ops.aten._assert_scalar.default(gt_1, "Runtime assertion failed for expression 0 < u1 + 1 on node 'gt_1'");  gt_1 = _assert_scalar_5 = None
        mul: "Sym(u2*u3)" = arg1_1 * arg2_1;  arg1_1 = arg2_1 = None
        mul_1: "Sym(u0*u1)" = _local_scalar_dense * _local_scalar_dense_1
        eq: "Sym(Eq(u2*u3, u0*u1))" = mul == mul_1;  mul = mul_1 = None
        _assert_scalar_6 = torch.ops.aten._assert_scalar.default(eq, "Runtime assertion failed for expression Eq(u2*u3, u0*u1) on node 'eq'");  eq = _assert_scalar_6 = None
        clone: "f32[u2, u3][Max(1, u3), 1]cpu" = torch.ops.aten.clone.default(arg3_1, memory_format = torch.contiguous_format);  arg3_1 = None
        view: "f32[u0, u1][Max(1, u1), 1]cpu" = torch.ops.aten.view.default(clone, [_local_scalar_dense, _local_scalar_dense_1]);  clone = _local_scalar_dense = _local_scalar_dense_1 = None
        mul_21: "f32[u0, u1][Max(1, u1), 1]cpu" = torch.ops.aten.mul.Tensor(view, 10);  view = None
        return (mul_21,)""",  # noqa: B950
            ignore_comments=True,
            ignore_empty_lines=True,
        )

        result_eager = func(x, torch.tensor([2, 50]))
        result_compiled = compiled_func(x, torch.tensor([2, 50]))
        self.assertEqual(result_compiled, result_eager)
        self.assertEqual(cnt.frame_count, 1)

        x = torch.randn(4, 4).t_()
        result_eager = func(x, torch.tensor([2, 8]))
        result_compiled = compiled_func(x, torch.tensor([2, 8]))
        self.assertEqual(result_compiled, result_eager)
        self.assertEqual(cnt.frame_count, 1)

        # Pass a contiguous tensor. A recompilation will happen due to 0/1 specialization on stride.
        log_stream, ctx = logs_to_string(
            "torch._functorch._aot_autograd.graph_capture", "aot_graphs"
        )
        with ctx():
            # This used to hit could guard on data-dependent expression Eq(10, u3) x.stride[0]==10. and x.size()=[u2, u3].
            # but not anymore since we use  contiguous_or_false .
            # We need a way to mark strides unbacked to avoid the recompilation here.
            x = torch.randn(10, 10)
            torch._dynamo.decorators.mark_unbacked(x, 0)
            torch._dynamo.decorators.mark_unbacked(x, 1)

        aot_graphs = "\n".join(log_stream.getvalue().strip().split("\n")[4:]).strip()
        self.assertExpectedInline(
            aot_graphs,
            """""",  # noqa: B950
            ignore_comments=True,
            ignore_empty_lines=True,
        )

        result_compiled = compiled_func(x, torch.tensor([2, 50]))
        result_eager = func(x, torch.tensor([2, 50]))

        self.assertEqual(result_compiled, result_eager)
        self.assertEqual(cnt.frame_count, 2)

        x = torch.randn(4, 4)

        result_eager = func(x, torch.tensor([2, 8]))
        result_compiled = compiled_func(x, torch.tensor([2, 8]))
        self.assertEqual(result_compiled, result_eager)
        self.assertEqual(cnt.frame_count, 2)

    @fresh_cache()
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_unbacked_slice(self):
        from torch.fx.experimental.symbolic_shapes import statically_known_true

        # standard slice
        def f1(x, xs):
            u0, u1 = xs.tolist()
            # in this test we add the torch checks not to avoid DDE but to ensure
            # that we pick specific path during compilation.
            torch._check(u0 >= 0)
            torch._check(u0 <= x.size(0))
            torch._check(u1 >= 0)
            torch._check(u1 <= x.size(0))
            torch._check(u0 <= u1)
            out = x[u0:u1]
            assert statically_known_true(out.size(0) == (u1 - u0))
            return out

        x, xs = torch.randn(10), torch.tensor([3, 6])
        fn1 = torch.compile(f1, fullgraph=True, backend="inductor")
        self.assertEqual(fn1(x, xs).size(0), 3)
        self.assertTrue(torch.allclose(fn1(x, xs), f1(x, xs)))
        with self.assertRaises(RuntimeError):
            fn1(x, torch.tensor([-1, 5]))

        # known negative slice
        def f2(x, n):
            u0 = n.item()
            torch._check(u0 > 1)
            torch._check(u0 <= x.size(0))
            out = x[-u0:]
            assert statically_known_true(out.size(0) == u0)
            return out

        x, n = torch.randn(10), torch.tensor([5])
        fn2 = torch.compile(f2, fullgraph=True, backend="inductor")
        self.assertEqual(fn2(x, n).size(0), 5)
        self.assertTrue(torch.allclose(fn2(x, n), f2(x, n)))
        with self.assertRaises(RuntimeError):
            fn2(x, torch.tensor([-5]))

        # general case: no known info
        def f3(x, xs):
            u0, u1 = xs.tolist()
            return x[u0:u1]

        log_stream, ctx = logs_to_string(
            "torch._inductor.compile_fx", "post_grad_graphs"
        )
        cnts = CompileCounterWithBackend("inductor")
        x, xs = torch.randn(10), torch.tensor([3, 6])
        with ctx():
            fn3 = torch.compile(f3, fullgraph=True, backend=cnts)
            xs = torch.tensor([-9, -1])  # negative case
            self.assertTrue(torch.allclose(fn3(x, xs), f3(x, xs)))
            xs = torch.tensor([-1000, 1000])  # out of bounds
            self.assertTrue(torch.allclose(fn3(x, xs), f3(x, xs)))
            xs = torch.tensor([2, -2])  # mixed
            self.assertTrue(torch.allclose(fn3(x, xs), f3(x, xs)))
            self.assertEqual(cnts.frame_count, 1)

        aot_graphs = "\n".join(log_stream.getvalue().strip().split("\n")[4:]).strip()
        self.assertExpectedInline(
            aot_graphs,
            """\
        select: "i64[][]cpu" = torch.ops.aten.select.int(arg0_1, 0, 0)
        _local_scalar_dense: "Sym(u0)" = torch.ops.aten._local_scalar_dense.default(select);  select = None
        select_1: "i64[][]cpu" = torch.ops.aten.select.int(arg0_1, 0, 1);  arg0_1 = None
        _local_scalar_dense_1: "Sym(u1)" = torch.ops.aten._local_scalar_dense.default(select_1);  select_1 = None
        slice_1: "f32[u2][1]cpu" = torch.ops.aten.slice.Tensor(arg1_1, 0, _local_scalar_dense, _local_scalar_dense_1);  arg1_1 = _local_scalar_dense = _local_scalar_dense_1 = None
        sym_size_int: "Sym(u2)" = torch.ops.aten.sym_size.int(slice_1, 0)
        ge_1: "Sym(u2 >= 0)" = sym_size_int >= 0
        _assert_scalar = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u2 >= 0 on node 'ge'");  ge_1 = _assert_scalar = None
        le: "Sym(u2 <= 10)" = sym_size_int <= 10;  sym_size_int = None
        _assert_scalar_1 = torch.ops.aten._assert_scalar.default(le, "Runtime assertion failed for expression u2 <= 10 on node 'le'");  le = _assert_scalar_1 = None
        sym_storage_offset_default: "Sym(u3)" = torch.ops.aten.sym_storage_offset.default(slice_1)
        ge_2: "Sym(u3 >= 0)" = sym_storage_offset_default >= 0;  sym_storage_offset_default = None
        _assert_scalar_2 = torch.ops.aten._assert_scalar.default(ge_2, "Runtime assertion failed for expression u3 >= 0 on node 'ge_1'");  ge_2 = _assert_scalar_2 = None
        return (slice_1,)""",  # noqa: B950
            ignore_comments=True,
            ignore_empty_lines=True,
        )

    @fresh_cache()
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @torch._inductor.config.patch("cpp_wrapper", True)
    def test_unbacked_slice_cpp_wrapper(self):
        self.test_unbacked_slice()

    @fresh_cache()
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_unbacked_slice_with_step(self):
        def f1(x, xs):
            u0, u1 = xs.tolist()
            out = x[u0:u1:5]
            return out

        x, xs = torch.randn(10), torch.tensor([2, -2])
        fn1 = torch.compile(f1, fullgraph=True, backend="inductor")
        self.assertTrue(torch.allclose(fn1(x, xs), f1(x, xs)))

    @fresh_cache()
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @torch._inductor.config.patch("cpp_wrapper", True)
    def test_unbacked_slice_with_step_cpp_wrapper(self):
        self.test_unbacked_slice_with_step()

    @fresh_cache()
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_slice_with_tensor_indices(self):
        for d in [True, False]:
            # Test slicing with tensor start/stop/step on RHS (reading)

            # Test 1: Basic slice with tensor start and stop
            def f1(x, start_t, stop_t):
                return x[start_t:stop_t]

            x = torch.randn(20)
            start_t = torch.tensor(5)
            stop_t = torch.tensor(15)
            fn1 = torch.compile(f1, fullgraph=True, dynamic=d, backend="inductor")
            self.assertTrue(
                torch.allclose(fn1(x, start_t, stop_t), f1(x, start_t, stop_t))
            )

            # Test 2: Slice with tensor step
            def f2(x, start_t, stop_t, step_t):
                return x[start_t:stop_t:step_t]

            step_t = torch.tensor(2)
            fn2 = torch.compile(f2, fullgraph=True, dynamic=d, backend="inductor")
            self.assertTrue(
                torch.allclose(
                    fn2(x, start_t, stop_t, step_t), f2(x, start_t, stop_t, step_t)
                )
            )

            # Test 3: Slice with only tensor start
            def f3(x, start_t):
                return x[start_t:]

            fn3 = torch.compile(f3, fullgraph=True, dynamic=d, backend="inductor")
            self.assertTrue(torch.allclose(fn3(x, start_t), f3(x, start_t)))

            # Test 4: Slice with only tensor stop
            def f4(x, stop_t):
                return x[:stop_t]

            fn4 = torch.compile(f4, fullgraph=True, dynamic=d, backend="inductor")
            self.assertTrue(torch.allclose(fn4(x, stop_t), f4(x, stop_t)))

            # Test 5: Negative indices with tensors
            def f5(x, start_t):
                return x[start_t:-1]

            start_t_neg = torch.tensor(-10)
            fn5 = torch.compile(f5, fullgraph=True, dynamic=d, backend="inductor")
            self.assertTrue(torch.allclose(fn5(x, start_t_neg), f5(x, start_t_neg)))

            # Test 6: Multidimensional slice with tensor indices
            def f6(x, start_t, stop_t):
                return x[:, start_t:stop_t]

            x_2d = torch.randn(10, 20)
            fn6 = torch.compile(f6, fullgraph=True, dynamic=d, backend="inductor")
            self.assertTrue(
                torch.allclose(fn6(x_2d, start_t, stop_t), f6(x_2d, start_t, stop_t))
            )

    @fresh_cache()
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @torch._inductor.config.patch("cpp_wrapper", True)
    def test_slice_with_tensor_indices_cpp_wrapper(self):
        self.test_slice_with_tensor_indices()

    @fresh_cache()
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_tensor_split(self):
        def f1(x, xs):
            xs = torch.tensor(xs.tolist())
            return torch.tensor_split(x, xs)

        x = torch.randn(20)
        xs = torch.tensor([5, 10, 15])
        fn = torch.compile(f1, fullgraph=True, backend="inductor")

        def compare(x, xs):
            for i, j in zip(f1(x, xs), fn(x, xs)):
                self.assertTrue(torch.allclose(i, j))

        compare(x, xs)
        xs = torch.tensor([-15, 9, 10, 11])
        compare(x, xs)
        xs = torch.tensor([-15, -10, -5, -2])
        compare(x, xs)

    @fresh_cache()
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @torch._inductor.config.patch("cpp_wrapper", True)
    def test_tensor_split_cpp_wrapper(self):
        self.test_tensor_split()

    @fresh_cache()
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @torch._dynamo.config.patch("capture_dynamic_output_shape_ops", True)
    def test_nonzero_slice(self):
        def f(x):
            nz = x.nonzero()
            return nz[:-1]

        x = torch.randn(3, 4)
        fn = torch.compile(f, fullgraph=True, backend="inductor")
        self.assertTrue(torch.allclose(f(x), fn(x)))
        y = torch.zeros(3, 4)
        self.assertTrue(torch.allclose(f(y), fn(y)))

    @fresh_cache()
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @torch._dynamo.config.patch("capture_dynamic_output_shape_ops", True)
    @torch._inductor.config.patch("cpp_wrapper", True)
    def test_nonzero_slice_cpp_wrapper(self):
        self.test_nonzero_slice()

    @fresh_cache()
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @torch._dynamo.config.patch("capture_dynamic_output_shape_ops", True)
    def test_nonzero_select(self):
        def f(x):
            nz = x.nonzero()
            return nz[-1] + nz[0]

        x = torch.randn(3, 4)
        fn = torch.compile(f, fullgraph=True, backend="inductor")
        self.assertTrue(torch.allclose(f(x), fn(x)))

    @fresh_cache()
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @torch._dynamo.config.patch("capture_dynamic_output_shape_ops", True)
    @torch._inductor.config.patch("cpp_wrapper", True)
    def test_nonzero_select_cpp_wrapper(self):
        self.test_nonzero_select()

    @fresh_cache()
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @torch._dynamo.config.patch("capture_dynamic_output_shape_ops", True)
    def test_padnd(self):
        import torch.nn.functional as F

        def f(x, xs, y):
            u0, u1 = xs.tolist()
            for u in [u0, u1]:
                torch._check(u >= 0)
            z = F.pad(x, (u0, u1, u0, u1))
            return z @ y

        x = torch.randn(8, 8)
        xs = torch.tensor([2, 2])
        y = torch.randn(12, 4)
        fn = torch.compile(f, fullgraph=True, backend="inductor")
        fn(x, xs, y)

    @unittest.skip("this test fails due to inductor/autograd issue #153041")
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_unbacked_non_contigious_reshape_failing(self):
        # reshape u1 -> (u0*u0)
        # this result in the tensor "i64[u0, u0][s7*u0, s7].
        # reshape happens in place reshape (no-clone)
        def func(x, y):
            f = y.item()
            t1 = x.view((f, f))
            t2 = x.reshape((f, f))
            return t1, t2

        # create a non-contiguous with data being even numbers in [0:cnt-1]
        def make_non_contiguous_tensor(cnt):
            # create a non-contiguous tensor x that is skipping odd indices.
            x = torch.arange(cnt * 2)
            x = x.as_strided((x.size()[0] // 2,), (2,))
            return x

        x = make_non_contiguous_tensor(4)
        torch._dynamo.decorators.mark_unbacked(x, 0)
        compiled_func = torch.compile(
            fullgraph=True,
            backend="inductor",
        )(func)

        compiled_result = compiled_func(x, torch.tensor([2]))
        eager_result = func(x, torch.tensor([2]))
        self.assertEqual(compiled_result, eager_result)

    @skipIfTorchDynamo("not allowed to trace mark_unbacked")
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_invalid_view_unbacked_view(self):
        cnt = CompileCounterWithBackend("inductor")

        # This view (u2, u3) -> (u0, u1) can't happen in general unless we know that input is contiguous or we have
        # hints to to compute strides.
        def func(x, y):
            u0, u1 = y.tolist()
            result2 = x.view(u0, u1) * 10
            return result2

        compiled_func = torch.compile(fullgraph=True, backend=cnt, dynamic=True)(func)

        x = torch.randn(10, 10)
        # make x not contiguous.
        x = x.t_()
        torch._dynamo.decorators.mark_unbacked(x, 0)
        torch._dynamo.decorators.mark_unbacked(x, 1)
        with self.assertRaises(torch._dynamo.exc.UserError):
            # throws a data dependent error.
            compiled_func(x, torch.tensor([5, 20]))

    @skipIfTorchDynamo()
    def test_unbind_not_dynamic(self):
        cnt = CompileCounter()

        @torch.compile(fullgraph=True, dynamic=True, backend=cnt)
        def func(y):
            return y.unbind(dim=2), y * 10

        func(torch.ones(5, 6, 7, 8))
        self.assertEqual(cnt.frame_count, 1)
        # it can be dynamic in all dimensions except dim=2
        func(torch.ones(4, 9, 7, 10))
        self.assertEqual(cnt.frame_count, 1)

        func(torch.ones(5, 6, 8, 8))
        func(torch.ones(5, 6, 9, 8))
        self.assertEqual(cnt.frame_count, 3)

    @skipIfTorchDynamo("not allowed to trace mark_unbacked")
    @fresh_cache()
    def test_unbacked_contiguous(self):
        cnt = CompileCounterWithBackend("inductor")

        def func(x):
            contig = x.contiguous()
            return (contig + 1) * 100

        compiled_func = torch.compile(fullgraph=True, backend=cnt, dynamic=True)(func)

        x = torch.randn(10, 10)
        # make x not contiguous.
        x = x.t_()
        torch._dynamo.decorators.mark_unbacked(x, 0)
        torch._dynamo.decorators.mark_unbacked(x, 1)
        log_stream, ctx = logs_to_string(
            "torch._inductor.compile_fx", "post_grad_graphs"
        )
        with ctx():
            compiled_func(x)
            self.assertEqual(compiled_func(x), func(x))
            y = torch.rand(20, 20).t()
            self.assertEqual(compiled_func(y), func(y))
            self.assertEqual(cnt.frame_count, 1)
        output = "\n".join(log_stream.getvalue().strip().split("\n")[4:]).strip()
        self.assertExpectedInline(
            output,
            """\
        ge_1: "Sym(u0 >= 0)" = arg0_1 >= 0;  arg0_1 = None
        _assert_scalar = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge_1 = _assert_scalar = None
        ge_3: "Sym(u1 >= 0)" = arg1_1 >= 0;  arg1_1 = None
        _assert_scalar_1 = torch.ops.aten._assert_scalar.default(ge_3, "Runtime assertion failed for expression u1 >= 0 on node 'ge_1'");  ge_3 = _assert_scalar_1 = None
        clone: "f32[u0, u1][Max(1, u1), 1]cpu" = torch.ops.aten.clone.default(arg2_1, memory_format = torch.contiguous_format);  arg2_1 = None
        add_3: "f32[u0, u1][Max(1, u1), 1]cpu" = torch.ops.aten.add.Tensor(clone, 1);  clone = None
        mul_6: "f32[u0, u1][Max(1, u1), 1]cpu" = torch.ops.aten.mul.Tensor(add_3, 100);  add_3 = None
        return (mul_6,)""",  # noqa: B950
            ignore_comments=True,
            ignore_empty_lines=True,
        )

        log_stream, ctx = logs_to_string(
            "torch._inductor.compile_fx", "post_grad_graphs"
        )
        with ctx():
            # recompilation will happen due to stride specialization.
            y = torch.rand(20, 20)
            torch._dynamo.decorators.mark_unbacked(y, 0)
            torch._dynamo.decorators.mark_unbacked(y, 1)
            self.assertEqual(compiled_func(y), func(y))
            self.assertEqual(cnt.frame_count, 2)

        output = "\n".join(log_stream.getvalue().strip().split("\n")[4:]).strip()

        # No clone this time since input is contiguous.
        self.assertExpectedInline(
            output,
            """\
        ge_1: "Sym(u0 >= 0)" = arg0_1 >= 0;  arg0_1 = None
        _assert_scalar = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge_1 = _assert_scalar = None
        ge_3: "Sym(u1 >= 0)" = arg1_1 >= 0;  arg1_1 = None
        _assert_scalar_1 = torch.ops.aten._assert_scalar.default(ge_3, "Runtime assertion failed for expression u1 >= 0 on node 'ge_1'");  ge_3 = _assert_scalar_1 = None
        add: "f32[u0, u1][Max(1, u1), 1]cpu" = torch.ops.aten.add.Tensor(arg2_1, 1);  arg2_1 = None
        mul_5: "f32[u0, u1][Max(1, u1), 1]cpu" = torch.ops.aten.mul.Tensor(add, 100);  add = None
        return (mul_5,)""",  # noqa: B950
            ignore_comments=True,
            ignore_empty_lines=True,
        )

    @fresh_cache()
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_unbacked_select_index(self):
        cnt = CompileCounterWithBackend("inductor")

        def func(x, y):
            u0 = y.item()
            return (
                torch.select(x, 0, u0),
                torch.select(x, 1, u0),
                torch.select(x, 2, u0),
            )

        compiled_func = torch.compile(fullgraph=True, backend=cnt, dynamic=True)(func)
        x = torch.rand(3, 3, 3)
        zero = torch.tensor([0])
        pos = torch.tensor([1])
        # code can handle both negative and positive indices.
        neg = torch.tensor([-1])

        log_stream, ctx = logs_to_string(
            "torch._inductor.compile_fx", "post_grad_graphs"
        )
        with ctx():
            self.assertEqual(compiled_func(x, zero), func(x, zero))
        output = "\n".join(log_stream.getvalue().strip().split("\n")[4:]).strip()
        self.assertExpectedInline(
            output,
            """\
        _local_scalar_dense: "Sym(u0)" = torch.ops.aten._local_scalar_dense.default(arg0_1);  arg0_1 = None
        select: "f32[s77, s77][s77, 1]cpu" = torch.ops.aten.select.int(arg2_1, 0, _local_scalar_dense)
        select_1: "f32[s77, s77][s77**2, 1]cpu" = torch.ops.aten.select.int(arg2_1, 1, _local_scalar_dense)
        select_2: "f32[s77, s77][s77**2, s77]cpu" = torch.ops.aten.select.int(arg2_1, 2, _local_scalar_dense);  arg2_1 = _local_scalar_dense = None
        return (select, select_1, select_2)""",  # noqa: B950
            ignore_comments=True,
            ignore_empty_lines=True,
        )
        self.assertEqual(compiled_func(x, pos), func(x, pos))
        self.assertEqual(compiled_func(x, neg), func(x, neg))
        self.assertEqual(cnt.frame_count, 1)

        def func2(x, y):
            u0, u1 = y.tolist()
            return torch.select(x, 0, u0 + u1)

        compiled_func2 = torch.compile(fullgraph=True, backend=cnt, dynamic=False)(
            func2
        )
        zero = torch.tensor([0, 0])
        pos = torch.tensor([1, 1])
        neg = torch.tensor([-1, -1])

        self.assertEqual(compiled_func2(x, pos), func2(x, pos))
        self.assertEqual(compiled_func2(x, neg), func2(x, neg))
        self.assertEqual(compiled_func2(x, zero), func2(x, zero))
        self.assertEqual(cnt.frame_count, 2)

    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_unbacked_select_2(self):
        class M(torch.nn.Module):
            def forward(self, x):
                nz = x.nonzero()
                return nz[-1]

        mod = M()
        x = torch.randn(4)
        self.assertEqual(torch.compile(mod)(x), mod(x))

    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_unbacked_select_index_with_check(self):
        def func3(x, y):
            u0 = y.item()
            # Test that taking the non-unbacked path works fine also.
            torch._check(u0 >= 0)
            return (torch.select(x, 1, u0),)

        compiled_func3 = torch.compile(
            fullgraph=True, backend="inductor", dynamic=True
        )(func3)
        x = torch.rand(3, 3, 3)
        zero = torch.tensor([0])
        pos = torch.tensor([1])
        print(compiled_func3(x, pos))

        self.assertEqual(compiled_func3(x, pos), func3(x, pos))
        self.assertEqual(compiled_func3(x, zero), func3(x, zero))

    @fresh_cache()
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @torch._inductor.config.patch("cpp_wrapper", True)
    def test_unbacked_select_index_cpp_wrapper(self):
        self.test_unbacked_select_index()

    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_unbacked_select2(self):
        def f(idx, x):
            x = x.select(0, idx.item())
            return x @ x

        x = torch.randn(3, 3, 3)
        idx = torch.tensor(1, dtype=torch.int64)
        out = torch.compile(f)(idx, x)
        self.assertEqual(out, f(idx, x))

    def test_trunc_int_div_true(self):
        @torch.compile(backend="inductor", dynamic=True, fullgraph=True)
        def f(x, s13, s57, s77):
            torch._check(s13 >= 0)
            torch._check(s57 >= 0)
            torch._check(s77 >= 0)
            if int(s13 * ((s57 // s13) + (s77 // s13)) / s13) >= 1:
                return x * 2
            else:
                return x * 100

        # ensure we compile this with no errors.
        x = torch.rand(10)
        f(x, 4, 4096, 3920)

    @skipIfTorchDynamo("not allowed to trace mark_unbacked")
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_unbacked_reshape3(self):
        def func(x):
            x = x.as_strided([x.size()[0], 1536], [2048, 1])
            result1 = x.view(x.size()[0], -1, 128)
            return result1 * 10

        compiled = torch.compile(fullgraph=True, backend="inductor")(func)
        x = torch.randn(10, 2048)

        torch._dynamo.decorators.mark_unbacked(x, 0)
        self.assertEqual(func(x), compiled(x))

    @fresh_cache()
    @skipIfTorchDynamo("not allowed to trace mark_unbacked")
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_unbacked_reshape_copy(self):
        cnt = CompileCounterWithBackend("inductor")

        # Reshape happens in place reshape (no-clone)
        # reshape u1 -> (u0*u0)
        def func(x, y):
            f = y.item()
            t3 = torch._ops.ops.aten._reshape_copy(x, (f, f))
            return t3

        compiled_func = torch.compile(
            fullgraph=True,
            backend=cnt,
            dynamic=True,
        )(func)

        # create a non-contiguous with data being even numbers in [0:cnt-1]
        # and reshape it into sqrt(cnt)*sqrt(cnt)
        def make_non_contiguous_tensor_and_test(cnt):
            # create a non-contiguous tensor x that is skipping odd indices.
            x = torch.arange(cnt * 2)
            x = x.as_strided((x.size()[0] // 2,), (2,))

            torch._dynamo.decorators.mark_unbacked(x, 0)
            sz = torch.tensor([int(math.sqrt(cnt))])
            compiled_result = compiled_func(x, sz)
            eager_result = func(x, sz)
            self.assertEqual(compiled_result, eager_result)

        log_stream, ctx = logs_to_string(
            "torch._functorch._aot_autograd.graph_capture", "aot_graphs"
        )
        with ctx():
            make_non_contiguous_tensor_and_test(4)
        aot_graphs = "\n".join(log_stream.getvalue().strip().split("\n")[4:]).strip()
        self.assertExpectedInline(
            aot_graphs,
            """\
def forward(self, arg0_1: "i64[1][1]cpu", arg1_1: "Sym(u1)", arg2_1: "Sym(s7)", arg3_1: "i64[u1][s7]cpu"):
        ge_1: "Sym(u1 >= 0)" = arg1_1 >= 0
        _assert_scalar = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u1 >= 0 on node 'ge'");  ge_1 = _assert_scalar = None
        _local_scalar_dense: "Sym(u0)" = torch.ops.aten._local_scalar_dense.default(arg0_1);  arg0_1 = None
        ge_2: "Sym(u0 >= 0)" = _local_scalar_dense >= 0
        _assert_scalar_1 = torch.ops.aten._assert_scalar.default(ge_2, "Runtime assertion failed for expression u0 >= 0 on node 'ge_1'");  ge_2 = _assert_scalar_1 = None
        pow_1: "Sym(u0**2)" = _local_scalar_dense ** 2
        eq: "Sym(Eq(u1, u0**2))" = arg1_1 == pow_1;  arg1_1 = pow_1 = None
        _assert_scalar_2 = torch.ops.aten._assert_scalar.default(eq, "Runtime assertion failed for expression Eq(u1, u0**2) on node 'eq'");  eq = _assert_scalar_2 = None
        _reshape_copy: "i64[u0, u0][Max(1, u0), 1]cpu" = torch.ops.aten._reshape_copy.default(arg3_1, [_local_scalar_dense, _local_scalar_dense]);  arg3_1 = _local_scalar_dense = None
        return (_reshape_copy,)""",  # noqa: B950
            ignore_comments=True,
            ignore_empty_lines=True,
        )

        make_non_contiguous_tensor_and_test(49)
        self.assertEqual(cnt.frame_count, 1)

        # Pass in a contiguous tensor, it will recompile due to stride being 1 (0/1 specialization).
        # marking strides unbacked would have avoided the recompilation here.
        x = torch.arange(100)
        torch._dynamo.decorators.mark_unbacked(x, 0)

        log_stream, ctx = logs_to_string(
            "torch._functorch._aot_autograd.graph_capture", "aot_graphs"
        )
        with ctx():
            compiled_result = compiled_func(x, torch.tensor([10]))
            eager_result = func(x, torch.tensor([10]))
            self.assertEqual(compiled_result, eager_result)
            self.assertEqual(cnt.frame_count, 2)

        aot_graphs = "\n".join(log_stream.getvalue().strip().split("\n")[4:]).strip()
        self.assertExpectedInline(
            aot_graphs,
            """\
def forward(self, arg0_1: "i64[1][1]cpu", arg1_1: "Sym(u1)", arg2_1: "i64[u1][1]cpu"):
        ge_1: "Sym(u1 >= 0)" = arg1_1 >= 0
        _assert_scalar = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u1 >= 0 on node 'ge'");  ge_1 = _assert_scalar = None
        _local_scalar_dense: "Sym(u0)" = torch.ops.aten._local_scalar_dense.default(arg0_1);  arg0_1 = None
        ge_2: "Sym(u0 >= 0)" = _local_scalar_dense >= 0
        _assert_scalar_1 = torch.ops.aten._assert_scalar.default(ge_2, "Runtime assertion failed for expression u0 >= 0 on node 'ge_1'");  ge_2 = _assert_scalar_1 = None
        pow_1: "Sym(u0**2)" = _local_scalar_dense ** 2
        eq: "Sym(Eq(u1, u0**2))" = arg1_1 == pow_1;  arg1_1 = pow_1 = None
        _assert_scalar_2 = torch.ops.aten._assert_scalar.default(eq, "Runtime assertion failed for expression Eq(u1, u0**2) on node 'eq'");  eq = _assert_scalar_2 = None
        _reshape_copy: "i64[u0, u0][Max(1, u0), 1]cpu" = torch.ops.aten._reshape_copy.default(arg2_1, [_local_scalar_dense, _local_scalar_dense]);  arg2_1 = _local_scalar_dense = None
        return (_reshape_copy,)""",  # noqa: B950
            ignore_comments=True,
            ignore_empty_lines=True,
        )

        x = torch.arange(25)
        compiled_result = compiled_func(x, torch.tensor([5]))
        eager_result = func(x, torch.tensor([5]))
        self.assertEqual(cnt.frame_count, 2)

    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_unbacked_item(self):
        def func():
            _x_ms = torch.tensor([True, False], dtype=torch.int64)
            _mask_ms = torch.zeros_like(_x_ms, dtype=torch.bool)
            _mask_ms[:1] = True
            var_node_2 = torch.masked_select(_x_ms, _mask_ms)
            var_node_0 = var_node_2.item()
            return var_node_0

        result_original = func()
        compiled_program = torch.compile(func, fullgraph=True, dynamic=True)
        result_compiled = compiled_program()
        self.assertEqual(result_original, result_compiled)

    def test_unbacked_item_set_item(self):
        def my_arithmetic(a, b):
            wrk = torch.zeros(a.size(0))
            for i in range(a.size(0)):
                idx = b[i].item()
                wrk[idx] += 1

            return wrk

        compiled = torch.compile(my_arithmetic, fullgraph=True, disable=False)
        a = torch.randn([9])
        b = torch.ones(9, dtype=torch.int32)
        compiled(a, b)
        self.assertEqual(compiled(a, b), my_arithmetic(a, b))

    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_unbacked_item_set_item2(self):
        def accumulate(X0, start):
            start = start.item()
            N = 3
            result = X0[start]
            for i in range(N):
                result += X0[start + 1 + i]
            return result

        compiled = torch.compile(accumulate, fullgraph=True)
        X0 = torch.randn(10, 10)
        self.assertEqual(
            accumulate(X0, torch.tensor([1])), compiled(X0, torch.tensor([1]))
        )

    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_unbacked_item_set_item3(self):
        def func(x, y):
            u0 = y.item()
            x[u0] = 0
            return x

        compiled = torch.compile(func, fullgraph=True, disable=False)
        b = torch.tensor([0])
        a = torch.ones(9, dtype=torch.int32)

        compiled(a, b)
        self.assertEqual(compiled(a, b), func(a, b))

    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_select_scatter_unbacked_index(self):
        def func(x, y):
            u0 = y.item()
            # Create a scalar tensor to scatter into the selected index
            scalar_src = torch.tensor(42, dtype=x.dtype)
            return x.select_scatter(scalar_src, 0, u0)

        compiled = torch.compile(func, fullgraph=True, dynamic=True, backend="inductor")
        b = torch.tensor([0])
        a = torch.ones(9, dtype=torch.int32)

        self.assertEqual(compiled(a, b), func(a, b))

    @fresh_cache()
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_narrow_unbacked_start(self):
        def func(x, start, length):
            # unbacked start
            u0 = start.item()
            return torch.narrow(x, 0, u0, length)

        compiled_func = torch.compile(func, fullgraph=True, backend="inductor")

        x = torch.tensor([1, 2, 3, 4, 5, 6])

        # Test cases: (start, length)
        test_cases = [
            # Negative starts
            (-2, 2),  # Start from second-to-last element
            (-1, 1),  # Start from last element
            (-3, 3),  # Start from third-to-last element
            (-6, 2),  # Start from beginning (negative)
            (-4, 1),  # Start from fourth-to-last element
            # Positive starts
            (0, 2),  # Start from beginning
            (1, 3),  # Start from second element
            (2, 2),  # Start from third element
            (4, 2),  # Start near end
            # Edge cases
            (0, 6),  # Full tensor
            (0, 1),  # Single element from start
            (5, 1),  # Single element from end
        ]

        for start_val, length in test_cases:
            with self.subTest(start=start_val, length=length):
                start = torch.tensor([start_val])

                # Test with compiled function
                result_compiled = compiled_func(x, start, length)

                # Test with eager function (expected behavior)
                result_eager = func(x, start, length)

                # Compare results
                self.assertEqual(result_compiled, result_eager)

    @fresh_cache()
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @torch._inductor.config.patch("cpp_wrapper", True)
    def test_narrow_unbacked_start_cpp_wrapper(self):
        """Test narrow with unbacked start with cpp_wrapper"""
        self.test_narrow_unbacked_start()

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_narrow_with_tensor_start(self):
        @torch.compile(backend="inductor", fullgraph=True)
        def f(x, start, end):
            return torch.narrow(x, 0, start, end)

        x = torch.tensor(
            [False], device="cuda:0" if torch.cuda.is_available() else "cpu"
        )
        start = torch.tensor(0)
        res = f(x, start, 0)
        self.assertEqual(res.shape, torch.Size([0]))

    @skipIfTorchDynamo()
    @torch.fx.experimental._config.patch("backed_size_oblivious", True)
    def test_backed_size_oblivious_broadcast(self):
        cnt = CompileCounterWithBackend("inductor")
        torch._dynamo.reset()

        def func(a, b):
            torch.broadcast_shapes(a.size(), b.size())
            return a + b

        compiled = torch.compile(func, fullgraph=True, backend=cnt, dynamic=True)

        def run(a, b):
            self.assertEqual(compiled(a, b), func(a, b))

        # No 0/1 specializations, no broadcasts.
        # but a[0] == b[0] and a[1] == b[1] are asserted.
        run(torch.rand(1, 10), torch.rand(1, 10))
        run(torch.rand(1, 1), torch.rand(1, 1))
        run(torch.rand(10, 10), torch.rand(10, 10))

        self.assertEqual(cnt.frame_count, 1)
        run(torch.rand(10, 10), torch.rand(1, 10))
        self.assertEqual(cnt.frame_count, 2)

        cnt.clear()
        torch._dynamo.reset()

        # specialize a[0] == 1. b[0] not specialized.
        run(torch.rand(1, 10), torch.rand(9, 10))
        run(torch.rand(1, 10), torch.rand(1, 10))
        self.assertEqual(cnt.frame_count, 1)
        # if we change a[0] we get recompilation.
        run(torch.rand(10, 10), torch.rand(10, 10))
        self.assertEqual(cnt.frame_count, 2)

        cnt.clear()
        torch._dynamo.reset()

        # TODO duck sizing shall be disabled when backed_size_oblivious
        # is on probably.
        # specialize b[0] == 1. a[0] not specialized.
        run(torch.rand(10, 11), torch.rand(1, 11))
        run(torch.rand(1, 10), torch.rand(1, 10))
        self.assertEqual(cnt.frame_count, 1)
        run(torch.rand(2, 10), torch.rand(2, 10))
        self.assertEqual(cnt.frame_count, 2)

    @torch._dynamo.config.patch("capture_dynamic_output_shape_ops", True)
    def test_unbacked_view_extra(self):
        def fn(x):
            i0 = x.nonzero().size(0)
            y = torch.zeros((i0, 192))
            return y.view([12, -1, 192])

        res1 = torch.compile(fn, fullgraph=True)(torch.ones((12,)))
        res2 = fn(torch.ones((12,)))
        self.assertEqual(res1, res2)


instantiate_parametrized_tests(TestUnbacked)


if __name__ == "__main__":
    run_tests()
