# Owner(s): ["oncall: jit"]

import contextlib
import copy
import itertools
import inspect
import math
import operator
import re

import sympy
import torch
import torch.fx
import torch.nn.functional as F
from torch import sym_int, SymBool, SymFloat, SymInt
from torch._C import _disabled_torch_function_impl
from torch.fx.experimental import symbolic_shapes
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import (
    DimConstraints,
    DimDynamic,
    expect_true,
    guard_bool,
    guard_float,
    guard_int,
    GuardOnDataDependentSymNode,
    ShapeEnv,
    sym_float,
    sym_sqrt,
    SymNode,
    to_node,
    is_symbolic,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map
from torch.utils._sympy.functions import FloorDiv, Mod

aten = torch.ops.aten

meta_funcs = {}


def register_meta(op):
    def decorator(f):
        def add_func(op):
            meta_funcs[op] = f
        tree_map(add_func, op)
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
    def __new__(cls, sym_shape, sym_strides, dtype, layout, requires_grad, device, storage_offset=0):
        # TODO: this is wrong in general
        sym_stride = create_contiguous(sym_shape)
        r = torch.Tensor._make_wrapper_subclass(
            cls, sym_shape,
            sym_stride, storage_offset,
            dtype=dtype, layout=layout, requires_grad=requires_grad,
            device=device,
        )
        return r

    __torch_function__ = _disabled_torch_function_impl

    def new_empty(self, shape):
        return FakeSymbolicTensor(shape, None, self.dtype, self.layout, self.requires_grad, self.device)

    @classmethod
    def __torch_dispatch__(cls, func_overload, types, args=(), kwargs=None):
        if func_overload in meta_funcs:
            return meta_funcs[func_overload](*args, **kwargs)

        if func_overload == torch.ops.aten.new_empty.default:
            self = args[0]
            shape = args[1]
            return FakeSymbolicTensor(shape, self.stride(), self.dtype, self.layout, self.requires_grad, self.device)

        raise RuntimeError(f"operator {func_overload} not supported")


def create_symbolic_tensor(name, arg, shape_env):
    from torch._dynamo.source import ConstantSource

    constraint_dims = [None] * arg.dim()
    dynamic_dims = [DimDynamic.DUCK] * arg.dim()
    sym_shapes, sym_strides, sym_storage_offset = \
        shape_env.create_symbolic_sizes_strides_storage_offset(
            arg,
            source=ConstantSource(name),
            dynamic_dims=dynamic_dims,
            constraint_dims=constraint_dims
        )
    return FakeSymbolicTensor(sym_shapes, sym_strides, arg.dtype, arg.layout, arg.requires_grad, arg.device, sym_storage_offset)

def create_symtype(cls, pytype, shape_env, val):
    from torch._dynamo.source import ConstantSource
    symbol = shape_env.create_symbol(
        val,
        source=ConstantSource(f"__testing_only{len(shape_env.var_to_val)}"),
        dynamic_dim=DimDynamic.DUCK,
        constraint_dim=None,
    )
    return cls(SymNode(
        symbol,
        shape_env,
        pytype,
        hint=val,
    ))

def create_symint(shape_env, i: int):
    return create_symtype(SymInt, int, shape_env, i)

def create_symbool(shape_env, b: bool):
    return create_symtype(SymBool, bool, shape_env, b)

def create_symfloat(shape_env, f: float):
    return create_symtype(SymFloat, float, shape_env, f)

@skipIfTorchDynamo("Creating ShapeEnv fails for confusing reasons (also we never expect dynamo to see code like this)")
class TestPySymInt(TestCase):

    def test_arith_ops(self):
        shape_env = ShapeEnv()
        symints = []
        for i in range(2, 5):
            symints.append((i, create_symint(shape_env, i)))

        ops = [operator.add, operator.sub, operator.floordiv, operator.mul, operator.mod]

        for op in ops:
            for args in itertools.permutations(symints, 2):
                if not isinstance(args[0][1], int) and ((op != operator.mod or op != operator.floordiv) and args[1][0] != 0):
                    self.assertTrue(op(args[0][1], args[1][1]) == op(args[0][0], args[1][0]))


    def test_reverse_arith_ops(self):
        shape_env = ShapeEnv()

        a = create_symint(shape_env, 2)
        self.assertTrue(5 // a == 5 // 2)

        a = create_symint(shape_env, 2)
        self.assertTrue(5 * a == 5 * 2)


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
        self.assertTrue(isinstance(x.size()[1].node.maybe_as_int(), int))  # due to guard above
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

        gt_op, _bt = shape_env.guards[-1]
        self.assertTrue(isinstance(gt_op, sympy.core.relational.StrictGreaterThan))
        self.assertTrue(str(x.shape[0]), str(gt_op.args[0]))
        self.assertTrue(str(expand_x.shape[1]), str(x.shape[0]))
        self.assertTrue(str(expand_x.shape[1]), str(result.shape[0]))

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
        r = sym_float(x.shape[0])
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
        r = torch.empty(a0, device='meta')
        self.assertIsInstance(r.shape[0], SymInt)

    def test_guard_int(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 2)
        self.assertEqual(guard_int(a0), 2)
        self.assertExpectedInline(str(shape_env.guards[0][0]), """Eq(s0, 2)""")

    def test_sym_int(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 5)
        r = sym_int(a0)
        self.assertEqual(r, 5)
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        self.assertExpectedInline(str(shape_env.guards[0][0]), """Eq(s0, 5)""")

        a1 = create_symint(shape_env, 7)
        r = sym_int(a1 / 2)
        self.assertEqual(guard_int(r), 3)
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        self.assertExpectedInline(str(shape_env.guards[1][0]), """Eq(floor(s1/2), 3)""")

        a3 = create_symint(shape_env, 3)
        r = sym_int(2.0 * sym_float(a3))
        self.assertEqual(guard_int(r), 6)
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        self.assertExpectedInline(str(shape_env.guards[2][0]), """Eq(2*s2, 6)""")

    def test_sym_sqrt(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 4)
        r = sym_sqrt(a0)
        self.assertEqual(r, 2)
        self.assertIsInstance(r, torch.SymFloat, msg=type(r))
        self.assertExpectedInline(str(shape_env.guards[0][0]), """Eq(sqrt(s0), 2)""")

    def test_sym_floor(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 5)
        r = math.floor(a0 / 2)
        self.assertEqual(r, 2)
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        self.assertExpectedInline(str(shape_env.guards[0][0]), """Eq(floor(s0/2), 2)""")
        r = math.floor(3.0 * a0)
        self.assertEqual(r, 15)
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        self.assertExpectedInline(str(shape_env.guards[1][0]), """Eq(3*s0, 15)""")

    def test_sym_ceil(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 5)
        r = math.ceil(a0 / 2)
        self.assertEqual(r, 3)
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        self.assertExpectedInline(str(shape_env.guards[0][0]), """Eq(ceiling(s0/2), 3)""")
        r = math.floor(3.0 * a0)
        self.assertEqual(r, 15)
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        self.assertExpectedInline(str(shape_env.guards[1][0]), """Eq(3*s0, 15)""")


    def test_int_conversion(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 2)
        int(a0)
        self.assertExpectedInline(str(shape_env.guards[0][0]), """Eq(s0, 2)""")

    def test_data_dependent_guard(self):
        shape_env = ShapeEnv()
        s0 = shape_env.create_unbacked_symint()
        self.assertRaises(GuardOnDataDependentSymNode, lambda: bool(s0 == 0))

    def test_expect_true_basic(self):
        shape_env = ShapeEnv()
        i0 = shape_env.create_unbacked_symint()
        # This doesn't error
        self.assertTrue(expect_true(i0 == 0))
        # This generates a deferred runtime assert
        self.assertExpectedInline(
            str([ra.expr for ra in shape_env.deferred_runtime_asserts[i0.node.expr]]),
            """[Eq(i0, 0)]"""
        )
        self.assertIn("test_dynamic_shapes.py", shape_env.deferred_runtime_asserts[i0.node.expr][0].msg)
        # After expecting true, guards now resolve given the runtime assert
        bool(i0 == 0)

    def test_expect_true_with_s0(self):
        shape_env = ShapeEnv()
        s0 = create_symint(shape_env, 5)
        i0 = shape_env.create_unbacked_symint()
        self.assertTrue(expect_true(i0 <= s0))
        self.assertExpectedInline(
            str([ra.expr for ra in shape_env.deferred_runtime_asserts[i0.node.expr]]),
            """[i0 <= s0]"""
        )
        self.assertTrue(i0 <= s0)
        self.assertFalse(i0 > s0)

    def test_expect_true_prefer_later(self):
        shape_env = ShapeEnv()
        i0 = shape_env.create_unbacked_symint()
        i1 = shape_env.create_unbacked_symint()
        self.assertTrue(expect_true(i0 + i1 == 10))
        # Importantly, this is put in i1, not i0!
        self.assertExpectedInline(
            str([ra.expr for ra in shape_env.deferred_runtime_asserts[i1.node.expr]]),
            """[Eq(i0 + i1, 10)]"""
        )
        self.assertTrue(i0 + i1 == 10)
        # NB: We currently don't support deriving that we can substitute
        # i0 + i1 with 10; maybe we should, but this means our rewriting
        # system is no longer confluent (it's probably OK though, because
        # you're unlikely to get other equalities like this on the
        # unbacked SymInts.)

    def test_expect_true_double_digits(self):
        shape_env = ShapeEnv()
        ia = [shape_env.create_unbacked_symint() for _ in range(11)]  # allocate 10
        self.assertEqual(str(ia[-1]), "i10")
        self.assertTrue(expect_true(sum(ia) == 20))
        self.assertEqual(len(shape_env.deferred_runtime_asserts[ia[-1].node.expr]), 1)

    def test_non_overlapping_and_dense(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 5)
        r = torch.empty_strided((a0, 7), (1, a0), device='meta')
        self.assertTrue(torch.ops.aten.is_non_overlapping_and_dense.default(r))

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

        self.assertExpectedInline(out.strip(), """\
class f(torch.nn.Module):
    def forward(self, a_1: f32[s0, s1], b_1: f32[s2, s1]):
        # No stacktrace found for following nodes
        sym_size: Sym(s0) = torch.ops.aten.sym_size(a_1, 0)
        sym_size_1: Sym(s2) = torch.ops.aten.sym_size(b_1, 0)
        add: Sym(s0 + s2) = sym_size + sym_size_1;  sym_size = sym_size_1 = None
        sym_size_2: Sym(s1) = torch.ops.aten.sym_size(a_1, 1)
        sym_size_3: Sym(s1) = torch.ops.aten.sym_size(b_1, 1);  b_1 = None
        add_1: Sym(2*s1) = sym_size_2 + sym_size_3;  sym_size_2 = sym_size_3 = None
        new_empty: f32[s0 + s2, 2*s1] = torch.ops.aten.new_empty.default(a_1, [add, add_1], pin_memory = False);  a_1 = add = add_1 = None
        native_dropout = torch.ops.aten.native_dropout.default(new_empty, 0.5, True);  new_empty = None
        getitem: f32[s0 + s2, 2*s1] = native_dropout[0]
        getitem_1: b8[s0 + s2, 2*s1] = native_dropout[1];  native_dropout = None
        return (getitem, getitem_1)""")  # noqa: B950

@skipIfTorchDynamo("Creating ShapeEnv fails for confusing reasons (also we never expect dynamo to see code like this)")
class TestSymNumberMagicMethods(TestCase):
    def _do_test(self, fn, inp1, inp2, shape_env, is_unary_fn):
        # Helper function
        # NB: don't use one as that will get specialized
        seed_node = (create_symint(shape_env, 2) / 2.).node
        bool_seed_node = (create_symint(shape_env, 2) == 2).node

        def get_sym_inp(inp):
            # NB: this must come before int
            if isinstance(inp, bool):
                return torch.SymBool(to_node(bool_seed_node, inp))
            elif isinstance(inp, int):
                return torch.SymInt(to_node(seed_node, inp))
            else:
                return torch.SymFloat(to_node(seed_node, inp))

        def maybe_xfail(inp1, inp2):
            if fn == "sym_sqrt" and inp1 < 0:
                # ValueError: math domain error
                return self.assertRaises((ValueError,))
            elif fn in ("truediv", "floordiv", "mod") and inp2 == 0:
                # ZeroDivisionError: division by zero
                return self.assertRaises((ZeroDivisionError,))
            elif fn == "pow" and inp1 == 0 and inp2 < 0:
                # ZeroDivisionError: 0.0 cannot be raised to a negative power
                return self.assertRaises((ZeroDivisionError,))
            elif fn == "pow" and inp1 < 0 and inp2 in (2.5, -2.5) and (
                type(inp1) in (SymFloat, SymInt) or
                type(inp2) in (SymFloat, SymInt)
            ):
                # Complex result, which we do not support:
                # TypeError: Cannot convert complex to float
                return self.assertRaises((TypeError,))
            elif fn in ("lshift", "rshift") and not (
                isinstance(inp1, (SymInt, int)) and
                isinstance(inp2, (SymInt, int))
            ):
                # TypeError: unsupported operand type(s)
                return self.assertRaises((TypeError,))
            elif fn in ("lshift", "rshift") and inp2 < 0:
                # ValueError: math domain error
                return self.assertRaises((ValueError,))
            else:
                return contextlib.nullcontext()

        if fn in symbolic_shapes.magic_methods_on_math:
            lambda_apply = getattr(math, fn)
        elif fn in symbolic_shapes.magic_methods_on_submodule:
            lambda_apply = getattr(symbolic_shapes, fn)
        elif fn in symbolic_shapes.magic_methods_on_operator_with_trailing_underscore:
            lambda_apply = getattr(operator, f"{fn}_")
        else:
            lambda_apply = getattr(operator, fn)

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
            if fn not in symbolic_shapes.alternate_impl_if_hinted_methods:
                self.assertTrue(isinstance(out, (SymInt, SymFloat, SymBool)))
            out = guard_fn(out)
            self.assertEqual(out, ref_out)

        if is_unary_fn:
            return

        # Symified second arg
        sym_inp2 = get_sym_inp(inp2)
        with maybe_xfail(inp1, sym_inp2):
            out = lambda_apply(inp1, sym_inp2)
            if fn not in symbolic_shapes.alternate_impl_if_hinted_methods:
                self.assertTrue(isinstance(out, (SymInt, SymFloat, SymBool)))
            out = guard_fn(out)
            self.assertEqual(out, ref_out)

        # Symified both args
        with maybe_xfail(sym_inp1, sym_inp2):
            out = lambda_apply(sym_inp1, sym_inp2)
            if fn not in symbolic_shapes.alternate_impl_if_hinted_methods:
                self.assertTrue(isinstance(out, (SymInt, SymFloat, SymBool)))
            out = guard_fn(out)
            self.assertEqual(out, ref_out)


    @parametrize("fn", list(symbolic_shapes.magic_methods.keys()))
    def test_bool_method(self, fn):
        if fn not in symbolic_shapes.bool_magic_methods:
            self.skipTest(f"{fn} is non-bool")

        is_unary_fn = fn in symbolic_shapes.unary_magic_methods
        shape_env = ShapeEnv()
        self._do_test(fn, True, False, shape_env, is_unary_fn)


    @parametrize("fn", list(symbolic_shapes.magic_methods.keys()))
    @parametrize("first_type", ["int", "float"])
    @parametrize("second_type", ["int", "float"])
    def test_method(self, fn, first_type, second_type):
        if first_type == "float":
            # TODO: Hmm, this looks like we skip all floats
            self.skipTest(f"{fn} is not a float magic method")

        is_unary_fn = fn in symbolic_shapes.unary_magic_methods
        # Second argument is ignored for unary function. So only run for one type
        if is_unary_fn and second_type == "float":
            self.skipTest(f"{fn} is unary and already tested")

        if fn in symbolic_shapes.bool_magic_methods:
            self.skipTest(f"{fn} is bool")

        # Only floats here since these will be converted to int if necessary.
        # We also ignore complex and bool.
        values = (
            0.0,
            1.0,
            2.5,
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

    def test_symnode_hashing(self):
        shape_env = ShapeEnv()

        # SymInt, SymBool, SymFloat are unhashable
        unhashable = (
            create_symint(shape_env, 3),
            create_symbool(shape_env, True),
            create_symfloat(shape_env, 4.2),
        )

        for x in unhashable:
            with self.assertRaisesRegex(TypeError, "unhashable"):
                hash(x)

        # Singleton SymInt, constant SymBool, SymNode are hashable
        j1 = torch._C._get_singleton_int(1)
        j1_copy = torch._C._get_singleton_int(1)
        j2 = torch._C._get_singleton_int(2)
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

    def test_non_symbolic_symnode(self):
        j1 = torch._C._get_singleton_int(1)
        j2 = torch._C._get_singleton_int(1)
        j3 = torch._C._get_singleton_int(3)

        self.assertIsInstance(j1, torch.SymInt)
        self.assertNotIsInstance(j1, int)

        with self.assertRaisesRegex(RuntimeError, "add not supported by SingletonSymNode"):
            j1 + 3

        self.assertFalse(j1 == 3)
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
        values = (
            (2.5, 2.1),
            (2.1, 2.5),
            (2.0, 2.1),
            (7, 2.5),
            (2.1, 7),
            (7, 2),
        )

        for x, y in TestFloorDiv.yield_test_cases(values):
            self.assertEqual(TestFloorDiv.python_floordiv(x, y), TestFloorDiv.torch_floordiv(x, y))

    def test_floordiv_bool(self):
        values = (
            (False, True),
            (True, 2.5),
            (2.5, True),
            (False, 7),
            (7, True),
        )

        for x, y in TestFloorDiv.yield_test_cases(values, negate=False):
            # Compares to int since our FloorDiv has no bool support
            self.assertEqual(TestFloorDiv.python_floordiv(x, y), TestFloorDiv.torch_floordiv(int(x), int(y)))
            # Tests that our impl throws
            self.assertRaisesRegex(
                TypeError,
                (rf"unsupported operand type\(s\) for //: "
                 rf"'{type(sympy.sympify(x)).__name__}' and '{type(sympy.sympify(y)).__name__}'"
                 rf", expected integer or real"),
                lambda: TestFloorDiv.torch_floordiv(x, y))

    def test_floordiv_complex(self):
        values = (
            (1.5 + 2.5j, 1.3 + 3.5j),
            (1.5 + 2.5j, 2.5),
            (2.5, 1.5 + 2.5j),
            (1.5 + 2.5j, 7),
            (7, 1.5 + 2.5j),
        )

        for x, y in TestFloorDiv.yield_test_cases(values):
            # We don't test error messages to avoid depending on Python
            # interpreter version
            self.assertRaises(TypeError, lambda: TestFloorDiv.python_floordiv(x, y))
            self.assertRaisesRegex(
                TypeError,
                (rf"unsupported operand type\(s\) for //: "
                 rf"'{type(sympy.sympify(x)).__name__}' and '{type(sympy.sympify(y)).__name__}'"
                 rf", expected integer or real"),
                lambda: TestFloorDiv.torch_floordiv(x, y))

    def test_floordiv_div_by_zero(self):
        values = (
            (2.5, 0),
            (2.1, 0.0),
            (2.3, sympy.Symbol("s", zero=True)),
        )

        for x, y in TestFloorDiv.yield_test_cases(values, negate=False):
            # We don't test error messages to avoid depending on Python
            # interpreter version
            if type(y) is not sympy.Symbol:
                self.assertRaises(ZeroDivisionError, lambda: TestFloorDiv.python_floordiv(x, y))
            self.assertRaisesRegex(
                ZeroDivisionError,
                "division by zero",
                lambda: TestFloorDiv.torch_floordiv(x, y))

    def test_floordiv_zero_base(self):
        values = (
            (0, 2.5),
            (0.0, 2.1),
            (sympy.Symbol("s", zero=True), 2.3),
        )

        for x, y in TestFloorDiv.yield_test_cases(values, negate=False):
            if type(x) is not sympy.Symbol:
                self.assertEqual(TestFloorDiv.python_floordiv(x, y), TestFloorDiv.torch_floordiv(x, y))
            else:
                self.assertEqual(0, TestFloorDiv.torch_floordiv(x, y))

    def test_floordiv_div_by_one(self):
        values = (
            (2.5, 1),
            (2.1, 1.0),
            (2, 1.0),
            (2, 1),
        )

        for x, y in TestFloorDiv.yield_test_cases(values):
            self.assertEqual(TestFloorDiv.python_floordiv(x, y), TestFloorDiv.torch_floordiv(x, y))

    def test_floordiv_simplify(self):
        # Tests how we simplify or evaluate FloorDiv without free variables
        shape_env = ShapeEnv()
        result = 21
        exprs = (
            7 * FloorDiv(6, 2),
            7 * FloorDiv(6.28, 2),
            7 * FloorDiv(6.28, 2.0),
            7 * FloorDiv(6.28, (FloorDiv(6.28, 3.14))),
        )

        for expr in exprs:
            self.assertEqual(expr, result)
            self.assertEqual(expr.doit(deep=False), result)
            self.assertEqual(expr.doit(deep=True), result)
            self.assertEqual(sympy.simplify(expr), result)
            self.assertEqual(shape_env.simplify(expr), result)
            self.assertEqual(shape_env.evaluate_expr(expr), result)

    def test_floordiv_simplify_rational(self):
        result = 21

        a = sympy.Symbol("a", integer=True)
        b = sympy.Symbol("b")

        cases = [
            (FloorDiv(a, sympy.Rational(1, 8)), 8 * a),
            (FloorDiv(b, sympy.Rational(1, 8)), sympy.floor(8 * b)),
        ]

        for expr, expected in cases:
            self.assertEqual(expr, expected)

    def test_floordiv_assumptions(self):
        # We define two Symbols (with different names) for each type to make
        # sure the behavior is consistent regardless of whether both arguments
        # are the same object or not.
        cases = (
            sympy.Symbol("i1", integer=True),
            sympy.Symbol("i2", integer=True),
            sympy.Symbol("r1", real=True),
            sympy.Symbol("r2", real=True),
            sympy.Symbol("c1", complex=True, real=False, integer=False),
            sympy.Symbol("c2", complex=True, real=False, integer=False),
            sympy.Symbol("s1"),
            sympy.Symbol("s2"),
        )

        for base, divisor in itertools.product(cases, repeat=2):
            def op():
                return FloorDiv(base, divisor)

            def is_complex(x):
                return x.is_integer is False and x.is_real is False and x.is_complex

            if is_complex(base) or is_complex(divisor):
                self.assertRaisesRegex(
                    TypeError,
                    (r"unsupported operand type\(s\) for //: 'Symbol' and 'Symbol',"
                     r" expected integer or real"),
                    op)
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
    def test_dim_constraints_reduce_congruences_simple(self):
        from sympy import Symbol
        from torch.fx.experimental.symbolic_shapes import DimConstraints

        s = Symbol("s", positive=True, integer=True)
        dim_constraints = DimConstraints({}, {}, set())
        dim_constraints._congruences[s] = {
            (s / 2) % 2,
            (s / 2) % 8,
            (s / 2) % 4,
            s % 2,
            ((s / 16) + 2) % 4,
        }
        congruences = dim_constraints.reduce_congruences()
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

    def test_dim_constraints_solve_full(self):
        from sympy import Eq, Integer, Ne, Symbol
        from torch._dynamo.source import LocalSource, TensorProperty, TensorPropertySource

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
        dim_constraints = DimConstraints(symbol_to_source, var_to_val, marked_dynamic)
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
        dim_constraints.add(Ne(64 * (Mod(FloorDiv((FloorDiv(s1, 2) - 1), 8) + 1, 4)), 0))
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
        dim_constraints.remove_redundant_dynamic_results()
        self.assertEqual(dim_constraints._static_results, {
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
        })
        self.assertEqual(dim_constraints._dynamic_results, {
            "dynamic_dim(L['e'], 1) == dynamic_dim(L['c'], 1)",
            "dynamic_dim(L['d'], 1) == dynamic_dim(L['c'], 1)",
        })

        def dummy_fn(a, b, c, d, e, f):
            pass

        action_code = dim_constraints.prettify_results(inspect.signature(dummy_fn))
        static_code, dynamic_code = re.findall(r"```(.*?)```", action_code, re.DOTALL)
        expected_static = '''
def specializations(a, b, c, d, e, f):
    # a:
    assert a.size()[0] == 8
    assert a.size()[1] == 22
    assert a.size()[2] == 96
    assert a.size()[3] == 96

    # b:
    assert b.size()[0] == 8
    assert b.size()[1] == 22
    assert b.size()[2] == 3

    # c:
    assert c.size()[0] == 8

    # d:
    assert d.size()[0] == 8

    # f:
    assert f.size()[1] == 1
'''
        expected_dynamic = '''
def specify_constraints(a, b, c, d, e, f):
    return [
        # d:
        dynamic_dim(d, 1) == dynamic_dim(c, 1),

        # e:
        dynamic_dim(e, 1) == dynamic_dim(c, 1),
    ]
'''

        self.assertEqual(static_code, expected_static)
        self.assertEqual(dynamic_code, expected_dynamic)



if __name__ == '__main__':
    run_tests()
