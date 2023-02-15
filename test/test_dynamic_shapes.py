# -*- coding: utf-8 -*-
# Owner(s): ["oncall: jit"]

from torch._C import _disabled_torch_function_impl
import torch.fx
import torch.nn.functional as F
from torch.testing._internal.common_utils import run_tests, TestCase, skipIfTorchDynamo, \
    IS_WINDOWS, parametrize, instantiate_parametrized_tests
import unittest
import torch
import operator
import itertools
import contextlib
import math
import copy
from torch.utils._pytree import tree_map
from torch.fx.experimental import symbolic_shapes
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import SymNode, \
    FloorDiv, ShapeEnv, sym_sqrt, sym_float, to_node, GuardOnDataDependentSymNode, \
    guard_bool, guard_int, guard_float
from torch.utils._python_dispatch import TorchDispatchMode
from torch import SymBool, SymInt, SymFloat, sym_int

aten = torch.ops.aten

try:
    import sympy
    # TODO(jansel): these tests fail on windows
    HAS_SYMPY = not IS_WINDOWS
except ImportError:
    HAS_SYMPY = False
skipIfNoSympy = unittest.skipIf(not HAS_SYMPY, "no sympy")


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
    sym_shapes, sym_strides, sym_storage_offset = \
        shape_env.create_symbolic_sizes_strides_storage_offset(arg, source=ConstantSource(name))
    return FakeSymbolicTensor(sym_shapes, sym_strides, arg.dtype, arg.layout, arg.requires_grad, arg.device, sym_storage_offset)

def create_symint(shape_env, i: int):
    from torch._dynamo.source import ConstantSource
    return shape_env.create_symintnode(
        shape_env.create_symbol(i, source=ConstantSource(f"__testing_only{len(shape_env.var_to_val)}")),
        hint=i
    )

@skipIfTorchDynamo("Creating ShapeEnv fails for confusing reasons (also we never expect dynamo to see code like this)")
class TestPySymInt(TestCase):

    @skipIfNoSympy
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


    @skipIfNoSympy
    def test_reverse_arith_ops(self):
        shape_env = ShapeEnv()

        a = create_symint(shape_env, 2)
        self.assertTrue(5 // a == 5 // 2)

        a = create_symint(shape_env, 2)
        self.assertTrue(5 * a == 5 * 2)


    @skipIfNoSympy
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
        self.assertTrue(isinstance(x.size()[1], SymInt))
        self.assertTrue(x.size()[2] == 3)

        self.assertTrue(x.size(0) == 5)
        self.assertTrue(x.size(1) == 4)
        self.assertTrue(x.size(2) == 3)
        self.assertTrue(isinstance(x.size(2), SymInt))

        y = create_symbolic_tensor("y", torch.randn(5, 4, 3)[1:], shape_env)
        self.assertTrue(isinstance(y.storage_offset(), SymInt))
        self.assertTrue(y.storage_offset() == 12)

    @skipIfNoSympy
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

    @skipIfNoSympy
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

    @skipIfNoSympy
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

    @skipIfNoSympy
    def test_stride(self):
        shape_env = ShapeEnv()
        x = create_symbolic_tensor("x", torch.randn(5, 5), shape_env)
        self.assertIsInstance(x.stride()[0], SymInt)

    @skipIfNoSympy
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

    @skipIfNoSympy
    def test_numel(self):
        shape_env = ShapeEnv()
        x = create_symbolic_tensor("x", torch.randn(5), shape_env)
        self.assertIsInstance(x.numel(), torch.SymInt)
        self.assertIsInstance(torch.numel(x), torch.SymInt)

        x = torch.rand(3, 3)
        self.assertIsInstance(x.numel(), int)
        self.assertIsInstance(torch.numel(x), int)

    @skipIfNoSympy
    def test_int_to_float(self):
        shape_env = ShapeEnv()
        x = create_symbolic_tensor("x", torch.randn(5), shape_env)
        r = sym_float(x.shape[0])
        self.assertIsInstance(r, torch.SymFloat, msg=type(r))

    @skipIfNoSympy
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

    @skipIfNoSympy
    def test_meta_symint(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 2)
        r = torch.empty(a0, device='meta')
        self.assertIsInstance(r.shape[0], SymInt)

    @skipIfNoSympy
    def test_guard_int(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 2)
        self.assertEqual(guard_int(a0), 2)
        self.assertExpectedInline(str(shape_env.guards[0][0]), """Eq(s0, 2)""")

    @skipIfNoSympy
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

        a2 = create_symint(shape_env, -3)
        r = sym_int(a2 / 2)
        self.assertEqual(guard_int(r), -1)
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        self.assertExpectedInline(str(shape_env.guards[2][0]), """Eq(ceiling(-s2/2), -1)""")

        a3 = create_symint(shape_env, 3)
        r = sym_int(2.0 * sym_float(a3))
        self.assertEqual(guard_int(r), 6)
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        self.assertExpectedInline(str(shape_env.guards[3][0]), """Eq(2*s2, 6)""")

    @skipIfNoSympy
    def test_sym_sqrt(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 4)
        r = sym_sqrt(a0)
        self.assertEqual(r, 2)
        self.assertIsInstance(r, torch.SymFloat, msg=type(r))
        self.assertExpectedInline(str(shape_env.guards[0][0]), """Eq(sqrt(s0), 2)""")

    @skipIfNoSympy
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

    @skipIfNoSympy
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


    @skipIfNoSympy
    def test_int_conversion(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 2)
        self.assertRaisesRegex(RuntimeError, "Trying to extract", lambda: int(a0))

    @skipIfNoSympy
    def test_data_dependent_guard(self):
        shape_env = ShapeEnv()
        s0 = shape_env.create_unbacked_symint()
        self.assertRaises(GuardOnDataDependentSymNode, lambda: bool(s0 == 0))

    @skipIfNoSympy
    def test_non_overlapping_and_dense(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 5)
        r = torch.empty_strided((a0, 7), (1, a0), device='meta')
        self.assertTrue(torch.ops.aten.is_non_overlapping_and_dense.default(r))

    @skipIfNoSympy
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

    @skipIfNoSympy
    def test_deepcopy(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 2)
        assert a0 < 4
        new_shape_env = copy.deepcopy(shape_env)
        self.assertEqual(len(new_shape_env.guards), 1)

    @skipIfNoSympy
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
        new_empty: f32[s0 + s2, 2*s1] = torch.ops.aten.new_empty.default(a_1, [add, add_1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False);  a_1 = add = add_1 = None
        native_dropout = torch.ops.aten.native_dropout.default(new_empty, 0.5, True);  new_empty = None
        getitem: f32[s0 + s2, 2*s1] = native_dropout[0]
        getitem_1: b8[s0 + s2, 2*s1] = native_dropout[1];  native_dropout = None
        return (getitem, getitem_1)""")  # noqa: B950

@skipIfTorchDynamo("Creating ShapeEnv fails for confusing reasons (also we never expect dynamo to see code like this)")
class TestSymNumberMagicMethods(TestCase):
    def _do_test(self, fn, inp1, inp2, shape_env, is_unary_fn):
        # Helper function
        seed_node = (create_symint(shape_env, 1) / 1.).node
        bool_seed_node = (create_symint(shape_env, 1) == 1).node

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
            try:
                if type(v) in (SymBool, bool):
                    return guard_bool(v)
                elif type(v) in (SymFloat, float):
                    return guard_float(v)
                else:  # SymInt, int
                    return guard_int(v)
            except Exception as e:
                raise e

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
            out = guard_fn(out)
            self.assertEqual(out, ref_out)

        if is_unary_fn:
            return

        # Symified second arg
        sym_inp2 = get_sym_inp(inp2)
        with maybe_xfail(inp1, sym_inp2):
            out = lambda_apply(inp1, sym_inp2)
            out = guard_fn(out)
            self.assertEqual(out, ref_out)

        # Symified both args
        with maybe_xfail(sym_inp1, sym_inp2):
            out = lambda_apply(sym_inp1, sym_inp2)
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

    @skipIfNoSympy
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

    @skipIfNoSympy
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

    @skipIfNoSympy
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

    @skipIfNoSympy
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

    @skipIfNoSympy
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

    @skipIfNoSympy
    def test_floordiv_div_by_one(self):
        values = (
            (2.5, 1),
            (2.1, 1.0),
            (2, 1.0),
            (2, 1),
        )

        for x, y in TestFloorDiv.yield_test_cases(values):
            self.assertEqual(TestFloorDiv.python_floordiv(x, y), TestFloorDiv.torch_floordiv(x, y))

    @skipIfNoSympy
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

    @skipIfNoSympy
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

if __name__ == '__main__':
    run_tests()
