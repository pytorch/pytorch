# -*- coding: utf-8 -*-
# Owner(s): ["oncall: jit"]

from torch._C import _disabled_torch_function_impl
import torch.fx
import torch.nn.functional as F
from torch.testing._internal.common_utils import run_tests, TestCase
import unittest
import torch
import operator
import itertools
from torch.utils._pytree import tree_map
from torch.fx.experimental.symbolic_shapes import ShapeEnv, PySymInt

aten = torch.ops.aten

try:
    import sympy
    HAS_SYMPY = True
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
    def __new__(cls, sym_shape, sym_strides, dtype, layout, requires_grad, device):
        # sym_strides doesn't work yet
        # TODO: this is wrong in general
        offset = 0
        r = torch.Tensor._make_wrapper_subclass(
            cls, sym_shape,
            create_contiguous(sym_shape), offset,
            dtype=dtype, layout=layout, requires_grad=requires_grad,
            device=device,
        )

        r.sym_shape = sym_shape
        return r

    __torch_function__ = _disabled_torch_function_impl

    def new_empty(self, shape):
        return FakeSymbolicTensor(shape, None, self.dtype, self.layout, self.requires_grad, self.device)

    @classmethod
    def __torch_dispatch__(cls, func_overload, types, args=(), kwargs=None):
        if func_overload in meta_funcs:
            return meta_funcs[func_overload](*args, **kwargs)

        if func_overload == torch.ops.aten.sym_size.default:
            self = args[0]
            return self.sym_shape

        # some calls can be redirected to `sym_size` rather than
        # `sym_sizes`. `sym_size` uses `dim` to canonicalize an index
        # so we need to implement both `sym_size` and `dim` for python
        # tensors
        if func_overload == torch.ops.aten.dim.default:
            self = args[0]
            return len(self.sym_shape)

        if func_overload == torch.ops.aten.new_empty.default:
            self = args[0]
            shape = args[1]
            return FakeSymbolicTensor(shape, self.stride(), self.dtype, self.layout, self.requires_grad, self.device)

        raise RuntimeError(f"operator {func_overload} not supported")


def create_symbolic_tensor(name, arg, shape_env):
    sym_shapes = tuple([shape_env.create_symint(f"{name}_{idx}", val) for idx, val in enumerate(arg.size())])
    sym_strides = tuple([shape_env.create_symint(f"{name}_{idx}_stride", val) for idx, val in enumerate(arg.stride())])
    return FakeSymbolicTensor(sym_shapes, sym_strides, arg.dtype, arg.layout, arg.requires_grad, arg.device)


CPP_SYMINT_CLASS = type(torch._C.SymIntNode.new_symint(1))


class TestPySymInt(TestCase):

    @skipIfNoSympy
    def test_arith_ops(self):
        shape_env = ShapeEnv()
        symints = []
        for i in range(5):
            symints.append((i, shape_env.create_symint(f"s{i}", i)))

        ops = [operator.add, operator.sub, operator.floordiv, operator.mul, operator.mod]

        for op in ops:
            for args in itertools.permutations(symints, 2):
                if not isinstance(args[0][1], int) and ((op != operator.mod or op != operator.floordiv) and args[1][0] != 0):
                    self.assertTrue(op(args[0][1], args[1][1]) == op(args[0][0], args[1][0]))


    @skipIfNoSympy
    def test_reverse_arith_ops(self):
        shape_env = ShapeEnv()

        a = shape_env.create_symint("s1", 2)
        self.assertTrue(5 // a == 5 // 2)

        a = shape_env.create_symint("s1", 2)
        self.assertTrue(5 * a == 5 * 2)


    @skipIfNoSympy
    def test_roundtrip(self):
        shape_env = ShapeEnv()
        x = create_symbolic_tensor("x", torch.randn(5, 4, 3), shape_env)
        self.assertTrue(not isinstance(x.shape[0], PySymInt))
        self.assertTrue(isinstance(x.shape[0], CPP_SYMINT_CLASS))

        self.assertTrue(x.shape[0] == 5)
        self.assertTrue(x.shape[1] == 4)
        self.assertTrue(x.shape[2], 3)

        self.assertTrue(x.size()[0], 5)
        self.assertTrue(x.size()[1], 4)
        self.assertTrue(isinstance(x.size()[1], CPP_SYMINT_CLASS))
        self.assertTrue(x.size()[2] == 3)

        self.assertTrue(x.size(0) == 5)
        self.assertTrue(x.size(1) == 4)
        self.assertTrue(x.size(2) == 3)
        self.assertTrue(isinstance(x.size(2), CPP_SYMINT_CLASS))

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
        y = create_symbolic_tensor("y", torch.randn(1, 4, 1), shape_env)
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
        self.assertTrue(z.shape[2] == int(y.shape[2]))

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
    def test_size_expressions(self):
        shape_env = ShapeEnv()
        x = create_symbolic_tensor("x", torch.randn(5), shape_env)
        expand_x = x.expand(x.shape[0], x.shape[0])
        if expand_x.shape[0] > 3:
            result = expand_x + expand_x
        else:
            result = expand_x + expand_x

        gt_op = shape_env.guards[0][0]
        self.assertTrue(isinstance(gt_op, sympy.core.relational.StrictGreaterThan))
        self.assertTrue(str(x.shape[0]), str(gt_op.args[0]))
        self.assertTrue(str(expand_x.shape[1]), str(x.shape[0]))
        self.assertTrue(str(expand_x.shape[1]), str(result.shape[0]))

    @skipIfNoSympy
    def test_aten_ops(self):

        shape_env = ShapeEnv()
        x = create_symbolic_tensor("x", torch.randn(5), shape_env)
        torch.ops.aten.narrow_copy.default(x, 0, 0, x.shape[0])

        shape_env = ShapeEnv()
        x = create_symbolic_tensor("x", torch.randn(5, 4, 3), shape_env)
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
        a0 = shape_env.create_symint("a0", 2)
        r = torch.empty(a0, device='meta')
        self.assertIsInstance(r.shape[0], CPP_SYMINT_CLASS)


if __name__ == '__main__':
    run_tests()
