from typing import Sequence
import sympy
import torch
import torch.fx as fx
from torch.utils._pytree import tree_map
import operator
from contextlib import contextmanager
from torch._meta_registrations import meta_funcs, register_meta
# from torch.fx.experimental.proxy_tensor import PySymInt
aten = torch.ops.aten

from torch._C import _disabled_torch_function_impl

@contextmanager
def no_dispatch():
    guard = torch._C._DisableTorchDispatch()  # type: ignore[attr-defined]
    try:
        yield
    finally:
        del guard

@register_meta(aten.detach.default)
def nop(x):
    return x
# meta_funcs = {}

# def register_meta(op):
#     def decorator(f):
#         def add_func(op):
#             meta_funcs[op] = f
#         tree_map(add_func, op)
#         return f
#     return decorator

@register_meta(aten.ones_like.default)
def n_like(arg, **kwargs):
    return arg.new_empty(arg.shape)

@register_meta([aten.add.Tensor, aten.sub.Tensor, aten.mul.Tensor])
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


@register_meta(aten.sum.default)
def sum_meta(x):
    return x.new_empty(())


@register_meta(aten.expand.SymInt)
def expand_symint_meta(a, size, implicit=False):
    return a.new_empty(size)


class PySymInt(object):
    def __init__(self, expr, shape_env):
        self.expr = expr
        self.shape_env = shape_env

    def wrap(self, num):
        return PySymInt(sympy.Integer(num), self.shape_env)

    def __str__(self):
        return f"PySymInt({self.expr})"

    def __int__(self):
        # import pdb; pdb.set_trace()
        return self.shape_env.evaluate_expr(self.expr)

    def __bool__(self):
        return bool(self.shape_env.evaluate_expr(self.expr))

magic_methods = {
    'add': lambda a, b: a + b,
    'radd': lambda a, b: a + b,
    'sub': lambda a, b: a - b,
    'mul': lambda a, b: a * b,
    'div': lambda a, b: a / b,
    'mod': lambda a, b: a % b,
    'eq': lambda a, b: sympy.Eq(a, b),
    'gt': lambda a, b: sympy.Gt(a, b),
    'lt': lambda a, b: sympy.Lt(a, b),
}

for method, func in magic_methods.items():
    method_name = f'{method}'
    def create_magic_impl(func):
        def magic_impl(self, other):
            if isinstance(other, PySymInt):
                other = other.expr
            return PySymInt(func(self.expr, other), self.shape_env)
        return magic_impl

    # this should be wrapped transparently into torch._C.SymbolicIntNode
    setattr(PySymInt, method_name, create_magic_impl(func))


class ShapeEnv(object):
    def __init__(self):
        self.guards = []
        self.shape_env = {}

    def create_symint(self, name, val):
        sympy_expr = sympy.Symbol(name)
        py_sym_int = PySymInt(sympy_expr, self)
        cpp_sym_int = torch._C.SymbolicIntNode.new_symint(py_sym_int)
        self.shape_env[sympy_expr] = val
        return cpp_sym_int

    def evaluate_expr(self, expr):
        concrete_val = expr.subs(self.shape_env)
        self.guards.append((expr, concrete_val))
        return concrete_val


def create_contiguous(shape):
    if len(shape) == 0:
        return []
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
        contiguous_strides = create_contiguous(sym_shape)
        print(sym_shape, contiguous_strides)
        r = torch.Tensor._make_wrapper_subclass(
            cls, sym_shape,
            contiguous_strides, offset,
            dtype=dtype, layout=layout, requires_grad=requires_grad,
            device=device
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
   sym_shapes = tuple([shape_env.create_symint(f"{name}_{idx}", val) for idx, val in enumerate(arg.size())])
   sym_strides = tuple([shape_env.create_symint(f"{name}_{idx}_stride", val) for idx, val in enumerate(arg.stride())])
   return FakeSymbolicTensor(sym_shapes, sym_strides, arg.dtype, arg.layout, arg.requires_grad, arg.device)

from functorch import make_fx

shape_env = ShapeEnv()
x = create_symbolic_tensor("x", torch.randn(3,4,5, requires_grad=True), shape_env)

def f(y):
    x = y * 2
    x = x.sum()
    # x = x.expand(y.shape)
    # return x
    return torch.autograd.grad(x, y)

traced_graph = make_fx(f, decomposition_table={torch.ops.aten.detach.default: lambda x: x})(x)
traced_graph.graph.eliminate_dead_code()
traced_graph.recompile()
print(traced_graph)
print(shape_env.guards)

exit(0)
y = (x + 2).sum()
out = torch.autograd.grad(y, x)
print(out[0].shape)
exit(0)
expand_x = x.expand(x.shape[0], x.shape[0])
if expand_x.shape[0] > 3:
    result = expand_x + expand_x
else:
    result = expand_x + expand_x
print(torch.cat([expand_x, expand_x]).shape[0])
print(shape_env.guards)
