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

from functorch import make_fx

x = torch.randn(3, 4, 5, requires_grad=True)

def f(y):
    x = y * 2
    assert x.shape[0] > 1
    x = x.sum()
    # return x
    return torch.autograd.grad(x, y)

traced_graph = make_fx(f, decomposition_table={torch.ops.aten.detach.default: lambda x: x})(x)
traced_graph.graph.eliminate_dead_code()
traced_graph.recompile()
print(traced_graph)
print(traced_graph.shape_env.guards)

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
