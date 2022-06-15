from typing import Sequence
import sympy
import torch
import torch.fx as fx
from torch.utils._pytree import tree_map
import operator
from contextlib import contextmanager
from torch._meta_registrations import meta_funcs, register_meta
from torch.fx.experimental.proxy_tensor import make_fx
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

@register_meta([aten.add.Tensor, aten.sub.Tensor, aten.mul.Tensor], register_dispatcher=False)
def binary_meta(a, b):
    return a.new_empty(a.shape)

@register_meta(aten.cat.default, register_dispatcher=False)
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



x = torch.randn(3, 4, 5)

def f(y):
    x = y * 2
    assert x.shape[0] > 1
    # x = x.sum()
    return x

traced_graph = make_fx(f, decomposition_table={torch.ops.aten.detach.default: lambda x: x})(x)
traced_graph.graph.eliminate_dead_code()
traced_graph.recompile()
print(traced_graph)
print(traced_graph.shape_env.guards)

exit(0)