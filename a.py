# import torch
# from torch.testing._internal.two_tensor import TwoTensor
# from torch.nested._internal.nested_tensor import jagged_from_list


# def get_jagged_tensor(nested_size, offsets, requires_grad=True):
#     # Makes a jagged tensor with N constituent tensors with size
#     # as specified ((S0, S1, S2), D)
#     D = nested_size[1]
#     out = []
#     for s in nested_size[0]:
#         out.append(torch.randn(s, D, requires_grad=requires_grad, dtype=torch.float64))
#     return jagged_from_list(out, offsets)


# i = torch.randn(2, 2, requires_grad=True)
# x = TwoTensor(i, i.clone())
# y = TwoTensor(x.clone(), x.clone())

# nt, _ = get_jagged_tensor(((2, 3, 4), 5), None, True)
# # z = TwoTensor(y, y)

# @torch.compile(backend='aot_eager', dynamic=True)
# def fn(t):
#     return t.sin() * t.shape[0]

# k = fn(nt)


# def f(x, i, y):
#     out1 = x.sin() + i.sin() + y.sin()
#     val1 = x.shape[0] * i.shape[1] * y.shape[0]
#     return out1 * val1

# out = f(x, i, y)

# x_test = x.clone().detach().requires_grad_(True)
# i_test = i.clone().detach().requires_grad_(True)
# y_test = y.clone().detach().requires_grad_(True)

# out_test = torch.compile(f, backend="aot_eager", dynamic=True)(x_test, i_test, y_test)
# print(torch.allclose(out, out_test))


# out.sum().backward()
# out_test.sum().backward()
# print(torch.allclose(x.grad, x_test.grad))
# print(torch.allclose(i.grad, i_test.grad))
# print(torch.allclose(y.grad, y_test.grad))


# run twice to exercise code path with a cache hit.
# ENABLE_AOT_AUTOGRAD_CACHE=1 python tmp.py
# ENABLE_AOT_AUTOGRAD_CACHE=1 python tmp.py

# code for tmp.py
# import torch
# from torch.testing._internal.two_tensor import TwoTensor

# @torch.compile
# def f(x):
#     tmp = x.sin()
#     s0 = tmp.shape[0]
#     return tmp.expand(s0, s0)


# x_a = torch.randn(4, requires_grad=True)
# x = TwoTensor(x_a, x_a.clone())
# out = f(x)
# out.sum().backward()


# @torch.compile(backend="aot_eager_decomp_partition")
# def f(x, y):
#     return torch.cat([x, y])

# x_a = torch.randn(4, 5, 6)
# x = TwoTensor(x_a, x_a.clone())

# y_a = torch.randn(7, 5, 6)
# y = TwoTensor(y_a, y_a.clone())

# # dims 0 and 2 are dynamic, dim 1 is static
# torch._dynamo.mark_dynamic(x, 0)
# torch._dynamo.mark_dynamic(x, 2)
# # also mark the inner tensors as dynamic
# torch._dynamo.mark_dynamic(x.a, 0)
# torch._dynamo.mark_dynamic(x.a, 2)
# torch._dynamo.mark_dynamic(x.b, 0)
# torch._dynamo.mark_dynamic(x.b, 2)

# # dims 0 and 2 are dynamic, dim 1 is static
# torch._dynamo.mark_dynamic(y, 0)
# torch._dynamo.mark_dynamic(y, 2)
# # also mark the inner tensors as dynamic
# torch._dynamo.mark_dynamic(y.a, 0)
# torch._dynamo.mark_dynamic(y.a, 2)
# torch._dynamo.mark_dynamic(y.b, 0)
# torch._dynamo.mark_dynamic(y.b, 2)

# out = f(x, y)
# out.sum().backward()

# import torch
# import torch.nn.functional as F

# def fn(query, key, value):
#     return F.scaled_dot_product_attention(query, key, value)

# B = 4
# query = torch.rand(32, B, 8, 128, dtype=torch.float16, device="cuda")
# key = torch.rand(B, 32, 8, 128, dtype=torch.float16, device="cuda")
# value = torch.rand(32, 8, 128, dtype=torch.float16, device="cuda")
# y = torch.vmap(fn, in_dims=(1, 0, None))(query, key, value)
# print(y.shape)

# import torch

# m = torch.nn.Linear(3, 3)
# x = torch.nn.Linear(3, 3)

# new_bias = torch.randn(3)
# new_weight = torch.randn(3, 3)

# @torch.compile(backend="eager", fullgraph=True)
# def fn(weight, bias, x):
#     return torch.func.functional_call(m, {"weight": new_weight, "bias": new_bias}, x)

# x = torch.randn(2, 3)
# y = fn(new_weight, new_bias, x)
# expected = torch.nn.functional.linear(x, new_weight, new_bias)
# assert torch.allclose(y, expected)


# import torch

# @torch.compile(backend="eager", fullgraph=True)
# def f(d, t):
#     d.pop(1)
#     d[2] = t

# t = torch.zeros(2)
# d = {1: t, 2: t}
# f(d, t)
# print(d)


# @torch.compile(backend="eager", fullgraph=True)
# def f2(x):
#     return dict(a=x+2, b=x+3)


# t = torch.zeros(2)
# print(f2(t))


# import torch
# from torch.testing._internal.two_tensor import TwoTensor

# @torch.compile(backend="eager", dynamic=True)
# def f(t):
#     tmp = t._base if t._is_view() else t
#     return tmp + 1

# x_a = torch.randn(4, 4, requires_grad=True)
# x = TwoTensor(x_a, x_a.clone())
# out = f(x[3])
# out = f(x.a[3])


# import torch


# src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
# index = torch.tensor([1, 1, 0, 1, 2, 1])
# input = torch.tensor([1.0, 2.0, 3.0, 4.0])

# # Simulate a batch dimension of 1
# src = src.unsqueeze(0)
# index = index.unsqueeze(0)
# input = input.unsqueeze(0)


# def _fn(inputs, reduce):
#     _src, _index, _input = inputs
#     return _input.scatter_reduce_(0, _index, _src, reduce=reduce)


# for reduce in ("sum", "prod", "mean", "amax", "amin"):
#     result = torch.vmap(_fn, in_dims=(0, None))((src, index, input), reduce)
#     print(result)

# import torch
# from contextlib import contextmanager

# def g(gen, x):
#     return next(gen) + x

# def f(gen, x):
#     return next(gen) + x

# @contextmanager
# def gen():
#     yield 1
#     yield 2

# def h(x):
#     return g(gen, x) + f(gen, x)


def f():
    y = 123
    @contextmanager
    def gen():
        yield 1 + y
    return gen

def g(gen, x):
    return next(gen) + x

def h(x):
    gen = f(x)
    g(gen, x)


# what happens if we change the parent of a InlineTracer (case above)

# X - rename the VT to be SingleYieldGenerator as we cannot handle multiple yields atm.
# how hard it would be to create a generic version of it?

# X - graph breaks should not be allowed(?) - No!
# if we have a graph break, the entire generator should run in eager mode or error(?)
def foo(args):

    with ctx:
        # ...
        print(args[0])
        # ...


    with ctx:
        # ...
    print(args[0])
    with ctx:
        # ...