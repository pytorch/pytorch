import contextlib

import torch
from torch.func import vmap
from torch._functorch.vmap import vmap_increment_nesting


@contextlib.contextmanager
def context():
    try:
        # torch.set_default_dtype(torch.float64)
        torch._dynamo.graph_break()
        yield
    finally:
        pass
        # torch.set_default_dtype(torch.float32)

a = 0

class context22:
    def __enter__(self):
        # torch.set_default_dtype(torch.float64)
        # torch._C._functorch._vmap_increment_nesting(1, 'same')
        global a
        a += 1
        torch._dynamo.graph_break()

    def __exit__(self, exc, typ, tb):
        global a
        a -= 1
        print(a)
        # torch._C._functorch._vmap_decrement_nesting()
        # torch.set_default_dtype(torch.float32)



@torch.compile(backend='aot_eager', fullgraph=False)
def f(x):
    with context22():
        return x.sin()
    # with vmap_increment_nesting(1, 'same'):
    #     torch._dynamo.graph_break()
    #     return x.sin()

# x = torch.randn(2, 3)
# y = f(x)
# print(a)
# print(y.shape)


# def bar():
#     yield 1
#     yield 2
#     yield 3


# @torch.compile(backend="eager", fullgraph=True)
# def foo(x):
#     for y in bar():
#         x = x + y
#     return x


# @torch.compile(backend="eager", fullgraph=False)
# def foo(x):
#     with context():
#         y = x + 1
#     return y
#     # return torch.func.vmap(lambda x: x.sin())(x)

# x = torch.tensor([1.0])
# y = foo(x)
# print(y)


# def bar():
#     yield 1
#     yield 2
#     yield 3

# @torch.compile(backend="eager", fullgraph=True)
# def foo(x):
#     for y in bar():
#         x = x + y
#     return x

# y = foo(x)
# print(y)


def f(x):
    print(y)
    yield 1

# g = f(1)

# next(g) # this call should error out, because there's no y

g = f(1)

def h():
    y = 3
    next(g)

h()   # this call should succeed