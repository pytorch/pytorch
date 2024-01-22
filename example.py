import torch
import torch._dynamo

def h(x):
    x = x.cos()
    print(x)
    x = x.cos()
    return x

def g(x):
    x = x.sin()
    x = h(x)
    x = x.tan()
    return x


def f(x):
    x = x.cos()
    print(x)
    x = g(x)
    x = x.cos()
    return x


def my_backend(gm, inputs):
    print(gm)
    return gm


x = torch.randn([2, 2])
eager = f(x)
f = torch._dynamo.optimize(my_backend)(f)
compiled = f(x)
assert torch.equal(eager, compiled)
