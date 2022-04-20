import torch


@torch.jit.script
def foo(a, b, c):
    return a * b + c

def inspect(g):
    print(len(list(g.inputs())))


torch._C._jit_register_python_callback(inspect)

a = torch.rand(5, 5)
b = torch.rand(5, 5)
c = torch.rand(5, 5)

foo(a, b, c)
foo(a, b, c)