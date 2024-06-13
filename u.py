import torch
import traceback
from torch._subclasses.fake_tensor import maybe_get_fake_mode

@torch.compiler.allow_in_graph
def f(x):
    if maybe_get_fake_mode(x):
        print("fake")
        return x
    print("real")
    return x

f._dynamo_split_before_autograd = True

@torch.compile(backend="aot_eager")
def g(x):
    return f(x + 1) + 1

g(torch.randn(3))
g(torch.randn(3))
