import torch
from torch import _inductor as inductor
from torch._dynamo import compiled_autograd

def compiler_fn(gm):
    """Same as torch.compile() but counts number of compiles"""

    def inner_compiler(gm_, example_inputs_):
        return inductor.compile(gm_, example_inputs_)

    return torch.compile(gm, backend=inner_compiler, fullgraph=True, dynamic=True)

def fn():
    x = torch.ones(10, requires_grad=True)
    out = torch.sin(x)
    loss = out.sum()
    loss.backward()
    yield x.grad

print(list(fn()))
with compiled_autograd.enable(compiler_fn):
    print(list(fn()))
