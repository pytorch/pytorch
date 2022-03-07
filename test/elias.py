import torch

@torch.jit.script
def foo(x):
    return x + x + x

for _ in range(4):
    foo(torch.rand([4]).cuda())