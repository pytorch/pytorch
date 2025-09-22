import torch

@torch.compile(fullgraph=True)
def foo(a):
    return torch.randn(5) * a.item()

foo(torch.tensor(2.0))
