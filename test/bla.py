import torch

@torch.jit.script
def func(t) :
    x = [1, 2, 3]
    y = x + [1]
    print(x)
    print(y)

func(torch.rand(3))
