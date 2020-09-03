import torch

a = 4

@torch.jit.script
def foo():
    return a
