import torch

@torch.jit.script
def func(x):
    # type: (int) -> int
    print(chr(x))
    return x
print(func(255))
print(chr(255))
