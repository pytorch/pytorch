import torch

@torch.jit.script
def func(t):
    s = t.size()
    v = t.reshape(s[1], s[0])
    v[
