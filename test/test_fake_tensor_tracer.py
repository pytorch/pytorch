from torch.fx.experimental.proxy_tensor import make_fx
import torch

def f(x):
    return x.cos()

print(make_fx(f)(torch.randn(5)))