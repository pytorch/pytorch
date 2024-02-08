import torch
import unittest
import torch._dynamo.config as config
from torch.func import grad, vjp, vmap

config.capture_func_transforms = True


# @torch.compile(backend="eager")
# def fn(x, p):
#     _, func = torch.func.vjp(torch.sin, x)
#     return func(p)

# x = torch.randn(4, 4)
# p = torch.randn(4, 4)
# y = fn(x, p)
# print(y)

@torch.compile(backend='eager')
def f(x):
    return vmap(torch.sin)(x)


x = torch.randn(3, 4, 5, 6)
# f(x)
vmap(vmap(f, randomness='same'), randomness='error')(x)
vmap(vmap(f, randomness='error'), randomness='same')(x)