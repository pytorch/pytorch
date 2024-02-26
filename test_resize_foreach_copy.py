"""
Produced AOT graphs: https://gist.github.com/yf225/3a9ef062a750dd02d7e08b87aa4bc864

Notice that there is no copy_ to copy the mutated values back to the input. Why?
"""

"""
TORCH_COMPILE_DEBUG=1 python3 test_resize_foreach_copy.py >output.txt 2>&1
"""
import functools
import contextlib
import logging
import os
import sys
import traceback

import torch
import torch._dynamo
import torch.nn as nn
from torch._dynamo import compiled_autograd


def unsafe_alloc_storage(tensor: torch.Tensor) -> None:
    tensor.untyped_storage().resize_(tensor.numel() * tensor.itemsize)


def unsafe_free_storage(tensor: torch.Tensor) -> None:
    tensor.untyped_storage().resize_(0)


def pre_backward(module, grad) -> None:
    unsafe_alloc_storage(module.x1)
    return grad


def post_backward_hook(param):
    unsafe_free_storage(param)


class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x1 = torch.nn.Parameter(torch.randn(4, 4))
        self.x1.register_hook(functools.partial(pre_backward, self))  # do alloc_storage before backward
        self.x1.register_post_accumulate_grad_hook(post_backward_hook)  # do free_storage after backward

    def forward(self, x3):
        unsafe_alloc_storage(self.x1)
        x5 = self.x1[:]
        with torch.no_grad():
            torch._foreach_copy_([x5], [x3])
        out = self.x1 * self.x1
        unsafe_free_storage(self.x1)
        return out

device = "cpu"

if __name__ == "__main__":
    x3 = torch.randn(4, 4, device=device, requires_grad=True)
    mod = TestModule()
    mod = mod.to(device)

    def compiler_fn(gm):
        print("Compiling autograd?")
        return torch.compile(gm, backend="aot_eager", fullgraph=True)
    with compiled_autograd.enable(compiler_fn):
        out = torch.compile(mod, backend="aot_eager", fullgraph=True)(x3)
        out.sum().backward()
