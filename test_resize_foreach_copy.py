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


should_resize_storage = True
run_eager = False
run_compiled = True


def print_if_eager(msg):
    if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
        print(msg)


def unsafe_alloc_storage(tensor: torch.Tensor) -> None:
    print_if_eager("unsafe_alloc_storage")
    tensor.untyped_storage().resize_(tensor.numel() * tensor.itemsize)


def unsafe_free_storage(tensor: torch.Tensor) -> None:
    print_if_eager("unsafe_free_storage")
    tensor.untyped_storage().resize_(0)


def post_forward_hook(module, args, output):
    print_if_eager("in post_forward_hook")
    output.register_hook(functools.partial(pre_backward_hook, module))  # do alloc_storage before backward
    module.x1.register_post_accumulate_grad_hook(post_backward_hook)  # do free_storage after backward
    print_if_eager("done post_forward_hook")
    return output


def pre_backward_hook(module, grad) -> None:
    print_if_eager("in pre_backward_hook")
    if should_resize_storage:
        unsafe_alloc_storage(module.x1)
    print_if_eager("done pre_backward_hook")
    return grad


def post_backward_hook(param):
    print_if_eager("in post_backward_hook")
    if should_resize_storage:
        unsafe_free_storage(param)
    print_if_eager("done pre_backward_hook")
    return


class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x1 = torch.nn.Parameter(torch.randn(4, 4))

    def forward(self, x3):
        if should_resize_storage:
            unsafe_alloc_storage(self.x1)
        x5 = self.x1[:]
        with torch.no_grad():
            torch._foreach_copy_([x5], [x3])
        out = torch.matmul(self.x1, self.x1)
        if should_resize_storage:
            unsafe_free_storage(self.x1)
        return out

device = "cpu"

if __name__ == "__main__":
    x3 = torch.randn(4, 4, device=device, requires_grad=True)
    mod = TestModule()
    mod = mod.to(device)
    mod.register_forward_hook(post_forward_hook, prepend=False)

    if run_eager:
        out = mod(x3)
        out.sum().backward()
        print(f"eager done: mod.x1.grad: {mod.x1.grad}")

    if run_compiled:
        def compiler_fn(gm):
            print("Compiling autograd?")
            return torch.compile(gm, backend="aot_eager", fullgraph=True)
        with compiled_autograd.enable(compiler_fn):
            compiled_mod = torch.compile(mod, backend="aot_eager", fullgraph=True)
            out = compiled_mod(x3)
            out.sum().backward()
        print(f"compiled done: mod.x1.grad: {mod.x1.grad}")

"""
Observations:

should_resize_storage = True
1. run_eager=True, run_compiled=False: can run through without error, does storage resizing correctly
2. run_eager=False, run_compiled=True: end of forward graph does resize_(0) first then t(), causing error:
   `setStorage: sizes [4, 4], strides [1, 4], storage offset 0, and itemsize 4 requiring a storage size of 64 are out of bounds for storage of size 0` error
   compile log: https://gist.github.com/yf225/49001d58ba3bc9cd67c00a499739021d

should_resize_storage = False
1. run_eager=True, run_compiled=False: can run through without error
2. run_eager=False, run_compiled=True: can run through without error
"""
