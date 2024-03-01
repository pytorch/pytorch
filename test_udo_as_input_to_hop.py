"""
TORCH_COMPILE_DEBUG=1 python3 test_udo_as_input_to_hop.py >output.txt 2>&1
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


run_eager = False
run_compiled = True


def print_if_eager(msg):
    if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
        print(msg)


def pre_forward_hook(module, args, kwargs):
    print_if_eager("in pre_forward_hook")
    assert len(kwargs) == 0
    args = RegisterPostBackwardFunction.apply(self, *args)
    print_if_eager("done pre_forward_hook")
    return args, kwargs


class FSDPParamGroup:
    def __init__(self, tensor):
        self.tensor = tensor

    def post_backward(self):
        with torch.no_grad():
            self.tensor.add_(1)


class RegisterPostBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, param_group: FSDPParamGroup, *inputs: torch.Tensor):
        # All tensors in `inputs` should require gradient
        ctx.param_group = param_group
        return inputs

    @staticmethod
    def backward(ctx, *grads: torch.Tensor):
        ctx.param_group.post_backward()
        return (None,) + grads


class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x1 = torch.nn.Parameter(torch.randn(4, 4))

    def forward(self, x3):
        x5 = self.x1[:]
        with torch.no_grad():
            torch._foreach_copy_([x5], [x3])
        out = torch.matmul(self.x1, self.x1)
        return out

device = "cpu"

if __name__ == "__main__":
    x3 = torch.randn(4, 4, device=device, requires_grad=True)
    mod = TestModule()
    mod = mod.to(device)
    mod.register_forward_pre_hook(pre_forward_hook, prepend=True, with_kwargs=True)

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
