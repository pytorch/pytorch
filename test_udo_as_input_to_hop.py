"""
TORCH_COMPILE_DEBUG=1 python3 test_udo_as_input_to_hop.py >output.txt 2>&1
TORCH_COMPILE_DEBUG=1 python3 test_udo_as_input_to_hop.py
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


pass_udo_into_hop = True
run_eager = True
run_compiled = False


def print_if_eager(msg):
    if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
        print(msg)


class FSDPParamGroup:
    def __init__(self):
        self.tensor = torch.randn(4, 4, requires_grad=True)

    def post_backward(self):
        print("post_backward is called!")
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
        print(f"grads: {grads}")
        return (None,) + grads


class RegisterPostBackwardNoUDOFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *inputs: torch.Tensor):
        with torch.no_grad():
            inputs[0].add_(1)
        return inputs

    @staticmethod
    def backward(ctx, *grads: torch.Tensor):
        return grads


def pre_forward_hook(param_group, module, args, kwargs):
    print_if_eager("in pre_forward_hook")
    assert len(kwargs) == 0
    if pass_udo_into_hop:
        args = RegisterPostBackwardFunction.apply(param_group, *args)
    else:
        args = RegisterPostBackwardNoUDOFunction.apply(*args)
    print_if_eager("done pre_forward_hook")
    return args, kwargs


class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x1 = torch.nn.Parameter(torch.randn(4, 4))

    def forward(self, x3, x4):
        out = torch.matmul(x4, x4)
        out = out * torch.matmul(x3, x3)
        out = out * torch.matmul(x3, x3)
        return out

device = "cpu"

if __name__ == "__main__":
    torch.manual_seed(0)
    x3 = torch.randn(4, 4, device=device, requires_grad=True)
    x4 = torch.randn(4, 4, device=device, requires_grad=True)
    x3.register_hook(lambda grad: print(f"x3 grad: {grad}"))
    x4.register_hook(lambda grad: print(f"x4 grad: {grad}"))
    fsdp_param_group = FSDPParamGroup()
    mod = TestModule()
    mod = mod.to(device)
    mod.register_forward_pre_hook(functools.partial(pre_forward_hook, fsdp_param_group), prepend=True, with_kwargs=True)

    if run_eager:
        out = mod(x3, x4)
        out.sum().backward()
        print(f"eager done: mod.x1.grad: {mod.x1.grad}")

    if run_compiled:
        def compiler_fn(gm):
            print("Compiling autograd?")
            return torch.compile(gm, backend="aot_eager", fullgraph=True)
        with compiled_autograd.enable(compiler_fn):
            mod = torch.compile(mod, backend="aot_eager", fullgraph=True)
            out = mod(x3, x4)
            out.sum().backward()
        print(f"compiled done: mod.x1.grad: {mod.x1.grad}")
