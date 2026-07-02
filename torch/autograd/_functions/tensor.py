# mypy: allow-untyped-defs
import operator
from functools import reduce
from typing_extensions import deprecated

import torch
import torch._utils
from torch.autograd.function import Function


class Type(Function):
    @staticmethod
    @deprecated(
        "`torch.autograd._functions.Type` is deprecated as of PyTorch 2.1, "
        "please use `torch.tensor.to(dtype=dtype)` instead.",
        category=FutureWarning,
    )
    def forward(ctx, i, dest_type):
        ctx.input_type = type(i)
        ctx.input_device = -1 if not i.is_cuda else i.get_device()
        return i.type(dest_type)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.input_device == -1:
            return grad_output.type(ctx.input_type), None
        else:
            # FIX: Changed device_index to device context manager
            with torch.accelerator.device(ctx.input_device):
                return grad_output.type(ctx.input_type), None


# TODO: deprecate this
class Resize(Function):
    @staticmethod
    def forward(ctx, tensor, sizes):
        ctx.sizes = sizes
        ctx.numel = reduce(operator.mul, sizes, 1)
        if tensor.numel() != ctx.numel:
            raise RuntimeError(
                f"requested resize to {'x'.join(map(str, sizes))} ({ctx.numel} elements in total), "
                f"but the given tensor has a size of {'x'.join(map(str, tensor.size()))} ({tensor.numel()} elements). "
                f"autograd's resize can only change the shape of a given tensor, while preserving the number of elements."
            )
        
        ctx.input_sizes = tensor.size()
        # FIX: Cleaned up redundant copy_ and obsolete tensor.new() branching
        return tensor.contiguous().view(*sizes)

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output.numel() != ctx.numel:
            raise AssertionError(
                f"Expected grad_output to have {ctx.numel} elements, but got {grad_output.numel()}"
            )
        return grad_output.contiguous().view(ctx.input_sizes), None
