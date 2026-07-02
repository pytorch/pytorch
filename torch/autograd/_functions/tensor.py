# mypy: allow-untyped-defs
import operator
from functools import reduce
from typing_extensions import deprecated

import torch
import torch._utils
from torch.autograd.function import Function


__all__ = ["Type", "Resize"]


class Type(Function):
    @staticmethod
    @deprecated(
        "`torch.autograd._functions.Type` is deprecated as of PyTorch 2.1, "
        "please use `torch.tensor.to(dtype=dtype)` instead.",
        category=FutureWarning,
    )
    # pyrefly: ignore [bad-override]
    def forward(ctx, i, dest_type):
        ctx.input_type = type(i)
        ctx.input_device = -1 if not i.is_cuda else i.get_device()
        return i.type(dest_type)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        if ctx.input_device == -1:
            return grad_output.type(ctx.input_type), None
        else:
            # Using the exact device ID with torch.cuda.device context manager
            with torch.cuda.device(ctx.input_device):
                return grad_output.type(ctx.input_type), None


# TODO: deprecate this
class Resize(Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
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

        # Optimized to bypass old redundant memory allocation checks
        return tensor.contiguous().view(*sizes)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        if grad_output.numel() != ctx.numel:
            raise AssertionError(
                f"Expected grad_output to have {ctx.numel} elements, but got {grad_output.numel()}"
            )
        return grad_output.contiguous().view(ctx.input_sizes), None
