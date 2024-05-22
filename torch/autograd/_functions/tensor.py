import operator
from functools import reduce
from typing_extensions import deprecated

import torch
import torch._utils
from ..function import Function


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
            with torch.cuda.device(ctx.input_device):
                return grad_output.type(ctx.input_type), None


# TODO: deprecate this
class Resize(Function):
    @staticmethod
    def forward(ctx, tensor, sizes):
        ctx.sizes = sizes
        ctx.numel = reduce(operator.mul, sizes, 1)
        if tensor.numel() != ctx.numel:
            raise RuntimeError(
                (
                    "requested resize to {} ({} elements in total), "
                    "but the given tensor has a size of {} ({} elements). "
                    "autograd's resize can only change the shape of a given "
                    "tensor, while preserving the number of elements. "
                ).format(
                    "x".join(map(str, sizes)),
                    ctx.numel,
                    "x".join(map(str, tensor.size())),
                    tensor.numel(),
                )
            )
        ctx.input_sizes = tensor.size()
        if tensor.is_quantized:
            tensor.copy_(tensor)
            return tensor.contiguous().view(*sizes)
        if tensor.is_contiguous():
            result = tensor.new(tensor).contiguous().view(*sizes)
            return result
        else:
            return tensor.contiguous().view(*sizes)

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.numel() == ctx.numel
        return grad_output.contiguous().view(ctx.input_sizes), None
