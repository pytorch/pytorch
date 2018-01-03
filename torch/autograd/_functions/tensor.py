from functools import reduce
import torch

from ..function import Function
from ..variable import Variable


class Type(Function):

    @staticmethod
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
        ctx.numel = reduce(lambda x, y: x * y, sizes, 1)
        if tensor.numel() != ctx.numel:
            raise RuntimeError(("requested resize to {} ({} elements in total), "
                                "but the given tensor has a size of {} ({} elements). "
                                "autograd's resize can only change the shape of a given "
                                "tensor, while preserving the number of elements. ").format(
                'x'.join(map(str, sizes)), ctx.numel,
                'x'.join(map(str, tensor.size())), tensor.numel()))
        ctx.input_sizes = tensor.size()
        if tensor.is_contiguous():
            result = tensor.new(tensor).contiguous().view(*sizes)
            ctx.mark_shared_storage((tensor, result))
            return result
        else:
            return tensor.contiguous().view(*sizes)

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.numel() == ctx.numel
        return grad_output.contiguous().view(ctx.input_sizes), None


class Repeat(Function):

    @staticmethod
    def forward(ctx, input, repeats):
        ctx.repeats = repeats
        ctx.input_dims = input.dim()
        return input.repeat(repeats)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        num_unsqueezed = grad_output.dim() - ctx.input_dims
        for _ in range(num_unsqueezed):
            grad_input = grad_input.sum(0, keepdim=False)
        for dim, repeat in enumerate(ctx.repeats[num_unsqueezed:]):
            if repeat == 1:
                continue
            grad_input = sum(grad_input.chunk(repeat, dim))
        return grad_input, None
