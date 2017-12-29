import torch
from ..function import Function, traceable
import math


def sort_args(a, b, key=torch.is_tensor):
    return (a, b, True) if key(a) else (b, a, False)


@traceable
class PowConstant(Function):

    @staticmethod
    def forward(ctx, a, b):
        tensor, ctx.constant, ctx.tensor_first = sort_args(a, b)
        if ctx.tensor_first:
            ctx.save_for_backward(tensor)
            return tensor.pow(ctx.constant)
        else:
            result = torch.pow(ctx.constant, tensor)
            ctx.save_for_backward(result)
            return result

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.tensor_first:
            var, = ctx.saved_variables
            return grad_output.mul(ctx.constant).mul(var.pow(ctx.constant - 1)), None
        else:
            var_result, = ctx.saved_variables
            return None, grad_output.mul(var_result).mul_(math.log(ctx.constant))
