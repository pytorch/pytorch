import torch
from ..function import Function, InplaceFunction, traceable
from .utils import maybe_unexpand, maybe_unexpand_or_view
import math


def sort_args(a, b, key=torch.is_tensor):
    return (a, b, True) if key(a) else (b, a, False)


def gen_inputs(g, a, b):
    tensor, constant, tensor_first = sort_args(a, b, key=is_node)
    assert tensor.hasType()
    type = str(tensor.type().scalarType())
    broadcast = False
    if len(tensor.type().sizes()) > 1:
        broadcast = True
    constant = g.constant(constant, [0], type).setTypeAs(tensor)
    return tensor, constant, broadcast, tensor_first


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


@traceable
class Negate(InplaceFunction):

    @staticmethod
    def symbolic(g, i, inplace=False):
        # See Note [Export inplace]
        return g.op("Scale", i, scale_f=-1)

    @staticmethod
    def forward(ctx, i, inplace=False):
        if inplace:
            ctx.mark_dirty(i)
            return i.neg_()
        else:
            return i.neg()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg(), None
