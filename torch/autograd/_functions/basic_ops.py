import torch
from ..function import Function, InplaceFunction
from .utils import maybe_unexpand, maybe_unexpand_or_view
import math


class Add(InplaceFunction):

    @staticmethod
    def primspec(g, a, b, inplace=False):
        # TODO: [Export inplace]
        return g.appendNode(g.create("Add", [a, b]))

    @staticmethod
    def forward(ctx, a, b, inplace=False):
        ctx.a_size = a.size()
        ctx.b_size = b.size()
        if inplace:
            ctx.mark_dirty(a)
            return a.add_(b)
        else:
            return a.add(b)

    @staticmethod
    def backward(ctx, grad_output):
        return maybe_unexpand(grad_output, ctx.a_size), maybe_unexpand_or_view(grad_output, ctx.b_size), None


class Sub(InplaceFunction):

    @staticmethod
    def primspec(g, a, b, inplace=False):
        # TODO: [Export inplace]
        return g.appendNode(g.create("Sub", [a, b]))

    @staticmethod
    def forward(ctx, a, b, inplace=False):
        ctx.a_size = a.size()
        ctx.b_size = b.size()
        if inplace:
            ctx.mark_dirty(a)
            return a.sub_(b)
        else:
            return a.sub(b)

    @staticmethod
    def backward(ctx, grad_output):
        return maybe_unexpand(grad_output, ctx.a_size), maybe_unexpand_or_view(grad_output.neg(), ctx.b_size), None


class Mul(Function):

    @staticmethod
    def primspec(g, a, b, inplace=False):
        # TODO: [Export inplace]
        return g.op("Mul", a, b)

    @staticmethod
    def forward(ctx, a, b):
        ctx.a_size = a.size()
        ctx.b_size = b.size()
        ctx.save_for_backward(a, b)
        return a.mul(b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_variables
        return maybe_unexpand(grad_output.mul(b), ctx.a_size), maybe_unexpand_or_view(grad_output.mul(a), ctx.b_size)


class Div(Function):

    @staticmethod
    def forward(ctx, a, b):
        ctx.a_size = a.size()
        ctx.b_size = b.size()
        ctx.save_for_backward(a, b)
        return a.div(b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_variables
        b_rec = b.reciprocal()
        grad_a = grad_output.mul(b_rec)
        grad_b = grad_output.neg().mul(a).mul(b_rec).mul(b_rec)
        return maybe_unexpand(grad_a, ctx.a_size), maybe_unexpand_or_view(grad_b, ctx.b_size)


class Pow(Function):

    @staticmethod
    def forward(ctx, a, b):
        ctx.a_size = a.size()
        ctx.b_size = b.size()
        ctx.save_for_backward(a, b)
        return a.pow(b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_variables
        grad_a = grad_output.mul(b).mul(a.pow(b - 1))
        grad_b = grad_output.mul(a.pow(b)).mul(a.log())
        return maybe_unexpand(grad_a, ctx.a_size), maybe_unexpand_or_view(grad_b, ctx.b_size)


def sort_args(a, b):
    return (a, b, True) if torch.is_tensor(a) else (b, a, False)


class AddConstant(InplaceFunction):

    @staticmethod
    def forward(ctx, a, b, inplace=False):
        tensor, constant, ctx.tensor_first = sort_args(a, b)
        if inplace:
            ctx.mark_dirty(tensor)
            return tensor.add_(constant)
        else:
            return tensor.add(constant)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.tensor_first:
            return grad_output, None, None
        else:
            return None, grad_output, None


class SubConstant(InplaceFunction):

    @staticmethod
    def forward(ctx, a, b, inplace=False):
        tensor, constant, ctx.tensor_first = sort_args(a, b)
        if ctx.tensor_first:
            if inplace:
                ctx.mark_dirty(tensor)
                return tensor.sub_(constant)
            else:
                return tensor.sub(constant)
        else:
            if inplace:
                ctx.mark_dirty(tensor)
                return tensor.neg_().add_(constant)
            else:
                return tensor.neg().add_(constant)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.tensor_first:
            return grad_output, None, None
        else:
            return None, grad_output.neg(), None


class MulConstant(InplaceFunction):

    @staticmethod
    def forward(ctx, a, b, inplace=False):
        tensor, ctx.constant, ctx.tensor_first = sort_args(a, b)
        if inplace:
            ctx.mark_dirty(tensor)
            return tensor.mul_(ctx.constant)
        else:
            return tensor.mul(ctx.constant)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.mul(ctx.constant)
        if ctx.tensor_first:
            return grad_input, None, None
        else:
            return None, grad_input, None


class DivConstant(InplaceFunction):

    @staticmethod
    def forward(ctx, a, b, inplace=False):
        tensor, ctx.constant, ctx.tensor_first = sort_args(a, b)
        ctx.inplace = inplace
        if ctx.tensor_first:
            if inplace:
                ctx.mark_dirty(tensor)
                return tensor.div_(ctx.constant)
            else:
                return tensor.div(ctx.constant)
        else:
            ctx.save_for_backward(tensor)
            if inplace:
                ctx.mark_dirty(tensor)
                return tensor.reciprocal_().mul_(ctx.constant)
            else:
                return tensor.reciprocal().mul_(ctx.constant)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.tensor_first:
            return grad_output.div(ctx.constant), None, None
        else:
            v, = ctx.saved_variables
            if ctx.inplace:
                return None, grad_output.mul(v).mul(v).div_(-ctx.constant), None
            else:
                v_rep = v.reciprocal()
                return None, grad_output.mul(v_rep).mul(v_rep).mul_(-ctx.constant), None


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


class Negate(InplaceFunction):

    @staticmethod
    def primspec(g, i, inplace=False):
        # TODO: [Export inplace]
        return g.appendNode(g.create("Scale", [i]).f_("scale", -1))

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
