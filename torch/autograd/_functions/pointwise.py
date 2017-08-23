import torch

from itertools import repeat

from ..._thnn import type2backend
from ..function import Function, InplaceFunction
from ..variable import Variable
from .utils import maybe_unexpand, maybe_unexpand_or_view


class Exp(InplaceFunction):

    @staticmethod
    def forward(ctx, i, inplace=False):
        if inplace:
            ctx.mark_dirty(i)
            result = i.exp_()
        else:
            result = i.exp()
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_variables
        return grad_output * result, None


class Log(Function):

    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i.log()

    @staticmethod
    def backward(ctx, grad_output):
        i, = ctx.saved_variables
        return grad_output.div(i)


class Log1p(Function):

    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i.log1p()

    @staticmethod
    def backward(ctx, grad_output):
        i, = ctx.saved_variables
        return grad_output.div(i.add(1))


class Tanh(InplaceFunction):

    @staticmethod
    def primspec(g, i, inplace=False):
        if inplace:
            return None
        return g.appendNode(g.create("Tanh", [i]))

    @staticmethod
    def forward(ctx, i, inplace=False):
        if inplace:
            ctx.mark_dirty(i)
            result = i.tanh_()
        else:
            result = i.tanh()
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_variables
        if grad_output.volatile:
            grad_input = Variable(grad_output.data.new(grad_output.size()), volatile=True)
            backend = type2backend[type(result.data)]
            backend.Tanh_updateGradInput(backend.library_state, None, grad_output.data,
                                         grad_input.data, result.data)
        else:
            grad_input = grad_output * (1 - result * result)
        return grad_input, None


class Sigmoid(InplaceFunction):

    @staticmethod
    def primspec(g, i, inplace=False):
        if inplace:
            return None
        return g.appendNode(g.create("Sigmoid", [i]))

    @staticmethod
    def forward(ctx, i, inplace=False):
        if inplace:
            ctx.mark_dirty(i)
            result = i.sigmoid_()
        else:
            result = i.sigmoid()
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_variables
        if grad_output.volatile:
            grad_input = Variable(grad_output.data.new(grad_output.size()), volatile=True)
            backend = type2backend[type(result.data)]
            backend.Sigmoid_updateGradInput(backend.library_state, None, grad_output.data,
                                            grad_input.data, result.data)
        else:
            grad_input = grad_output * ((1 - result) * result)
        return grad_input, None


class Sinh(Function):

    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i.sinh()

    @staticmethod
    def backward(ctx, grad_output):
        i, = ctx.saved_variables
        return grad_output * i.cosh()


class Cosh(Function):

    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i.cosh()

    @staticmethod
    def backward(ctx, grad_output):
        i, = ctx.saved_variables
        return grad_output * i.sinh()


class Abs(Function):

    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i.abs()

    @staticmethod
    def backward(ctx, grad_output):
        i, = ctx.saved_variables
        return grad_output * i.sign()


class Clamp(Function):

    @staticmethod
    def forward(ctx, i, min_val, max_val):
        ctx._mask = (i.ge(min_val) * i.le(max_val))
        return i.clamp(min_val, max_val)

    @staticmethod
    def backward(ctx, grad_output):
        mask = Variable(ctx._mask.type_as(grad_output.data))
        return grad_output * mask, None, None


class Sqrt(Function):

    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i.sqrt()

    @staticmethod
    def backward(ctx, grad_output):
        i, = ctx.saved_variables
        return grad_output.mul(i.pow(-0.5)).div_(2)


class Sin(Function):

    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i.sin()

    @staticmethod
    def backward(ctx, grad_output):
        i, = ctx.saved_variables
        return grad_output * i.cos()


class Cos(Function):

    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i.cos()

    @staticmethod
    def backward(ctx, grad_output):
        i, = ctx.saved_variables
        return grad_output.mul(i.sin()).neg_()


class Tan(Function):

    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i.tan()

    @staticmethod
    def backward(ctx, grad_output):
        i, = ctx.saved_variables
        return grad_output.div(i.cos().pow(2))


class Asin(Function):

    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i.asin()

    @staticmethod
    def backward(ctx, grad_output):
        i, = ctx.saved_variables
        return grad_output * (1 - i.mul(i)).sqrt().reciprocal()


class Acos(Function):

    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i.acos()

    @staticmethod
    def backward(ctx, grad_output):
        i, = ctx.saved_variables
        return grad_output.mul((1 - i.mul(i)).sqrt().reciprocal()).neg_()


class Atan(Function):

    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i.atan()

    @staticmethod
    def backward(ctx, grad_output):
        i, = ctx.saved_variables
        return grad_output * i.mul(i).add_(1).reciprocal()


class Atan2(Function):

    @staticmethod
    def forward(ctx, y, x):
        ctx.save_for_backward(y, x)
        return y.atan2(x)

    @staticmethod
    def backward(ctx, grad_output):
        y, x, = ctx.saved_variables
        denominator = y.mul(y).add(x.mul(x)).reciprocal()
        return grad_output * x.mul(denominator), grad_output * y.neg().mul(denominator)


# TODO: make inplace and update grad formulas
class Reciprocal(Function):

    @staticmethod
    def forward(ctx, i):
        result = i.reciprocal()
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_variables
        return grad_output * result.mul(result).neg_()


class Cmax(Function):

    @staticmethod
    def forward(ctx, a, b):
        ctx._a_size = a.size()
        ctx._b_size = b.size()
        ctx._mask = a.gt(b)
        return a.max(b)

    @staticmethod
    def backward(ctx, grad_output):
        mask = Variable(ctx._mask.type_as(grad_output.data))
        return (
            maybe_unexpand(grad_output * mask, ctx._a_size),
            maybe_unexpand_or_view(grad_output * Variable(ctx._mask.eq(0).type_as(grad_output.data)), ctx._b_size)
        )


class CmaxConstant(Function):

    @staticmethod
    def forward(ctx, i, constant):
        ctx._mask = i.gt(constant)
        return i.clamp(min=constant)

    @staticmethod
    def backward(ctx, grad_output):
        mask = Variable(ctx._mask.type_as(grad_output.data))
        return grad_output * mask, None


class Cmin(Function):

    @staticmethod
    def forward(ctx, a, b):
        ctx._a_size = a.size()
        ctx._b_size = b.size()
        ctx._mask = a.lt(b).type_as(a)
        return a.min(b)

    @staticmethod
    def backward(ctx, grad_output):
        mask = Variable(ctx._mask.type_as(grad_output.data))
        return (
            maybe_unexpand(grad_output * mask, ctx._a_size),
            maybe_unexpand_or_view(grad_output * Variable(ctx._mask.eq(0).type_as(grad_output.data)), ctx._b_size)
        )


class CminConstant(Function):

    @staticmethod
    def forward(ctx, i, constant):
        ctx._mask = i.lt(constant)
        return i.clamp(max=constant)

    @staticmethod
    def backward(ctx, grad_output):
        mask = Variable(ctx._mask.type_as(grad_output.data))
        return grad_output * mask, None


class _ConstantGrad(Function):
    grad_value = 0

    @classmethod
    def forward(cls, ctx, *args):
        ctx._num_args = len(args)
        ctx._args0_size = args[0].size()
        return getattr(args[0], cls.__name__.lower())(*args[1:])

    @classmethod
    def backward(cls, ctx, grad_output):
        return (maybe_unexpand(grad_output.mul(cls.grad_value), ctx._args0_size),) + (ctx._num_args - 1) * (None,)


class Floor(_ConstantGrad):
    pass


class Ceil(_ConstantGrad):
    pass


class Round(_ConstantGrad):
    pass


class Sign(_ConstantGrad):
    pass


class Trunc(_ConstantGrad):
    pass


class Frac(_ConstantGrad):
    grad_value = 1


class Fmod(_ConstantGrad):
    grad_value = 1


class Remainder(_ConstantGrad):
    grad_value = 1


class Lerp(Function):

    @staticmethod
    def forward(ctx, a, b, weight):
        ctx._a_size = a.size()
        ctx._b_size = b.size()
        ctx._weight = float(weight)
        return a.lerp(b, ctx._weight)

    @staticmethod
    def backward(ctx, grad_output):
        return (maybe_unexpand(grad_output.mul(1 - ctx._weight), ctx._a_size),
                maybe_unexpand_or_view(grad_output.mul(ctx._weight), ctx._b_size), None)


class Rsqrt(InplaceFunction):

    @staticmethod
    def forward(ctx, i, inplace=False):
        if inplace:
            ctx.mark_dirty(i)
            result = i.rsqrt_()
        else:
            result = i.rsqrt()
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_variables
        return result.pow(3).div_(-2).mul(grad_output), None


class Addcmul(InplaceFunction):

    @staticmethod
    def forward(ctx, add_tensor, mul_tensor1, mul_tensor2, scale=1.0, inplace=False):
        ctx._scale = scale
        ctx._add_tensor_size = add_tensor.size()
        ctx.save_for_backward(mul_tensor1, mul_tensor2)
        if inplace:
            ctx.mark_dirty(add_tensor)
            return add_tensor.addcmul_(scale, mul_tensor1, mul_tensor2)
        else:
            return add_tensor.addcmul(scale, mul_tensor1, mul_tensor2)

    @staticmethod
    def backward(ctx, grad_output):
        grad_add = grad_mul1 = grad_mul2 = None
        mul_tensor1, mul_tensor2 = ctx.saved_variables

        if ctx.needs_input_grad[0]:
            grad_add = maybe_unexpand(grad_output, ctx._add_tensor_size)

        if ctx.needs_input_grad[1]:
            grad_mul1 = maybe_unexpand_or_view(grad_output.mul(mul_tensor2).mul_(ctx._scale), mul_tensor1.size())

        if ctx.needs_input_grad[2]:
            grad_mul2 = maybe_unexpand_or_view(grad_output.mul(mul_tensor1).mul_(ctx._scale), mul_tensor2.size())

        return grad_add, grad_mul1, grad_mul2, None, None


class Addcdiv(InplaceFunction):

    @staticmethod
    def forward(ctx, add_tensor, div_tensor1, div_tensor2, scale=1.0, inplace=False):
        ctx._scale = scale
        ctx._add_tensor_size = add_tensor.size()
        ctx.save_for_backward(div_tensor1, div_tensor2)
        if inplace:
            ctx.mark_dirty(add_tensor)
            return add_tensor.addcdiv_(ctx._scale, div_tensor1, div_tensor2)
        else:
            return add_tensor.addcdiv(ctx._scale, div_tensor1, div_tensor2)

    @staticmethod
    def backward(ctx, grad_output):
        grad_add = grad_div1 = grad_div2 = None
        div_tensor1, div_tensor2 = ctx.saved_variables

        if ctx.needs_input_grad[0]:
            grad_add = maybe_unexpand(grad_output, ctx._add_tensor_size)

        if ctx.needs_input_grad[1]:
            grad_div1 = maybe_unexpand_or_view(grad_output.div(div_tensor2).mul_(ctx._scale), div_tensor1.size())

        if ctx.needs_input_grad[2]:
            div_tensor2_sq = div_tensor2.mul(div_tensor2)
            grad_div2 = maybe_unexpand_or_view(grad_output.mul(div_tensor1).div(div_tensor2_sq).mul(-ctx._scale),
                                               div_tensor2.size())

        return grad_add, grad_div1, grad_div2, None, None

# TODO: atan2 + inplace
