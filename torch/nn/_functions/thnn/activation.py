import torch
from torch.autograd.function import Function, InplaceFunction
from torch._thnn import type2backend
from torch.autograd.variable import Variable

from . import _all_functions


class RReLU(InplaceFunction):

    @staticmethod
    def forward(ctx, input, lower, upper, train, inplace):
        ctx.lower = lower
        ctx.upper = upper
        ctx.train = train
        ctx.inplace = inplace
        ctx._backend = type2backend[type(input)]
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.new(input.size())
        ctx.noise = input.new()
        ctx._backend.RReLU_updateOutput(
            ctx._backend.library_state,
            input,
            output,
            ctx.noise,
            ctx.lower,
            ctx.upper,
            ctx.train,
            ctx.inplace,
            torch.default_generator if not input.is_cuda else 0
        )
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_variables
        return (RReLUBackward.apply(input, grad_output, ctx.noise, ctx.lower, ctx.upper, ctx.train),
                None, None, None, None)


class RReLUBackward(Function):

    @staticmethod
    def forward(ctx, input, grad_output, noise, lower, upper, train):
        ctx.noise = noise
        ctx.lower = lower
        ctx.upper = upper
        ctx.train = train
        ctx._backend = type2backend[type(input)]
        ctx.save_for_backward(input)

        grad_input = input.new()
        ctx._backend.RReLU_updateGradInput(
            ctx._backend.library_state,
            input,
            grad_output,
            grad_input,
            ctx.noise,
            ctx.lower,
            ctx.upper,
            ctx.train,
            False
        )
        return grad_input

    @staticmethod
    def backward(ctx, ggI):
        input, = ctx.saved_variables

        gI = None

        positive_mask = (input > 0).type_as(ggI)
        nonpositive_mask = (input <= 0).type_as(ggI)
        mask = positive_mask + nonpositive_mask * Variable(ctx.noise)
        ggO = ggI * mask
        return gI, ggO, None, None, None, None


class SELU(InplaceFunction):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946

    @staticmethod
    def forward(ctx, input, inplace):
        backend = type2backend[type(input)]
        if inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.new(input.size())
        backend.ELU_updateOutput(
            backend.library_state,
            input,
            output,
            SELU.alpha,
            inplace,
        )
        output.mul_(SELU.scale)
        ctx.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_variables
        if grad_output.volatile:
            grad_input = Variable(input.data.new(input.size()), volatile=True)
            backend = type2backend[type(input.data)]
            backend.ELU_updateGradInput(
                backend.library_state,
                grad_output.data.mul(SELU.scale),
                grad_input.data,
                output.data.div(SELU.scale),
                SELU.alpha,
                False
            )
        else:
            positive_mask = (output > 0).type_as(grad_output)
            negative_mask = (output <= 0).type_as(grad_output)
            grad_input = grad_output * SELU.scale * (positive_mask +
                                                     negative_mask * (output / SELU.scale + SELU.alpha))
        return grad_input, None


_all_functions.append(RReLU)
_all_functions.append(RReLUBackward)
_all_functions.append(SELU)
