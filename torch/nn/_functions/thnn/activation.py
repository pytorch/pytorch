import torch
from torch.autograd.function import Function, InplaceFunction
from torch._thnn import type2backend
from torch.autograd.variable import Variable

from . import _all_functions


class PReLU(Function):

    @staticmethod
    def symbolic(g, input, weight):
        # TODO: Properly support numel in type()
        if all(s == 1 for s in weight.type().sizes()):
            raise RuntimeError("single weight shared among input channels not supported")
        return g.appendNode(g.create("PRelu", [input, weight]))

    @staticmethod
    def forward(ctx, input, weight):
        ctx._backend = type2backend[type(input)]
        output = input.new()
        ctx._backend.PReLU_updateOutput(
            ctx._backend.library_state,
            input,
            output,
            weight
        )
        ctx.save_for_backward(input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_variables
        # alternatively, we could recalculate _backend
        return PReLUBackward.apply(input, weight, grad_output, ctx._backend)


class PReLUBackward(Function):
    @staticmethod
    def forward(ctx, input, weight, grad_output, backend):
        ctx.save_for_backward(input, weight, grad_output)
        grad_input = input.new()
        backend.PReLU_updateGradInput(
            backend.library_state,
            input,
            grad_output,
            grad_input,
            weight
        )

        # TODO: this won't have to be zeroed in the future
        grad_weight = weight.new().resize_as_(weight).zero_()
        backend.PReLU_accGradParameters(
            backend.library_state,
            input,
            grad_output,
            grad_input,
            weight,
            grad_weight,
            1
        )
        return grad_input, grad_weight

    @staticmethod
    def backward(ctx, ggI, ggW):
        input, weight, gO = ctx.saved_variables
        positive_mask = (input > 0).type_as(ggI)
        nonpositive_mask = (input <= 0).type_as(ggW)
        # Explanation: Let input be i, weight be w, grad_output be gO.
        # f(i, w) = i  if i > 0
        #         = wi if i <= 0
        # df/di * gO  = gO      if i > 0      df/dw * g0 = 0      if i > 0
        #             = g0 * w  if i <= 0                = g0 * i  if i <= 0
        # The rest is taking derivatives of these wrt i, w, gO and summing/expanding properly.
        if weight.numel() == 1:
            # from PReLU.forward: num_parameters == 0 is used indicate that a
            # single weight is shared among all input channels.
            mask = positive_mask + nonpositive_mask * weight.expand_as(input)
            ggO = ggI * mask + ggW.expand_as(gO) * (nonpositive_mask * input)
            return ggW.expand_as(gO) * gO * nonpositive_mask, (ggI * gO * nonpositive_mask).sum(), ggO, None, None
        else:
            # Expand ggW to match size of ggI; a simple expand doesn't work because
            # ggW is the size of the input channel (dim==1 unless there is only 1 dimension).  For example,
            # let ggI be size (3,4,5,6,7) and ggW be size (4).  Then we unsqueeze ggW to be size (4,1,1,1)
            # so the expand succeeds.
            dims_to_unsqueeze = max(input.dim() - 2, 0)
            ggW_expanded = ggW
            for _ in range(dims_to_unsqueeze):
                ggW_expanded = ggW_expanded.unsqueeze(1)
            ggW_expanded = ggW_expanded.expand_as(ggI)

            gI = ggW_expanded * gO * nonpositive_mask

            gW = ggI * gO * nonpositive_mask
            if input.dim() > 1:
                gW = gW.sum(0)
            while gW.dim() > 1:
                gW = gW.sum(1)

            ggO = None
            if gO.requires_grad:
                # expand weight as input as in ggW/ggI above
                weight_expanded = weight
                for _ in range(dims_to_unsqueeze):
                    weight_expanded = weight_expanded.unsqueeze(1)
                weight_expanded = weight_expanded.expand_as(input)

                mask = positive_mask + nonpositive_mask * weight_expanded
                ggO = ggI * mask + ggW_expanded * nonpositive_mask * input
            return gI, gW, ggO, None, None


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
    def symbolic(g, input, inplace):
        return g.op("Selu", input)

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


_all_functions.append(PReLU)
_all_functions.append(PReLUBackward)
_all_functions.append(RReLU)
_all_functions.append(RReLUBackward)
_all_functions.append(SELU)
