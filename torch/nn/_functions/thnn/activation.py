import torch
from torch.autograd.function import Function, InplaceFunction
from torch._thnn import type2backend
from torch.autograd.variable import Variable

from . import _all_functions


class PReLU(Function):

    def forward(self, input, weight):
        self._backend = type2backend[type(input)]
        output = input.new()
        self.num_parameters = weight.numel()
        if self.num_parameters == 1:
            self.num_parameters = 0
        self._backend.PReLU_updateOutput(
            self._backend.library_state,
            input,
            output,
            weight,
            self.num_parameters
        )
        self.save_for_backward(input, weight)
        return output

    def backward(self, grad_output):
        input, weight = self.saved_tensors
        # TODO: check if requires grad
        grad_input = input.new()
        self._backend.PReLU_updateGradInput(
            self._backend.library_state,
            input,
            grad_output,
            grad_input,
            weight,
            self.num_parameters
        )

        buf = weight.new()
        buf2 = weight.new()
        # TODO: this won't have to be zeroed in the future
        grad_weight = weight.new().resize_as_(weight).zero_()
        self._backend.PReLU_accGradParameters(
            self._backend.library_state,
            input,
            grad_output,
            grad_input,
            weight,
            grad_weight,
            buf,
            buf2,
            self.num_parameters,
            1
        )
        return grad_input, grad_weight


class RReLU(InplaceFunction):

    def __init__(self, lower, upper, train, inplace=False):
        super(RReLU, self).__init__(inplace)
        self.lower = lower
        self.upper = upper
        self.train = train

    def forward(self, input):
        self._backend = type2backend[type(input)]
        if self.inplace:
            self.mark_dirty(input)
            output = input
        else:
            output = input.new(input.size())
        self.noise = input.new()
        self._backend.RReLU_updateOutput(
            self._backend.library_state,
            input,
            output,
            self.noise,
            self.lower,
            self.upper,
            self.train,
            self.inplace,
            torch.default_generator if not input.is_cuda else 0
        )
        self.save_for_backward(input)
        return output

    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = input.new()
        self._backend.RReLU_updateGradInput(
            self._backend.library_state,
            input,
            grad_output,
            grad_input,
            self.noise,
            self.lower,
            self.upper,
            self.train,
            False
        )
        return grad_input


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
                input.data,
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


class Softmin(Function):

    def forward(self, input):
        self._backend = type2backend[type(input)]
        self.mininput = input.clone().mul(-1)
        output = input.new()
        self._backend.SoftMax_updateOutput(
            self._backend.library_state,
            self.mininput,
            output
        )
        self.save_for_backward(output)
        return output

    def backward(self, grad_output):
        output, = self.saved_tensors
        grad_input = grad_output.new()
        self._backend.SoftMax_updateGradInput(
            self._backend.library_state,
            self.mininput,
            grad_output,
            grad_input,
            output
        )
        return grad_input.mul(-1)


# TODO: This class should be removed once THNN function support Variable backward
class Threshold(Function):

    @staticmethod
    def forward(ctx, input, threshold, value, inplace):
        if inplace:
            if value > threshold:
                raise RuntimeError('in-place processing requires value ({}) to not '
                                   'exceed threshold ({})'.format(value, threshold))
        ctx.threshold = threshold
        ctx.value = value
        ctx.inplace = inplace

        if inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.new(input.size())
        ctx.save_for_backward(input)

        backend = type2backend[type(input)]
        backend.Threshold_updateOutput(
            backend.library_state,
            input,
            output,
            threshold,
            value,
            inplace
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_variables
        if grad_output.volatile:
            grad_input = Variable(input.data.new(input.size()), volatile=True)
            backend = type2backend[type(input.data)]
            backend.Threshold_updateGradInput(
                backend.library_state,
                input.data,
                grad_output.data,
                grad_input.data,
                ctx.threshold,
                ctx.value,
                False
            )
        else:
            grad_input = grad_output.masked_fill(input <= ctx.threshold, 0)
        return grad_input, None, None, None


# TODO: This class should be removed once THNN function support Variable backward
class LeakyReLU(Function):

    @staticmethod
    def forward(ctx, input, negative_slope, inplace):
        ctx.negative_slope = negative_slope
        ctx.inplace = inplace

        if inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.new(input.size())
        ctx.save_for_backward(input)

        backend = type2backend[type(input)]
        backend.LeakyReLU_updateOutput(
            backend.library_state,
            input,
            output,
            negative_slope,
            inplace
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_variables
        if grad_output.volatile:
            grad_input = Variable(input.data.new(input.size()), volatile=True)
            backend = type2backend[type(input.data)]
            backend.LeakyReLU_updateGradInput(
                backend.library_state,
                input.data,
                grad_output.data,
                grad_input.data,
                ctx.negative_slope,
                False
            )
        else:
            positive_mask = input > 0
            negative_mask = input <= 0
            mask = positive_mask.type_as(grad_output) + negative_mask.type_as(grad_output) * ctx.negative_slope
            grad_input = mask * grad_output
        return grad_input, None, None

_all_functions.append(PReLU)
_all_functions.append(RReLU)
_all_functions.append(SELU)
_all_functions.append(Softmin)
_all_functions.append(Threshold)
_all_functions.append(LeakyReLU)
