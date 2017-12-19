from torch.autograd import Variable
from torch.autograd.function import Function
from torch._thnn import type2backend

from . import _all_functions
from torch.nn.modules.utils import _single


# NB: Looking for MaxPool2d or AvgPool2d?  They're natively implemented by ATen.
# Look at tools/autograd/derivatives.yaml


class FractionalMaxPool2d(Function):

    @staticmethod
    def forward(ctx, input, kh, kw, output_size=None, output_ratio=None,
                _random_samples=None):
        # Pool size (how wide the pooling for each output unit is)
        ctx.kw, ctx.kh = kw, kh

        # Random samples are drawn for all
        # batch * plane * (height, width; i.e., 2) points. This determines
        # the 2d "pseudorandom" overlapping pooling regions for each
        # (batch element x input plane).
        ctx.random_samples = _random_samples

        if output_size is not None:
            ctx.oh, ctx.ow = output_size
            ctx.rh, ctx.rw = None, None
        elif output_ratio is not None:
            ctx.oh, ctx.ow = None, None
            ctx.rh, ctx.rw = output_ratio
            assert 0 < ctx.rh < 1
            assert 0 < ctx.rw < 1
        else:
            assert False

        if ctx.random_samples is None:
            random_samples = input.new().resize_(input.size(0),
                                                 input.size(1), 2).uniform_()
        else:
            random_samples = ctx.random_samples
            ctx.random_samples = None

        if ctx.oh is None:
            ctx.oh = int(input.size(2) * ctx.rh)
            ctx.ow = int(input.size(3) * ctx.rw)
        assert isinstance(ctx.oh, int) and isinstance(ctx.ow, int)

        indices = input.new().long()
        output = input.new()
        ctx._backend = type2backend[type(input)]
        ctx._backend.SpatialFractionalMaxPooling_updateOutput(
            ctx._backend.library_state,
            input,
            output,
            ctx.ow, ctx.oh,
            ctx.kw, ctx.kh,
            indices,
            random_samples
        )

        ctx.random_samples = None  # Free unnecessary buffers
        ctx.save_for_backward(input, indices)
        ctx.mark_non_differentiable(indices)
        return output, indices

    @staticmethod
    def backward(ctx, grad_output, _grad_indices=None):
        input, indices = ctx.saved_variables

        return (FractionalMaxPool2dBackward.apply(input, indices, grad_output, ctx.oh, ctx.ow, ctx.kh, ctx.kw),
                None, None, None, None, None, None)


class FractionalMaxPool2dBackward(Function):

    @staticmethod
    def forward(ctx, input, indices, grad_output, oh, ow, kh, kw):
        ctx._backend = type2backend[type(input)]
        ctx.oh = oh
        ctx.ow = ow
        ctx.kh = kh
        ctx.kw = kw
        ctx.save_for_backward(indices)

        grad_input = grad_output.new()
        ctx._backend.SpatialFractionalMaxPooling_updateGradInput(
            ctx._backend.library_state,
            input,
            grad_output,
            grad_input,
            ctx.ow, ctx.oh,
            ctx.kw, ctx.kh,
            indices)

        return grad_input

    @staticmethod
    def backward(ctx, ggI):
        indices, = ctx.saved_variables

        gI = Variable(ggI.data.new(ggI.size()).zero_())
        # ggO is equivalent to the 1d case, but the indices are given wrt the last two dimensions combined
        indices_view = indices.view(indices.size()[:-2] + (-1,))
        ggO = ggI.contiguous().view(ggI.size()[:-2] + (-1,)).gather(dim=2, index=indices_view).view_as(indices)
        return gI, None, ggO, None, None, None, None, None, None


class AdaptiveMaxPool1d(Function):

    @staticmethod
    def forward(ctx, input, output_size):
        if input.dim() != 3:
            raise ValueError('expected 3D input (got {}D input)'
                             .format(input.dim()))

        ctx.output_size = _single(output_size)
        input2d = input.unsqueeze(2)    # size = N*C*1*L
        backend = type2backend[type(input)]
        indices, output = input2d.new().long(), input2d.new()
        backend.SpatialAdaptiveMaxPooling_updateOutput(backend.library_state,
                                                       input2d, output, indices,
                                                       ctx.output_size[0], 1)
        indices = indices.squeeze(2)
        output = output.squeeze(2)
        ctx.save_for_backward(input, indices)
        ctx.mark_non_differentiable(indices)
        return output, indices

    @staticmethod
    def backward(ctx, grad_output, _indices_grad=None):
        input, indices = ctx.saved_variables

        grad_input = AdaptiveMaxPool1dBackward.apply(input, indices, grad_output)
        return grad_input, None, None


class AdaptiveMaxPool1dBackward(Function):

    @staticmethod
    def forward(ctx, input, indices, grad_output):
        backend = type2backend[type(input)]
        ctx.save_for_backward(indices)

        input2d = input.unsqueeze(2)
        indices2d = indices.unsqueeze(2)
        grad_output2d = grad_output.unsqueeze(2)
        grad_input = grad_output2d.new()
        backend = type2backend[type(input)]
        backend.SpatialAdaptiveMaxPooling_updateGradInput(backend.library_state,
                                                          input2d, grad_output2d, grad_input, indices2d)
        grad_input = grad_input.squeeze(2)
        return grad_input

    @staticmethod
    def backward(ctx, ggI):
        indices, = ctx.saved_variables
        gI = Variable(ggI.data.new(ggI.size()).zero_())
        ggO = ggI.gather(dim=2, index=indices)
        return gI, None, ggO, None, None, None, None, None, None


class AdaptiveAvgPool1d(Function):

    @staticmethod
    def forward(ctx, input, output_size):
        if input.dim() != 3:
            raise ValueError('expected 3D input (got {}D input)'
                             .format(input.dim()))

        ctx.output_size = _single(output_size)
        input2d = input.unsqueeze(2)    # size = N*C*1*L
        backend = type2backend[type(input)]
        output = input2d.new()
        ctx.save_for_backward(input)
        backend.SpatialAdaptiveAveragePooling_updateOutput(
            backend.library_state,
            input2d, output,
            ctx.output_size[0], 1)
        output = output.squeeze(2)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_variables
        return AdaptiveAvgPool1dBackward.apply(input, grad_output), None


class AdaptiveAvgPool1dBackward(Function):

    @staticmethod
    def forward(ctx, input, grad_output):
        backend = type2backend[type(grad_output)]
        ctx.output_size = grad_output.size(-1)
        input2d = input.unsqueeze(2)
        grad_output2d = grad_output.unsqueeze(2)
        grad_input = grad_output2d.new()
        backend.SpatialAdaptiveAveragePooling_updateGradInput(
            backend.library_state,
            input2d, grad_output2d, grad_input)
        grad_input = grad_input.squeeze(2)
        return grad_input

    @staticmethod
    def backward(ctx, ggI):
        gI = Variable(ggI.data.new(ggI.size()).zero_())
        ggO = AdaptiveAvgPool1d.apply(ggI, ctx.output_size)
        return gI, ggO, None, None


_all_functions.append(FractionalMaxPool2d)
_all_functions.append(FractionalMaxPool2dBackward)
_all_functions.append(AdaptiveMaxPool1d)
_all_functions.append(AdaptiveMaxPool1dBackward)
_all_functions.append(AdaptiveAvgPool1d)
_all_functions.append(AdaptiveAvgPool1dBackward)
