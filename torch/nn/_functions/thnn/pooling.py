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


_all_functions.append(FractionalMaxPool2d)
_all_functions.append(FractionalMaxPool2dBackward)
