import torch
from torch.autograd.function import Function
from torch._thnn import type2backend

from . import _all_functions


class CrossMapLRN2d(Function):

    @staticmethod
    def forward(ctx, input, size, alpha=1e-4, beta=0.75, k=1):
        ctx.size = size
        ctx.alpha = alpha
        ctx.beta = beta
        ctx.k = k
        ctx._backend = None
        ctx.scale = None

        assert input.dim() == 4

        ctx.scale = ctx.scale or input.new()
        output = input.new()

        backend = type2backend[input.type()]
        if backend is not None:
            try:
                backend.SpatialCrossMapLRN_updateOutput
                ctx._backend = backend
            except NotImplementedError:
                pass

        if ctx._backend is not None:
            ctx._backend.SpatialCrossMapLRN_updateOutput(
                ctx._backend.library_state,
                input,
                output,
                ctx.scale,
                ctx.size,
                ctx.alpha,
                ctx.beta,
                ctx.k
            )
        else:
            batch_size = input.size(0)
            channels = input.size(1)
            input_height = input.size(2)
            input_width = input.size(3)

            output.resize_as_(input)
            ctx.scale.resize_as_(input)

            # use output storage as temporary buffer
            input_square = output
            torch.pow(input, 2, out=input_square)

            pre_pad = int((ctx.size - 1) / 2 + 1)
            pre_pad_crop = channels if pre_pad > channels else pre_pad

            scale_first = ctx.scale.select(1, 0)
            scale_first.zero_()
            # compute first feature map normalization
            for c in range(pre_pad_crop):
                scale_first.add_(input_square.select(1, c))

            # reuse computations for next feature maps normalization
            # by adding the next feature map and removing the previous
            for c in range(1, channels):
                scale_previous = ctx.scale.select(1, c - 1)
                scale_current = ctx.scale.select(1, c)
                scale_current.copy_(scale_previous)
                if c < channels - pre_pad + 1:
                    square_next = input_square.select(1, c + pre_pad - 1)
                    scale_current.add_(1, square_next)

                if c > pre_pad:
                    square_previous = input_square.select(1, c - pre_pad)
                    scale_current.add_(-1, square_previous)

            ctx.scale.mul_(ctx.alpha / ctx.size).add_(ctx.k)

            torch.pow(ctx.scale, -ctx.beta, out=output)
            output.mul_(input)

        ctx.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        grad_input = grad_output.new()

        if ctx._backend is not None:
            ctx._backend.SpatialCrossMapLRN_updateGradInput(
                ctx._backend.library_state,
                input,
                grad_output,
                grad_input,
                ctx.scale,
                output,
                ctx.size,
                ctx.alpha,
                ctx.beta,
                ctx.k
            )
        else:
            batch_size = input.size(0)
            channels = input.size(1)
            input_height = input.size(2)
            input_width = input.size(3)

            paddded_ratio = input.new(channels + ctx.size - 1, input_height,
                                      input_width)
            accum_ratio = input.new(input_height, input_width)

            cache_ratio_value = 2 * ctx.alpha * ctx.beta / ctx.size
            inversePrePad = int(ctx.size - (ctx.size - 1) / 2)

            grad_input.resize_as_(input)
            torch.pow(ctx.scale, -ctx.beta, out=grad_input).mul_(grad_output)

            paddded_ratio.zero_()
            padded_ratio_center = paddded_ratio.narrow(0, inversePrePad,
                                                       channels)
            for n in range(batch_size):
                torch.mul(grad_output[n], output[n], out=padded_ratio_center)
                padded_ratio_center.div_(ctx.scale[n])
                torch.sum(
                    paddded_ratio.narrow(0, 0, ctx.size - 1), 0, keepdim=False, out=accum_ratio)
                for c in range(channels):
                    accum_ratio.add_(paddded_ratio[c + ctx.size - 1])
                    grad_input[n][c].addcmul_(-cache_ratio_value, input[n][c],
                                              accum_ratio)
                    accum_ratio.add_(-1, paddded_ratio[c])

        return grad_input, None, None, None, None


_all_functions.append(CrossMapLRN2d)
