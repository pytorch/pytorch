import torch
from torch.autograd.function import Function
from torch._thnn import type2backend

from . import _all_functions


class CrossMapLRN2d(Function):

    def __init__(self, size, alpha=1e-4, beta=0.75, k=1):
        super(CrossMapLRN2d, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self._backend = None
        self.scale = None

    def forward(self, input):
        assert input.dim() == 4

        self.scale = self.scale or input.new()
        output = input.new()

        backend = type2backend[type(input)]
        if backend is not None:
            try:
                backend.SpatialCrossMapLRN_updateOutput
                self._backend = backend
            except NotImplementedError:
                pass

        if self._backend is not None:
            self._backend.SpatialCrossMapLRN_updateOutput(
                self._backend.library_state,
                input,
                output,
                self.scale,
                self.size,
                self.alpha,
                self.beta,
                self.k
            )
        else:
            batch_size = input.size(0)
            channels = input.size(1)
            input_height = input.size(2)
            input_width = input.size(3)

            output.resize_as_(input)
            self.scale.resize_as_(input)

            # use output storage as temporary buffer
            input_square = output
            torch.pow(input, 2, out=input_square)

            pre_pad = int((self.size - 1) / 2 + 1)
            pre_pad_crop = channels if pre_pad > channels else pre_pad

            scale_first = self.scale.select(1, 0)
            scale_first.zero_()
            # compute first feature map normalization
            for c in range(pre_pad_crop):
                scale_first.add_(input_square.select(1, c))

            # reuse computations for next feature maps normalization
            # by adding the next feature map and removing the previous
            for c in range(1, channels):
                scale_previous = self.scale.select(1, c - 1)
                scale_current = self.scale.select(1, c)
                scale_current.copy_(scale_previous)
                if c < channels - pre_pad + 1:
                    square_next = input_square.select(1, c + pre_pad - 1)
                    scale_current.add_(1, square_next)

                if c > pre_pad:
                    square_previous = input_square.select(1, c - pre_pad)
                    scale_current.add_(-1, square_previous)

            self.scale.mul_(self.alpha / self.size).add_(self.k)

            torch.pow(self.scale, -self.beta, out=output)
            output.mul_(input)

        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = grad_output.new()

        if self._backend is not None:
            self._backend.SpatialCrossMapLRN_updateGradInput(
                self._backend.library_state,
                input,
                grad_output,
                grad_input,
                self.scale,
                output,
                self.size,
                self.alpha,
                self.beta,
                self.k
            )
        else:
            batch_size = input.size(0)
            channels = input.size(1)
            input_height = input.size(2)
            input_width = input.size(3)

            paddded_ratio = input.new(channels + self.size - 1, input_height,
                                      input_width)
            accum_ratio = input.new(input_height, input_width)

            cache_ratio_value = 2 * self.alpha * self.beta / self.size
            inversePrePad = int(self.size - (self.size - 1) / 2)

            grad_input.resize_as_(input)
            torch.pow(self.scale, -self.beta, out=grad_input).mul_(grad_output)

            paddded_ratio.zero_()
            padded_ratio_center = paddded_ratio.narrow(0, inversePrePad,
                                                       channels)
            for n in range(batch_size):
                torch.mul(grad_output[n], output[n], out=padded_ratio_center)
                padded_ratio_center.div_(self.scale[n])
                torch.sum(
                    paddded_ratio.narrow(0, 0, self.size - 1), 0, out=accum_ratio)
                for c in range(channels):
                    accum_ratio.add_(paddded_ratio[c + self.size - 1])
                    grad_input[n][c].addcmul_(-cache_ratio_value, input[n][c],
                                              accum_ratio)
                    accum_ratio.add_(-1, paddded_ratio[c])

        return grad_input


_all_functions.append(CrossMapLRN2d)
