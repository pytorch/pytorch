import torch
from torch._thnn import type2backend
from torch.autograd import Function

from .auto import Conv2d
from . import _all_functions


def interleave(iterable, obj):
    for item in iterable:
        yield obj
        yield item


class Conv1d(Conv2d):
    def __init__(self, *args):
        super(Conv1d, self).__init__(*args)
        self.additional_args = self.additional_args[:2] + \
            list(interleave(self.additional_args[2:], 1))

    def _reshape(self, t):
        t.resize_(t.size(0), t.size(1), 1, t.size(2))

    def _revert_reshape(self, t):
        t.resize_(t.size(0), t.size(1), t.size(3))

    def forward(self, input, *params):
        self._reshape(input)
        result = super(Conv1d, self).forward(input, *params)
        self._revert_reshape(input)
        return result

    def backward(self, grad_output):
        self._reshape(self.saved_variables[0].data)
        result = super(Conv1d, self).backward(grad_output)
        self._revert_reshape(self.saved_variables[0].data)
        return result


class Conv3d(Function):
    def __init__(self, *args):
        super(Conv3d, self).__init__()
        self.additional_args = args

    def forward(self, input, weight, bias=None):
        self._backend = type2backend[type(input)]
        # TODO: free buffers when not needed
        self.buffer1 = input.new()
        self.buffer2 = input.new()
        output = input.new()
        self.with_bias = bias is not None
        if torch.typename(input) == 'torch.cuda.FloatTensor':
            self._backend.VolumetricConvolution_updateOutput(
                self._backend.library_state, input, output, weight, bias,
                self.buffer1, self.buffer2, *self.additional_args[3:])
        else:
            self._backend.VolumetricConvolutionMM_updateOutput(
                self._backend.library_state, input, output, weight,
                bias, self.buffer1, *self.additional_args)
        if self.with_bias:
            self.save_for_backward(input, weight, bias)
        else:
            self.save_for_backward(input, weight)
        return output

    def _get_saved_tensors(self):
        if self.with_bias:
            input, weight, bias = self.saved_tensors
        else:
            input, weight = self.saved_tensors
            bias = None
        return input, weight, bias

    def _compute_grad_input(self, grad_output):
        input, weight, bias = self._get_saved_tensors()
        # TODO: no zero needed in the future
        grad_input = input.new().resize_as_(input).zero_()
        if torch.typename(input) == 'torch.cuda.FloatTensor':
            self._backend.VolumetricConvolution_updateGradInput(
                self._backend.library_state, input, grad_output, grad_input,
                weight, self.buffer1, *self.additional_args[3:])
        else:
            self._backend.VolumetricConvolutionMM_updateGradInput(
                self._backend.library_state, input, grad_output, grad_input,
                weight, self.buffer1, self.buffer2, *self.additional_args)
        return grad_input

    def _compute_grad_weight(self, grad_output):
        input, weight, bias = self._get_saved_tensors()
        # TODO: no zero needed in the future
        grad_weight = weight.new().resize_as_(weight).zero_()
        grad_bias = bias.new().resize_as_(bias).zero_()
        if torch.typename(input) == 'torch.cuda.FloatTensor':
            args = self.additional_args[3:] + (1,)
            self._backend.VolumetricConvolution_accGradParameters(
                self._backend.library_state, input, grad_output, grad_weight,
                grad_bias, self.buffer1, self.buffer2,
                *args)
        else:
            self._backend.VolumetricConvolutionMM_accGradParameters(
                self._backend.library_state, input, grad_output, grad_weight,
                grad_bias, self.buffer1, 1)
        return grad_weight, grad_bias

    def backward(self, grad_output):
        grad = tuple()
        if self.needs_input_grad[0]:
            grad += (self._compute_grad_input(grad_output),)
        else:
            grad += (None,)
        if any(self.needs_input_grad[1:]):
            grad_weight, grad_bias = self._compute_grad_weight(grad_output)
            grad += (grad_weight if self.needs_input_grad[1] else None,)
            if self.with_bias:
                grad += (grad_bias if self.needs_input_grad[2] else None,)
        return grad


_all_functions.append(Conv1d)
_all_functions.append(Conv3d)

