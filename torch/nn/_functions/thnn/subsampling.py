import torch
from torch.autograd import Function
from torch._thnn import type2backend

from . import _all_functions


class Subsampling2d(Function):

    def __init__(self, size, stride):
        self.size = size
        self.stride = stride

    def forward(self, input, weight, bias):
        assert input.dim() == 4
        assert weight.dim() == 1
        assert bias.dim() == 1

        output = input.new()
        backend = type2backend[type(input)]
        self.save_for_backward(input, weight, bias)
        backend.SpatialSubSampling_updateOutput(
            backend.library_state,
            input,
            output,
            weight,
            bias,
            self.size[0], self.size[1],
            self.stride[0], self.stride[1]
        )
        return output

    def backward(self, grad_output, scale=1):
        input, weight, bias = self.saved_tensors

        backend = type2backend[type(input)]

        grad_input = None
        if self.needs_input_grad[0]:
            grad_input = input.new().resize_as_(input)
            backend.SpatialSubSampling_updateGradInput(
                backend.library_state,
                input,
                grad_output,
                grad_input,
                weight,
                self.size[0], self.size[1],
                self.stride[0], self.stride[1]
            )

        grad_weight, grad_bias = (None, None)
        if any(self.needs_input_grad[1:]):
            grad_weight = weight.new().resize_as_(weight).zero_()
            grad_bias = bias.new().resize_as_(bias).zero_()
            backend.SpatialSubSampling_accGradParameters(
                backend.library_state,
                input,
                grad_output,
                grad_weight,
                grad_bias,
                self.size[0], self.size[1],
                self.stride[0], self.stride[1],
                scale
            )

        return grad_input, grad_weight, grad_bias


class Subsampling3d(Function):

    def __init__(self, size, stride):
        self.size = size
        self.stride = stride

    def forward(self, input, weight, bias):
        assert input.dim() == 5
        assert weight.dim() == 1
        assert bias.dim() == 1

        output = input.new()
        backend = type2backend[type(input)]
        self.save_for_backward(input, weight, bias)
        backend.VolumetricSubSampling_updateOutput(
            backend.library_state,
            input,
            output,
            weight,
            bias,
            self.size[0], self.size[1], self.size[2],
            self.stride[0], self.stride[1], self.stride[2]
        )
        return output

    def backward(self, grad_output, scale=1):
        input, weight, bias = self.saved_tensors

        backend = type2backend[type(input)]

        grad_input = None
        if self.needs_input_grad[0]:
            grad_input = input.new().resize_as_(input)
            backend.VolumetricSubSampling_updateGradInput(
                backend.library_state,
                input,
                grad_output,
                grad_input,
                weight,
                self.size[0], self.size[1], self.size[2],
                self.stride[0], self.stride[1], self.stride[2]
            )

        grad_weight, grad_bias = (None, None)
        if any(self.needs_input_grad[1:]):
            grad_weight = weight.new().resize_as_(weight).zero_()
            grad_bias = bias.new().resize_as_(bias).zero_()
            backend.VolumetricSubSampling_accGradParameters(
                backend.library_state,
                input,
                grad_output,
                grad_weight,
                grad_bias,
                self.size[0], self.size[1], self.size[2],
                self.stride[0], self.stride[1], self.stride[2],
                scale
            )

        return grad_input, grad_weight, grad_bias


_all_functions.append(Subsampling2d)
_all_functions.append(Subsampling3d)
