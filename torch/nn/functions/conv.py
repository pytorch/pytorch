import torch
from torch.autograd import Function
from torch._thnn import type2backend
import torch.backends.cudnn as cudnn
from torch.nn.modules.utils import _ntuple


class ConvBase(Function):
    def __init__(self, ndim, stride=1, padding=0, groups=1):
        super(ConvBase, self).__init__()
        self.ndim = ndim
        self.stride = _ntuple(self.ndim)(stride)
        self.padding = _ntuple(self.ndim)(padding)
        self.groups = groups

    def forward(self, input, weight, bias=None):
        output = input.new(*self._output_size(input, weight))
        self.save_for_backward(input, weight, bias)
        self._update_output(input, weight, bias, output)
        return output

    def backward(self, grad_output):
        input, weight, bias = self.saved_tensors
        grad_input = (self._grad_input(input, weight, bias, grad_output)
                      if self.needs_input_grad[0] else None)
        grad_weight, grad_bias = (
            self._grad_params(input, weight, bias, grad_output)
            if any(self.needs_input_grad[1:]) else (None, None))
        return grad_input, grad_weight, grad_bias


def _thnn_size(size):
    # THNN uses [T] x W x H instead of [T] x H x W
    if len(size) == 3:
        return (size[0], size[2], size[1])
    elif len(size) == 2:
        return (size[1], size[0])
    else:
        raise ValueError("invalid size")


class Conv(ConvBase):
    def _output_size(self, input, weight):
        output_size = (input.size(0), weight.size(0),)
        for d in range(self.ndim):
            k = weight.size(d + 2)
            s = (input.size(d + 2) + 2 * self.padding[d] - k) // self.stride[d] + 1
            output_size += (s,)
        return output_size

    def _conv_size(self, weight):
        return (_thnn_size(weight.size()[2:]), _thnn_size(self.stride),
                _thnn_size(self.padding))

    def _update_output(self, input, weight, bias, output):
        self.use_cudnn = cudnn.is_acceptable(input)
        if self.use_cudnn:
            self._cudnn_info = torch._C._cudnn_convolution_full_forward(
                input, weight, bias, output, self.padding,
                self.stride, self.groups, cudnn.benchmark)
            return

        if self.groups != 1:
            # TODO: implement groups for THNN
            raise ValueError('THNN does not support groups')

        backend = type2backend[type(input)]
        self._finput = input.new()
        self._fgrad_input = input.new()
        kernel, stride, padding = self._conv_size(weight)
        if self.ndim == 2:
            backend.SpatialConvolutionMM_updateOutput(
                backend.library_state, input, output, weight, bias,
                self._finput, self._fgrad_input, *(kernel + stride + padding))
        elif self.ndim == 3 and input.is_cuda:
            backend.VolumetricConvolution_updateOutput(
                backend.library_state, input, output, weight, bias,
                self._finput, self._fgrad_input, *(stride + padding))
        else:
            assert(self.ndim == 3)
            backend.VolumetricConvolutionMM_updateOutput(
                backend.library_state, input, output, weight, bias,
                self._finput, *(kernel + stride + padding))

    def _grad_input(self, input, weight, bias, grad_output):
        if self.use_cudnn:
            grad_input = input.new().resize_as_(input)
            torch._C._cudnn_convolution_backward_data(
                grad_output, grad_input, weight, self._cudnn_info,
                cudnn.benchmark)
            return grad_input

        backend = type2backend[type(input)]
        grad_input = input.new().resize_as_(input).zero_()
        kernel, stride, padding = self._conv_size(weight)
        if self.ndim == 2:
            backend.SpatialConvolutionMM_updateGradInput(
                backend.library_state, input, grad_output, grad_input,
                weight, self._finput, self._fgrad_input,
                *(kernel + stride + padding))
        elif self.ndim == 3 and input.is_cuda:
            backend.VolumetricConvolution_updateGradInput(
                backend.library_state, input, grad_output, grad_input,
                weight, self._finput, *(stride + padding))
        else:
            assert(self.ndim == 3)
            backend.VolumetricConvolutionMM_updateGradInput(
                backend.library_state, input, grad_output, grad_input,
                weight, self._finput, self._fgrad_input,
                *(kernel + stride + padding))
        return grad_input

    def _grad_params(self, input, weight, bias, grad_output):
        grad_weight = grad_bias = None
        if self.use_cudnn:
            if self.needs_input_grad[1]:
                grad_weight = weight.new().resize_as_(weight)
                torch._C._cudnn_convolution_backward_filter(
                    grad_output, input, grad_weight, self._cudnn_info,
                    cudnn.benchmark)

            if bias is not None and self.needs_input_grad[2]:
                grad_bias = bias.new().resize_as_(bias)
                torch._C._cudnn_convolution_backward_bias(
                    grad_output, grad_bias, self._cudnn_info)

            return grad_weight, grad_bias

        backend = type2backend[type(input)]
        kernel, stride, padding = self._conv_size(weight)
        grad_weight = weight.new().resize_as_(weight).zero_()
        if bias is not None and self.needs_input_grad[2]:
            grad_bias = bias.new().resize_as_(bias).zero_()
        if self.ndim == 2:
            backend.SpatialConvolutionMM_accGradParameters(
                backend.library_state, input, grad_output, grad_weight,
                grad_bias, self._finput, self._fgrad_input,
                *(kernel + stride + padding + (1.0,)))
        elif self.ndim == 3 and input.is_cuda:
            backend.VolumetricConvolution_accGradParameters(
                backend.library_state, input, grad_output, grad_weight,
                grad_bias, self._finput, self._fgrad_input,
                *(stride + padding + (1.0,)))
        else:
            assert(self.ndim == 3)
            backend.VolumetricConvolutionMM_accGradParameters(
                backend.library_state, input, grad_output, grad_weight,
                grad_bias, self._finput,
                *(kernel + stride + padding + (1.0,)))
        return grad_weight, grad_bias


class ConvTranspose(ConvBase):
    def __init__(self, ndim, stride=1, padding=0, groups=1, output_padding=0):
        super(ConvTranspose, self).__init__(ndim, stride, padding, groups)
        self.output_padding = _ntuple(ndim)(output_padding)

    def _output_size(self, input, weight):
        output_size = (input.size(0), weight.size(1),)
        for d in range(self.ndim):
            s = ((input.size(d + 2) - 1) * self.stride[d] -
                 self.padding[d] * 2 + weight.size(d + 2) + self.output_padding[d])
            output_size += (s,)
        return output_size

    def _conv_size(self, weight):
        return (_thnn_size(weight.size()[2:]), _thnn_size(self.stride),
                _thnn_size(self.padding), _thnn_size(self.output_padding))

    def _update_output(self, input, weight, bias, output):
        self.use_cudnn = cudnn.is_acceptable(input)
        if self.use_cudnn:
            self._cudnn_info = \
                torch._C._cudnn_convolution_transpose_full_forward(
                    input, weight, bias, output, self.padding, self.stride,
                    self.groups, cudnn.benchmark)
            return

        if self.groups != 1:
            raise ValueError('THNN does not support groups')

        backend = type2backend[type(input)]
        kernel, stride, padding, output_padding = self._conv_size(weight)
        _finput = input.new()
        _fgrad_input = input.new()
        if self.ndim == 2:
            backend.SpatialFullConvolution_updateOutput(
                backend.library_state, input, output, weight, bias,
                _finput, _fgrad_input, *(kernel + stride + padding + output_padding))
        else:
            backend.VolumetricFullConvolution_updateOutput(
                backend.library_state, input, output, weight, bias,
                _finput, _fgrad_input, *(stride + padding + output_padding))

    def _grad_input(self, input, weight, bias, grad_output):
        if self.use_cudnn:
            grad_input = input.new().resize_as_(input)
            # ConvTranspose uses the same kernels as regular convolution
            # but swaps forward and backward calls
            torch._C._cudnn_convolution_forward(
                grad_output, weight, grad_input, self._cudnn_info,
                cudnn.benchmark)
            return grad_input

        backend = type2backend[type(input)]
        kernel, stride, padding, output_padding = self._conv_size(weight)
        grad_input = input.new().resize_as_(input).zero_()
        grad_columns = input.new()
        if self.ndim == 2:
            backend.SpatialFullConvolution_updateGradInput(
                backend.library_state, input, grad_output, grad_input,
                weight, grad_columns, *(kernel + stride + padding + output_padding))
        else:
            tmp = input.new()  # not actually used by THNN/THCUNN
            backend.VolumetricFullConvolution_updateGradInput(
                backend.library_state, input, grad_output, grad_input,
                weight, grad_columns, tmp, *(stride + padding + output_padding))

        return grad_input

    def _grad_params(self, input, weight, bias, grad_output):
        grad_weight = grad_bias = None
        if self.use_cudnn:
            if self.needs_input_grad[1]:
                grad_weight = weight.new().resize_as_(weight)
                torch._C._cudnn_convolution_backward_filter(
                    grad_output, input, grad_weight, self._cudnn_info,
                    cudnn.benchmark)

            if bias is not None and self.needs_input_grad[2]:
                grad_bias = bias.new().resize_as_(bias)
                torch._C._cudnn_convolution_backward_bias(
                    grad_output, grad_bias, self._cudnn_info)

            return grad_weight, grad_bias

        backend = type2backend[type(input)]
        kernel, stride, padding, output_padding = self._conv_size(weight)
        grad_weight = weight.new().resize_as_(weight).zero_()
        if bias is not None and self.needs_input_grad[2]:
            grad_bias = bias.new().resize_as_(bias).zero_()
        _finput = input.new()
        _fgrad_input = input.new()
        if self.ndim == 2:
            backend.SpatialFullConvolution_accGradParameters(
                backend.library_state, input, grad_output, grad_weight,
                grad_bias, _finput, _fgrad_input,
                *(kernel + stride + padding + output_padding + (1,)))
        else:
            backend.VolumetricFullConvolution_accGradParameters(
                backend.library_state, input, grad_output, grad_weight,
                grad_bias, _finput, _fgrad_input,
                *(stride + padding + output_padding + (1,)))
        return grad_weight, grad_bias
