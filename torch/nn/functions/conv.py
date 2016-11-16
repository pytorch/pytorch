import torch
from torch.autograd import Function
from torch._thnn import type2backend
import torch.backends.cudnn as cudnn


class _Conv2dBase(Function):

    def forward(self, input, weight, bias=None):
        output = input.new(*self._output_size(input, weight))
        if bias is not None:
            self.save_for_backward(input, weight, bias)
        else:
            self.save_for_backward(input, weight)

        self._update_output(input, weight, bias, output)
        return output

    def backward(self, grad_output):
        tensors = self.saved_tensors
        if len(tensors) == 2:
            input, weight = tensors
            bias = None
        else:
            input, weight, bias = tensors

        grad_input, grad_weight, grad_bias = None, None, None

        if self.needs_input_grad[0]:
            grad_input = self._grad_input(input, weight, bias, grad_output)
        if any(self.needs_input_grad[1:]):
            grad_weight, grad_bias = self._grad_params(input, weight, bias, grad_output)

        if bias is not None:
            return grad_input, grad_weight, grad_bias
        else:
            return grad_input, grad_weight


class Conv2d(_Conv2dBase):

    def __init__(self, stride, pad, groups):
        super(Conv2d, self).__init__()
        self.stride = stride
        self.pad = pad
        self.groups = groups

    def _output_size(self, input, weight):
        kh, kw = weight.size(2), weight.size(3)
        h = (input.size(2) + 2 * self.pad[0] - kh) // self.stride[0] + 1
        w = (input.size(3) + 2 * self.pad[1] - kw) // self.stride[1] + 1

        return input.size(0), weight.size(0), h, w

    def _update_output(self, input, weight, bias, output):
        self.use_cudnn = cudnn.is_acceptable(input)
        if self.use_cudnn:
            self._cudnn_info = torch._C._cudnn_convolution_full_forward(
                input, weight, bias, output, self.pad[0], self.pad[1],
                self.stride[0], self.stride[1], self.groups, cudnn.benchmark)
        else:
            # TODO: implement groups for THNN
            if self.groups != 1:
                raise ValueError('THNN does not support groups')
            backend = type2backend[type(input)]
            self._finput = input.new()
            self._fgrad_input = input.new()
            backend.SpatialConvolutionMM_updateOutput(
                backend.library_state, input, output, weight, bias,
                self._finput, self._fgrad_input, weight.size(3), weight.size(2),
                self.stride[1], self.stride[0], self.pad[1], self.pad[0])

    def _grad_input(self, input, weight, bias, grad_output):
        if self.use_cudnn:
            grad_input = input.new().resize_as_(input)
            torch._C._cudnn_convolution_backward_data(
                grad_output, grad_input, weight, self._cudnn_info,
                cudnn.benchmark)
        else:
            backend = type2backend[type(input)]
            grad_input = input.new().resize_as_(input).zero_()
            backend.SpatialConvolutionMM_updateGradInput(
                backend.library_state, input, grad_output, grad_input,
                weight, self._finput, self._fgrad_input, weight.size(3),
                weight.size(2), self.stride[1], self.stride[0], self.pad[1],
                self.pad[0])
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
        else:
            backend = type2backend[type(input)]
            grad_weight = weight.new().resize_as_(weight).zero_()
            if bias is not None and self.needs_input_grad[2]:
                grad_bias = bias.new().resize_as_(bias).zero_()
            backend.SpatialConvolutionMM_accGradParameters(
                backend.library_state, input, grad_output, grad_weight,
                grad_bias, self._finput, self._fgrad_input, weight.size(3),
                weight.size(2), self.stride[1], self.stride[0], self.pad[1],
                self.pad[0], 1)
        return grad_weight, grad_bias


class ConvTranspose2d(_Conv2dBase):
    def __init__(self, kw, kh, dw, dh, padw, padh, out_padw, out_padh, groups):
        super(ConvTranspose2d, self).__init__()
        self.stride = (dh, dw)
        self.pad = (padh, padw)
        self.out_pad = (out_padh, out_padw)
        self.groups = groups

    def _output_size(self, input, weight):
        kh, kw = weight.size(2), weight.size(3)
        h = (input.size(2) - 1) * self.stride[0] - 2 * self.pad[0] + kh + self.out_pad[0]
        w = (input.size(3) - 1) * self.stride[1] - 2 * self.pad[1] + kw + self.out_pad[1]
        return input.size(0), weight.size(1), h, w

    def _update_output(self, input, weight, bias, output):
        self.use_cudnn = cudnn.is_acceptable(input)
        if self.use_cudnn:
            self._cudnn_info = \
                torch._C._cudnn_convolution_transpose_full_forward(
                    input, weight, bias, output, self.pad[0], self.pad[1],
                    self.stride[0], self.stride[1], self.groups, cudnn.benchmark)
        else:
            # TODO: implement groups for THNN
            if self.groups != 1:
                raise ValueError('THNN does not support groups')
            backend = type2backend[type(input)]
            _finput = input.new()
            _fgrad_input = input.new()
            backend.SpatialFullConvolution_updateOutput(
                backend.library_state, input, output, weight, bias,
                _finput, _fgrad_input, weight.size(3), weight.size(2),
                self.stride[1], self.stride[0], self.pad[1], self.pad[0],
                self.out_pad[1], self.out_pad[0])

    def _grad_input(self, input, weight, bias, grad_output):
        if self.use_cudnn:
            grad_input = input.new().resize_as_(input)
            # ConvTranspose uses the same kernels as regular convolution
            # but swaps forward and backward calls
            torch._C._cudnn_convolution_forward(
                grad_output, weight, grad_input, self._cudnn_info,
                cudnn.benchmark)
        else:
            backend = type2backend[type(input)]
            grad_input = input.new().resize_as_(input).zero_()
            grad_columns = input.new()
            backend.SpatialFullConvolution_updateGradInput(
                backend.library_state, input, grad_output, grad_input,
                weight, grad_columns, weight.size(3),
                weight.size(2), self.stride[1], self.stride[0], self.pad[1],
                self.pad[0], self.out_pad[1], self.out_pad[0])
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
        else:
            backend = type2backend[type(input)]
            grad_weight = weight.new().resize_as_(weight).zero_()
            if bias is not None and self.needs_input_grad[2]:
                grad_bias = bias.new().resize_as_(bias).zero_()
            _finput = input.new()
            _fgrad_input = input.new()
            backend.SpatialFullConvolution_accGradParameters(
                backend.library_state, input, grad_output, grad_weight,
                grad_bias, _finput, _fgrad_input, weight.size(3),
                weight.size(2), self.stride[1], self.stride[0], self.pad[1],
                self.pad[0], self.out_pad[1], self.out_pad[0], 1)
        return grad_weight, grad_bias

