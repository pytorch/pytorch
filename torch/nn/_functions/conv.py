import torch
from torch.autograd import Function
from torch._thnn import type2backend
from .thnn.auto import function_by_name
import torch.backends.cudnn as cudnn


_thnn_convs = {}


class ConvNd(Function):

    def __init__(self, stride, padding, dilation, transposed, output_padding,
                 groups):
        super(ConvNd, self).__init__()
        if len(stride) == 1:
            # view 1d convolutions as 2d
            stride = (1,) + stride
            padding = (0,) + padding
            dilation = (1,) + dilation
            output_padding = (0,) + output_padding
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups

    def forward(self, input, weight, bias=None):
        k = input.dim()
        self.save_for_backward(input, weight, bias)
        input = input.contiguous()
        if k == 3:
            input, weight = _view4d(input, weight)
        output = self._update_output(input, weight, bias)
        if k == 3:
            output, = _view3d(output)
        return output

    def backward(self, grad_output):
        k = grad_output.dim()
        grad_output = grad_output.contiguous()
        input, weight, bias = self.saved_tensors
        input = input.contiguous()
        if k == 3:
            grad_output, input, weight = _view4d(grad_output, input, weight)
        grad_input = (self._grad_input(input, weight, grad_output)
                      if self.needs_input_grad[0] else None)
        grad_weight, grad_bias = (
            self._grad_params(input, weight, bias, grad_output)
            if any(self.needs_input_grad[1:]) else (None, None))
        if k == 3:
            grad_input, grad_weight, = _view3d(grad_input, grad_weight)
        return grad_input, grad_weight, grad_bias

    def is_dilated(self):
        return self.dilation != (1,) * len(self.dilation)

    def _output_size(self, input, weight):
        channels = (weight.size(1) * self.groups if self.transposed
                    else weight.size(0))
        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = self.padding[d]
            kernel = self.dilation[d] * (weight.size(d + 2) - 1) + 1
            stride = self.stride[d]
            if self.transposed:
                out_pad = self.output_padding[d]
                output_size += (
                    (in_size - 1) * stride - (2 * pad) + kernel + out_pad,)
            else:
                output_size += ((in_size + (2 * pad) - kernel) // stride + 1,)
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError("convolution input is too small (output would be {})".format(
                             'x'.join(map(str, output_size))))
        return output_size

    def _update_output(self, input, weight, bias):
        self.use_cudnn = cudnn.is_acceptable(input)
        if self.use_cudnn and cudnn.version() < 6000:
            self.use_cudnn = not self.is_dilated()
        if self.use_cudnn:
            output = input.new(*self._output_size(input, weight))
            if self.transposed:
                self._cudnn_info = (
                    torch._C._cudnn_convolution_transpose_full_forward(
                        input, weight, bias, output, self.padding, self.stride, self.dilation,
                        self.groups, cudnn.benchmark))
            else:
                self._cudnn_info = torch._C._cudnn_convolution_full_forward(
                    input, weight, bias, output, self.padding, self.stride, self.dilation,
                    self.groups, cudnn.benchmark)
            if not self.requires_grad:
                del self._cudnn_info
            return output

        self._bufs = [[] for g in range(self.groups)]
        output = self._thnn('update_output', input, weight, bias)
        if not self.requires_grad:
            del self._bufs
        return output

    def _grad_input(self, input, weight, grad_output):
        if self.use_cudnn:
            grad_input = input.new().resize_as_(input)
            if self.transposed:
                # ConvTranspose uses the same kernels as regular convolution
                # but swaps forward and backward calls
                torch._C._cudnn_convolution_forward(
                    grad_output, weight, grad_input, self._cudnn_info,
                    cudnn.benchmark)
            else:
                torch._C._cudnn_convolution_backward_data(
                    grad_output, grad_input, weight, self._cudnn_info,
                    cudnn.benchmark)
            return grad_input

        return self._thnn('grad_input', input, weight, grad_output)

    def _grad_params(self, input, weight, bias, grad_output):
        if self.use_cudnn:
            grad_weight = grad_bias = None
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

        return self._thnn('grad_params', input, weight, bias, grad_output)

    def thnn_class_name(self, input):
        assert input.dim() == 4 or input.dim() == 5
        if self.transposed:
            if input.dim() == 4:
                return 'SpatialFullConvolution'
            else:
                return 'VolumetricFullConvolution'
        elif self.is_dilated():
            if input.dim() == 4:
                return 'SpatialDilatedConvolution'
            else:
                return 'VolumetricDilatedConvolution'
        elif input.dim() == 4:
            return 'SpatialConvolutionMM'
        elif input.dim() == 5 and input.is_cuda:
            return 'VolumetricConvolution'
        else:
            return 'VolumetricConvolutionMM'

    def _thnn(self, fn_name, input, weight, *args):
        impl = _thnn_convs[self.thnn_class_name(input)]
        if self.groups == 1:
            return impl[fn_name](self, self._bufs[0], input, weight, *args)
        else:
            res = []
            for g in range(self.groups):
                def group(tensor, dim=None):
                    if tensor is None:
                        return None
                    if dim is None:
                        dim = 0 if tensor.dim() == 1 else 1
                    n = tensor.size(dim) // self.groups
                    return tensor.narrow(dim, n * g, n).contiguous()

                grouped_args = [group(input, 1), group(weight, 0)]
                grouped_args += [group(t) for t in args]
                res.append(impl[fn_name](self, self._bufs[g], *grouped_args))
            if fn_name == 'grad_params':
                return [torch.cat(t, 0) if t[0] is not None else None
                        for t in zip(*res)]
            else:
                return torch.cat(res, 1)


def _view4d(*tensors):
    # view 3d tensor as 4d (conv1d as conv2d)
    output = []
    for t in tensors:
        assert t.dim() == 3
        size = list(t.size())
        size.insert(2, 1)
        output += [t.view(*size)]
    return output


def _view3d(*tensors):
    # view 4d tensor as 3d
    output = []
    for t in tensors:
        if t is None:
            output += [None]
        else:
            assert t.dim() == 4 and t.size(2) == 1
            output += [t.squeeze(2)]
    return output


def parse_arguments(self, arguments, buffers, kernel_size):
    idx = {'T': -3, 'H': -2, 'W': -1}
    buf_idx = 0
    params = []
    for arg in arguments:
        if arg.type == 'THTensor*':
            params.append(buffers[buf_idx])
            buf_idx += 1
        elif arg.name == 'scale':
            params.append(1.0)
        elif arg.name.startswith('dil'):
            params.append(self.dilation[idx[arg.name[-1]]])
        elif arg.name[0] == 'k':
            params.append(kernel_size[idx[arg.name[-1]]])
        elif arg.name[0] == 'd':
            params.append(self.stride[idx[arg.name[-1]]])
        elif arg.name[0] == 'p':
            params.append(self.padding[idx[arg.name[-1]]])
        elif arg.name[0] == 'a':
            params.append(self.output_padding[idx[arg.name[-1]]])
        else:
            raise RuntimeError('unexpected argument in THNN header: ' + arg)
    return params


def make_update_output(fn):
    def call_update_output(self, bufs, input, weight, bias):
        backend = type2backend[type(input)]
        bufs.extend([input.new(), input.new()])
        output = input.new(*self._output_size(input, weight))
        kernel_size = weight.size()[2:]
        args = parse_arguments(self, fn.arguments[5:], bufs, kernel_size)
        getattr(backend, fn.name)(backend.library_state, input, output, weight,
                                  bias, *args)
        return output
    return call_update_output


def make_grad_input(fn):
    def call_grad_input(self, bufs, input, weight, grad_output):
        backend = type2backend[type(input)]
        grad_input = input.new().resize_as_(input)
        kernel_size = weight.size()[2:]
        args = parse_arguments(self, fn.arguments[5:], bufs, kernel_size)
        getattr(backend, fn.name)(backend.library_state, input, grad_output,
                                  grad_input, weight, *args)
        return grad_input
    return call_grad_input


def make_grad_params(fn):
    def call_grad_params(self, bufs, input, weight, bias, grad_output):
        backend = type2backend[type(input)]
        grad_weight = weight.new().resize_as_(weight).zero_()
        grad_bias = None
        if bias is not None and self.needs_input_grad[2]:
            grad_bias = bias.new().resize_as_(bias).zero_()
        kernel_size = weight.size()[2:]
        args = parse_arguments(self, fn.arguments[5:], bufs, kernel_size)
        getattr(backend, fn.name)(backend.library_state, input, grad_output,
                                  grad_weight, grad_bias, *args)
        return grad_weight, grad_bias
    return call_grad_params


def _bind_functions():
    classes = [
        'SpatialConvolutionMM',
        'VolumetricConvolution',
        'VolumetricConvolutionMM',
        'SpatialDilatedConvolution',
        'VolumetricDilatedConvolution',
        'SpatialFullConvolution',
        'VolumetricFullConvolution',
    ]
    fns = function_by_name
    for name in classes:
        _thnn_convs[name] = {
            'update_output': make_update_output(fns[name + '_updateOutput']),
            'grad_input': make_grad_input(fns[name + '_updateGradInput']),
            'grad_params': make_grad_params(fns[name + '_accGradParameters']),
        }


_bind_functions()
