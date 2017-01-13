from torch.autograd.function import Function
from torch._thnn import type2backend

from . import _all_functions
from torch.nn.modules.utils import _pair, _triple


class MaxPool1d(Function):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.pad = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input):
        backend = type2backend[type(input)]
        indices, output = input.new().long(), input.new()
        backend.SpatialDilatedMaxPooling_updateOutput(backend.library_state,
                input, output, indices,
                self.kernel_size, 1,
                self.stride, 1,
                self.pad, 0,
                self.dilation, 1,
                self.ceil_mode)
        if indices.dim() == 4:
            # TODO: fix when THCUNN handles 3D indices properly
            indices = indices.squeeze(0)
        if self.return_indices:
            self.save_for_backward(input, indices)
            self.mark_non_differentiable(indices)
            return output, indices
        else:
            self.save_for_backward(input)
            self.indices = indices
            return output

    def backward(self, grad_output, _indices_grad=None):
        if self.return_indices:
            input, indices = self.saved_tensors
        else:
            input, = self.saved_tensors
            indices = self.indices
        if indices.is_cuda and indices.dim() == 3:
            # TODO: fix when THCUNN handles 3D indices properly
            indices = indices.unsqueeze(0)
        grad_input = grad_output.new()
        backend = type2backend[type(input)]
        backend.SpatialDilatedMaxPooling_updateGradInput(backend.library_state,
                input, grad_output, grad_input, indices,
                self.kernel_size, 1,
                self.stride, 1,
                self.pad, 0,
                self.dilation, 1,
                self.ceil_mode)
        return grad_input

class MaxPool2d(Function):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride if stride is not None else kernel_size)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input):
        backend = type2backend[type(input)]
        indices, output = input.new().long(), input.new()
        backend.SpatialDilatedMaxPooling_updateOutput(backend.library_state,
                input, output, indices,
                self.kernel_size[1], self.kernel_size[0],
                self.stride[1], self.stride[0],
                self.padding[1], self.padding[0],
                self.dilation[1], self.dilation[0],
                self.ceil_mode)
        if self.return_indices:
            self.save_for_backward(input, indices)
            self.mark_non_differentiable(indices)
            return output, indices
        else:
            self.save_for_backward(input)
            self.indices = indices
            return output

    def backward(self, grad_output, _indices_grad=None):
        if self.return_indices:
            input, indices = self.saved_tensors
        else:
            input, = self.saved_tensors
            indices = self.indices
        grad_input = grad_output.new()
        backend = type2backend[type(input)]
        backend.SpatialDilatedMaxPooling_updateGradInput(backend.library_state,
                input, grad_output, grad_input, indices,
                self.kernel_size[1], self.kernel_size[0],
                self.stride[1], self.stride[0],
                self.padding[1], self.padding[0],
                self.dilation[1], self.dilation[0],
                self.ceil_mode)
        return grad_input

class MaxPool3d(Function):

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride if stride is not None else kernel_size)
        self.padding  = _triple(padding)
        self.dilation = _triple(dilation)
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input):
        backend = type2backend[type(input)]
        indices, output = input.new().long(), input.new()
        backend.VolumetricDilatedMaxPooling_updateOutput(backend.library_state,
                input, output, indices,
                self.kernel_size[0], self.kernel_size[2], self.kernel_size[1],
                self.stride[0], self.stride[2], self.stride[1],
                self.padding[0], self.padding[2], self.padding[1],
                self.dilation[0], self.dilation[2], self.dilation[1],
                self.ceil_mode)
        if self.return_indices:
            self.save_for_backward(input, indices)
            self.mark_non_differentiable(indices)
            return output, indices
        else:
            self.save_for_backward(input)
            self.indices = indices
            return output

    def backward(self, grad_output, _indices_grad=None):
        if self.return_indices:
            input, indices = self.saved_tensors
        else:
            input, = self.saved_tensors
            indices = self.indices
        grad_input = grad_output.new()
        backend = type2backend[type(input)]
        backend.VolumetricDilatedMaxPooling_updateGradInput(backend.library_state,
                input, grad_output, grad_input, indices,
                self.kernel_size[0], self.kernel_size[2], self.kernel_size[1],
                self.stride[0], self.stride[2], self.stride[1],
                self.padding[0], self.padding[2], self.padding[1],
                self.dilation[0], self.dilation[2], self.dilation[1],
                self.ceil_mode)
        return grad_input


class MaxUnpool2d(Function):
    def __init__(self, out_w, out_h):
        super(MaxUnpool2d, self).__init__()
        self.out_w = out_w
        self.out_h = out_h

    def forward(self, input, indices):
        self.save_for_backward(input, indices)
        self._backend = type2backend[type(input)]
        output = input.new()
        self._backend.SpatialMaxUnpooling_updateOutput(
                self._backend.library_state, input, output, indices,
                self.out_w, self.out_h)
        return output

    def backward(self, grad_output):
        input, indices = self.saved_tensors
        grad_input = grad_output.new()
        self._backend.SpatialMaxUnpooling_updateGradInput(
                self._backend.library_state, input, grad_output, grad_input,
                indices, self.out_w, self.out_h)
        return grad_input, None


class MaxUnpool3d(Function):
    def __init__(self, out_t, out_w, out_h, *args):
        super(MaxUnpool3d, self).__init__()
        self.out_t = out_t
        self.out_w = out_w
        self.out_h = out_h
        self.args = args
        assert len(args) == 6

    def forward(self, input, indices):
        self.save_for_backward(input, indices)
        self._backend = type2backend[type(input)]
        output = input.new()
        self._backend.VolumetricMaxUnpooling_updateOutput(
                self._backend.library_state, input, output, indices,
                self.out_t, self.out_w, self.out_h, *self.args)
        return output

    def backward(self, grad_output):
        input, indices = self.saved_tensors
        grad_input = grad_output.new()
        self._backend.VolumetricMaxUnpooling_updateGradInput(
                self._backend.library_state, input, grad_output, grad_input,
                indices, self.out_t, self.out_w, self.out_h, *self.args)
        return grad_input, None


class FractionalMaxPool2d(Function):

    def __init__(self, kh, kw, output_size=None, output_ratio=None,
            return_indices=False, _random_samples=None):
        super(FractionalMaxPool2d, self).__init__()

        # Pool size (how wide the pooling for each output unit is)
        self.kw, self.kh = kw, kh

        # Random samples are drawn for all
        # batch * plane * (height, width; i.e., 2) points. This determines
        # the 2d "pseudorandom" overlapping pooling regions for each
        # (batch element x input plane).
        self.random_samples = _random_samples

        self.return_indices = return_indices

        if output_size is not None:
            self.oh, self.ow = output_size
            self.rh, self.rw = None, None
        elif output_ratio is not None:
            self.oh, self.ow = None, None
            self.rh, self.rw = output_ratio
            assert 0 < self.rh < 1
            assert 0 < self.rw < 1
        else:
            assert False

    def forward(self, input):
        if self.random_samples is None:
            random_samples = input.new().resize_(input.size(0),
                    input.size(1), 2).uniform_()
        else:
            random_samples = self.random_samples
            self.random_samples = None

        if self.oh is None:
            self.oh = int(input.size(2) * self.rh)
            self.ow = int(input.size(3) * self.rw)
        assert isinstance(self.oh, int) and isinstance(self.ow, int)

        indices = input.new().long()
        output = input.new()
        self._backend = type2backend[type(input)]
        self._backend.SpatialFractionalMaxPooling_updateOutput(
            self._backend.library_state,
            input,
            output,
            self.ow, self.oh,
            self.kw, self.kh,
            indices,
            random_samples
        )

        self.random_samples = None # Free unnecessary buffers
        if self.return_indices:
            self.save_for_backward(input, indices)
            return output, indices
        else:
            self.indices = indices
            self.save_for_backward(input)
            return output

    def backward(self, grad_output, _grad_indices=None):
        if self.return_indices:
            input, indices = self.saved_tensors
        else:
            input, = self.saved_tensors
            indices = self.indices

        grad_input = grad_output.new()
        self._backend.SpatialFractionalMaxPooling_updateGradInput(
            self._backend.library_state,
            input,
            grad_output,
            grad_input,
            self.ow, self.oh,
            self.kw, self.kh,
            indices)

        return grad_input


class AvgPool2d(Function):
    def __init__(self, kernel_size, stride=None, padding=0,
                 ceil_mode=False, count_include_pad=True):
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride if stride is not None else kernel_size)
        self.padding = _pair(padding)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input):
        backend = type2backend[type(input)]
        output = input.new()
        # can avoid this with cudnn
        self.save_for_backward(input)
        backend.SpatialAveragePooling_updateOutput(
            backend.library_state,
            input, output,
            self.kernel_size[1], self.kernel_size[0],
            self.stride[1], self.stride[0],
            self.padding[1], self.padding[0],
            self.ceil_mode, self.count_include_pad)
        return output

    def backward(self, grad_output):
        backend = type2backend[type(grad_output)]
        input, = self.saved_tensors
        grad_input = grad_output.new()
        backend.SpatialAveragePooling_updateGradInput(
            backend.library_state,
            input, grad_output, grad_input,
            self.kernel_size[1], self.kernel_size[0],
            self.stride[1], self.stride[0],
            self.padding[1], self.padding[0],
            self.ceil_mode, self.count_include_pad)
        return grad_input

class AvgPool3d(Function):
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride if stride is not None else kernel_size)

    def forward(self, input):
        backend = type2backend[type(input)]
        output = input.new()
        # can avoid this with cudnn
        self.save_for_backward(input)
        backend.VolumetricAveragePooling_updateOutput(backend.library_state,
                input, output,
                self.kernel_size[0], self.kernel_size[2], self.kernel_size[1],
                self.stride[0], self.stride[2], self.stride[1])
        return output

    def backward(self, grad_output):
        backend = type2backend[type(grad_output)]
        input, = self.saved_tensors
        grad_input = grad_output.new()
        backend.VolumetricAveragePooling_updateGradInput(backend.library_state,
                input, grad_output, grad_input,
                self.kernel_size[0], self.kernel_size[2], self.kernel_size[1],
                self.stride[0], self.stride[2], self.stride[1])
        return grad_input

_all_functions.append(AvgPool2d)
_all_functions.append(AvgPool3d)
_all_functions.append(MaxPool1d)
_all_functions.append(MaxPool2d)
_all_functions.append(MaxPool3d)
_all_functions.append(MaxUnpool2d)
_all_functions.append(MaxUnpool3d)
_all_functions.append(FractionalMaxPool2d)
