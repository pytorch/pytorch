from numbers import Integral
import torch
from torch.autograd import Function
from torch._thnn import type2backend

from . import _all_functions
from ...modules.utils import _pair, _triple


class _UpsamplingBase(Function):

    def __init__(self, size=None, scale_factor=None):
        super(_UpsamplingBase, self).__init__()
        if size is None and scale_factor is None:
            raise ValueError('either size or scale_factor should be defined')
        if scale_factor is not None and not isinstance(scale_factor, (Integral, tuple)):
            raise ValueError('scale_factor must be of integer type or a tuple of integer types')
        self.size = size
        self.scale_factor = scale_factor


class UpsamplingNearest2d(_UpsamplingBase):

    def __init__(self, size=None, scale_factor=None):
        super(UpsamplingNearest2d, self).__init__(size, scale_factor)

        if self.scale_factor is not None and not isinstance(scale_factor, Integral):
            raise ValueError('scale_factor must be a single Integer value for nearest neighbor sampling')

    def forward(self, input):
        assert input.dim() == 4

        if self.scale_factor is None:
            if (self.size[0] % input.size(2) != 0 or
                    self.size[1] % input.size(3) != 0):
                raise RuntimeError("output size specified in UpsamplingNearest "
                                   "({}) has to be divisible by the input size, but got: "
                                   "{}".format('x'.join(map(str, self.size)),
                                               'x'.join(map(str, input.size()))))
            self.scale_factor = self.size[0] // input.size(2)
            if self.scale_factor != self.size[1] // input.size(3):
                raise RuntimeError("input aspect ratio doesn't match the "
                                   "output ratio")

        output = input.new()
        backend = type2backend[type(input)]
        self.save_for_backward(input)
        backend.SpatialUpSamplingNearest_updateOutput(
            backend.library_state,
            input,
            output,
            self.scale_factor
        )
        return output

    def backward(self, grad_output):
        assert grad_output.dim() == 4

        input, = self.saved_tensors
        grad_input = grad_output.new()
        backend = type2backend[type(input)]
        backend.SpatialUpSamplingNearest_updateGradInput(
            backend.library_state,
            input,
            grad_output,
            grad_input,
            self.scale_factor
        )
        return grad_input


def _check_linear_scale_factor(scale_factor, dim=2):
    if dim == 2:
        scale_factor = _pair(scale_factor)
    elif dim == 3:
        scale_factor = _triple(scale_factor)
    else:
        raise ValueError("dim has to be 2 or 3")

    try:
        assert len(scale_factor) == 2 or len(scale_factor) == 3
        assert all(isinstance(s, Integral) and s >= 1 for s in scale_factor)
    except AssertionError as e:
        raise ValueError('scale_factor must be a non-negative integer, '
                         'or a tuple of non-negative integers for bilinear and trilinear upsampling, but got: '
                         '{}'.format(scale_factor))
    return scale_factor


class UpsamplingBilinear2d(_UpsamplingBase):

    def __init__(self, size=None, scale_factor=None):
        super(UpsamplingBilinear2d, self).__init__(size, scale_factor)

        if self.scale_factor is not None:
            self.scale_factor = _check_linear_scale_factor(self.scale_factor, dim=2)

    def forward(self, input):
        assert input.dim() == 4

        if self.scale_factor is not None:
            self.output_size = (
                input.size(2) * self.scale_factor[0],
                input.size(3) * self.scale_factor[1],
            )
        else:
            self.output_size = self.size

        self.input_size = input.size()
        output = input.new()
        backend = type2backend[type(input)]
        backend.SpatialUpSamplingBilinear_updateOutput(
            backend.library_state,
            input,
            output,
            self.output_size[0],
            self.output_size[1],
        )
        return output

    def backward(self, grad_output):
        assert grad_output.dim() == 4

        grad_output = grad_output.contiguous()
        grad_input = grad_output.new()
        backend = type2backend[type(grad_output)]
        backend.SpatialUpSamplingBilinear_updateGradInput(
            backend.library_state,
            grad_output,
            grad_input,
            self.input_size[0],
            self.input_size[1],
            self.input_size[2],
            self.input_size[3],
            self.output_size[0],
            self.output_size[1],
        )
        return grad_input


class UpsamplingNearest3d(_UpsamplingBase):
    def __init__(self, size=None, scale_factor=None):
        super(UpsamplingNearest3d, self).__init__(size, scale_factor)

        if self.scale_factor is not None and not isinstance(scale_factor, Integral):
            raise ValueError('scale_factor must be a single Integer value for nearest neighbor sampling')

    def forward(self, input):
        assert input.dim() == 5

        if self.scale_factor is None:
            if (self.size[0] % input.size(2) != 0 or self.size[1] % input.size(3) != 0 or
               self.size[2] % input.size(4) != 0):
                raise RuntimeError("output size specified in UpSamplingNearest "
                                   "({}) has to be divisible by the input size, but got: "
                                   "{}".format('x'.join(map(str, self.size)),
                                               'x'.join(map(str, input.size()))))
            self.scale_factor = self.size[0] // input.size(2)
            if (self.scale_factor != self.size[1] // input.size(3) or
               self.scale_factor != self.size[2] // input.size(4)):
                raise RuntimeError("input aspect ratio doesn't match the "
                                   "output ratio")

        output = input.new()
        backend = type2backend[type(input)]
        self.save_for_backward(input)
        backend.VolumetricUpSamplingNearest_updateOutput(backend.library_state,
                                                         input,
                                                         output,
                                                         self.scale_factor)
        return output

    def backward(self, grad_output):
        assert grad_output.dim() == 5
        input, = self.saved_tensors
        grad_input = grad_output.new()
        backend = type2backend[type(input)]
        backend.VolumetricUpSamplingNearest_updateGradInput(backend.library_state,
                                                            input,
                                                            grad_output,
                                                            grad_input,
                                                            self.scale_factor)
        return grad_input


class UpsamplingTrilinear3d(_UpsamplingBase):
    def __init__(self, size=None, scale_factor=None):
        super(UpsamplingTrilinear3d, self).__init__(size, scale_factor)

        if self.scale_factor is not None:
            self.scale_factor = _check_linear_scale_factor(self.scale_factor, dim=3)

    def forward(self, input):
        assert input.dim() == 5

        if self.scale_factor is not None:
            self.output_size = (
                input.size(2) * self.scale_factor[0],
                input.size(3) * self.scale_factor[1],
                input.size(4) * self.scale_factor[2],
            )
        else:
            self.output_size = self.size

        self.input_size = input.size()
        output = input.new()
        backend = type2backend[type(input)]
        backend.VolumetricUpSamplingTrilinear_updateOutput(
            backend.library_state,
            input,
            output,
            self.output_size[0],
            self.output_size[1],
            self.output_size[2]
        )
        return output

    def backward(self, grad_output):
        assert grad_output.dim() == 5

        grad_output = grad_output.contiguous()
        grad_input = grad_output.new()
        backend = type2backend[type(grad_output)]
        backend.VolumetricUpSamplingTrilinear_updateGradInput(
            backend.library_state,
            grad_output,
            grad_input,
            self.input_size[0],
            self.input_size[1],
            self.input_size[2],
            self.input_size[3],
            self.input_size[4],
            self.output_size[0],
            self.output_size[1],
            self.output_size[2]
        )
        return grad_input


_all_functions.append(UpsamplingNearest2d)
_all_functions.append(UpsamplingBilinear2d)
_all_functions.append(UpsamplingNearest3d)
_all_functions.append(UpsamplingTrilinear3d)
