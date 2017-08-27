from numbers import Integral
import torch
from torch.autograd.function import Function
from torch._thnn import type2backend

from . import _all_functions
from ...modules.utils import _pair, _triple


def _check_size_scale_factor(size, scale_factor):
    if size is None and scale_factor is None:
        raise ValueError('either size or scale_factor should be defined')
    if scale_factor is not None and not isinstance(scale_factor, (Integral, tuple)):
        raise ValueError('scale_factor must be of integer type or a tuple of integer types')


class UpsamplingNearest2d(Function):

    @staticmethod
    def forward(ctx, input, size=None, scale_factor=None):
        assert input.dim() == 4

        _check_size_scale_factor(size, scale_factor)

        ctx.size = size
        ctx.scale_factor = scale_factor

        if ctx.scale_factor is not None and not isinstance(ctx.scale_factor, Integral):
            raise ValueError('scale_factor must be a single Integer value for nearest neighbor sampling')

        if ctx.scale_factor is None:
            if (ctx.size[0] % input.size(2) != 0 or
                    ctx.size[1] % input.size(3) != 0):
                raise RuntimeError("output size specified in UpsamplingNearest "
                                   "({}) has to be divisible by the input size, but got: "
                                   "{}".format('x'.join(map(str, ctx.size)),
                                               'x'.join(map(str, input.size()))))
            ctx.scale_factor = ctx.size[0] // input.size(2)
            if ctx.scale_factor != ctx.size[1] // input.size(3):
                raise RuntimeError("input aspect ratio doesn't match the "
                                   "output ratio")

        output = input.new()
        backend = type2backend[type(input)]
        ctx.save_for_backward(input)
        backend.SpatialUpSamplingNearest_updateOutput(
            backend.library_state,
            input,
            output,
            ctx.scale_factor
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_variables
        grad_input = UpsamplingNearest2dBackward.apply(input, grad_output, ctx.scale_factor)
        return grad_input, None, None


class UpsamplingNearest2dBackward(Function):

    @staticmethod
    def forward(ctx, input, grad_output, scale_factor):
        assert grad_output.dim() == 4
        ctx.scale_factor = scale_factor

        grad_input = grad_output.new()
        backend = type2backend[type(input)]
        backend.SpatialUpSamplingNearest_updateGradInput(
            backend.library_state,
            input,
            grad_output,
            grad_input,
            ctx.scale_factor
        )
        return grad_input

    @staticmethod
    def backward(ctx, ggI):
        gI = None
        ggO = UpsamplingNearest2d.apply(ggI, None, ctx.scale_factor)

        return gI, ggO, None


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


class UpsamplingBilinear2d(Function):

    @staticmethod
    def forward(ctx, input, size=None, scale_factor=None):
        assert input.dim() == 4

        ctx.size = size
        ctx.scale_factor = scale_factor

        if ctx.scale_factor is not None:
            ctx.scale_factor = _check_linear_scale_factor(ctx.scale_factor, dim=2)

        if ctx.scale_factor is not None:
            ctx.output_size = (
                input.size(2) * ctx.scale_factor[0],
                input.size(3) * ctx.scale_factor[1],
            )
        else:
            ctx.output_size = ctx.size

        ctx.input_size = input.size()
        output = input.new()
        backend = type2backend[type(input)]
        backend.SpatialUpSamplingBilinear_updateOutput(
            backend.library_state,
            input,
            output,
            ctx.output_size[0],
            ctx.output_size[1],
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = UpsamplingBilinear2dBackward.apply(grad_output, ctx.input_size, ctx.output_size)
        return grad_input, None, None


class UpsamplingBilinear2dBackward(Function):

    @staticmethod
    def forward(ctx, grad_output, input_size, output_size):
        assert grad_output.dim() == 4

        ctx.input_size = input_size
        ctx.output_size = output_size

        grad_output = grad_output.contiguous()
        grad_input = grad_output.new()
        backend = type2backend[type(grad_output)]
        backend.SpatialUpSamplingBilinear_updateGradInput(
            backend.library_state,
            grad_output,
            grad_input,
            ctx.input_size[0],
            ctx.input_size[1],
            ctx.input_size[2],
            ctx.input_size[3],
            ctx.output_size[0],
            ctx.output_size[1],
        )
        return grad_input

    @staticmethod
    def backward(ctx, ggI):
        ggO = UpsamplingBilinear2d.apply(ggI, ctx.output_size, None)

        return ggO, None, None


class UpsamplingNearest3d(Function):

    @staticmethod
    def forward(ctx, input, size=None, scale_factor=None):
        assert input.dim() == 5

        ctx.size = size
        ctx.scale_factor = scale_factor

        if ctx.scale_factor is not None and not isinstance(ctx.scale_factor, Integral):
            raise ValueError('scale_factor must be a single Integer value for nearest neighbor sampling')

        if ctx.scale_factor is None:
            if (ctx.size[0] % input.size(2) != 0 or ctx.size[1] % input.size(3) != 0 or
               ctx.size[2] % input.size(4) != 0):
                raise RuntimeError("output size specified in UpSamplingNearest "
                                   "({}) has to be divisible by the input size, but got: "
                                   "{}".format('x'.join(map(str, ctx.size)),
                                               'x'.join(map(str, input.size()))))
            ctx.scale_factor = ctx.size[0] // input.size(2)
            if (ctx.scale_factor != ctx.size[1] // input.size(3) or
               ctx.scale_factor != ctx.size[2] // input.size(4)):
                raise RuntimeError("input aspect ratio doesn't match the "
                                   "output ratio")

        output = input.new()
        backend = type2backend[type(input)]
        ctx.save_for_backward(input)
        backend.VolumetricUpSamplingNearest_updateOutput(backend.library_state,
                                                         input,
                                                         output,
                                                         ctx.scale_factor)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_variables

        grad_input = UpsamplingNearest3dBackward.apply(input, grad_output, ctx.scale_factor)
        return grad_input, None, None


class UpsamplingNearest3dBackward(Function):

    @staticmethod
    def forward(ctx, input, grad_output, scale_factor):
        assert grad_output.dim() == 5

        ctx.scale_factor = scale_factor
        grad_input = grad_output.new()
        backend = type2backend[type(input)]
        backend.VolumetricUpSamplingNearest_updateGradInput(backend.library_state,
                                                            input,
                                                            grad_output,
                                                            grad_input,
                                                            ctx.scale_factor)
        return grad_input

    @staticmethod
    def backward(ctx, ggI):
        gI = None
        ggO = UpsamplingNearest3d.apply(ggI, None, ctx.scale_factor)

        return gI, ggO, None


class UpsamplingTrilinear3d(Function):

    @staticmethod
    def forward(ctx, input, size=None, scale_factor=None):
        assert input.dim() == 5

        ctx.size = size
        ctx.scale_factor = scale_factor

        if ctx.scale_factor is not None:
            ctx.scale_factor = _check_linear_scale_factor(ctx.scale_factor, dim=3)

        if ctx.scale_factor is not None:
            ctx.output_size = (
                input.size(2) * ctx.scale_factor[0],
                input.size(3) * ctx.scale_factor[1],
                input.size(4) * ctx.scale_factor[2],
            )
        else:
            ctx.output_size = ctx.size

        ctx.input_size = input.size()
        output = input.new()
        backend = type2backend[type(input)]
        backend.VolumetricUpSamplingTrilinear_updateOutput(
            backend.library_state,
            input,
            output,
            ctx.output_size[0],
            ctx.output_size[1],
            ctx.output_size[2]
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = UpsamplingTrilinear3dBackward.apply(grad_output, ctx.input_size, ctx.output_size)
        return grad_input, None, None


class UpsamplingTrilinear3dBackward(Function):

    @staticmethod
    def forward(ctx, grad_output, input_size, output_size):
        assert grad_output.dim() == 5

        ctx.input_size = input_size
        ctx.output_size = output_size
        grad_output = grad_output.contiguous()
        grad_input = grad_output.new()
        backend = type2backend[type(grad_output)]
        backend.VolumetricUpSamplingTrilinear_updateGradInput(
            backend.library_state,
            grad_output,
            grad_input,
            ctx.input_size[0],
            ctx.input_size[1],
            ctx.input_size[2],
            ctx.input_size[3],
            ctx.input_size[4],
            ctx.output_size[0],
            ctx.output_size[1],
            ctx.output_size[2]
        )
        return grad_input

    @staticmethod
    def backward(ctx, ggI):
        ggO = UpsamplingTrilinear3d.apply(ggI, ctx.output_size, None)

        return ggO, None, None


_all_functions.append(UpsamplingNearest2d)
_all_functions.append(UpsamplingNearest2dBackward)
_all_functions.append(UpsamplingBilinear2d)
_all_functions.append(UpsamplingBilinear2dBackward)
_all_functions.append(UpsamplingNearest3d)
_all_functions.append(UpsamplingNearest3dBackward)
_all_functions.append(UpsamplingTrilinear3d)
_all_functions.append(UpsamplingTrilinear3dBackward)
