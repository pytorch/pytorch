from torch.autograd import Variable
from torch.autograd.function import Function, once_differentiable
from torch._thnn import type2backend

from . import _all_functions
from torch.nn.modules.utils import _single, _pair, _triple


class MaxPool1d(Function):

    @staticmethod
    def forward(ctx, input, kernel_size, stride=None, padding=0, dilation=1,
                ceil_mode=False):
        if (input.dim() != 3):
            raise ValueError('expected 3D input (got {}D input)'
                             .format(input.dim()))
        ctx.kernel_size = kernel_size
        ctx.stride = stride if stride is not None else kernel_size
        ctx.pad = padding
        ctx.dilation = dilation
        ctx.ceil_mode = ceil_mode

        input2d = input.unsqueeze(2)    # size = N*C*1*L
        backend = type2backend[type(input)]
        indices, output = input2d.new().long(), input2d.new()
        backend.SpatialDilatedMaxPooling_updateOutput(backend.library_state,
                                                      input2d, output, indices,
                                                      ctx.kernel_size, 1,
                                                      ctx.stride, 1,
                                                      ctx.pad, 0,
                                                      ctx.dilation, 1,
                                                      ctx.ceil_mode)
        indices = indices.squeeze(2)
        output = output.squeeze(2)
        ctx.save_for_backward(input, indices)
        ctx.mark_non_differentiable(indices)
        return output, indices

    @staticmethod
    def backward(ctx, grad_output, _indices_grad=None):
        input, indices = ctx.saved_variables

        grad_input = MaxPool1dBackward.apply(input, indices, grad_output, ctx.kernel_size, ctx.stride, ctx.pad,
                                             ctx.dilation, ctx.ceil_mode)
        return grad_input, None, None, None, None, None, None


class MaxPool1dBackward(Function):

    @staticmethod
    def forward(ctx, input, indices, grad_output, kernel_size, stride, padding, dilation, ceil_mode):
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.pad = padding
        ctx.dilation = dilation
        ctx.ceil_mode = ceil_mode
        input2d = input.unsqueeze(2)
        indices2d = indices.unsqueeze(2)
        grad_output2d = grad_output.unsqueeze(2)
        grad_input = grad_output2d.new()
        ctx.save_for_backward(indices)
        backend = type2backend[type(input)]
        backend.SpatialDilatedMaxPooling_updateGradInput(backend.library_state,
                                                         input2d, grad_output2d, grad_input, indices2d,
                                                         ctx.kernel_size, 1,
                                                         ctx.stride, 1,
                                                         ctx.pad, 0,
                                                         ctx.dilation, 1,
                                                         ctx.ceil_mode)
        grad_input = grad_input.squeeze(2)
        return grad_input

    @staticmethod
    def backward(ctx, ggI, ggIndices=None):
        indices, = ctx.saved_variables
        gI = Variable(ggI.data.new(ggI.size()).zero_())
        ggO = ggI.gather(dim=2, index=indices)
        return gI, None, ggO, None, None, None, None, None, None


class MaxPool2d(Function):

    @staticmethod
    def forward(ctx, input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
        ctx.kernel_size = _pair(kernel_size)
        ctx.stride = _pair(stride if stride is not None else kernel_size)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.ceil_mode = ceil_mode
        backend = type2backend[type(input)]
        indices, output = input.new().long(), input.new()
        backend.SpatialDilatedMaxPooling_updateOutput(backend.library_state,
                                                      input, output, indices,
                                                      ctx.kernel_size[1], ctx.kernel_size[0],
                                                      ctx.stride[1], ctx.stride[0],
                                                      ctx.padding[1], ctx.padding[0],
                                                      ctx.dilation[1], ctx.dilation[0],
                                                      ctx.ceil_mode)
        ctx.save_for_backward(input, indices)
        ctx.mark_non_differentiable(indices)
        return output, indices

    @staticmethod
    def backward(ctx, grad_output, _indices_grad=None):
        input, indices = ctx.saved_variables
        grad_input = MaxPool2dBackward.apply(input, indices, grad_output, ctx.kernel_size, ctx.stride, ctx.padding,
                                             ctx.dilation, ctx.ceil_mode)
        return grad_input, None, None, None, None, None, None


class MaxPool2dBackward(Function):

    @staticmethod
    def forward(ctx, input, indices, grad_output, kernel_size, stride, padding, dilation,
                ceil_mode):
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.ceil_mode = ceil_mode

        grad_input = grad_output.new()
        ctx.save_for_backward(indices)
        backend = type2backend[type(input)]
        backend.SpatialDilatedMaxPooling_updateGradInput(backend.library_state,
                                                         input, grad_output, grad_input, indices,
                                                         ctx.kernel_size[1], ctx.kernel_size[0],
                                                         ctx.stride[1], ctx.stride[0],
                                                         ctx.padding[1], ctx.padding[0],
                                                         ctx.dilation[1], ctx.dilation[0],
                                                         ctx.ceil_mode)
        return grad_input

    @staticmethod
    def backward(ctx, ggI, _ggIndices=None):
        indices, = ctx.saved_variables

        gI = Variable(ggI.data.new(ggI.size()).zero_())
        # ggO is equivalent to the 1d case, but the indices are given wrt the last two dimensions combined
        indices_view = indices.view(indices.size()[:-2] + (-1,))
        ggO = ggI.contiguous().view(ggI.size()[:-2] + (-1,)).gather(dim=2, index=indices_view).view_as(indices)
        return gI, None, ggO, None, None, None, None, None, None


class MaxPool3d(Function):

    @staticmethod
    def forward(ctx, input, kernel_size, stride=None, padding=0, dilation=1,
                ceil_mode=False):
        ctx.kernel_size = _triple(kernel_size)
        ctx.stride = _triple(stride if stride is not None else kernel_size)
        ctx.padding = _triple(padding)
        ctx.dilation = _triple(dilation)
        ctx.ceil_mode = ceil_mode
        backend = type2backend[type(input)]
        indices, output = input.new().long(), input.new()
        backend.VolumetricDilatedMaxPooling_updateOutput(backend.library_state,
                                                         input, output, indices,
                                                         ctx.kernel_size[0], ctx.kernel_size[2], ctx.kernel_size[1],
                                                         ctx.stride[0], ctx.stride[2], ctx.stride[1],
                                                         ctx.padding[0], ctx.padding[2], ctx.padding[1],
                                                         ctx.dilation[0], ctx.dilation[2], ctx.dilation[1],
                                                         ctx.ceil_mode)
        ctx.save_for_backward(input, indices)
        ctx.mark_non_differentiable(indices)
        return output, indices

    @staticmethod
    def backward(ctx, grad_output, _indices_grad=None):
        input, indices = ctx.saved_variables
        grad_input = MaxPool3dBackward.apply(input, indices, grad_output, ctx.kernel_size, ctx.stride,
                                             ctx.padding, ctx.dilation, ctx.ceil_mode)
        return grad_input, None, None, None, None, None, None


class MaxPool3dBackward(Function):

    @staticmethod
    def forward(ctx, input, indices, grad_output, kernel_size, stride, padding, dilation,
                ceil_mode):
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.ceil_mode = ceil_mode
        grad_input = grad_output.new()
        backend = type2backend[type(input)]
        backend.VolumetricDilatedMaxPooling_updateGradInput(backend.library_state,
                                                            input, grad_output, grad_input, indices,
                                                            ctx.kernel_size[0], ctx.kernel_size[
                                                                2], ctx.kernel_size[1],
                                                            ctx.stride[0], ctx.stride[2], ctx.stride[1],
                                                            ctx.padding[0], ctx.padding[2], ctx.padding[1],
                                                            ctx.dilation[0], ctx.dilation[2], ctx.dilation[1],
                                                            ctx.ceil_mode)
        return grad_input

    @staticmethod
    def backward(ctx, ggI, _ggIndices=None):
        raise ValueError("MaxPool3d cannot be differentiated twice")


class MaxUnpool2d(Function):

    @staticmethod
    def forward(ctx, input, indices, output_size):
        ctx.output_size = output_size
        ctx.save_for_backward(input, indices)
        ctx._backend = type2backend[type(input)]
        output = input.new()
        ctx._backend.SpatialMaxUnpooling_updateOutput(
            ctx._backend.library_state, input, output, indices,
            ctx.output_size[1], ctx.output_size[0])
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, indices = ctx.saved_tensors
        grad_input = grad_output.new()
        ctx._backend.SpatialMaxUnpooling_updateGradInput(
            ctx._backend.library_state, input, grad_output, grad_input,
            indices, ctx.output_size[1], ctx.output_size[0])
        return grad_input, None, None


class MaxUnpool3d(Function):

    @staticmethod
    def forward(ctx, input, indices, output_size, stride, padding):
        ctx.output_size = output_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.save_for_backward(input, indices)
        ctx._backend = type2backend[type(input)]
        output = input.new()
        ctx._backend.VolumetricMaxUnpooling_updateOutput(
            ctx._backend.library_state, input, output, indices,
            ctx.output_size[0], ctx.output_size[2], ctx.output_size[1],
            ctx.stride[0], ctx.stride[2], ctx.stride[1],
            ctx.padding[0], ctx.padding[2], ctx.padding[1])
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, indices = ctx.saved_tensors
        grad_input = grad_output.new()
        ctx._backend.VolumetricMaxUnpooling_updateGradInput(
            ctx._backend.library_state, input, grad_output, grad_input, indices,
            ctx.output_size[0], ctx.output_size[2], ctx.output_size[1],
            ctx.stride[0], ctx.stride[2], ctx.stride[1],
            ctx.padding[0], ctx.padding[2], ctx.padding[1])
        return grad_input, None, None, None, None


class FractionalMaxPool2d(Function):

    @staticmethod
    def forward(ctx, input, kh, kw, output_size=None, output_ratio=None,
                return_indices=False, _random_samples=None):
        # Pool size (how wide the pooling for each output unit is)
        ctx.kw, ctx.kh = kw, kh

        # Random samples are drawn for all
        # batch * plane * (height, width; i.e., 2) points. This determines
        # the 2d "pseudorandom" overlapping pooling regions for each
        # (batch element x input plane).
        ctx.random_samples = _random_samples

        ctx.return_indices = return_indices

        if output_size is not None:
            ctx.oh, ctx.ow = output_size
            ctx.rh, ctx.rw = None, None
        elif output_ratio is not None:
            ctx.oh, ctx.ow = None, None
            ctx.rh, ctx.rw = output_ratio
            assert 0 < ctx.rh < 1
            assert 0 < ctx.rw < 1
        else:
            assert False

        if ctx.random_samples is None:
            random_samples = input.new().resize_(input.size(0),
                                                 input.size(1), 2).uniform_()
        else:
            random_samples = ctx.random_samples
            ctx.random_samples = None

        if ctx.oh is None:
            ctx.oh = int(input.size(2) * ctx.rh)
            ctx.ow = int(input.size(3) * ctx.rw)
        assert isinstance(ctx.oh, int) and isinstance(ctx.ow, int)

        indices = input.new().long()
        output = input.new()
        ctx._backend = type2backend[type(input)]
        ctx._backend.SpatialFractionalMaxPooling_updateOutput(
            ctx._backend.library_state,
            input,
            output,
            ctx.ow, ctx.oh,
            ctx.kw, ctx.kh,
            indices,
            random_samples
        )

        ctx.random_samples = None  # Free unnecessary buffers
        if ctx.return_indices:
            ctx.save_for_backward(input, indices)
            return output, indices
        else:
            ctx.indices = indices
            ctx.save_for_backward(input)
            return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output, _grad_indices=None):
        if ctx.return_indices:
            input, indices = ctx.saved_tensors
        else:
            input, = ctx.saved_tensors
            indices = ctx.indices

        grad_input = grad_output.new()
        ctx._backend.SpatialFractionalMaxPooling_updateGradInput(
            ctx._backend.library_state,
            input,
            grad_output,
            grad_input,
            ctx.ow, ctx.oh,
            ctx.kw, ctx.kh,
            indices)

        return grad_input, None, None, None, None, None, None


class AvgPool2d(Function):

    @staticmethod
    def forward(ctx, input, kernel_size, stride=None, padding=0,
                ceil_mode=False, count_include_pad=True):
        ctx.kernel_size = _pair(kernel_size)
        ctx.stride = _pair(stride if stride is not None else kernel_size)
        ctx.padding = _pair(padding)
        ctx.ceil_mode = ceil_mode
        ctx.count_include_pad = count_include_pad
        backend = type2backend[type(input)]
        output = input.new()
        # can avoid this with cudnn
        ctx.save_for_backward(input)
        backend.SpatialAveragePooling_updateOutput(
            backend.library_state,
            input, output,
            ctx.kernel_size[1], ctx.kernel_size[0],
            ctx.stride[1], ctx.stride[0],
            ctx.padding[1], ctx.padding[0],
            ctx.ceil_mode, ctx.count_include_pad)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_variables
        grad_input = AvgPool2dBackward.apply(input, grad_output, ctx.kernel_size, ctx.stride,
                                             ctx.padding, ctx.ceil_mode, ctx.count_include_pad)
        return grad_input, None, None, None, None, None


class AvgPool2dBackward(Function):

    @staticmethod
    def forward(ctx, input, grad_output, kernel_size, stride, padding, ceil_mode, count_include_pad):
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.ceil_mode = ceil_mode
        ctx.count_include_pad = count_include_pad
        backend = type2backend[type(grad_output)]
        grad_input = grad_output.new()
        ctx.save_for_backward(input)
        backend.SpatialAveragePooling_updateGradInput(
            backend.library_state,
            input, grad_output, grad_input,
            ctx.kernel_size[1], ctx.kernel_size[0],
            ctx.stride[1], ctx.stride[0],
            ctx.padding[1], ctx.padding[0],
            ctx.ceil_mode, ctx.count_include_pad)
        return grad_input

    @staticmethod
    def backward(ctx, ggI):
        input, = ctx.saved_variables
        gI = Variable(ggI.data.new(ggI.size()).zero_())
        ggO = AvgPool2d.apply(ggI, ctx.kernel_size, ctx.stride, ctx.padding, ctx.ceil_mode, ctx.count_include_pad)
        return gI, ggO, None, None, None, None, None


class AvgPool3d(Function):

    @staticmethod
    def forward(ctx, input, kernel_size, stride=None):
        ctx.kernel_size = _triple(kernel_size)
        ctx.stride = _triple(stride if stride is not None else kernel_size)
        backend = type2backend[type(input)]
        output = input.new()
        # can avoid this with cudnn
        ctx.save_for_backward(input)
        backend.VolumetricAveragePooling_updateOutput(backend.library_state,
                                                      input, output,
                                                      ctx.kernel_size[0], ctx.kernel_size[2], ctx.kernel_size[1],
                                                      ctx.stride[0], ctx.stride[2], ctx.stride[1])
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_variables
        grad_input = AvgPool3dBackward.apply(input, grad_output, ctx.kernel_size, ctx.stride)
        return grad_input, None, None


class AvgPool3dBackward(Function):

    @staticmethod
    def forward(ctx, input, grad_output, kernel_size, stride):
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        backend = type2backend[type(grad_output)]
        grad_input = grad_output.new()
        ctx.save_for_backward(input)
        backend.VolumetricAveragePooling_updateGradInput(backend.library_state,
                                                         input, grad_output, grad_input,
                                                         ctx.kernel_size[0], ctx.kernel_size[2], ctx.kernel_size[1],
                                                         ctx.stride[0], ctx.stride[2], ctx.stride[1])
        return grad_input

    @staticmethod
    def backward(ctx, ggI):
        input, = ctx.saved_variables
        gI = Variable(ggI.data.new(ggI.size()).zero_())
        ggO = AvgPool3d.apply(ggI, ctx.kernel_size, ctx.stride)
        return gI, ggO, None, None


class AdaptiveMaxPool1d(Function):

    @staticmethod
    def forward(ctx, input, output_size, return_indices=False):
        if input.dim() != 3:
            raise ValueError('expected 3D input (got {}D input)'
                             .format(input.dim()))

        ctx.output_size = _single(output_size)
        ctx.return_indices = return_indices
        input2d = input.unsqueeze(2)    # size = N*C*1*L
        backend = type2backend[type(input)]
        indices, output = input2d.new().long(), input2d.new()
        backend.SpatialAdaptiveMaxPooling_updateOutput(backend.library_state,
                                                       input2d, output, indices,
                                                       ctx.output_size[0], 1)
        indices = indices.squeeze(2)
        output = output.squeeze(2)
        if ctx.return_indices:
            ctx.save_for_backward(input, indices)
            ctx.mark_non_differentiable(indices)
            return output, indices
        else:
            ctx.save_for_backward(input)
            ctx.indices = indices
            return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output, _indices_grad=None):
        if ctx.return_indices:
            input, indices = ctx.saved_tensors
        else:
            input, = ctx.saved_tensors
            indices = ctx.indices

        input2d = input.unsqueeze(2)
        indices2d = indices.unsqueeze(2)
        grad_output2d = grad_output.unsqueeze(2)
        grad_input = grad_output2d.new()
        backend = type2backend[type(input)]
        backend.SpatialAdaptiveMaxPooling_updateGradInput(backend.library_state,
                                                          input2d, grad_output2d, grad_input, indices2d)
        grad_input = grad_input.squeeze(2)
        return grad_input, None, None


class AdaptiveMaxPool2d(Function):

    @staticmethod
    def forward(ctx, input, output_size, return_indices=False):
        ctx.output_size = _pair(output_size)
        ctx.return_indices = return_indices
        backend = type2backend[type(input)]
        indices, output = input.new().long(), input.new()
        backend.SpatialAdaptiveMaxPooling_updateOutput(backend.library_state,
                                                       input, output, indices,
                                                       ctx.output_size[1], ctx.output_size[0])
        if ctx.return_indices:
            ctx.save_for_backward(input, indices)
            ctx.mark_non_differentiable(indices)
            return output, indices
        else:
            ctx.save_for_backward(input)
            ctx.indices = indices
            return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output, _indices_grad=None):
        if ctx.return_indices:
            input, indices = ctx.saved_tensors
        else:
            input, = ctx.saved_tensors
            indices = ctx.indices
        grad_input = grad_output.new()
        backend = type2backend[type(input)]
        backend.SpatialAdaptiveMaxPooling_updateGradInput(backend.library_state,
                                                          input, grad_output, grad_input, indices)
        return grad_input, None, None


class AdaptiveAvgPool1d(Function):

    @staticmethod
    def forward(ctx, input, output_size):
        if input.dim() != 3:
            raise ValueError('expected 3D input (got {}D input)'
                             .format(input.dim()))

        ctx.output_size = _single(output_size)
        input2d = input.unsqueeze(2)    # size = N*C*1*L
        backend = type2backend[type(input)]
        output = input2d.new()
        ctx.save_for_backward(input)
        backend.SpatialAdaptiveAveragePooling_updateOutput(
            backend.library_state,
            input2d, output,
            ctx.output_size[0], 1)
        output = output.squeeze(2)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        backend = type2backend[type(grad_output)]
        input, = ctx.saved_tensors
        input2d = input.unsqueeze(2)
        grad_output2d = grad_output.unsqueeze(2)
        grad_input = grad_output2d.new()
        backend.SpatialAdaptiveAveragePooling_updateGradInput(
            backend.library_state,
            input2d, grad_output2d, grad_input)
        grad_input = grad_input.squeeze(2)
        return grad_input, None


class AdaptiveAvgPool2d(Function):

    @staticmethod
    def forward(ctx, input, output_size):
        ctx.output_size = _pair(output_size)
        backend = type2backend[type(input)]
        output = input.new()
        ctx.save_for_backward(input)
        backend.SpatialAdaptiveAveragePooling_updateOutput(
            backend.library_state,
            input, output,
            ctx.output_size[1], ctx.output_size[0])
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        backend = type2backend[type(grad_output)]
        input, = ctx.saved_tensors
        grad_input = grad_output.new()
        backend.SpatialAdaptiveAveragePooling_updateGradInput(
            backend.library_state,
            input, grad_output, grad_input)
        return grad_input, None

_all_functions.append(AvgPool2d)
_all_functions.append(AvgPool2dBackward)
_all_functions.append(AvgPool3d)
_all_functions.append(AvgPool3dBackward)
_all_functions.append(MaxPool1d)
_all_functions.append(MaxPool1dBackward)
_all_functions.append(MaxPool2d)
_all_functions.append(MaxPool2dBackward)
_all_functions.append(MaxPool3d)
_all_functions.append(MaxPool3dBackward)
_all_functions.append(MaxUnpool2d)
_all_functions.append(MaxUnpool3d)
_all_functions.append(FractionalMaxPool2d)
_all_functions.append(AdaptiveMaxPool1d)
_all_functions.append(AdaptiveMaxPool2d)
_all_functions.append(AdaptiveAvgPool1d)
_all_functions.append(AdaptiveAvgPool2d)
