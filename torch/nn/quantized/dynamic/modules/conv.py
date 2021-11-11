# coding=utf-8
r"""Quantized convolution modules."""

from typing import Optional, List, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.nn.intrinsic as nni
# import torch.nn.intrinsic.qat as nniqat

from torch._ops import ops
from torch.nn.common_types import _size_1_t
from torch.nn.modules.utils import _single, _pair, _triple
# from torch.nn.quantized.modules.utils import _quantize_weight
# from torch.nn.utils import fuse_conv_bn_weights
from torch.nn.quantized.modules.conv import _reverse_repeat_padding
import torch.nn.quantized.modules as nnq


_SUPPORTED_PADDING = {
    'zeros',
    'reflect'
}

# #why did I think I needed this ????????
# def dynamic_conv_from_float_helper(cls, mod):
#     r"""Creates a quantized module from a float module or qparams_dict.
#     Args:
#         mod (Module): a float module, either produced by torch.ao.quantization
#         utilities or provided by the user
#     """
#     # derived classes override cls._FLOAT_MODULE attribute
#     msg = ' nnq.' + cls.__name__ + '.from_float only works for ' + \
#         cls._FLOAT_MODULE.__name__  # type: ignore[attr-defined]
#     assert type(mod) == cls._FLOAT_MODULE, msg
#     assert hasattr(mod, 'qconfig'), \
#         'Input float module must have qconfig defined.'
#     weight_post_process = mod.qconfig.weight()
#     weight_post_process(mod.weight)
#     assert weight_post_process.dtype == torch.qint8, \
#         'Weight observer must have a dtype of qint8'
#     qweight = _quantize_weight(mod.weight.float(), weight_post_process)
#     # the __init__ call used is the one from derived classes and not the one from _ConvTransposeNd
#     qconv = cls(mod.in_channels, mod.out_channels, mod.kernel_size,  # type: ignore[call-arg]
#                 mod.stride, mod.padding, mod.output_padding, mod.groups,
#                 mod.bias is not None, mod.dilation, mod.padding_mode)
#     qconv.set_weight_bias(qweight, mod.bias)
#     return qconv

# TODO Fix docstring
class Conv1d(nnq.Conv1d):
    r"""Applies a 1D convolution over a quantized input signal composed of
    several quantized input planes.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.Conv1d`.

    .. note::
        Only `zeros` is supported for the :attr:`padding_mode` argument.

    .. note::
        Only `torch.quint8` is supported for the input data type.


    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point

    See :class:`~torch.nn.Conv1d` for other attributes.

    Examples::

        >>> m = nn.quantized.Conv1d(16, 33, 3, stride=2)
        >>> input = torch.randn(20, 16, 100)
        >>> # quantize input to quint8
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0,
                                                dtype=torch.quint8)
        >>> output = m(q_input)

    """

    _FLOAT_MODULE = nn.Conv1d
    _NNIQAT_CONV_BN_MODULE = None  # type: ignore
    _NNI_CONV_RELU_MODULE = None  # type: ignore
    # _NNIQAT_CONV_BN_MODULE = nniqat.ConvBn1d
    # _NNI_CONV_RELU_MODULE = nni.ConvReLU1d

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_1_t,
                 stride: _size_1_t = 1,
                 padding: _size_1_t = 0,
                 dilation: _size_1_t = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = padding if isinstance(padding, str) else _single(padding)
        dilation = _single(dilation)

        super(Conv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode, **factory_kwargs)

    def _get_name(self):
        return 'DynamicQuantizedConv1d'

    # @reviewer unsure whether reduce range should be true or false, here I mimic qlinear_dynamic using true
    def forward(self, input, reduce_range=True):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 3:
            raise ValueError("Input shape must be `(N, C, L)`!")
        if self.padding_mode != 'zeros':
            # Padding in Conv1d is stored as (p, p), need to get (p,)
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding[:1])
            input = F.pad(input, _reversed_padding_repeated_twice,
                          mode=self.padding_mode)
        return ops.quantized.conv1d_dynamic(input, self._packed_params, reduce_range=reduce_range)

# TODO Fix docstring
class Conv2d(nnq.Conv2d):
    r"""Applies a 2D convolution over a quantized input signal composed of
    several quantized input planes.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.Conv2d`.

    .. note::
        Only `zeros` is supported for the :attr:`padding_mode` argument.

    .. note::
        Only `torch.quint8` is supported for the input data type.


    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point

    See :class:`~torch.nn.Conv2d` for other attributes.

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.quantized.Conv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.quantized.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.quantized.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> # quantize input to quint8
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> output = m(q_input)

    """
    _FLOAT_MODULE = nn.Conv2d
    _NNIQAT_CONV_BN_MODULE = None  # type: ignore
    _NNI_CONV_RELU_MODULE = None  # type: ignore
    # _NNIQAT_CONV_BN_MODULE = nniqat.ConvBn2d
    # _NNI_CONV_RELU_MODULE = nni.ConvReLU2d

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        # Subclasses of _ConvNd need to call _init rather than __init__. See
        # discussion on PR #49702
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode, **factory_kwargs)

    def _get_name(self):
        return 'DynamicQuantizedConv2d'

    # @reviewer unsure whether reduce range should be true or false, here I mimic qlinear_dynamic using true
    def forward(self, input, reduce_range=True):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        if self.padding_mode != 'zeros':
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding)
            input = F.pad(input, _reversed_padding_repeated_twice,
                          mode=self.padding_mode)
        return ops.quantized.conv2d_dynamic(
            input, self._packed_params, reduce_range=reduce_range)

# TODO Fix docstring
class Conv3d(nnq.Conv3d):
    r"""Applies a 3D convolution over a quantized input signal composed of
    several quantized input planes.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.Conv3d`.

    .. note::
        Only `zeros` is supported for the :attr:`padding_mode` argument.

    .. note::
        Only `torch.quint8` is supported for the input data type.


    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point

    See :class:`~torch.nn.Conv3d` for other attributes.

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.quantized.Conv3d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.quantized.Conv3d(16, 33, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.quantized.Conv3d(16, 33, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), dilation=(1, 2, 2))
        >>> input = torch.randn(20, 16, 56, 56, 56)
        >>> # quantize input to quint8
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> output = m(q_input)

    """
    _FLOAT_MODULE = nn.Conv3d
    _NNIQAT_CONV_BN_MODULE = None  # type: ignore
    _NNI_CONV_RELU_MODULE = None  # type: ignore
    # _NNIQAT_CONV_BN_MODULE = nniqat.ConvBn3d
    # _NNI_CONV_RELU_MODULE = nni.ConvReLU3d

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', device=None, dtype=None):
        assert padding_mode != 'reflect', "Conv3d does not support reflection padding"
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        # Subclasses of _ConvNd need to call _init rather than __init__. See
        # discussion on PR #49702
        super(Conv3d, self)._init(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _triple(0), groups, bias, padding_mode, **factory_kwargs)

    def _get_name(self):
        return 'QuantizedConv3d'

    # @reviewer unsure whether reduce range should be true or false, here I mimic qlinear_dynamic using true
    def forward(self, input, reduce_range=True):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 5:
            raise ValueError("Input shape must be `(N, C, D, H, W)`!")
        if self.padding_mode != 'zeros':
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding)
            input = F.pad(input, _reversed_padding_repeated_twice,
                          mode=self.padding_mode)
        return ops.quantized.conv3d_dynamic(
            input, self._packed_params, reduce_range=reduce_range)

class ConvTranspose1d(nnq.ConvTranspose1d):
    r"""Applies a 1D transposed convolution operator over an input image
    composed of several input planes.
    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.ConvTranspose1d`.

    .. note:: Currently only the QNNPACK engine is implemented.
        Please, set the `torch.backends.quantized.engine = 'qnnpack'`

    For special notes, please, see :class:`~torch.nn.quantized.Conv1d`

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point
    See :class:`~torch.nn.ConvTranspose2d` for other attributes.

    Examples::

        >>> torch.backends.quantized.engine = 'qnnpack'
        >>> # With square kernels and equal stride
        >>> m = nnq.ConvTranspose1d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nnq.ConvTranspose1d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> input = torch.randn(20, 16, 50)
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> output = m(q_input)
        >>> # exact output size can be also specified as an argument
        >>> input = torch.randn(1, 16, 12)
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> downsample = nnq.Conv1d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nnq.ConvTranspose1d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(q_input)
        >>> h.size()
        torch.Size([1, 16, 6])
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12])
    """

    _FLOAT_MODULE = nn.ConvTranspose1d

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, padding_mode='zeros', device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}

        super(ConvTranspose1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, output_padding, 
            groups, bias, dilation, padding_mode, **factory_kwargs)

    def _get_name(self):
        return 'DynamicQuantizedConvTranpose1d'

    # @reviewer unsure whether reduce range should be true or false, here I mimic qlinear_dynamic using true
    def forward(self, input, reduce_range=True):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 3:
            raise ValueError("Input shape must be `(N, C, L)`!")
        return torch.ops.quantized.conv_transpose1d_dynamic(
            input, self._packed_params, reduce_range=reduce_range)


class ConvTranspose2d(nnq.ConvTranspose2d):
    r"""Applies a 2D transposed convolution operator over an input image
    composed of several input planes.
    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.ConvTranspose2d`.

    For special notes, please, see :class:`~torch.nn.quantized.Conv2d`

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point
    See :class:`~torch.nn.ConvTranspose2d` for other attributes.

    Examples::

        >>> # QNNPACK or FBGEMM as backend
        >>> torch.backends.quantized.engine = 'qnnpack'
        >>> # With square kernels and equal stride
        >>> m = nnq.ConvTranspose2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nnq.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> output = m(q_input)
        >>> # exact output size can be also specified as an argument
        >>> input = torch.randn(1, 16, 12, 12)
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> downsample = nnq.Conv2d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nnq.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(q_input)
        >>> h.size()
        torch.Size([1, 16, 6, 6])
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12, 12])
    """

    _FLOAT_MODULE = nn.ConvTranspose2d

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, padding_mode='zeros', device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}

        super(ConvTranspose2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, output_padding, 
            groups, bias, dilation, padding_mode, **factory_kwargs)

    def _get_name(self):
        return 'QuantizedConvTranpose2d'

    # @reviewer unsure whether reduce range should be true or false, here I mimic qlinear_dynamic using true
    def forward(self, input, reduce_range=True):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        return ops.quantized.conv_transpose2d_dynamic(
            input, self._packed_params, reduce_range=reduce_range)

class ConvTranspose3d(nnq.ConvTranspose3d):
    r"""Applies a 3D transposed convolution operator over an input image
    composed of several input planes.
    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.ConvTranspose3d`.

    .. note:: Currently only the FBGEMM engine is implemented.
        Please, set the `torch.backends.quantized.engine = 'fbgemm'`

    For special notes, please, see :class:`~torch.nn.quantized.Conv3d`

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point
    See :class:`~torch.nn.ConvTranspose3d` for other attributes.

    Examples::

        >>> torch.backends.quantized.engine = 'fbgemm'
        >>> # With cubic kernels and equal stride
        >>> m = nnq.ConvTranspose3d(16, 33, 3, stride=2)
        >>> # non-cubic kernels and unequal stride and with padding
        >>> m = nnq.ConvTranspose3d(16, 33, (3, 3, 5), stride=(2, 1, 1), padding=(4, 2, 2))
        >>> input = torch.randn(20, 16, 50, 100, 100)
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> output = m(q_input)
        >>> # exact output size can be also specified as an argument
        >>> input = torch.randn(1, 16, 12, 12, 12)
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> downsample = nnq.Conv3d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nnq.ConvTranspose3d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(q_input)
        >>> h.size()
        torch.Size([1, 16, 6, 6, 6])
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12, 12, 12])
    """

    _FLOAT_MODULE = nn.ConvTranspose3d

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, padding_mode='zeros', device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}

        super(ConvTranspose3d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, output_padding, 
            groups, bias, dilation, padding_mode, **factory_kwargs)

    def _get_name(self):
        return 'DynamicQuantizedConvTranpose3d'

    # # @reviewer unsure whether reduce range should be true or false, here I mimic qlinear_dynamic using trueeviewer unsure whether reduce range should be true or false, here I mimic qlinear_dynamic using true
    def forward(self, input, reduce_range=True):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 5:
            raise ValueError("Input shape must be `(N, C, T, H, W)`!")
        return ops.quantized.conv_transpose3d_dynamic(
            input, self._packed_params, reduce_range=reduce_range)
