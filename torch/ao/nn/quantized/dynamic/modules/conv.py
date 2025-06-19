# mypy: allow-untyped-defs
r"""Dynamically quantized convolution modules."""

import warnings
from typing import ClassVar, Optional

import torch
import torch.ao.nn.quantized as nnq
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch._ops import ops
from torch.ao.nn.quantized.modules.conv import _reverse_repeat_padding
from torch.nn.common_types import _size_1_t
from torch.nn.modules.utils import _pair, _single, _triple


__all__ = [
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
]


class Conv1d(nnq.Conv1d):
    r"""A dynamically quantized conv module with floating point tensors as inputs and outputs.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.Conv1d` and :class:`~torch.ao.nn.quantized.dynamic.Conv1d` and

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point

    See :class:`~torch.nn.Conv1d` for other attributes.

    Examples::

        >>> # xdoctest: +SKIP
        >>> m = nn.quantized.dynamic.Conv1d(16, 33, 3, stride=2)
        >>> input = torch.randn(20, 16, 100)
        >>> output = m(input)

    """

    _FLOAT_MODULE: ClassVar[type[nn.Conv1d]] = nn.Conv1d
    _NNIQAT_CONV_BN_MODULE: ClassVar[Optional[type[nn.Module]]] = None
    _NNI_CONV_RELU_MODULE: ClassVar[Optional[type[nn.Module]]] = None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        reduce_range=True,
    ):
        warnings.warn(
            f"The current implementation of the {self._get_name()} module has poor numerical accuracy and its use is not recommended"  # noqa: B950
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = padding if isinstance(padding, str) else _single(padding)
        dilation = _single(dilation)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _get_name(self):
        return "DynamicQuantizedConv1d"

    def forward(self, input: Tensor, reduce_range: bool = True) -> Tensor:
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 3:
            raise ValueError("Input shape must be `(N, C, L)`!")
        if self.padding_mode != "zeros":
            # Padding in Conv1d is stored as (p, p), need to get (p,)
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding[:1])
            input = F.pad(
                input, _reversed_padding_repeated_twice, mode=self.padding_mode
            )
        return ops.quantized.conv1d_dynamic(input, self._packed_params, reduce_range)


class Conv2d(nnq.Conv2d):
    r"""A dynamically quantized conv module with floating point tensors as inputs and outputs.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.Conv2d` and :class:`~torch.ao.nn.quantized.dynamic.Conv2d` and

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point

    See :class:`~torch.nn.Conv2d` for other attributes.

    Examples::

        >>> # xdoctest: +SKIP
        >>> # With square kernels and equal stride
        >>> m = nn.quantized.dynamic.Conv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.quantized.dynamic.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.quantized.dynamic.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> output = m(input)

    """

    _FLOAT_MODULE: ClassVar[type[nn.Conv2d]] = nn.Conv2d
    _NNIQAT_CONV_BN_MODULE: ClassVar[Optional[type[nn.Module]]] = None
    _NNI_CONV_RELU_MODULE: ClassVar[Optional[type[nn.Module]]] = None

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        warnings.warn(
            f"The current implementation of the {self._get_name()} module "
            "has poor numerical accuracy and its use is not recommended"
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _get_name(self):
        return "DynamicQuantizedConv2d"

    def forward(self, input: Tensor, reduce_range: bool = True) -> Tensor:
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        if self.padding_mode != "zeros":
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding)
            input = F.pad(
                input, _reversed_padding_repeated_twice, mode=self.padding_mode
            )
        return ops.quantized.conv2d_dynamic(input, self._packed_params, reduce_range)


class Conv3d(nnq.Conv3d):
    r"""A dynamically quantized conv module with floating point tensors as inputs and outputs.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.Conv3d` and :class:`~torch.ao.nn.quantized.dynamic.Conv3d` and

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point

    See :class:`~torch.nn.Conv3d` for other attributes.

    Examples::

        >>> # xdoctest: +SKIP
        >>> # With square kernels and equal stride
        >>> m = nn.quantized.dynamic.Conv3d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.quantized.dynamic.Conv3d(16, 33, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.quantized.dynamic.Conv3d(16, 33, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), dilation=(1, 2, 2))
        >>> input = torch.randn(20, 16, 56, 56, 56)
        >>> output = m(input)

    """

    _FLOAT_MODULE: ClassVar[type[nn.Conv3d]] = nn.Conv3d
    _NNIQAT_CONV_BN_MODULE: ClassVar[Optional[type[nn.Module]]] = None
    _NNI_CONV_RELU_MODULE: ClassVar[Optional[type[nn.Module]]] = None

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        warnings.warn(
            f"The current implementation of the {self._get_name()} module has poor numerical accuracy and its use is not recommended"  # noqa: B950
        )
        assert padding_mode != "reflect", "Conv3d does not support reflection padding"
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        super()._init(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _triple(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _get_name(self):
        return "DynamicQuantizedConv3d"

    def forward(self, input: Tensor, reduce_range: bool = True) -> Tensor:
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 5:
            raise ValueError("Input shape must be `(N, C, D, H, W)`!")
        if self.padding_mode != "zeros":
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding)
            input = F.pad(
                input, _reversed_padding_repeated_twice, mode=self.padding_mode
            )
        return ops.quantized.conv3d_dynamic(input, self._packed_params, reduce_range)


class ConvTranspose1d(nnq.ConvTranspose1d):
    r"""A dynamically quantized transposed convolution module with floating point tensors as inputs and outputs.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.ConvTranspose1d`.

    For special notes, please, see :class:`~torch.ao.nn.quantized.dynamic.Conv1d`

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point
    See :class:`~torch.nn.ConvTranspose1d` for other attributes.

    Examples::

        >>> # xdoctest: +SKIP
        >>> # With square kernels and equal stride
        >>> m = nndq.ConvTranspose1d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nndq.ConvTranspose1d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> output = m(input)
        >>> # exact output size can be also specified as an argument
        >>> downsample = nndq.Conv1d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nndq.ConvTranspose1d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(input)
        >>> h.size()
        torch.Size([1, 16, 6])
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12])
    """

    _FLOAT_MODULE: ClassVar[type[nn.ConvTranspose1d]] = nn.ConvTranspose1d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        warnings.warn(
            f"The current implementation of the {self._get_name()} module has poor numerical accuracy and its use is not recommended"  # noqa: B950
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            padding_mode,
            **factory_kwargs,
        )

    def _get_name(self):
        return "DynamicQuantizedConvTranspose1d"

    def forward(self, input: Tensor, reduce_range: bool = True) -> Tensor:
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 3:
            raise ValueError("Input shape must be `(N, C, L)`!")
        return torch.ops.quantized.conv_transpose1d_dynamic(
            input, self._packed_params, reduce_range
        )


class ConvTranspose2d(nnq.ConvTranspose2d):
    r"""A dynamically quantized transposed convolution module with floating point tensors as inputs and outputs.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.ConvTranspose2d`.

    For special notes, please, see :class:`~torch.ao.nn.quantized.dynamic.Conv2d`

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point
    See :class:`~torch.nn.ConvTranspose2d` for other attributes.

    Examples::

        >>> # xdoctest: +SKIP
        >>> # With square kernels and equal stride
        >>> m = nnq.ConvTranspose2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nnq.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> output = m(input)
        >>> # exact output size can be also specified as an argument
        >>> downsample = nnq.Conv2d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nnq.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(input)
        >>> h.size()
        torch.Size([1, 16, 6, 6])
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12, 12])
    """

    _FLOAT_MODULE: ClassVar[type[nn.ConvTranspose2d]] = nn.ConvTranspose2d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        warnings.warn(
            f"The current implementation of the {self._get_name()} module has poor numerical accuracy and its use is not recommended"  # noqa: B950
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            padding_mode,
            **factory_kwargs,
        )

    def _get_name(self):
        return "DynamicQuantizedConvTranspose2d"

    def forward(self, input: Tensor, reduce_range: bool = True) -> Tensor:
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        return ops.quantized.conv_transpose2d_dynamic(
            input, self._packed_params, reduce_range
        )


class ConvTranspose3d(nnq.ConvTranspose3d):
    r"""A dynamically quantized transposed convolution module with floating point tensors as inputs and outputs.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.ConvTranspose3d`.

    For special notes, please, see :class:`~torch.ao.nn.quantized.dynamic.Conv3d`

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point
    See :class:`~torch.nn.ConvTranspose3d` for other attributes.

    Examples::

        >>> # xdoctest: +SKIP
        >>> # With cubic kernels and equal stride
        >>> m = nnq.ConvTranspose3d(16, 33, 3, stride=2)
        >>> # non-cubic kernels and unequal stride and with padding
        >>> m = nnq.ConvTranspose3d(16, 33, (3, 3, 5), stride=(2, 1, 1), padding=(4, 2, 2))
        >>> output = m(input)
        >>> # exact output size can be also specified as an argument
        >>> downsample = nnq.Conv3d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nnq.ConvTranspose3d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(input)
        >>> h.size()
        torch.Size([1, 16, 6, 6, 6])
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12, 12, 12])
    """

    _FLOAT_MODULE: ClassVar[type[nn.ConvTranspose3d]] = nn.ConvTranspose3d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        warnings.warn(
            f"The current implementation of the {self._get_name()} module has poor numerical accuracy and its use is not recommended"  # noqa: B950
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            padding_mode,
            **factory_kwargs,
        )

    def _get_name(self):
        return "DynamicQuantizedConvTranspose3d"

    def forward(self, input: Tensor, reduce_range: bool = True) -> Tensor:
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 5:
            raise ValueError("Input shape must be `(N, C, T, H, W)`!")
        return ops.quantized.conv_transpose3d_dynamic(
            input, self._packed_params, reduce_range
        )
