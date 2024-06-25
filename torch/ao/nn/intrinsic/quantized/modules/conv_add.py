# mypy: allow-untyped-defs
import torch
import torch.ao.nn.intrinsic
import torch.ao.nn.intrinsic.qat
import torch.ao.nn.quantized as nnq
import torch.nn.functional as F


_reverse_repeat_padding = nnq.modules.conv._reverse_repeat_padding


class ConvAdd2d(nnq.Conv2d):
    r"""
    A ConvAdd2d module is a fused module of Conv2d and Add

    We adopt the same interface as :class:`torch.ao.nn.quantized.Conv2d`.

    Attributes:
        Same as torch.ao.nn.quantized.Conv2d

    """
    _FLOAT_MODULE = torch.ao.nn.intrinsic.ConvAdd2d  # type: ignore[assignment]

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
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def forward(self, input, extra_input):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        if self.padding_mode != "zeros":
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding)
            input = F.pad(
                input, _reversed_padding_repeated_twice, mode=self.padding_mode
            )
        return torch.ops.quantized.conv2d_add(
            input, extra_input, self._packed_params, self.scale, self.zero_point
        )

    def _get_name(self):
        return "QuantizedConvAdd2d"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        return super().from_float(
            mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )

    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point):
        return super().from_reference(ref_qconv[0], output_scale, output_zero_point)


class ConvAddReLU2d(nnq.Conv2d):
    r"""
    A ConvAddReLU2d module is a fused module of Conv2d, Add and Relu

    We adopt the same interface as :class:`torch.ao.nn.quantized.Conv2d`.

    Attributes:
        Same as torch.ao.nn.quantized.Conv2d

    """
    _FLOAT_MODULE = torch.ao.nn.intrinsic.ConvAddReLU2d  # type: ignore[assignment]

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
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def forward(self, input, extra_input):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        if self.padding_mode != "zeros":
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding)
            input = F.pad(
                input, _reversed_padding_repeated_twice, mode=self.padding_mode
            )
        return torch.ops.quantized.conv2d_add_relu(
            input, extra_input, self._packed_params, self.scale, self.zero_point
        )

    def _get_name(self):
        return "QuantizedConvAddReLU2d"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        return super().from_float(
            mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )

    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point):
        return super().from_reference(ref_qconv[0], output_scale, output_zero_point)
