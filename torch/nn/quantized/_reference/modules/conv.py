import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from torch.nn.common_types import _size_1_t
from .utils import _quantize_and_dequantize_weight
from .utils import _save_weight_qparams
from .utils import _get_weight_qparam_keys

class _ConvNd(torch.nn.modules.conv._ConvNd):
    """ A reference version of nn.quantized.Conv2d
        we will not pack the parameters in this module, since weight packing is an
        optimization for quantized backends supported in PyTorch (fbgemm/qnnpack),
        this is useful when user want to use this module in other backends like Glow.
    """
    __annotations__ = {"bias": Optional[torch.Tensor]}

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        _save_weight_qparams(
            destination, prefix, self.weight_qscheme, self.weight_dtype,
            self.weight_scale, self.weight_zero_point, self.weight_axis)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        for key in _get_weight_qparam_keys(state_dict, prefix):
            setattr(self, key, state_dict[prefix + key])
            state_dict.pop(prefix + key)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, False,
            missing_keys, unexpected_keys, error_msgs)

    def _init_weight_qparams(self, weight_qparams, device):
        if weight_qparams is None:
            weight_qparams = {
                "qscheme": torch.per_tensor_affine,
                "dtype": torch.quint8,
                "scale": 1.0,
                "zero_point": 0
            }
        self.weight_qscheme = weight_qparams["qscheme"]
        self.weight_dtype = weight_qparams["dtype"]
        assert self.weight_qscheme in [None, torch.per_tensor_affine, torch.per_channel_affine], \
            Exception(f"qscheme: {self.weight_qscheme} is not support in reference quantized linear module")
        if self.weight_qscheme is not None:
            self.register_buffer(
                "weight_scale",
                torch.tensor(weight_qparams["scale"], dtype=torch.float, device=device))
            self.register_buffer(
                "weight_zero_point",
                torch.tensor(weight_qparams["zero_point"], dtype=torch.int, device=device))
            if self.weight_qscheme == torch.per_channel_affine:
                self.register_buffer(
                    "weight_axis",
                    torch.tensor(weight_qparams["axis"], dtype=torch.int, device=device))
            else:
                # added for TorchScriptability, not used
                self.register_buffer(
                    "weight_axis", torch.tensor(0, dtype=torch.int, device=device))

    def get_weight(self):
        """
        Fake quantize (quantize and dequantize) the weight with
        the quantization parameters for weight, this is used to
        simulate the numerics for the quantized weight in a quantized
        model
        """
        # supress mypy warning
        assert isinstance(self.weight_scale, torch.Tensor)
        assert isinstance(self.weight_zero_point, torch.Tensor)
        assert isinstance(self.weight_axis, torch.Tensor)
        return _quantize_and_dequantize_weight(
            self.weight, self.weight_qscheme,
            self.weight_dtype, self.weight_scale, self.weight_zero_point, self.weight_axis)

    @staticmethod
    def from_float(cls, float_conv, weight_qparams):
        qref_conv = cls(
            float_conv.in_channels,
            float_conv.out_channels,
            float_conv.kernel_size,  # type: ignore[arg-type]
            float_conv.stride,  # type: ignore[arg-type]
            float_conv.padding,  # type: ignore[arg-type]
            float_conv.dilation,  # type: ignore[arg-type]
            float_conv.groups,
            float_conv.bias is not None,  # type: ignore[arg-type]
            float_conv.padding_mode,
            device=float_conv.weight.device,
            dtype=float_conv.weight.dtype,
            weight_qparams=weight_qparams)
        qref_conv.weight = torch.nn.Parameter(float_conv.weight.detach())
        if float_conv.bias is not None:
            qref_conv.bias = torch.nn.Parameter(float_conv.bias.detach())
        return qref_conv

class Conv1d(_ConvNd, nn.Conv1d):
    def __init__(self,
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
                 weight_qparams: Optional[Dict[str, Any]] = None):
        nn.Conv1d.__init__(
            self, in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode, device, dtype)
        self._init_weight_qparams(weight_qparams, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        we have:
        w(float) -- quant - dequant \
        x(float) ------------- F.conv1d ---

        In the full model, we will see
        w(float) -- quant - *dequant \
        x -- quant --- *dequant --  *F.conv1d --- *quant - dequant
        and the backend should be able to fuse the ops with `*` into a quantized conv1d
        """
        weight_dequant = self.get_weight()
        result = F.conv1d(
            x, weight_dequant, self.bias, self.stride,
            self.padding, self.dilation, self.groups)
        return result

    def _get_name(self):
        return "QuantizedConv1d(Reference)"

    @classmethod
    def from_float(cls, float_conv, weight_qparams):
        return _ConvNd.from_float(cls, float_conv, weight_qparams)

class Conv2d(_ConvNd, nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros',
                 device=None,
                 dtype=None,
                 weight_qparams: Optional[Dict[str, Any]] = None):
        nn.Conv2d.__init__(
            self, in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode, device, dtype)
        self._init_weight_qparams(weight_qparams, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        we have:
        w(float) -- quant - dequant \
        x(float) ------------- F.conv2d ---

        In the full model, we will see
        w(float) -- quant - *dequant \
        x -- quant --- *dequant --  *F.conv2d --- *quant - dequant
        and the backend should be able to fuse the ops with `*` into a quantized conv2d
        """
        weight_dequant = self.get_weight()
        result = F.conv2d(
            x, weight_dequant, self.bias, self.stride,
            self.padding, self.dilation, self.groups)
        return result

    def _get_name(self):
        return "QuantizedConv2d(Reference)"

    @classmethod
    def from_float(cls, float_conv, weight_qparams):
        return _ConvNd.from_float(cls, float_conv, weight_qparams)

class Conv3d(_ConvNd, nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode="zeros",
                 device=None,
                 dtype=None,
                 weight_qparams: Optional[Dict[str, Any]] = None):
        nn.Conv3d.__init__(
            self, in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)
        self._init_weight_qparams(weight_qparams, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        we have:
        w(float) -- quant - dequant \
        x(float) ------------- F.conv3d ---

        In the full model, we will see
        w(float) -- quant - *dequant \
        x -- quant --- *dequant --  *F.conv3d --- *quant - dequant
        and the backend should be able to fuse the ops with `*` into a quantized conv3d
        """
        weight_dequant = self.get_weight()
        result = F.conv3d(
            x, weight_dequant, self.bias, self.stride,
            self.padding, self.dilation, self.groups)
        return result

    def _get_name(self):
        return "QuantizedConv3d(Reference)"

    @classmethod
    def from_float(cls, float_conv, weight_qparams):
        return _ConvNd.from_float(cls, float_conv, weight_qparams)
