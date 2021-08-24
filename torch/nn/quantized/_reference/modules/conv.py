import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from torch.nn.common_types import _size_1_t
from torch.nn.modules.utils import _single
from .utils import _init_weight_qparams
from .utils import _quantize_and_dequantize_weight
from .utils import _save_weight_qparams
from .utils import _load_weight_qparams

class _ConvNd(torch.nn.modules.conv._ConvNd):
    """ A reference version of nn.quantized.Conv2d
        we will not pack the parameters in this module, since weight packing is an
        optimization for quantized backends supported in PyTorch (fbgemm/qnnpack),
        this is useful when user want to use this module in other backends like Glow.
    """
    __annotations__ = {"_bias": Optional[torch.Tensor]}

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        _save_weight_qparams(destination, prefix, self.weight_qscheme, self.weight_dtype, self.weight_scale, self.weight_zero_point, self.weight_axis)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        _load_weight_qparams(self, state_dict)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, False,
            missing_keys, unexpected_keys, error_msgs)

    def get_weight(self):
        return _quantize_and_dequantize_weight(
            self.weight, self.weight_qscheme,
            self.weight_dtype, self.weight_scale, self.weight_zero_point, self.weight_axis)

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
                 weight_qparams: Optional[Dict[str, Any]] = None):
        nn.Conv1d.__init__(
            self, in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)
        _init_weight_qparams(self, weight_qparams)

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
            x, weight_dequant, self._bias, self.stride,
            self.padding, self.dilation, self.groups)
        return result

    def _get_name(self):
        return "QuantizedConv1d(Reference)"

class Conv2d(_ConvNd, nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros',
                 weight_qparams: Optional[Dict[str, Any]] = None):
        nn.Conv2d.__init__(
            self, in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)

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
            x, weight_dequant, self._bias, self.stride,
            self.padding, self.dilation, self.groups)
        return result

    def _get_name(self):
        return "QuantizedConv2d(Reference)"

class Conv3d(_ConvNd, nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode="zeros",
                 weight_qparams: Optional[Dict[str, Any]] = None):
        nn.Conv3d.__init__(
            self, in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)

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
            x, weight_dequant, self._bias, self.stride,
            self.padding, self.dilation, self.groups)
        return result

    def _get_name(self):
        return "QuantizedConv3d(Reference)"
