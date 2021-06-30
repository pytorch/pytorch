import torch
import torch.nn.quantized as nnq
import torch.nn.functional as F
from typing import Optional
from torch.nn.common_types import _size_1_t
from torch.nn.modules.utils import _single

class _ConvNd(nnq._ConvNd):
    """ A reference version of nn.quantized.Conv2d
        we will not pack the parameters in this module, since weight packing is an
        optimization for quantized backends supported in PyTorch (fbgemm/qnnpack),
        this is useful when user want to use this module in other backends like Glow.
    """
    __annotations__ = {"_bias": Optional[torch.Tensor]}

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + '_qweight'] = self._qweight
        destination[prefix + '_bias'] = self._bias

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self._qweight = state_dict[prefix + '_qweight']
        self._bias = state_dict[prefix + '_bias']
        state_dict.pop(prefix + '_qweight')
        state_dict.pop(prefix + '_bias')

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, False,
            missing_keys, unexpected_keys, error_msgs)

    def _weight_bias(self):
        return self._qweight, self._bias

    def set_weight_bias(self, w: torch.Tensor, b: Optional[torch.Tensor]) -> None:
        self._qweight = w
        self._bias = b

class Conv1d(_ConvNd, nnq.Conv1d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_1_t,
                 stride: _size_1_t = 1,
                 padding: _size_1_t = 0,
                 dilation: _size_1_t = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros'):
        nnq.Conv1d.__init__(
            self, in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)
        # self.stride, self.padding, self.dilation are 2d tuple since
        # current quantized conv1d is using Conv2dPackedParams
        # TODO: we should fix this if we implemenet Conv1dPackedParams
        self._conv1d_stride = _single(self.stride[0])
        self._conv1d_padding = _single(self.padding[0])
        self._conv1d_dilation = _single(self.dilation[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dequant = x.dequantize()
        weight_dequant = self._qweight.dequantize()
        float_result = F.conv1d(
            x_dequant, weight_dequant, self._bias, self._conv1d_stride,
            self._conv1d_padding, self._conv1d_dilation, self.groups)
        # NEEDFIX: we don't have dtype in the Linear module APIs right now!
        result = torch.quantize_per_tensor(
            float_result, self.scale, self.zero_point, torch.quint8)
        return result

    def _get_name(self):
        return 'QuantizedConv1d(Reference)'

    @torch.jit.export
    def __setstate__(self, state):
        self.in_channels = state[0]
        self.out_channels = state[1]
        self.kernel_size = state[2]
        self.stride = state[3]
        self.padding = state[4]
        self.dilation = state[5]
        self.transposed = state[6]
        self.output_padding = state[7]
        self.groups = state[8]
        self.padding_mode = state[9]
        self.set_weight_bias(state[10], state[11])
        self.scale = state[12]
        self.zero_point = state[13]
        self.training = state[14]
        self._conv1d_stride = (self.stride[0],)
        self._conv1d_padding = (self.padding[0],)
        self._conv1d_dilation = (self.dilation[0],)

class Conv2d(_ConvNd, nnq.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        nnq.Conv2d.__init__(
            self, in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dequant = x.dequantize()
        weight_dequant = self._qweight.dequantize()
        float_result = F.conv2d(
            x_dequant, weight_dequant, self._bias, self.stride,
            self.padding, self.dilation, self.groups)
        # NEEDFIX: we don't have dtype in the Linear module APIs right now!
        result = torch.quantize_per_tensor(
            float_result, self.scale, self.zero_point, torch.quint8)
        return result

    def _get_name(self):
        return 'QuantizedConv2d(Reference)'

class Conv3d(_ConvNd, nnq.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        nnq.Conv3d.__init__(
            self, in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dequant = x.dequantize()
        weight_dequant = self._qweight.dequantize()
        float_result = F.conv3d(
            x_dequant, weight_dequant, self._bias, self.stride,
            self.padding, self.dilation, self.groups)
        # NEEDFIX: we don't have dtype in the Linear module APIs right now!
        result = torch.quantize_per_tensor(
            float_result, self.scale, self.zero_point, torch.quint8)
        return result

    def _get_name(self):
        return 'QuantizedConv3d(Reference)'
