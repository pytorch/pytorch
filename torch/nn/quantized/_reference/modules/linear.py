import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import _init_weight_qparams
from .utils import _quantize_and_dequantize_weight
from .utils import _save_weight_qparams
from .utils import _load_weight_qparams

class Linear(nn.Linear):
    """ A reference quantized linear module that fits into the FX
    Graph Mode Quantization workflow
    activation will be floating point Tensor, we will store floating
    point weight as well in the module, but in forward we'll quantize
    and dequantize the weight before running the floating point functional
    linear operator.
    """
    def __init__(self, in_features, out_features, bias_=True, weight_qparams=None):
        super().__init__(in_features, out_features, bias_)
        _init_weight_qparams(self, weight_qparams)

    def _get_name(self):
        return "QuantizedLinear(Reference)"

    def get_weight(self):
        return _quantize_and_dequantize_weight(
            self.weight, self.weight_qscheme, self.weight_dtype, self.weight_scale,
            self.weight_zero_point, self.weight_axis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        we have:
        w(float) -- quant - dequant \
        x(float) ------------- F.linear ---

        In the full model, we will see
        w(float) -- quant - *dequant \
        x -- quant --- *dequant --  *F.linear --- *quant - dequant
        and the backend should be able to fuse the ops with `*` into a quantized linear
        """
        weight_dequant = self.get_weight()
        result = F.linear(x, weight_dequant, self.bias)
        return result

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        _save_weight_qparams(destination, prefix, self.weight_qscheme, self.weight_dtype, self.weight_scale, self.weight_zero_point, self.weight_axis)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        _load_weight_qparams(self, state_dict)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, False,
            missing_keys, unexpected_keys, error_msgs)

    @classmethod
    def from_float(cls, float_linear, weight_qparams):
        qref_linear = Linear(float_linear.in_features, float_linear.out_features, float_linear.bias is not None, weight_qparams)
        qref_linear.weight = torch.nn.Parameter(float_linear.weight.detach())
        if float_linear.bias is not None:
            qref_linear.bias = torch.nn.Parameter(float_linear.bias.detach())
        return qref_linear
