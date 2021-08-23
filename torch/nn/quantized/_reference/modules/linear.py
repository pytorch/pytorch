import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import quantize_and_dequantize_weight

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
            self.register_buffer("weight_scale", torch.tensor(weight_qparams["scale"]))
            self.register_buffer("weight_zero_point", torch.tensor(weight_qparams["zero_point"]))
            if self.weight_qscheme == torch.per_channel_affine:
                self.register_buffer("weight_axis", torch.tensor(weight_qparams["axis"]))
            else:
                # added for TorchScriptability, not used
                self.register_buffer("weight_axis", torch.tensor(0))
    def _get_name(self):
        return "QuantizedLinear(Reference)"

    def get_weight(self):
        return _quantize_and_dequantize_weight(weight, weight_qscheme, weight_dtype, weight_scale, weight_zero_point, weight_axis)

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
        destination[prefix + "weight_qscheme"] = self.weight_qscheme
        destination[prefix + "weight_dtype"] = self.weight_dtype
        if self.weight_qscheme is not None:
            destination[prefix + "weight_scale"] = self.weight_scale
            destination[prefix + "weight_zero_point"] = self.weight_zero_point
            if self.weight_qscheme == torch.per_channel_affine:
                destination[prefix + "weight_axis"] = self.weight_axis

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self.weight_qscheme = state_dict[prefix + "weight_qscheme"]
        self.weight_dtype = state_dict[prefix + "weight_dtype"]
        state_dict.pop(prefix + "weight_qscheme")
        state_dict.pop(prefix + "weight_dtype")
        if self.weight_qscheme is not None:
            self.weight_scale = state_dict[prefix + "weight_scale"]
            self.weight_zero_point = state_dict[prefix + "weight_zero_point"]
            state_dict.pop(prefix + "weight_scale")
            state_dict.pop(prefix + "weight_zero_point")
            if self.weight_qscheme == torch.quantize_per_channel:
                self.weight_axis = state_dict[prefix + "weight_axis"]

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
