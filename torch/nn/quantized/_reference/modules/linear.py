import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from .utils import _quantize_and_dequantize_weight
from .utils import _quantize_weight
from .utils import _save_weight_qparams
from .utils import _get_weight_qparam_keys

class Linear(nn.Linear):
    """ A reference quantized linear module that fits into the FX
    Graph Mode Quantization workflow
    activation will be floating point Tensor, we will store floating
    point weight as well in the module, but in forward we'll quantize
    and dequantize the weight before running the floating point functional
    linear operator.
    """
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias_: bool = True,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            weight_qparams: Optional[Dict[str, Any]] = None):
        super().__init__(in_features, out_features, bias_, device, dtype)
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
                torch.tensor(
                    weight_qparams["zero_point"],
                    dtype=torch.int, device=device))
            if self.weight_qscheme == torch.per_channel_affine:
                self.register_buffer(
                    "weight_axis",
                    torch.tensor(weight_qparams["axis"], dtype=torch.int, device=device))
            else:
                # added for TorchScriptability, not used
                self.register_buffer(
                    "weight_axis",
                    torch.tensor(0, dtype=torch.int, device=device))

    def _get_name(self):
        return "QuantizedLinear(Reference)"

    def get_weight(self):
        """
        Fake quantize (quantize and dequantize) the weight with
        the quantization parameters for weight, this is used to
        simulate the numerics for the quantized weight in a quantized
        model
        """
        # suppress mypy warning
        assert isinstance(self.weight_scale, torch.Tensor)
        assert isinstance(self.weight_zero_point, torch.Tensor)
        assert isinstance(self.weight_axis, torch.Tensor)
        return _quantize_and_dequantize_weight(
            self.weight, self.weight_qscheme, self.weight_dtype, self.weight_scale,
            self.weight_zero_point, self.weight_axis)

    def get_quantized_weight(self):
        # suppress mypy warning
        assert isinstance(self.weight_scale, torch.Tensor)
        assert isinstance(self.weight_zero_point, torch.Tensor)
        assert isinstance(self.weight_axis, torch.Tensor)
        return _quantize_weight(
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

    @classmethod
    def from_float(cls, float_linear, weight_qparams):
        qref_linear = Linear(
            float_linear.in_features, float_linear.out_features,
            float_linear.bias is not None, device=float_linear.weight.device,
            dtype=float_linear.weight.dtype, weight_qparams=weight_qparams)
        qref_linear.weight = torch.nn.Parameter(float_linear.weight.detach())
        if float_linear.bias is not None:
            qref_linear.bias = torch.nn.Parameter(float_linear.bias.detach())
        return qref_linear
