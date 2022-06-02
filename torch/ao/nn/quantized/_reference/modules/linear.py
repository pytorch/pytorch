import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from .utils import ReferenceQuantizedModule

class Linear(nn.Linear, ReferenceQuantizedModule):
    """ A reference quantized linear module that fits into the FX
    Graph Mode Quantization workflow
    activation will be floating point Tensor, we will store floating
    point weight as well in the module, but in forward we'll quantize
    and dequantize the weight before running the floating point functional
    linear operator.
    """
    _IS_REFERENCE = True

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias_: bool = True,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            weight_qparams: Optional[Dict[str, Any]] = None):
        super().__init__(in_features, out_features, bias_, device, dtype)
        self._init_weight_qparams(weight_qparams, device)

    def _get_name(self):
        return "QuantizedLinear(Reference)"

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
        weight_quant_dequant = self.get_weight()
        result = F.linear(x, weight_quant_dequant, self.bias)
        return result

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
