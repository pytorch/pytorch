import torch
import torch.nn.intrinsic as nni
import torch.nn.quantized._reference as nnqr
import torch.nn.functional as F
from typing import Optional, Dict, Any

class LinearReLU(nnqr.Linear):
    _FLOAT_MODULE = nni.LinearReLU

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            weight_qparams: Optional[Dict[str, Any]] = None):
        super().__init__(in_features, out_features, bias, weight_qparams)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_dequant = self.get_weight()
        result = F.linear(x, weight_dequant, self.bias)
        result = F.relu(result, inplace=True)
        return result

    def _get_name(self):
        return "QuantizedLinearReLU(Reference)"
