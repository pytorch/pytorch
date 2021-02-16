import torch
import torch.nn.quantized._reference as nnqr
import torch.nn.functional as F

class LinearReLU(nnqr.Linear):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            dtype=torch.qint8):
        super().__init__(in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dequant = x.dequantize()
        weight_dequant = self._qweight.dequantize()
        float_result = F.linear(x_dequant, weight_dequant, self._bias)
        float_result = F.relu(float_result, inplace=True)
        # NEEDFIX: we don't have dtype in the Linear module APIs right now!
        result = torch.quantize_per_tensor(
            float_result, self.scale, self.zero_point, torch.quint8)
        return result

    def _get_name(self):
        return "QuantizedLinearReLU(Reference)"
