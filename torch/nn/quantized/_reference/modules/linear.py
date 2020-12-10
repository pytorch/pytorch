import torch
import torch.nn.quantized as nnq
import torch.nn.functional as F
from typing import Optional

class Linear(nnq.Linear):
    """ A backend independent version of nn.quantized.Linear
        we will not pack the parameters in this module, since weight packing is an
        optimization for quantized backends supported in PyTorch (fbgemm/qnnpack),
        this is useful when user want to use this module in other backends like Glow.
    """
    def __init__(self, in_features, out_features, bias_=True,
                 dtype=torch.qint8):
        super().__init__(in_features, out_features, bias_, dtype)
        self._qweight, self._bias = self._packed_params._weight_bias()
        del self._packed_params

    def _get_name(self):
        return "QuantizedLinear(Reference)"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dequant = x.dequantize()
        weight_dequant = self._qweight.dequantize()
        float_result = F.linear(x_dequant, weight_dequant, self._bias)
        # NEEDFIX: we don't have dtype in the Linear module APIs right now!
        result = torch.quantize_per_tensor(
            float_result, self.scale, self.zero_point, torch.quint8)
        return result

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
