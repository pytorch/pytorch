import torch
import torch.nn.quantized as nnq
import torch.nn.intrinsic as nni
import torch.nn.functional as F


class LinearReLU(nnq.Linear):
    r"""
    A LinearReLU module fused from Linear and ReLU modules

    We adopt the same interface as :class:`torch.nn.quantized.Linear`.

    Attributes:
        Same as torch.nn.quantized.Linear

    Examples::

        >>> m = nn.intrinsic.LinearReLU(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    _FLOAT_MODULE = nni.LinearReLU

    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            dtype=torch.qint8,
            backend_independent=False):
        super().__init__(in_features, out_features, bias, dtype, backend_independent)

    def forward(self, x):
        if self.backend_independent:
            x_dequant = x.dequantize()
            weight_dequant = self._qweight.dequantize()
            float_result = F.linear(x_dequant, weight_dequant, self._bias)
            float_result = F.relu(float_result, inplace=True)
            # NEEDFIX: we don't have dtype in the Linear module APIs right now!
            result = torch.quantize_per_tensor(
                float_result, self.scale, self.zero_point, torch.quint8)
        else:
            result = torch.ops.quantized.linear_relu(
                x, self._packed_params._packed_params,
                self.scale,
                self.zero_point)
        return result

    def _get_name(self):
        return 'QuantizedLinearReLU'

    @classmethod
    def from_float(cls, mod, backend_independent=False):
        return super(LinearReLU, cls).from_float(mod, backend_independent)
