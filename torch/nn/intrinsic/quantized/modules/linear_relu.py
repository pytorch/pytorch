import torch
import torch.nn.quantized as nnq
import torch.nn.intrinsic as nni

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

    def __init__(self, in_features, out_features, bias=True, dtype=torch.qint8):
        super().__init__(in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.quantized.linear_relu(
            x, self._packed_params._packed_params, self.scale, self.zero_point)

    def _get_name(self):
        return 'QuantizedLinearReLU'

    @classmethod
    def from_float(cls, mod):
        return super(LinearReLU, cls).from_float(mod)
