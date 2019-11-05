from __future__ import absolute_import, division, print_function, unicode_literals
import torch.nn.quantized as nnq
import torch.nn.intrinsic
import torch

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
    _FLOAT_MODULE = torch.nn.intrinsic.LinearReLU

    def __init__(self, in_features, out_features, bias=True):
        super(LinearReLU, self).__init__(in_features, out_features, bias)

    def forward(self, input):
        Y_q = torch.ops.quantized.linear_relu(
            input, self._packed_params,
            float(self.scale),
            int(self.zero_point))
        return Y_q

    def _get_name(self):
        return 'QuantizedLinearReLU'

    @classmethod
    def from_float(cls, mod):
        return super(LinearReLU, cls).from_float(mod)
