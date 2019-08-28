from __future__ import absolute_import, division, print_function, unicode_literals
import torch.nn.quantized as nnq
import torch.nn._intrinsic
import torch

class LinearReLU(nnq.Linear):
    r"""
    A LinearReLU module fused from Linear and ReLU modules

    We adopt the same interface as :class:`torch.nn.quantized.Linear`.

    Attributes:
        Same as torch.nn.quantized.Linear

    Examples::

        >>> m = nn._intrinsic.LinearReLU(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    _FLOAT_MODULE = torch.nn._intrinsic.LinearReLU

    def __init__(self, in_features, out_features, bias=True):
        super(LinearReLU, self).__init__(in_features, out_features, bias)

    def forward(self, input):
        bias = self.bias
        if bias is not None:
            bias = torch.quantize_linear(bias.dequantize(), float(self.weight_scale) * input.q_scale(), 0, torch.qint32)

        Y_q = torch.ops.quantized.fbgemm_linear_relu(
            input, self._packed_weight,
            bias,
            float(self.scale),
            int(self.zero_point))
        return Y_q

    @classmethod
    def from_float(cls, mod):
        return super(LinearReLU, self).from_float(mod)
