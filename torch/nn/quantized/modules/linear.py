from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from ...modules.linear import Linear as NNLinear
from ...._jit_internal import weak_module

@weak_module
class Linear(NNLinear):
    r"""
    A quantized linear module with quantized tensor as inputs
    and outputs.
    We adopt the same interface as `torch.nn.Linear`, please see https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
    for documentation.

    Similar to `torch.nn.Linear`, attributes will be randomly initialized at
        module creation time and will be overwritten later

    Attributes:
        weight: the non-learnable quantized weights of the
                module which are of shape :math:`(\text{out\_features}, \text{in\_features})`.
        bias:   the non-learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized to zero.
        out_scale: `scale` parameter of output Quantized Tensor, type: float
        out_zero_point: `zero_point` parameter for output Quantized Tensor, type: long

    Examples::

        >>> m = nn.quantized.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        assert bias, 'nobias is not supported in Quantized Linear module yet'
        super(Linear, self).__init__(in_features, out_features, bias)
        del self.weight
        del self.bias
        qweight = torch._empty_affine_quantized(
            [out_features, in_features], scale=1, zero_point=0,
            dtype=torch.qint8)
        qbias = torch._empty_affine_quantized(
            [out_features], scale=1, zero_point=0, dtype=torch.qint32)
        self.register_buffer('_packed_weight',
                             torch.ops.quantized.fbgemm_linear_prepack(qweight))
        self.register_buffer('bias', qbias)
        self.register_buffer('out_scale', torch.Tensor([1]))
        self.register_buffer('out_zero_point', torch.Tensor([0]))

    @property
    def weight(self):
        return torch.ops.quantized.fbgemm_linear_unpack(self._packed_weight)

    @weight.setter
    def weight(self, w):
        self._packed_weight = torch.ops.quantized.fbgemm_linear_prepack(w)

    def forward(self, x):
        Y_q = torch.ops.quantized.fbgemm_linear(
            x, self._packed_weight,
            self.bias,
            self.out_scale,
            self.out_zero_point)
        return Y_q
