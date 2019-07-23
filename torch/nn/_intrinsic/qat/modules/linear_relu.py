from __future__ import absolute_import, division, print_function, unicode_literals
from torch.nn.qat import Linear as QATLinear
from torch.nn._intrinsic import LinearReLU2d as NNLinearReLU2d
from torch.quantization.QConfig import default_qat_qconfig
import torch.nn.functional as F

class LinearReLU(QATLinear):
    r"""
    A LinearReLU module fused from Linear and ReLU modules, attached with
    FakeQuantize modules for output activation and weight, used in
    quantization aware training.

    We adopt the same interface as :class:`torch.nn.Linear`.

    Similar to `torch.nn._intrinsic.LinearReLU`, with FakeQuantize modules initialized to
    default.

    Attributes:
        observer: fake quant module for output activation, it's called observer
            to align with post training flow, TODO: rename?
        weight: fake quant module for weight

    Examples::

        >>> m = nn.qat.LinearReLU(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __FLOAT_MODULE__ = NNLinearReLU2d

    def __init__(self, in_features, out_features, bias=True,
                 activation_fake_quant=default_qat_qconfig.activation,
                 weight_fake_quant=default_qat_qconfig.weight):
        super(LinearReLU, self).__init__(in_features, out_features, bias, activation_fake_quant, weight_fake_quant)

    def forward(self, input):
        return self.observer(F.relu(F.linear(input, self.weight_fake_quant(self.weight), self.bias)))
