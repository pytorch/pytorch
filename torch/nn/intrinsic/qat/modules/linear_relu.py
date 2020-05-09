from __future__ import absolute_import, division, print_function, unicode_literals
import torch.nn.qat as nnqat
import torch.nn.intrinsic
import torch.nn.functional as F

class LinearReLU(nnqat.Linear):
    r"""
    A LinearReLU module fused from Linear and ReLU modules, attached with
    FakeQuantize modules for output activation and weight, used in
    quantization aware training.

    We adopt the same interface as :class:`torch.nn.Linear`.

    Similar to `torch.nn.intrinsic.LinearReLU`, with FakeQuantize modules initialized to
    default.

    Attributes:
        activation_post_process: fake quant module for output activation
        weight: fake quant module for weight

    Examples::

        >>> m = nn.qat.LinearReLU(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    _FLOAT_MODULE = torch.nn.intrinsic.LinearReLU

    def __init__(self, in_features, out_features, bias=True,
                 qconfig=None):
        super(LinearReLU, self).__init__(in_features, out_features, bias, qconfig)

    def forward(self, input):
        return self.activation_post_process(F.relu(
            F.linear(input, self.weight_fake_quant(self.weight), self.bias)))

    @classmethod
    def from_float(cls, mod, qconfig=None):
        return super(LinearReLU, cls).from_float(mod, qconfig)
