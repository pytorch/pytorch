from __future__ import absolute_import, division, print_function, unicode_literals
from ...modules.linear import Linear as NNLinear
from torch.quantization.QConfig import default_qat_qconfig
import torch.nn.functional as F

class Linear(NNLinear):
    r"""
    A linear module attached with FakeQuantize modules for both output
    activation and weight, used for quantization aware training.

    We adopt the same interface as `torch.nn.Linear`, please see https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
    for documentation.

    Similar to `torch.nn.Linear`, with FakeQuantize modules initialized to
    default.

    Attributes:
        observer: fake quant module for output activation, it's called observer
            to align with post training flow
        weight: fake quant module for weight

    Examples::

        >>> m = nn.qat.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True,
                 activation_fake_quant=default_qat_config.activation(),
                 weight_fake_quant=default_qat_qconfig.weight()):
        assert bias, 'nobias is not supported in Quantized Linear module yet'
        super(Linear, self).__init__(in_features, out_features, bias)
        self.observer = activation_fake_quant
        self.weight_fake_quant = weight_fake_quant

    def forward(self, input):
        return self.observer(F.linear(input, self.weight_fake_quant(self.weight), self.bias))

    # TODO: support initializing from qconfig
    @staticmethod
    def from_float(mod):
        r"""Create a qat module from a float module or qparams_dict

            Args: `mod` a float module, either produced by torch.quantization utilities
            or directly from user
        """
        assert type(mod) == NNLinear, 'qat.Linear.from_float only works for nn.Linear'
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert hasattr(mod, 'observer'), 'Input float module must have observer attached'
        qat_linear = Linear(mod.in_features, mod.out_features,
                            activation_fake_quant=mod.qconfig.activation(),
                            weight_fake_quant=mod.qconfig.weight())
        qat_linear.weight = mod.weight
        qat_linear.bias = mod.bias
        return qat_linear
