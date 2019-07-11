from __future__ import absolute_import, division, print_function, unicode_literals
from ...modules.linear import Linear as NNLinear
from torch.quantization.QConfig import default_qat_qconfig
import torch.nn.functional as F

class Linear(NNLinear):
    # TODO: update docs
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
        self.observer = default_qat_qconfig.activation()
        self.weight_fake_quant = default_qat_qconfig.weight()

    def forward(self, input):
        return self.observer(F.linear(input, self.weight_fake_quant(self.weight), self.bias))

    # TODO: support initializing from qconfig
    @staticmethod
    def from_float(mod):
        r"""Create a qat module from a float module or qparams_dict

            Args: `mod` a float module, either produced by torch.quantization utilities
            or directly from user
        """
        assert type(mod) == NNLinear, 'nnq.Linear.from_float only works for nn.Linear'
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert hasattr(mod, 'observer'), 'Input float module must have observer attached'
        qat_linear = Linear(mod.in_features, mod.out_features)
        qat_linear.weight = mod.weight
        qat_linear.bias = mod.bias
        qat_linear.observer = mod.qconfig.activation()
        qat_linear.weight_fake_quant = mod.qconfig.weight()
        return qat_linear
