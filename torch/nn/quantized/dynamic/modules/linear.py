from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from ....modules.module import Module
from ....modules.linear import Linear as NNLinear

class Linear(NNLinear):
    r"""
    A dynamic quantized linear module with quantized tensor as inputs
    and outputs.
    We adopt the same interface as `torch.nn.Linear`, please see
    https://pytorch.org/docs/stable/nn.html#torch.nn.Linear for documentation.

    Similar to `torch.nn.Linear`, attributes will be randomly initialized at
        module creation time and will be overwritten later

    Attributes:
        weight: the non-learnable quantized weights of the
                module which are of shape :math:`(\text{out\_features}, \text{in\_features})`.
        bias:   the non-learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized to zero.
        scale: `scale` parameter of weight Quantized Tensor, type: double
        zero_point: `zero_point` parameter for weight Quantized Tensor, type: long

    Examples::

        >>> m = nn.quantized.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    __constants__ = ['bias', 'in_features', 'out_features', 'scale', 'zero_point']

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__(in_features, out_features, bias)
        del self.weight
        del self.bias
        bias_fp32 = torch.Tensor(out_features).float()
        self.register_buffer('bias', bias_fp32)

        qweight = torch._empty_affine_quantized(
            [out_features, in_features], scale=1, zero_point=0,
            dtype=torch.qint8)
        self.register_buffer('_packed_weight',
                             torch.ops.quantized.fbgemm_linear_prepack(qweight))

    def forward(self, x):
        # Note that we can handle self.bias == None case.
        Y = torch.ops.quantized.fbgemm_linear_dynamic(
            x, self._packed_weight,
            self.bias)
        return Y.to(x.dtype)

    # TODO: support initializing from quantization parameters when Quantizer is
    # exposed in python
    @staticmethod
    def from_float(mod):
        r"""Create a dynamic quantized module from a float module or qparams_dict

            Args: `mod` a float module, either produced by torch.quantization utilities
            or directly from user
        """
        assert type(mod) == NNLinear, 'nnq.Linear.from_float only works for nn.Linear'
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert hasattr(mod, 'observer'), 'Input float module must have observer attached'
        weight_observer = mod.qconfig.weight()
        weight_observer(mod.weight)
        wt_qparams = weight_observer.calculate_qparams()
        bias_scale = (wt_qparams[0]).float()
        qweight = torch.quantize_linear(mod.weight.float(), wt_qparams[0], wt_qparams[1].long().item(), torch.qint8)
        qlinear = Linear(mod.in_features, mod.out_features)
        qlinear._packed_weight = torch.ops.quantized.fbgemm_linear_prepack(qweight)
        qlinear.bias = mod.bias.float()

        return qlinear
