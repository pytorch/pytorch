from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from ...modules.module import Module
from ...._jit_internal import weak_module



@weak_module
class Linear(Module):
    r"""
    A module that wraps the quantized fbgemm linear operator function
    We adopt the same interface as `torch.nn.Linear`, please see https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
    for documentation.

    Similar to `torch.nn.Linear`, attributes will be randomly initialized at
        module creation time and will be overwritten later

    Attributes:
        _packed_weight: the weight that is first quantized and then packed using
            fbgemm_linear_pack function
        bias:   the quantized bias
        output_scale: `scale` parameter of output Quantized Tensor
        output_zero_point: `zero_point` parameter for output Quantized Tensor
    """

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        weight = torch.randn(out_features, in_features, dtype=torch.float32)
        weight = torch.quantize_linear(weight, 1.0, 0, torch.qint8)
        _packed_weight = torch.ops.quantized.fbgemm_linear_prepack(weight)

        output_scale = 1.0
        self.register_buffer('output_scale', torch.Tensor([output_scale]))
        output_zero_point = 0
        self.register_buffer('output_zero_point', torch.Tensor([output_zero_point]))
        self.register_buffer('_packed_weight', _packed_weight)
        _bias = torch.quantize_linear(torch.zeros(out_features).float(), output_scale,
                                      output_zero_point, torch.qint32)
        self.register_buffer('bias', _bias)


    def forward(self, x):
        Y_q = torch.ops.quantized.fbgemm_linear(x, self._packed_weight, self.bias, self.output_scale, self.output_zero_point)
        return Y_q
