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
    def __init__(self, qweight, qbias, output_scale, output_zero_point):
        super(Linear, self).__init__()
        self.register_buffer('_packed_weight', torch.ops.quantized.fbgemm_linear_prepack(qweight))
        self.register_buffer('output_scale', torch.Tensor([output_scale]))
        self.register_buffer('output_zero_point', torch.Tensor([output_zero_point]))
        self.register_buffer('bias', qbias)

    def forward(self, x):
        Y_q = torch.ops.quantized.fbgemm_linear(
            x, self._packed_weight,
            self.bias,
            self.output_scale,
            self.output_zero_point)
        return Y_q

    @staticmethod
    def from_float(mod):
        if hasattr(mod, 'qConfig'):
            weight_observer = mod.qConfig.weight()
            weight_observer(mod.weight)
            wt_qparams = weight_observer.calculate_qparams()
            bias_qparams = torch.zeros(2)
            bias_scale = (wt_qparams[0] * mod.qparams[0]).float()
            qweight = torch.quantize_linear(mod.weight.float(), wt_qparams[0], wt_qparams[1].long(), torch.qint8)
            qbias = torch.quantize_linear(mod.bias.float(), bias_scale, 0, torch.qint32)
            output_scale = mod.qparams[0]
            output_zero_point = mod.qparams[1]
        else:
            output_scale, output_zero_point = 1, 0
            weight = torch.randn(mod.out_features, mod.in_features, dtype=torch.float32)
            qweight = torch.quantize_linear(weight, 1, 0, torch.qint8)
            bias = torch.zeros(mod.out_features, dtype=torch.float)
            qbias = torch.quantize_linear(
                bias, output_scale, output_zero_point, torch.qint32)
        return Linear(qweight, qbias, output_scale, output_zero_point)
