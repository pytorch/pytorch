from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from ...modules.linear import Linear as NNLinear
from ...._jit_internal import weak_module


@weak_module
class Linear(NNLinear):
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
        out_scale: `scale` parameter of output Quantized Tensor, type: float
        out_zero_point: `zero_point` parameter for output Quantized Tensor, type: long
    """
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        assert bias, 'Quantized Linear module always has bias'
        super(Linear, self).__init__(in_features, out_features, bias)
        self._parameters.pop('weight')
        self._parameters.pop('bias')
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

    @staticmethod
    def from_float(mod):
        assert type(mod) == NNLinear, 'nnq.Linear.from_float only works for nn.Linear'
        assert hasattr(mod, 'qconfig'), 'Float Module must have qconfig defined'
        weight_observer = mod.qconfig.weight()
        weight_observer(mod.weight)
        wt_qparams = weight_observer.calculate_qparams()
        bias_qparams = torch.zeros(2)
        bias_scale = (wt_qparams[0] * mod.qparams[0]).float()
        qweight = torch.quantize_linear(mod.weight.float(), wt_qparams[0], wt_qparams[1].long(), torch.qint8)
        qbias = torch.quantize_linear(mod.bias.float(), bias_scale, 0, torch.qint32)
        qlinear = Linear(mod.out_features, mod.in_features)
        qlinear._packed_weight = torch.ops.quantized.fbgemm_linear_prepack(qweight)
        qlinear._bias = qbias
        qlinear.out_scale = torch.tensor([mod.qparams[0]])
        qlinear.out_zero_point = torch.tensor([mod.qparams[1]])
        return qlinear
