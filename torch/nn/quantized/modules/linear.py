from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from torch.nn.modules.module import Module
from torch.nn.modules.linear import Linear as NNLinear

from torch._jit_internal import Optional

class Quantize(Module):
    r"""Quantizes an incoming tensor
    Args:
     `out_scale`: scale of the output Quantized Tensor
     `out_zero_point`: zero_point of output Quantized Tensor
     `out_dtype`: data type of output Quantized Tensor

    Attributes:
      `out_scale`, `out_zero_point`, `out_dtype`

    Examples::
        >>> t = torch.tensor([[1., -1.], [1., -1.]])
        >>> scale, zero_point, dtype = 1.0, 2, torch.qint8
        >>> qm = Quantize(scale, zero_point, dtype)
        >>> qt = qm(t)
        >>> print(qt)
        tensor([[ 1., -1.],
                [ 1., -1.]], size=(2, 2), dtype=torch.qint8, scale=1.0, zero_point=2)
    """

    def __init__(self, out_scale, out_zero_point, out_dtype):
        super(Quantize, self).__init__()
        self.register_buffer('_scale', torch.tensor([out_scale]))
        self.register_buffer('_zero_point', torch.tensor([out_zero_point], dtype=torch.long))
        self._dtype = out_dtype

    def forward(self, X):
        return torch.quantize_linear(X, float(self._scale),
                                     int(self._zero_point), self._dtype)

    @staticmethod
    def from_float(mod):
        assert hasattr(mod, 'observer')
        qparams = mod.observer.calculate_qparams()
        return Quantize(qparams[0].item(), qparams[1].item(), mod.observer.dtype)

class DeQuantize(Module):
    r"""Dequantizes an incoming tensor

    Examples::
        >>> input = torch.tensor([[1., -1.], [1., -1.]])
        >>> scale, zero_point, dtype = 1.0, 2, torch.qint8
        >>> qm = Quantize(scale, zero_point, dtype)
        >>> quantized_input = qm(input)
        >>> dqm = DeQuantize()
        >>> dequantized = dqm(quantized_input)
        >>> print(dequantized)
        tensor([[ 1., -1.],
                [ 1., -1.]], dtype=torch.float32)
    """

    def __init__(self):
        super(DeQuantize, self).__init__()

    def forward(self, Xq):
        return Xq.dequantize()

    @staticmethod
    def from_float(mod):
        return DeQuantize()

class Linear(torch.nn.Module):
    r"""
    A quantized linear module with quantized tensor as inputs and outputs.
    We adopt the same interface as `torch.nn.Linear`, please see
    https://pytorch.org/docs/stable/nn.html#torch.nn.Linear for documentation.

    Similar to :class:`~torch.nn.Linear`, attributes will be randomly
    initialized at module creation time and will be overwritten later

    Attributes:
        weight (Tensor): the non-learnable quantized weights of the module of
                         shape :math:`(\text{out\_features}, \text{in\_features})`.
        bias (Tensor): the non-learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized to zero.
        scale: `scale` parameter of output Quantized Tensor, type: double
        zero_point: `zero_point` parameter for output Quantized Tensor, type: long

    Examples::

        >>> m = nn.quantized.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> input = torch.quantize_linear(input, 1.0, 0, torch.quint8)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    def __init__(self, in_features, out_features, bias_=True):
        super(Linear, self).__init__()
        # We don't muck around with buffers or attributes or anything here
        # to keep the module simple. *everything* is simply a Python attribute.
        self.in_features = in_features
        self.out_features = out_features
        if bias_:
            self.bias = torch.jit.annotate(
                Optional[torch.Tensor],
                torch._empty_affine_quantized(
                    [out_features], scale=1, zero_point=0, dtype=torch.qint32))
        else:
            self.bias = torch.jit.annotate(Optional[torch.Tensor], None)

        qweight = torch._empty_affine_quantized(
            [out_features, in_features], scale=1, zero_point=0, dtype=torch.qint8)

        self._packed_weight = torch.ops.quantized.fbgemm_linear_prepack(qweight)
        self.scale = 1.0
        self.zero_point = 0

    def forward(self, x):
        return torch.ops.quantized.fbgemm_linear(
            x, self._packed_weight, self.bias, self.scale, self.zero_point)

    @staticmethod
    def from_float(mod):
        r"""Create a quantized module from a float module or qparams_dict

        Args:
            mod (Module): a float module, either produced by torch.quantization
                          utilities or provided by the user
        """
        if hasattr(mod, 'weight_fake_quant'):
            # assert type(mod) == QATLinear, 'training mode nnq.Linear.from_float only works for nn.qat.Linear'
            weight_observer = mod.weight_fake_quant
        else:
            assert type(mod) == NNLinear, 'nnq.Linear.from_float only works for nn.Linear'
            assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
            assert hasattr(mod, 'observer'), 'Input float module must have observer attached'
            weight_observer = mod.qconfig.weight()
            weight_observer(mod.weight)
        activation_observer = mod.observer
        act_scale, act_zp = activation_observer.calculate_qparams()
        wt_scale, wt_zp = weight_observer.calculate_qparams()
        bias_scale = (wt_scale * act_scale).float()
        qweight = torch.quantize_linear(mod.weight.float(), wt_scale, wt_zp.long().item(), torch.qint8)
        if mod.bias is not None:
            qbias = torch.quantize_linear(mod.bias.float(), bias_scale, 0, torch.qint32)
        else:
            qbias = None
        qlinear = Linear(mod.in_features, mod.out_features)
        qlinear._packed_weight = torch.ops.quantized.fbgemm_linear_prepack(qweight)
        qlinear.bias = qbias
        qlinear.scale = float(act_scale)
        qlinear.zero_point = int(act_zp)
        return qlinear
