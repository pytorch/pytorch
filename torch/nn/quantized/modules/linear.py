from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from ...modules.module import Module
from ...modules.linear import Linear as NNLinear

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
        >>> tensor([[ 1., -1.],
>         [ 1., -1.]], size=(2, 2), dtype=torch.qint8, scale=1.0, zero_point=2)
    """

    def __init__(self, out_scale, out_zero_point, out_dtype):
        super(Quantize, self).__init__()
        self.register_buffer('out_scale', torch.tensor([out_scale]))
        self.register_buffer('out_zero_point', torch.tensor([out_zero_point], dtype=torch.long))
        self.out_dtype = out_dtype

    def forward(self, X):
        return torch.quantize_linear(X, self.out_scale.item(),
                                     self.out_zero_point.item(), self.out_dtype)

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
        >>> tensor([[ 1., -1.],
            [ 1., -1.]], dtype=torch.float32)
    """

    def __init__(self):
        super(DeQuantize, self).__init__()

    def forward(self, Xq):
        return Xq.dequantize()

    @staticmethod
    def from_float(mod):
        return DeQuantize()

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
        out_scale: `scale` parameter of output Quantized Tensor, type: double
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
        super(Linear, self).__init__(in_features, out_features, bias)
        if bias:
            del self.bias
            qbias = torch._empty_affine_quantized(
                [out_features], scale=1, zero_point=0, dtype=torch.qint32)
            self.register_buffer('bias', qbias)
        else:
            self.register_buffer('bias', None)
        del self.weight
        qweight = torch._empty_affine_quantized(
            [out_features, in_features], scale=1, zero_point=0,
            dtype=torch.qint8)
        self.register_buffer('_packed_weight',
                             torch.ops.quantized.fbgemm_linear_prepack(qweight))
        self.register_buffer('out_scale',
                             torch.Tensor([1.0]).to(torch.double))
        self.register_buffer('out_zero_point',
                             torch.Tensor([0]).to(torch.long))

    @property
    def weight(self):
        return torch.ops.quantized.fbgemm_linear_unpack(self._packed_weight)

    @weight.setter
    def weight(self, w):
        self._packed_weight = torch.ops.quantized.fbgemm_linear_prepack(w)

    def forward(self, x):
        # Note that we can handle self.bias == None case.
        Y_q = torch.ops.quantized.fbgemm_linear(
            x, self._packed_weight,
            self.bias,
            self.out_scale,
            self.out_zero_point)
        return Y_q

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'weight'] = torch.ops.quantized.fbgemm_linear_unpack(destination[prefix + '_packed_weight'])
        destination.pop(prefix + '_packed_weight')

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self._packed_weight = torch.ops.quantized.fbgemm_linear_prepack(state_dict[prefix + 'weight'])
        if prefix + 'bias' in state_dict:
            self.bias.copy_(state_dict[prefix + 'bias'])
            state_dict.pop(prefix + 'bias')
        state_dict.pop(prefix + 'weight')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, False,
                                      missing_keys, unexpected_keys, error_msgs)
        return

    # TODO: support initializing from quantization parameters when Quantizer is
    # exposed in python
    @staticmethod
    def from_float(mod):
        r"""Create a quantized module from a float module or qparams_dict

            Args: `mod` a float module, either produced by torch.quantization utilities
            or directly from user
        """
        assert type(mod) == NNLinear, 'nnq.Linear.from_float only works for nn.Linear'
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert hasattr(mod, 'observer'), 'Input float module must have observer attached'
        activation_observer = mod.observer
        act_qparams = activation_observer.calculate_qparams()
        weight_observer = mod.qconfig.weight()
        weight_observer(mod.weight)
        wt_qparams = weight_observer.calculate_qparams()
        bias_scale = (wt_qparams[0] * act_qparams[0]).float()
        qweight = torch.quantize_linear(mod.weight.float(), wt_qparams[0], wt_qparams[1].long().item(), torch.qint8)
        qbias = torch.quantize_linear(mod.bias.float(), bias_scale, 0, torch.qint32)
        qlinear = Linear(mod.in_features, mod.out_features)
        qlinear._packed_weight = torch.ops.quantized.fbgemm_linear_prepack(qweight)
        qlinear.bias = qbias
        qlinear.out_scale = torch.tensor([act_qparams[0]])
        qlinear.out_zero_point = torch.tensor([act_qparams[1]])
        return qlinear
