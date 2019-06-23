from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from ...modules.module import Module
from ...._jit_internal import weak_module

@weak_module
class Quantize(Module):
    r"""Quantizes an incoming tensor
    Args:
     `output_scale`: scale of the output Quantized Tensor
     `output_zero_point`: zero_point of output Quantized Tensor
     `output_dtype`: data type of output Quantized Tensor

    Attributes:
      `output_scale`, `output_zero_point`, `output_dtype`

    Examples::
        >>> t = torch.tensor([[1., -1.], [1., -1.]])
        >>> scale, zero_point, dtype = 1.0, 2, torch.qint8
        >>> qm = Quantize(scale, zero_point, dtype)
        >>> qt = qm(t)
        >>> print(qt)
        >>> tensor([[ 1., -1.],
>         [ 1., -1.]], size=(2, 2), dtype=torch.qint8, scale=1.0, zero_point=2)
    """

    def __init__(self, output_scale, output_zero_point, output_dtype):
        super(Quantize, self).__init__()
        self.register_buffer('output_scale', torch.tensor([output_scale]))
        self.register_buffer('output_zero_point', torch.tensor([output_zero_point], dtype=torch.long))
        self.output_dtype = output_dtype

    def forward(self, X):
        Xq = torch.quantize_linear(X, self.output_scale.item(), self.output_zero_point.item(), self.output_dtype)
        return Xq

    @staticmethod
    def from_float(mod):
        return Quantize(mod.qparams[0].item(), mod.qparams[1].item(), torch.quint8)

@weak_module
class DeQuantize(Module):
    r"""Dequantizes an incoming tensor
    Examples::
        >>> t = torch.tensor([[1., -1.], [1., -1.]])
        >>> scale, zero_point, dtype = 1.0, 2, torch.qint8
        >>> qm = Quantize(scale, zero_point, dtype)
        >>> qt = qm(t)
        >>> deqm = DeQuantize()
        >>> t2 = deqm(qt)
        >>> print(t2)
        >>> tensor([[ 1., -1.],
            [ 1., -1.]], dtype=torch.float32)
    """

    def __init__(self):
        super(DeQuantize, self).__init__()

    def forward(self, Xq):
        X = Xq.dequantize()
        return X

    @staticmethod
    def from_float(mod):
        return DeQuantize()

@weak_module
class Linear(Module):
    r"""
    A module that wraps the quantized fbgemm linear operator function
    We adopt the same interface as `torch.nn.Linear`, please see https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
    for documentation.

    Similar to `torch.nn.Linear`, attributes will be randomly initialized at
        module creation time and will be overwritten later

    Attributes:
        _packed_weight: the non-learnable packed weights of the
            module which are of shape :math:`(\text{out\_features}, \text{in\_features})`.
        bias:   the non-learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized to zero.
        output_scale: `scale` parameter of output Quantized Tensor
        output_zero_point: `zero_point` parameter for output Quantized Tensor

    Examples::

        >>> m = nn.quantized.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['bias', 'in_features', 'out_features']

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

    # TODO: remove after https://github.com/pytorch/pytorch/pull/21933 is landed
    # def state_dict(self, destination=None, prefix='', keep_vars=False):
    #     r"""
    #     Example::
    #
    #         >>> module.state_dict().keys()
    #         ['bias', 'weight']
    #
    #     """
    #     raw_dict = super().state_dict(destination, prefix, keep_vars)
    #     weight = torch.ops.quantized.fbgemm_linear_unpack(raw_dict[prefix + '_packed_weight'])
    #     raw_dict[prefix + 'weight'] = weight
    #     raw_dict.pop(prefix + '_packed_weight')
    #     return raw_dict

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'weight'] = torch.ops.quantized.fbgemm_linear_unpack(destination[prefix + '_packed_weight'])
        destination.pop(prefix + '_packed_weight')

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        r"""
            Modify state_dict first and then use default load function
        """
        self._packed_weight = torch.ops.quantized.fbgemm_linear_prepack(state_dict[prefix + 'weight'])
        self.bias = state_dict[prefix + 'bias']
        # state_dict.pop(prefix + 'weight')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, False,
                                      missing_keys, unexpected_keys, error_msgs)
        return
