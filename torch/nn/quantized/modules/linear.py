from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from ...modules.module import Module
from ...modules.linear import Linear as NNLinear
# from ...qat.modules.linear import Linear as QATLinear

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
        self.register_buffer('_scale', torch.tensor([out_scale]))
        self.register_buffer('_zero_point', torch.tensor([out_zero_point], dtype=torch.long))
        self._dtype = out_dtype

    def forward(self, X):
        return torch.quantize_linear(X, self._scale.item(),
                                     self._zero_point.item(), self._dtype)

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
        self.register_buffer('scale',
                             torch.tensor([1.0], dtype=torch.double))
        self.register_buffer('zero_point',
                             torch.tensor([0], dtype=torch.long))

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
            self.scale,
            self.zero_point)
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
        qbias = torch.quantize_linear(mod.bias.float(), bias_scale, 0, torch.qint32)
        qlinear = Linear(mod.in_features, mod.out_features)
        qlinear._packed_weight = torch.ops.quantized.fbgemm_linear_prepack(qweight)
        qlinear.bias = qbias
        qlinear.scale = torch.tensor([act_scale], dtype=torch.double)
        qlinear.zero_point = torch.tensor([act_zp], dtype=torch.long)
        return qlinear

class DynamicLinear(NNLinear):
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

        >>> m = nn.quantized.DynamicLinear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    __constants__ = ['bias', 'in_features', 'out_features', 'scale', 'zero_point']

    def __init__(self, in_features, out_features, bias=True):
        super(DynamicLinear, self).__init__(in_features, out_features, bias)
        del self.weight
        # weight_fp32 = torch.Tensor(out_features, in_features).float()
        # self.register_buffer('weight', weight_fp32)
        del self.bias
        bias_fp32 = torch.Tensor(out_features).float()
        self.register_buffer('bias', bias_fp32)

        # weight_prepack, col_offsets, self.scale, self.zero_point = torch.fbgemm_linear_quantize_weight(weight_fp32)
        # self.register_buffer(
        #     '_packed_weight',
        #     torch.fbgemm_pack_quantized_matrix(weight_prepack))
        # self.register_buffer('col_offsets', col_offsets)
        qweight = torch._empty_affine_quantized(
            [out_features, in_features], scale=1, zero_point=0,
            dtype=torch.qint8)
        self.register_buffer('_packed_weight',
                             torch.ops.quantized.fbgemm_linear_prepack(qweight))


    def forward(self, x):
        # Y = torch.fbgemm_linear_int8_weight_fp32_activation(
        #     x.float(), self.weight, self._packed_weight, self.col_offsets,
        #     self.scale, self.zero_point, self.bias)
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
        # activation_observer = mod.observer
        # act_qparams = activation_observer.calculate_qparams()
        weight_observer = mod.qconfig.weight()
        weight_observer(mod.weight)
        wt_qparams = weight_observer.calculate_qparams()
        bias_scale = (wt_qparams[0]).float()
        qweight = torch.quantize_linear(mod.weight.float(), wt_qparams[0], wt_qparams[1].long().item(), torch.qint8)
        # weight_prepack, col_offsets, self.scale, self.zero_point = torch.fbgemm_linear_quantize_weight(weight_fp32)
        qlinear = Linear(mod.in_features, mod.out_features)
        qlinear._packed_weight = torch.ops.quantized.fbgemm_linear_prepack(qweight)
        qlinear.bias = mod.bias.float()
        return qlinear

