from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from torch._jit_internal import Optional  # noqa: F401
import torch.nn as nn
import torch.nn.intrinsic as nni
from torch.nn.quantized.modules.utils import _quantize_weight


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
        >>> input = torch.quantize_per_tensor(input, 1.0, 0, torch.quint8)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    _FLOAT_MODULE = nn.Linear

    def __init__(self, in_features, out_features, bias_=True):
        super(Linear, self).__init__()
        # We don't muck around with buffers or attributes or anything here
        # to keep the module simple. *everything* is simply a Python attribute.
        # Serialization logic is explicitly handled in the below serialization and
        # deserialization modules
        self.in_features = in_features
        self.out_features = out_features
        bias = None
        if bias_:
            bias = torch.zeros(out_features, dtype=torch.float)


        qweight = torch._empty_affine_quantized(
            [out_features, in_features], scale=1, zero_point=0, dtype=torch.qint8)

        self.set_weight_bias(qweight, bias)
        self.scale = 1.0
        self.zero_point = 0

    def _get_name(self):
        return 'QuantizedLinear'

    def extra_repr(self):
        return 'in_features={}, out_features={}, scale={}, zero_point={}'.format(
            self.in_features, self.out_features, self.scale, self.zero_point
        )

    def forward(self, x):
        return torch.ops.quantized.linear(
            x, self._packed_params, self.scale, self.zero_point)

    # ===== Serialization methods =====
    # The special consideration here is that we have to unpack the weights into their
    # regular QTensor form for serialization. Packed weights should not live
    # outside the process in which they were created, rather they should be derived
    # from the QTensor weight.
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super(Linear, self)._save_to_state_dict(destination, prefix, keep_vars)
        (w, b) = self._weight_bias()
        destination[prefix + 'weight'] = w
        destination[prefix + 'scale'] = torch.tensor(self.scale)
        destination[prefix + 'zero_point'] = torch.tensor(self.zero_point)
        destination[prefix + 'bias'] = b

    @torch.jit.export
    def __getstate__(self):
        if not torch.jit.is_scripting():
            raise RuntimeError('torch.save() is not currently supported for quantized modules.'
                               ' See https://github.com/pytorch/pytorch/issues/24045.'
                               ' Please use state_dict or torch.jit serialization.')
        (w, b) = self._weight_bias()
        return (
            self.in_features,
            self.out_features,
            b,
            w,
            self.scale,
            self.zero_point,
            self.training
        )

    # ===== Deserialization methods =====
    # Counterpart to the serialization methods, we must pack the serialized QTensor
    # weight into its packed format for use by the FBGEMM ops.
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self.set_weight_bias(state_dict[prefix + 'weight'], state_dict[prefix + 'bias'])
        state_dict.pop(prefix + 'weight')
        state_dict.pop(prefix + 'bias')

        self.scale = float(state_dict[prefix + 'scale'])
        state_dict.pop(prefix + 'scale')

        self.zero_point = int(state_dict[prefix + 'zero_point'])
        state_dict.pop(prefix + 'zero_point')

        super(Linear, self)._load_from_state_dict(state_dict, prefix, local_metadata, False,
                                                  missing_keys, unexpected_keys, error_msgs)

    @torch.jit.export
    def __setstate__(self, state):
        self.in_features = state[0]
        self.out_features = state[1]
        self.set_weight_bias(state[3], state[2])
        self.scale = state[4]
        self.zero_point = state[5]
        self.training = state[6]

    # Function rather than property to make sure that JIT serialization doesn't
    # register this as an attribute
    def _weight_bias(self):
        return torch.ops.quantized.linear_unpack(self._packed_params)

    def weight(self):
        (w, b) = torch.ops.quantized.linear_unpack(self._packed_params)
        return w

    def bias(self):
        (w, b) = torch.ops.quantized.linear_unpack(self._packed_params)
        return b

    def set_weight_bias(self, w, b):
        # type: (torch.Tensor, Optional[torch.Tensor]) -> None
        self._packed_params = torch.ops.quantized.linear_prepack(w, b)

    @classmethod
    def from_float(cls, mod):
        r"""Create a quantized module from a float module or qparams_dict

        Args:
            mod (Module): a float module, either produced by torch.quantization
                          utilities or provided by the user
        """
        if hasattr(mod, 'weight_fake_quant'):
            # assert type(mod) == QATLinear, 'training mode nnq.Linear.from_float only works for nn.qat.Linear'
            weight_post_process = mod.weight_fake_quant
            activation_post_process = mod.activation_post_process
        else:
            assert type(mod) == cls._FLOAT_MODULE, ' nnq.' + cls.__name__ + '.from_float only works for ' + \
                cls._FLOAT_MODULE.__name__
            assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
            if type(mod) == nni.LinearReLU:
                activation_post_process = mod[1].activation_post_process
                mod = mod[0]
            else:
                activation_post_process = mod.activation_post_process
            weight_post_process = mod.qconfig.weight()
            weight_post_process(mod.weight)
        act_scale, act_zp = activation_post_process.calculate_qparams()
        assert weight_post_process.dtype == torch.qint8, 'Weight observer must have dtype torch.qint8'
        qweight = _quantize_weight(mod.weight.float(), weight_post_process)
        qlinear = cls(mod.in_features, mod.out_features)
        qlinear.set_weight_bias(qweight, mod.bias)
        qlinear.scale = float(act_scale)
        qlinear.zero_point = int(act_zp)
        return qlinear
