# coding=utf-8
r"""Quantized convolution modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch._ops import ops
from torch.nn import Conv2d as NNConv2d
from torch.nn.modules.utils import _pair

from torch._jit_internal import Optional

class Conv2d(torch.nn.Module):
    r"""Applies a 2D convolution over a quantized input signal composed of
    several quantized input planes.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.Conv2d`.

    .. note::
        Only `zeros` is supported for the :attr:`padding_mode` argument.

    .. note::
        Only `torch.quint8` is supported for the input data type.


    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point

    See :class:`~torch.nn.Conv2d` for other attributes.

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.quantized.Conv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.quantized.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.quantized.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> # quantize input to qint8
        >>> q_input = torch.quantize_linear(input, scale=1.0, zero_point=0, dtype=torch.qint32)
        >>> output = m(input)

    """

    __FLOAT_MODULE = NNConv2d
    __annotations__ = {'bias' : Optional[torch.Tensor]}

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(Conv2d, self).__init__()
        if padding_mode != 'zeros':
            raise NotImplementedError(
                "Currently only zero-padding is supported by quantized conv")
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.transposed = False
        self.output_padding = 0
        self.groups = groups
        self.padding_mode = padding_mode
        # Initialize as NCHW. set_weight will internally transpose to
        # NHWC
        qweight = torch._empty_affine_quantized(
            [out_channels, in_channels // self.groups, kernel_size[0],
             kernel_size[1]],
            scale=1, zero_point=0, dtype=torch.qint8)
        self.set_weight(qweight)
        self.bias = torch._empty_affine_quantized([out_channels],
                                                  scale=1, zero_point=0,
                                                  dtype=torch.qint32)
        self.scale = 1.0
        self.zero_point = 0

    def set_weight(self, w):
        self._packed_weight = torch.ops.quantized.fbgemm_conv_prepack(
            w.permute([0, 2, 3, 1]), self.stride, self.padding, self.dilation, self.groups)

    def weight(self):
        return torch.ops.quantized.fbgemm_conv_unpack(
            self._packed_weight).permute([0, 3, 1, 2])

    def forward(self, input):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        output = ops.quantized.fbgemm_conv2d(input.permute([0, 2, 3, 1]),
                                             self._packed_weight, self.bias,
                                             self.stride, self.padding,
                                             self.dilation, self.groups,
                                             self.scale, self.zero_point)
        return output.permute([0, 3, 1, 2])

    # ===== Serialization methods =====
    # The special consideration here is that we have to unpack the weights into their
    # regular QTensor form for serialization. Packed weights should not live
    # outside the process in which they were created, rather they should be derived
    # from the QTensor weight.
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super(Conv2d, self)._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'weight'] = self.weight()
        destination[prefix + 'scale'] = torch.tensor(self.scale)
        destination[prefix + 'zero_point'] = torch.tensor(self.zero_point)
        destination[prefix + 'bias'] = self.bias

    @torch.jit.export
    def __getstate__(self):
        return (
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.transposed,
            self.output_padding,
            self.groups,
            self.padding_mode,
            self.weight(),
            self.bias,
            self.scale,
            self.zero_point,
        )

    # ===== Deserialization methods =====
    # Counterpart to the serialization methods, we must pack the serialized QTensor
    # weight into its packed format for use by the FBGEMM ops.
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self.set_weight(state_dict[prefix + 'weight'])
        state_dict.pop(prefix + 'weight')

        self.bias = state_dict[prefix + 'bias']
        state_dict.pop(prefix + 'bias')


        self.scale = float(state_dict[prefix + 'scale'])
        state_dict.pop(prefix + 'scale')

        self.zero_point = int(state_dict[prefix + 'zero_point'])
        state_dict.pop(prefix + 'zero_point')

        super(Conv2d, self)._load_from_state_dict(state_dict, prefix, local_metadata, False,
                                                  missing_keys, unexpected_keys, error_msgs)

    @torch.jit.export
    def __setstate__(self, state):
        # type: (Tuple[int, int, Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int], bool, int, int, str, Tensor, Optional[Tensor], float, int])  # noqa
        self.in_channels = state[0]
        self.out_channels = state[1]
        self.kernel_size = state[2]
        self.stride = state[3]
        self.padding = state[4]
        self.dilation = state[5]
        self.transposed = state[6]
        self.output_padding = state[7]
        self.groups = state[8]
        self.padding_mode = state[9]
        self.set_weight(state[10])
        self.bias = state[11]
        self.scale = state[12]
        self.zero_point = state[13]

    @classmethod
    def from_float(cls, mod):
        r"""Creates a quantized module from a float module or qparams_dict.

        Args:
            mod (Module): a float module, either produced by torch.quantization
                          utilities or provided by the user
        """
        if hasattr(mod, 'weight_fake_quant'):
            # assert type(mod) == cls.__QAT_MODULE, ' nnq.' + cls.__name__ + '.from_float only works for ' + \
            #     cls.__QAT_MODULE.__name__
            assert hasattr(mod, 'observer'), 'Input float module must have observer attached'
            weight_observer = mod.weight_fake_quant
        else:
            assert type(mod) == cls.__FLOAT_MODULE, ' nnq.' + cls.__name__ + '.from_float only works for ' + \
                cls.__FLOAT_MODULE.__name__
            assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
            assert hasattr(mod, 'observer'), 'Input float module must have observer attached'
            weight_observer = mod.qconfig.weight()
            weight_observer(mod.weight)
        activation_observer = mod.observer
        act_scale, act_zp = activation_observer.calculate_qparams()
        wt_scale, wt_zp = weight_observer.calculate_qparams()
        bias_scale = (wt_scale * act_scale).float()
        qweight = torch.quantize_linear(
            mod.weight.float().contiguous(),
            wt_scale, wt_zp.long().item(), torch.qint8)
        qconv = Conv2d(mod.in_channels, mod.out_channels, mod.kernel_size,
                       mod.stride, mod.padding, mod.dilation, mod.groups,
                       mod.bias is not None, mod.padding_mode)
        qconv.set_weight(qweight)
        if mod.bias is not None:
            qconv.bias = torch.quantize_linear(mod.bias, bias_scale, 0, torch.qint32)
        else:
            qconv.bias = None
        qconv.scale = float(act_scale)
        qconv.zero_point = int(act_zp)
        return qconv
