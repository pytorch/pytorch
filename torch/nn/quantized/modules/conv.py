# coding=utf-8
r"""Quantized convolution modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.intrinsic as nni
import torch.nn.intrinsic.qat as nniqat

from torch._ops import ops
from torch.nn.modules.utils import _pair, _triple
from torch.nn.quantized.modules.utils import _quantize_weight
from torch.nn.utils import fuse_conv_bn_weights


class ConvPackedParams(torch.nn.Module):
    def __init__(self, spatial_dim):
        super(ConvPackedParams, self).__init__()
        shape = [1] * (4 if spatial_dim == 2 else 5)
        wq = torch._empty_affine_quantized(shape, scale=1.0, zero_point=0, dtype=torch.qint8)
        self.stride = [1] * (2 if spatial_dim == 2 else 3)
        self.padding = [0] * (2 if spatial_dim == 2 else 3)
        self.dilation = [1] * (2 if spatial_dim == 2 else 3)
        self.groups = 1
        self.spatial_dim = spatial_dim
        self.set_weight_bias(wq, None)

    @torch.jit.export
    def set_conv_params(self, stride, padding, dilation, groups):
        # type: (List[int], List[int], List[int], int) -> None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    @torch.jit.export
    def set_weight_bias(self, weight, bias):
        # type: (torch.Tensor, Optional[torch.Tensor]) -> None
        if self.spatial_dim == 2:
            self._packed_params = torch.ops.quantized.conv2d_prepack(
                weight, bias, self.stride, self.padding, self.dilation, self.groups)
        elif self.spatial_dim == 3:
            self._packed_params = torch.ops.quantized.conv3d_prepack(
                weight, bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            raise RuntimeError('Unsupported spatial dimension!')

    @torch.jit.export
    def _weight_bias(self):
        if self.spatial_dim == 2:
            return torch.ops.quantized.conv2d_unpack(self._packed_params)
        elif self.spatial_dim == 3:
            return torch.ops.quantized.conv3d_unpack(self._packed_params)
        else:
            raise RuntimeError('Unsupported spatial dimension!')

    def forward(self, x):
        return x

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super(ConvPackedParams, self)._save_to_state_dict(destination, prefix, keep_vars)
        (w, b) = self._weight_bias()
        destination[prefix + 'weight'] = w
        destination[prefix + 'bias'] = b
        destination[prefix + 'spatial_dim'] = self.spatial_dim

    @torch.jit.export
    def __getstate__(self):
        if not torch.jit.is_scripting():
            raise RuntimeError(
                'torch.save() is not currently supported for quantized modules.'
                ' See https://github.com/pytorch/pytorch/issues/24045.'
                ' Please use state_dict or torch.jit serialization.')
        qweight, bias = self._weight_bias()
        return (qweight,
                bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
                self.training,
                self.spatial_dim)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self.spatial_dim = state_dict[prefix + 'spatial_dim']
        self.set_weight_bias(
            state_dict[prefix + 'weight'], state_dict[prefix + 'bias'])
        state_dict.pop(prefix + 'spatial_dim')
        state_dict.pop(prefix + 'weight')
        state_dict.pop(prefix + 'bias')
        super(ConvPackedParams, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, False, missing_keys,
            unexpected_keys, error_msgs)

    @torch.jit.export
    def __setstate__(self, state):
        self.stride = state[2]
        self.padding = state[3]
        self.dilation = state[4]
        self.groups = state[5]
        self.spatial_dim = state[7]  # NB needs to be before set_weight_bias
        self.set_weight_bias(state[0],
                             state[1])
        self.training = state[6]


class _ConvNd(nn.Module):
    _version = 2

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', spatial_dim=2):
        super(_ConvNd, self).__init__()
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
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = False
        self.output_padding = 0
        self.groups = groups
        self.padding_mode = padding_mode
        # Initialize as NCHW. set_weight will internally transpose to NHWC.
        qweight = torch._empty_affine_quantized(
            [out_channels, in_channels // self.groups] + list(kernel_size),
            scale=1, zero_point=0, dtype=torch.qint8)
        bias_float = (
            torch.zeros(out_channels, dtype=torch.float) if bias else None)

        self._packed_params = ConvPackedParams(spatial_dim=spatial_dim)
        self._packed_params.set_conv_params(list(stride), list(padding), list(dilation), groups)
        self._packed_params.set_weight_bias(qweight, bias_float)
        self.scale = 1.0
        self.zero_point = 0

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}, scale={scale}, zero_point={zero_point}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias() is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

    # ===== Serialization methods =====
    # The special consideration here is that we have to unpack the weights into
    # their regular QTensor form for serialization. Packed weights should not
    # live outside the process in which they were created, rather they should be
    # derived from the QTensor weight.
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super(_ConvNd, self)._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'scale'] = torch.tensor(self.scale)
        destination[prefix + 'zero_point'] = torch.tensor(self.zero_point)

    # ===== Deserialization methods =====
    # Counterpart to the serialization methods, we must pack the serialized
    # QTensor weight into its packed format for use by the FBGEMM ops.
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        if version is None or version == 1:
            weight = state_dict.pop(prefix + 'weight')
            bias = state_dict.pop(prefix + 'bias')
            if isinstance(self, Conv2d):
                spatial_dim = 2
            elif isinstance(self, Conv3d):
                spatial_dim = 3
            state_dict.update({
                prefix + '_packed_params.weight' : weight,
                prefix + '_packed_params.bias' : bias,
                prefix + '_packed_params.spatial_dim': spatial_dim
            })
        self.scale = float(state_dict[prefix + 'scale'])
        state_dict.pop(prefix + 'scale')
        self.zero_point = int(state_dict[prefix + 'zero_point'])
        state_dict.pop(prefix + 'zero_point')
        super(_ConvNd, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, False, missing_keys,
            unexpected_keys, error_msgs)

class Conv2d(_ConvNd):
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
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.qint32)
        >>> output = m(input)

    """

    _FLOAT_MODULE = nn.Conv2d

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode, spatial_dim=2)

    def _get_name(self):
        return 'QuantizedConv2d'

    def set_weight_bias(self, w, b):
        # type: (torch.Tensor, Optional[torch.Tensor]) -> None
        self._packed_params.set_weight_bias(w, b)

    def _weight_bias(self):
        return self._packed_params._weight_bias()

    def weight(self):
        return self._packed_params._weight_bias()[0]

    def bias(self):
        return self._packed_params._weight_bias()[1]

    def forward(self, input):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        return ops.quantized.conv2d(
            input, self._packed_params._packed_params, self.stride, self.padding,
            self.dilation, self.groups, self.scale, self.zero_point)

    @classmethod
    def from_float(cls, mod):
        r"""Creates a quantized module from a float module or qparams_dict.

        Args:
            mod (Module): a float module, either produced by torch.quantization
              utilities or provided by the user
        """
        if hasattr(mod, 'weight_fake_quant'):
            # assert type(mod) == cls.__QAT_MODULE, ' nnq.' + cls.__name__ + \
            # '.from_float only works for ' + cls.__QAT_MODULE.__name__
            if type(mod) == nniqat.ConvBn2d:
                mod.weight, mod.bias = fuse_conv_bn_weights(
                    mod.weight, mod.bias, mod.running_mean, mod.running_var,
                    mod.eps, mod.gamma, mod.beta)
            assert hasattr(mod, 'activation_post_process'), \
                'Input QAT module must have observer attached'
            weight_post_process = mod.weight_fake_quant
            activation_post_process = mod.activation_post_process
        else:
            assert type(mod) == cls._FLOAT_MODULE, \
                ' nnq.' + cls.__name__ + '.from_float only works for ' + \
                cls._FLOAT_MODULE.__name__
            assert hasattr(mod, 'qconfig'), \
                'Input float module must have qconfig defined.'
            # workaround for sequential, ConvReLU2d should probably
            # inherit from Conv2d instead
            if type(mod) == nni.ConvReLU2d:
                activation_post_process = mod[1].activation_post_process
                mod = mod[0]
            else:
                activation_post_process = mod.activation_post_process
            weight_post_process = mod.qconfig.weight()
            weight_post_process(mod.weight)
        act_scale, act_zp = activation_post_process.calculate_qparams()
        assert weight_post_process.dtype == torch.qint8, \
            'Weight observer must have a dtype of qint8'
        qweight = _quantize_weight(mod.weight.float(), weight_post_process)
        qconv = cls(mod.in_channels, mod.out_channels, mod.kernel_size,
                    mod.stride, mod.padding, mod.dilation, mod.groups,
                    mod.bias is not None, mod.padding_mode)
        qconv.set_weight_bias(qweight, mod.bias)
        qconv.scale = float(act_scale)
        qconv.zero_point = int(act_zp)

        return qconv


class Conv3d(_ConvNd):
    r"""Applies a 3D convolution over a quantized input signal composed of
    several quantized input planes.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.Conv3d`.

    .. note::
        Only `zeros` is supported for the :attr:`padding_mode` argument.

    .. note::
        Only `torch.quint8` is supported for the input data type.


    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point

    See :class:`~torch.nn.Conv3d` for other attributes.

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.quantized.Conv3d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.quantized.Conv3d(16, 33, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.quantized.Conv3d(16, 33, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), dilation=(1, 2, 2))
        >>> input = torch.randn(20, 16, 56, 56, 56)
        >>> # quantize input to qint8
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.qint32)
        >>> output = m(input)

    """

    _FLOAT_MODULE = nn.Conv3d

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        super(Conv3d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode, spatial_dim=3)

    def _get_name(self):
        return 'QuantizedConv3d'

    def set_weight_bias(self, w, b):
        # type: (torch.Tensor, Optional[torch.Tensor]) -> None
        self._packed_params.set_weight_bias(w, b)

    def _weight_bias(self):
        return self._packed_params._weight_bias()

    def weight(self):
        return self._packed_params._weight_bias()[0]

    def bias(self):
        return self._packed_params._weight_bias()[1]

    def forward(self, input):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 5:
            raise ValueError("Input shape must be `(N, C, D, H, W)`!")
        return ops.quantized.conv3d(
            input, self._packed_params._packed_params, self.stride, self.padding,
            self.dilation, self.groups, self.scale, self.zero_point)

    @classmethod
    def from_float(cls, mod):
        r"""Creates a quantized module from a float module or qparams_dict.

        Args:
            mod (Module): a float module, either produced by torch.quantization
              utilities or provided by the user
        """
        assert type(mod) == cls._FLOAT_MODULE, \
            ' nnq.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        assert hasattr(mod, 'qconfig'), \
            'Input float module must have qconfig defined.'
        # Workaround for sequential, ConvReLU3d should probably inherit from
        # Conv3d instead
        if type(mod) == nni.ConvReLU3d:
            activation_post_process = mod[1].activation_post_process
            mod = mod[0]
        else:
            activation_post_process = mod.activation_post_process
        weight_post_process = mod.qconfig.weight()
        weight_post_process(mod.weight)
        act_scale, act_zp = activation_post_process.calculate_qparams()
        assert weight_post_process.dtype == torch.qint8, \
            'Weight observer must have a dtype of qint8'
        qweight = _quantize_weight(mod.weight.float(), weight_post_process)
        qconv = cls(mod.in_channels, mod.out_channels, mod.kernel_size,
                    mod.stride, mod.padding, mod.dilation, mod.groups,
                    mod.bias is not None, mod.padding_mode)
        qconv.set_weight_bias(qweight, mod.bias)
        qconv.scale = float(act_scale)
        qconv.zero_point = int(act_zp)

        return qconv
