# coding=utf-8
r"""Quantized convolution modules."""

from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.intrinsic as nni
import torch.nn.intrinsic.qat as nniqat

from torch._ops import ops
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.quantized.modules.utils import _pair_from_first
from torch.nn.quantized.modules.utils import _quantize_weight
from torch.nn.utils import fuse_conv_bn_weights

class _ConvNd(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation,
                 transposed, output_padding,
                 groups, bias,
                 padding_mode='zeros'):
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
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        # Initialize as NCHW. set_weight will internally transpose to NHWC.
        if self.transposed:
            weight_shape = [in_channels, out_channels // self.groups]
        else:
            weight_shape = [out_channels, in_channels // self.groups]
        qweight = torch._empty_affine_quantized(
            weight_shape + list(kernel_size),
            scale=1, zero_point=0, dtype=torch.qint8)
        bias_float = (
            torch.zeros(out_channels, dtype=torch.float) if bias else None)

        self.set_weight_bias(qweight, bias_float)
        self.scale = 1.0
        self.zero_point = 0

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}, scale={scale}, zero_point={zero_point}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
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
    #   self
    #   |--- weight : Tensor
    #   |--- bias : Tensor
    #
    # TODO: maybe change to this when https://github.com/pytorch/pytorch/pull/32958 is landed
    #   self
    #   |--- _packed_params : Conv2dPackedParamsBase or Conv3dPackedParamsBase
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super(_ConvNd, self)._save_to_state_dict(destination, prefix, keep_vars)
        (w, b) = self._weight_bias()
        destination[prefix + 'weight'] = w
        destination[prefix + 'bias'] = b
        destination[prefix + 'scale'] = torch.tensor(self.scale)
        destination[prefix + 'zero_point'] = torch.tensor(self.zero_point)

    @torch.jit.export
    def __getstate__(self):
        (w, b) = self._weight_bias()
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
            w,
            b,
            self.scale,
            self.zero_point,
            self.training
        )

    # ===== Deserialization methods =====
    # Counterpart to the serialization methods, we must pack the serialized
    # QTensor weight into its packed format for use by the FBGEMM ops.
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self.set_weight_bias(
            state_dict[prefix + 'weight'], state_dict[prefix + 'bias'])
        state_dict.pop(prefix + 'weight')
        state_dict.pop(prefix + 'bias')
        self.scale = float(state_dict[prefix + 'scale'])
        state_dict.pop(prefix + 'scale')
        self.zero_point = int(state_dict[prefix + 'zero_point'])
        state_dict.pop(prefix + 'zero_point')
        super(_ConvNd, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, False, missing_keys,
            unexpected_keys, error_msgs)

    @torch.jit.export
    def __setstate__(self, state):
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
        self.set_weight_bias(state[10], state[11])
        self.scale = state[12]
        self.zero_point = state[13]
        self.training = state[14]

    @classmethod
    def get_qconv(cls, mod, activation_post_process, weight_post_process=None):
        r"""Creates a qconv object and returns it.
        """
        if weight_post_process is None:
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


class Conv1d(_ConvNd):
    r"""Applies a 1D convolution over a quantized input signal composed of
    several quantized input planes.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.Conv1d`.

    .. note::
        Only `zeros` is supported for the :attr:`padding_mode` argument.

    .. note::
        Only `torch.quint8` is supported for the input data type.


    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point

    See :class:`~torch.nn.Conv1d` for other attributes.

    Examples::

        >>> m = nn.quantized.Conv1d(16, 33, 3, stride=2)
        >>> input = torch.randn(20, 16, 100)
        >>> # quantize input to quint8
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0,
                                                dtype=torch.quint8)
        >>> output = m(q_input)

    """

    _FLOAT_MODULE = nn.Conv1d

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        kernel_size = _pair_from_first(kernel_size)
        stride = _pair_from_first(stride)
        padding = _pair_from_first(padding)
        dilation = _pair_from_first(dilation)

        super(Conv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias, padding_mode)

    def _get_name(self):
        return 'QuantizedConv1d'

    def set_weight_bias(self, w, b):
        # type: (torch.Tensor, Optional[torch.Tensor]) -> None
        self._packed_params = torch.ops.quantized.conv1d_prepack(
            w, b, self.stride, self.padding, self.dilation, self.groups)

    def _weight_bias(self):
        w, b = torch.ops.quantized.conv1d_unpack(self._packed_params)
        return w, b

    def weight(self):
        return self._weight_bias()[0]

    def bias(self):
        return self._weight_bias()[1]

    def forward(self, input):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 3:
            raise ValueError("Input shape must be `(N, C, L)`!")
        return ops.quantized.conv1d(input, self._packed_params, self.scale, self.zero_point)

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
        if type(mod) == nni.ConvReLU1d:
            activation_post_process = mod[1].activation_post_process
            mod = mod[0]
        else:
            activation_post_process = mod.activation_post_process
        return cls.get_qconv(mod, activation_post_process)


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
        >>> # quantize input to quint8
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> output = m(q_input)

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
            False, _pair(0), groups, bias, padding_mode)

    def _get_name(self):
        return 'QuantizedConv2d'

    def set_weight_bias(self, w, b):
        # type: (torch.Tensor, Optional[torch.Tensor]) -> None
        self._packed_params = torch.ops.quantized.conv2d_prepack(
            w, b, self.stride, self.padding, self.dilation, self.groups)

    def _weight_bias(self):
        return self._packed_params.unpack()

    def weight(self):
        return self._weight_bias()[0]

    def bias(self):
        return self._weight_bias()[1]

    def forward(self, input):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        return ops.quantized.conv2d(
            input, self._packed_params, self.scale, self.zero_point)

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
                    mod.weight, mod.bias, mod.bn.running_mean, mod.bn.running_var,
                    mod.bn.eps, mod.bn.weight, mod.bn.bias)
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

        return cls.get_qconv(mod, activation_post_process, weight_post_process)


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
        >>> # quantize input to quint8
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> output = m(q_input)

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
            False, _triple(0), groups, bias, padding_mode)

    def _get_name(self):
        return 'QuantizedConv3d'

    def set_weight_bias(self, w, b):
        # type: (torch.Tensor, Optional[torch.Tensor]) -> None
        self._packed_params = torch.ops.quantized.conv3d_prepack(
            w, b, self.stride, self.padding, self.dilation, self.groups)

    def _weight_bias(self):
        return self._packed_params.unpack()

    def weight(self):
        return self._weight_bias()[0]

    def bias(self):
        return self._weight_bias()[1]

    def forward(self, input):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 5:
            raise ValueError("Input shape must be `(N, C, D, H, W)`!")
        return ops.quantized.conv3d(
            input, self._packed_params, self.scale, self.zero_point)

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
        return cls.get_qconv(mod, activation_post_process)

# === Transposed Convolutions ===

class _ConvTransposeNd(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode):
        if padding_mode != 'zeros':
            raise ValueError('Only "zeros" padding mode is supported for {}'.format(self.__class__.__name__))

        super(_ConvTransposeNd, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, transposed, output_padding,
            groups, bias, padding_mode)

    def _input_padding(self, kernel_size, dilation, padding):
        # type: (List[int], List[int], List[int]) -> List[int]
        res = torch.jit.annotate(List[int], [])
        for kdx in range(len(kernel_size)):
            pad = (dilation[kdx] * (kernel_size[kdx] - 1) - padding[kdx])
            res.append(pad)
        return res

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
        weight_post_process = mod.qconfig.weight()
        weight_post_process(mod.weight)
        act_scale, act_zp = mod.activation_post_process.calculate_qparams()
        assert weight_post_process.dtype == torch.qint8, \
            'Weight observer must have a dtype of qint8'
        qweight = _quantize_weight(mod.weight.float(), weight_post_process)
        qconv = cls(mod.in_channels, mod.out_channels, mod.kernel_size,
                    mod.stride, mod.padding, mod.output_padding, mod.groups,
                    mod.bias is not None, mod.dilation, mod.padding_mode)
        qconv.set_weight_bias(qweight, mod.bias)
        qconv.scale = float(act_scale)
        qconv.zero_point = int(act_zp)

        return qconv


class ConvTranspose1d(_ConvTransposeNd):
    r"""Applies a 1D transposed convolution operator over an input image
    composed of several input planes.
    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.ConvTranspose1d`.

    .. note:: Currently only the QNNPACK engine is implemented.
        Please, set the `torch.backends.quantized.engine = 'qnnpack'`

    For special notes, please, see :class:`~torch.nn.quantized.Conv1d`

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point
    See :class:`~torch.nn.ConvTranspose2d` for other attributes.

    Examples::

        >>> torch.backends.quantized.engine = 'qnnpack'
        >>> # With square kernels and equal stride
        >>> m = nnq.ConvTranspose1d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nnq.ConvTranspose1d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> input = torch.randn(20, 16, 50)
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> output = m(q_input)
        >>> # exact output size can be also specified as an argument
        >>> input = torch.randn(1, 16, 12)
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> downsample = nnq.Conv1d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nnq.ConvTranspose1d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(q_input)
        >>> h.size()
        torch.Size([1, 16, 6])
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12])
    """

    _FLOAT_MODULE = nn.ConvTranspose1d

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)

        super(ConvTranspose1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias, padding_mode)

    def _get_name(self):
        return 'QuantizedConvTranpose1d'

    def set_weight_bias(self, w, b):
        # type: (torch.Tensor, Optional[torch.Tensor]) -> None
        self._packed_params = torch.ops.quantized.conv_transpose1d_prepack(
            w, b, self.stride, self.padding, self.output_padding, self.dilation,
            self.groups)

    def _weight_bias(self):
        w, b = torch.ops.quantized.conv_transpose1d_unpack(self._packed_params)
        return w, b

    def weight(self):
        (w, _) = self._weight_bias()
        return w

    def bias(self):
        (_, b) = self._weight_bias()
        return b

    def forward(self, input):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 3:
            raise ValueError("Input shape must be `(N, C, L)`!")
        return torch.ops.quantized.conv_transpose1d(
            input, self._packed_params, self.scale, self.zero_point)


class ConvTranspose2d(_ConvTransposeNd):
    r"""Applies a 2D transposed convolution operator over an input image
    composed of several input planes.
    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.ConvTranspose2d`.

    .. note:: Currently only the QNNPACK engine is implemented.
        Please, set the `torch.backends.quantized.engine = 'qnnpack'`

    For special notes, please, see :class:`~torch.nn.quantized.Conv2d`

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point
    See :class:`~torch.nn.ConvTranspose2d` for other attributes.

    Examples::

        >>> torch.backends.quantized.engine = 'qnnpack'
        >>> # With square kernels and equal stride
        >>> m = nnq.ConvTranspose2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nnq.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> output = m(q_input)
        >>> # exact output size can be also specified as an argument
        >>> input = torch.randn(1, 16, 12, 12)
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> downsample = nnq.Conv2d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nnq.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(q_input)
        >>> h.size()
        torch.Size([1, 16, 6, 6])
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12, 12])
    """

    _FLOAT_MODULE = nn.ConvTranspose2d

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)

        super(ConvTranspose2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias, padding_mode)

    def _get_name(self):
        return 'QuantizedConvTranpose2d'

    def set_weight_bias(self, w, b):
        # type: (torch.Tensor, Optional[torch.Tensor]) -> None
        self._packed_params = torch.ops.quantized.conv_transpose2d_prepack(
            w, b, self.stride, self.padding, self.output_padding, self.dilation,
            self.groups)

    def _weight_bias(self):
        w, b = torch.ops.quantized.conv2d_unpack(self._packed_params)
        return w, b

    def weight(self):
        (w, _) = self._weight_bias()
        return w

    def bias(self):
        (_, b) = self._weight_bias()
        return b

    def forward(self, input):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        return ops.quantized.conv_transpose2d(
            input, self._packed_params, self.scale, self.zero_point)
