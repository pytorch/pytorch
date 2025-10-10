# mypy: allow-untyped-defs
r"""Quantized convolution modules."""

from typing import ClassVar, Literal, Optional

import torch
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.intrinsic.qat as nniqat
import torch.nn as nn
import torch.nn.functional as F
from torch._ops import ops
from torch.nn.common_types import _size_1_t
from torch.nn.modules.utils import _pair, _single, _triple
from torch.nn.utils import fuse_conv_bn_weights

from .utils import _quantize_weight, WeightedQuantizedModule


__all__ = [
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
]

_SUPPORTED_PADDING = {"zeros", "reflect"}


def _reverse_repeat_padding(padding: list[int]) -> list[int]:
    _reversed_padding_repeated_twice: list[int] = []
    N = len(padding)
    for idx in range(N):
        _reversed_padding_repeated_twice.extend(padding[N - idx - 1] for _ in range(2))
    return _reversed_padding_repeated_twice


class _ConvNd(WeightedQuantizedModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        # All subclasses have this signature - See PR #49702s
        raise NotImplementedError

    def _init(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        bias,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if padding_mode not in _SUPPORTED_PADDING:
            raise ValueError(
                f"'padding_mode' {padding_mode} is not supported by quantized convolution"
            )
        self.padding_mode = padding_mode
        # Initialize as NCHW. set_weight will internally transpose to NHWC.
        if self.transposed:
            weight_shape = [in_channels, out_channels // self.groups]
        else:
            weight_shape = [out_channels, in_channels // self.groups]
        qweight = torch._empty_affine_quantized(
            weight_shape + list(kernel_size),
            scale=1,
            zero_point=0,
            dtype=torch.qint8,
            **{k: v for k, v in factory_kwargs.items() if k != "dtype"},
        )
        bias_float = (
            torch.zeros(
                out_channels,
                dtype=torch.float,
                **{k: v for k, v in factory_kwargs.items() if k != "dtype"},
            )
            if bias
            else None
        )

        self.set_weight_bias(qweight, bias_float)
        self.scale = 1.0
        self.zero_point = 0

    def set_weight_bias(self, qweight, bias_float):
        raise NotImplementedError

    def bias(self):
        raise NotImplementedError

    def _weight_bias(self):
        raise NotImplementedError

    def extra_repr(self):
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}, scale={scale}, zero_point={zero_point}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias() is None:
            s += ", bias=False"
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
        super()._save_to_state_dict(destination, prefix, keep_vars)
        (w, b) = self._weight_bias()
        destination[prefix + "weight"] = w
        destination[prefix + "bias"] = b
        destination[prefix + "scale"] = torch.tensor(self.scale)
        destination[prefix + "zero_point"] = torch.tensor(self.zero_point)

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
            self.training,
        )

    # ===== Deserialization methods =====
    # Counterpart to the serialization methods, we must pack the serialized
    # QTensor weight into its packed format for use by the FBGEMM ops.
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        self.set_weight_bias(state_dict[prefix + "weight"], state_dict[prefix + "bias"])
        state_dict.pop(prefix + "weight")
        state_dict.pop(prefix + "bias")
        self.scale = float(state_dict[prefix + "scale"])
        state_dict.pop(prefix + "scale")
        self.zero_point = int(state_dict[prefix + "zero_point"])
        state_dict.pop(prefix + "zero_point")
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            False,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

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

    def __deepcopy__(self, memo):
        new_instance = type(self).__new__(type(self))
        torch.nn.Module.__init__(new_instance)
        state = self.__getstate__()
        new_instance.__setstate__(state)
        return new_instance

    def __copy__(self):
        return self.__deepcopy__({})

    @classmethod
    def get_qconv(cls, mod, activation_post_process, weight_post_process=None):
        r"""Creates a qconv object and returns it."""
        if weight_post_process is None:
            weight_post_process = mod.qconfig.weight()
        weight_post_process(mod.weight)
        assert weight_post_process.dtype == torch.qint8, (
            "Weight observer must have a dtype of qint8"
        )
        qweight = _quantize_weight(mod.weight.float(), weight_post_process)
        # the __init__ call used is the one from derived classes and not the one from _ConvNd
        qconv = cls(
            mod.in_channels,
            mod.out_channels,
            mod.kernel_size,
            mod.stride,
            mod.padding,
            mod.dilation,
            mod.groups,
            mod.bias is not None,
            mod.padding_mode,
        )
        qconv.set_weight_bias(qweight, mod.bias)
        if (
            activation_post_process is None
            or activation_post_process.dtype == torch.float
        ):
            return qconv  # dynamic quantization doesn't need scale/zero_point
        else:
            act_scale, act_zp = activation_post_process.calculate_qparams()
            qconv.scale = float(act_scale)
            qconv.zero_point = int(act_zp)
            return qconv

    @staticmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        if hasattr(mod, "weight_fake_quant"):
            # assert type(mod) == cls.__QAT_MODULE, " nnq." + cls.__name__ + \
            # ".from_float only works for " + cls.__QAT_MODULE.__name__
            if type(mod) == cls._NNIQAT_CONV_BN_MODULE:
                mod.weight, mod.bias = fuse_conv_bn_weights(
                    mod.weight,
                    mod.bias,
                    mod.bn.running_mean,
                    mod.bn.running_var,
                    mod.bn.eps,
                    mod.bn.weight,
                    mod.bn.bias,
                )
            assert hasattr(mod, "activation_post_process"), (
                "Input QAT module must have observer attached"
            )
            weight_post_process = mod.weight_fake_quant
            activation_post_process = mod.activation_post_process
        else:
            assert type(mod) == cls._FLOAT_MODULE, (
                " nnq."
                + cls.__name__
                + ".from_float only works for "
                + cls._FLOAT_MODULE.__name__
                + " but got:"
                + str(type(mod))
            )
            assert hasattr(mod, "qconfig"), (
                "Input float module must have qconfig defined."
            )
            activation_post_process = (
                None
                if not hasattr(mod, "activation_post_process")
                else mod.activation_post_process
            )
            if type(mod) in [
                cls._NNI_CONV_RELU_MODULE,
                cls._NNI_CONV_ADD_MODULE,
                cls._NNI_CONV_ADD_RELU_MODULE,
            ]:
                mod = mod[0]
            weight_post_process = mod.qconfig.weight()
        return cls.get_qconv(mod, activation_post_process, weight_post_process)

    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point):
        r"""Create a (fbgemm/qnnpack) quantized module from a reference quantized module
        Args:
            ref_qconv (Module): a reference quantized  module, either produced by torch.ao.quantization
                                utilities or provided by the user
            output_scale (float): scale for output Tensor
            output_zero_point (int): zero point for output Tensor
        """
        qconv = cls(
            ref_qconv.in_channels,
            ref_qconv.out_channels,
            ref_qconv.kernel_size,  # type: ignore[arg-type]
            ref_qconv.stride,  # type: ignore[arg-type]
            ref_qconv.padding,  # type: ignore[arg-type]
            ref_qconv.dilation,  # type: ignore[arg-type]
            ref_qconv.groups,
            ref_qconv.bias is not None,  # type: ignore[arg-type]
            ref_qconv.padding_mode,
            device=ref_qconv.weight.device,
            dtype=ref_qconv.weight.dtype,
        )
        qweight = ref_qconv.get_quantized_weight()
        qconv.set_weight_bias(qweight, ref_qconv.bias)
        qconv.scale = float(output_scale)
        qconv.zero_point = int(output_zero_point)
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

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> m = nn.quantized.Conv1d(16, 33, 3, stride=2)
        >>> input = torch.randn(20, 16, 100)
        >>> # quantize input to quint8
        >>> # xdoctest: +SKIP
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0,
        ...                                     dtype=torch.quint8)
        >>> output = m(q_input)

    """

    _FLOAT_MODULE: ClassVar[type[nn.Conv1d]] = nn.Conv1d
    _NNIQAT_CONV_BN_MODULE: ClassVar[Optional[type[nn.Module]]] = nniqat.ConvBn1d
    _NNI_CONV_RELU_MODULE: ClassVar[Optional[type[nn.Module]]] = nni.ConvReLU1d
    _NNI_CONV_ADD_MODULE: ClassVar[Optional[type[nn.Module]]] = None
    _NNI_CONV_ADD_RELU_MODULE: ClassVar[Optional[type[nn.Module]]] = None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        # pyrefly: ignore  # bad-assignment
        padding = padding if isinstance(padding, str) else _single(padding)
        dilation = _single(dilation)

        # Subclasses of _ConvNd needs to call _init rather than __init__. See
        # discussion on PR #49702
        super()._init(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _single(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _get_name(self):
        return "QuantizedConv1d"

    def set_weight_bias(self, w: torch.Tensor, b: Optional[torch.Tensor]) -> None:
        if self.padding_mode == "zeros":
            self._packed_params = torch.ops.quantized.conv1d_prepack(
                w, b, self.stride, self.padding, self.dilation, self.groups
            )
        else:
            self._packed_params = torch.ops.quantized.conv1d_prepack(
                w, b, self.stride, _pair(0), self.dilation, self.groups
            )

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
        if self.padding_mode != "zeros":
            # Padding in Conv1d is stored as (p, p), need to get (p,)
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding[:1])
            input = F.pad(
                input, _reversed_padding_repeated_twice, mode=self.padding_mode
            )
        return ops.quantized.conv1d(
            input, self._packed_params, self.scale, self.zero_point
        )

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        r"""Creates a quantized module from a float module or qparams_dict.

        Args:
            mod (Module): a float module, either produced by torch.ao.quantization
              utilities or provided by the user
        """
        return _ConvNd.from_float(
            cls, mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )


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

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> # With square kernels and equal stride
        >>> m = nn.quantized.Conv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.quantized.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.quantized.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> # quantize input to quint8
        >>> # xdoctest: +SKIP
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> output = m(q_input)

    """

    _FLOAT_MODULE: ClassVar[type[nn.Conv2d]] = nn.Conv2d
    _NNIQAT_CONV_BN_MODULE: ClassVar[Optional[type[nn.Module]]] = nniqat.ConvBn2d
    _NNI_CONV_RELU_MODULE: ClassVar[Optional[type[nn.Module]]] = nni.ConvReLU2d
    _NNI_CONV_ADD_MODULE: ClassVar[type[nni.ConvAdd2d]] = nni.ConvAdd2d
    _NNI_CONV_ADD_RELU_MODULE: ClassVar[type[nni.ConvAddReLU2d]] = nni.ConvAddReLU2d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        # Subclasses of _ConvNd need to call _init rather than __init__. See
        # discussion on PR #49702
        super()._init(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _get_name(self):
        return "QuantizedConv2d"

    def set_weight_bias(self, w: torch.Tensor, b: Optional[torch.Tensor]) -> None:
        if self.padding_mode == "zeros":
            self._packed_params = torch.ops.quantized.conv2d_prepack(
                w, b, self.stride, self.padding, self.dilation, self.groups
            )
        else:
            self._packed_params = torch.ops.quantized.conv2d_prepack(
                w, b, self.stride, _pair(0), self.dilation, self.groups
            )

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
        if self.padding_mode != "zeros":
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding)
            input = F.pad(
                input, _reversed_padding_repeated_twice, mode=self.padding_mode
            )
        return ops.quantized.conv2d(
            input, self._packed_params, self.scale, self.zero_point
        )

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        r"""Creates a quantized module from a float module or qparams_dict.

        Args:
            mod (Module): a float module, either produced by torch.ao.quantization
              utilities or provided by the user
        """
        return _ConvNd.from_float(
            cls, mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )


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

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> # With square kernels and equal stride
        >>> m = nn.quantized.Conv3d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.quantized.Conv3d(16, 33, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.quantized.Conv3d(16, 33, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), dilation=(1, 2, 2))
        >>> input = torch.randn(20, 16, 56, 56, 56)
        >>> # quantize input to quint8
        >>> # xdoctest: +SKIP
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> output = m(q_input)

    """

    _FLOAT_MODULE: ClassVar[type[nn.Conv3d]] = nn.Conv3d
    _NNIQAT_CONV_BN_MODULE: ClassVar[Optional[type[nn.Module]]] = nniqat.ConvBn3d
    _NNI_CONV_RELU_MODULE: ClassVar[Optional[type[nn.Module]]] = nni.ConvReLU3d
    _NNI_CONV_ADD_MODULE: ClassVar[Optional[type[nn.Module]]] = None
    _NNI_CONV_ADD_RELU_MODULE: ClassVar[Optional[type[nn.Module]]] = None

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        assert padding_mode != "reflect", "Conv3d does not support reflection padding"
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        # Subclasses of _ConvNd need to call _init rather than __init__. See
        # discussion on PR #49702
        super()._init(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _triple(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _get_name(self):
        return "QuantizedConv3d"

    def set_weight_bias(self, w: torch.Tensor, b: Optional[torch.Tensor]) -> None:
        if self.padding_mode == "zeros":
            self._packed_params = torch.ops.quantized.conv3d_prepack(
                w, b, self.stride, self.padding, self.dilation, self.groups
            )
        else:
            self._packed_params = torch.ops.quantized.conv3d_prepack(
                w, b, self.stride, _triple(0), self.dilation, self.groups
            )

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
        if self.padding_mode != "zeros":
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding)
            input = F.pad(
                input, _reversed_padding_repeated_twice, mode=self.padding_mode
            )
        return ops.quantized.conv3d(
            input, self._packed_params, self.scale, self.zero_point
        )

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        r"""Creates a quantized module from a float module or qparams_dict.

        Args:
            mod (Module): a float module, either produced by torch.ao.quantization
              utilities or provided by the user
        """
        return _ConvNd.from_float(
            cls, mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )


# === Transposed Convolutions ===


class _ConvTransposeNd(_ConvNd):
    _FLOAT_MODULE: ClassVar[type[nn.modules.conv._ConvNd]]

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        bias,
        padding_mode,
        device=None,
        dtype=None,
    ):
        if padding_mode != "zeros":
            raise ValueError(
                f'Only "zeros" padding mode is supported for {self.__class__.__name__}'
            )
        factory_kwargs = {"device": device, "dtype": dtype}
        # Subclasses of _ConvNd need to call _init rather than __init__. See
        # discussion on PR #49702
        super()._init(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _input_padding(
        self, kernel_size: list[int], dilation: list[int], padding: list[int]
    ) -> list[int]:
        res = torch.jit.annotate(list[int], [])
        for kdx in range(len(kernel_size)):
            pad = dilation[kdx] * (kernel_size[kdx] - 1) - padding[kdx]
            res.append(pad)
        return res

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        r"""Creates a quantized module from a float module or qparams_dict.
        Args:
            mod (Module): a float module, either produced by torch.ao.quantization
              utilities or provided by the user
        """
        # derived classes override cls._FLOAT_MODULE attribute
        msg = (
            " nnq."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__  # type: ignore[attr-defined]
        )
        assert type(mod) == cls._FLOAT_MODULE, msg
        assert hasattr(mod, "qconfig"), "Input float module must have qconfig defined."
        weight_post_process = mod.qconfig.weight()  # type: ignore[operator, union-attr]
        weight_post_process(mod.weight)
        assert weight_post_process.dtype == torch.qint8, (
            "Weight observer must have a dtype of qint8"
        )
        qweight = _quantize_weight(mod.weight.float(), weight_post_process)
        # the __init__ call used is the one from derived classes and not the one from _ConvTransposeNd
        qconv = cls(
            mod.in_channels,
            mod.out_channels,
            mod.kernel_size,  # type: ignore[call-arg]
            mod.stride,
            mod.padding,
            mod.output_padding,
            mod.groups,
            mod.bias is not None,
            mod.dilation,
            mod.padding_mode,
        )
        qconv.set_weight_bias(qweight, mod.bias)
        if (
            not hasattr(mod, "activation_post_process")
            or mod.activation_post_process.dtype == torch.float
        ):
            return qconv  # dynamic quantization doesn't need scale/zero_point
        else:
            act_scale, act_zp = mod.activation_post_process.calculate_qparams()  # type: ignore[operator, union-attr]
            qconv.scale = float(act_scale)
            qconv.zero_point = int(act_zp)
            return qconv

    @staticmethod
    def from_reference(cls, ref_qconvt, output_scale, output_zero_point):  # type: ignore[override]
        r"""Create a (fbgemm/qnnpack) quantized module from a reference quantized module
        Args:
            ref_qconvt (Module): a reference quantized  module, either produced by torch.ao.quantization
                                 utilities or provided by the user
            output_scale (float): scale for output Tensor
            output_zero_point (int): zero point for output Tensor
        """
        qconv = cls(
            ref_qconvt.in_channels,
            ref_qconvt.out_channels,
            ref_qconvt.kernel_size,  # type: ignore[arg-type]
            ref_qconvt.stride,  # type: ignore[arg-type]
            ref_qconvt.padding,  # type: ignore[arg-type]
            ref_qconvt.output_padding,  # type: ignore[arg-type]
            ref_qconvt.groups,
            ref_qconvt.bias is not None,  # type: ignore[arg-type]
            ref_qconvt.dilation,  # type: ignore[arg-type]
            ref_qconvt.padding_mode,
            device=ref_qconvt.weight.device,
            dtype=ref_qconvt.weight.dtype,
        )
        qweight = ref_qconvt.get_quantized_weight()
        qconv.set_weight_bias(qweight, ref_qconvt.bias)
        qconv.scale = float(output_scale)
        qconv.zero_point = int(output_zero_point)
        return qconv


class ConvTranspose1d(_ConvTransposeNd):
    r"""Applies a 1D transposed convolution operator over an input image
    composed of several input planes.
    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.ConvTranspose1d`.

    .. note:: Currently only the QNNPACK engine is implemented.
        Please, set the `torch.backends.quantized.engine = 'qnnpack'`

    For special notes, please, see :class:`~torch.ao.nn.quantized.Conv1d`

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point
    See :class:`~torch.nn.ConvTranspose2d` for other attributes.

    Examples::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> torch.backends.quantized.engine = 'qnnpack'
        >>> from torch.ao.nn import quantized as nnq
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
        >>> # xdoctest: +SKIP("FIXME: output_size is not a parameter)
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12])
    """

    _FLOAT_MODULE: ClassVar[type[nn.ConvTranspose1d]] = nn.ConvTranspose1d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        output_padding = _single(output_padding)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            True,
            output_padding,
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _get_name(self):
        return "QuantizedConvTranspose1d"

    def set_weight_bias(self, w: torch.Tensor, b: Optional[torch.Tensor]) -> None:
        self._packed_params = torch.ops.quantized.conv_transpose1d_prepack(
            w,
            b,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
            self.groups,
        )

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
            input, self._packed_params, self.scale, self.zero_point
        )

    @classmethod
    def from_reference(cls, ref_qconvt, output_scale, output_zero_point):  # type: ignore[override]
        return _ConvTransposeNd.from_reference(
            cls, ref_qconvt, output_scale, output_zero_point
        )


class ConvTranspose2d(_ConvTransposeNd):
    r"""Applies a 2D transposed convolution operator over an input image
    composed of several input planes.
    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.ConvTranspose2d`.

    For special notes, please, see :class:`~torch.ao.nn.quantized.Conv2d`

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point
    See :class:`~torch.nn.ConvTranspose2d` for other attributes.

    Examples::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> # QNNPACK or FBGEMM as backend
        >>> torch.backends.quantized.engine = 'qnnpack'
        >>> # With square kernels and equal stride
        >>> import torch.ao.nn.quantized as nnq
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
        >>> # xdoctest: +SKIP("FIXME: output_size is not a parameter)
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12, 12])
    """

    _FLOAT_MODULE: ClassVar[type[nn.ConvTranspose2d]] = nn.ConvTranspose2d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            True,
            output_padding,
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _get_name(self):
        return "QuantizedConvTranspose2d"

    def set_weight_bias(self, w: torch.Tensor, b: Optional[torch.Tensor]) -> None:
        self._packed_params = torch.ops.quantized.conv_transpose2d_prepack(
            w,
            b,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
            self.groups,
        )

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
            input, self._packed_params, self.scale, self.zero_point
        )

    @classmethod
    def from_reference(cls, ref_qconvt, output_scale, output_zero_point):  # type: ignore[override]
        return _ConvTransposeNd.from_reference(
            cls, ref_qconvt, output_scale, output_zero_point
        )


class ConvTranspose3d(_ConvTransposeNd):
    r"""Applies a 3D transposed convolution operator over an input image
    composed of several input planes.
    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.ConvTranspose3d`.

    .. note:: Currently only the FBGEMM engine is implemented.
        Please, set the `torch.backends.quantized.engine = 'fbgemm'`

    For special notes, please, see :class:`~torch.ao.nn.quantized.Conv3d`

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point
    See :class:`~torch.nn.ConvTranspose3d` for other attributes.

    Examples::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> torch.backends.quantized.engine = 'fbgemm'
        >>> from torch.ao.nn import quantized as nnq
        >>> # With cubic kernels and equal stride
        >>> m = nnq.ConvTranspose3d(16, 33, 3, stride=2)
        >>> # non-cubic kernels and unequal stride and with padding
        >>> m = nnq.ConvTranspose3d(16, 33, (3, 3, 5), stride=(2, 1, 1), padding=(4, 2, 2))
        >>> input = torch.randn(20, 16, 50, 100, 100)
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> output = m(q_input)
        >>> # exact output size can be also specified as an argument
        >>> input = torch.randn(1, 16, 12, 12, 12)
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> downsample = nnq.Conv3d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nnq.ConvTranspose3d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(q_input)
        >>> h.size()
        torch.Size([1, 16, 6, 6, 6])
        >>> # xdoctest: +SKIP("FIXME: output_size is not a parameter)
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12, 12, 12])
    """

    _FLOAT_MODULE: ClassVar[type[nn.ConvTranspose3d]] = nn.ConvTranspose3d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        output_padding = _triple(output_padding)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            True,
            output_padding,
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _get_name(self):
        return "QuantizedConvTranspose3d"

    def set_weight_bias(self, w: torch.Tensor, b: Optional[torch.Tensor]) -> None:
        self._packed_params = torch.ops.quantized.conv_transpose3d_prepack(
            w,
            b,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
            self.groups,
        )

    def _weight_bias(self):
        w, b = torch.ops.quantized.conv3d_unpack(self._packed_params)
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
        if len(input.shape) != 5:
            raise ValueError("Input shape must be `(N, C, T, H, W)`!")
        return ops.quantized.conv_transpose3d(
            input, self._packed_params, self.scale, self.zero_point
        )

    @classmethod
    def from_reference(cls, ref_qconvt, output_scale, output_zero_point):  # type: ignore[override]
        return _ConvTransposeNd.from_reference(
            cls, ref_qconvt, output_scale, output_zero_point
        )
