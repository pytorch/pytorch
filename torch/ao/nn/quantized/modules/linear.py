# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
from collections.abc import Iterable
from typing import Optional

import torch
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.intrinsic.qat as nniqat
import torch.nn as nn
from torch.nn.utils.fusion import fuse_linear_bn_weights
from torch.nn.utils.parametrize import type_before_parametrizations

from .utils import _hide_packed_params_repr, _quantize_weight, WeightedQuantizedModule


__all__ = ["LinearPackedParams", "Linear"]


class LinearPackedParams(torch.nn.Module):
    _version = 3

    def __init__(self, dtype=torch.qint8):
        super().__init__()
        self.dtype = dtype
        if self.dtype == torch.qint8:
            wq = torch._empty_affine_quantized(
                [1, 1], scale=1.0, zero_point=0, dtype=torch.qint8
            )
        elif self.dtype == torch.float16:
            wq = torch.zeros([1, 1], dtype=torch.float)
        self.set_weight_bias(wq, None)  # type: ignore[possibly-undefined]

    @torch.jit.export
    def set_weight_bias(
        self, weight: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> None:
        if self.dtype == torch.qint8:
            self._packed_params = torch.ops.quantized.linear_prepack(weight, bias)
        elif self.dtype == torch.float16:
            self._packed_params = torch.ops.quantized.linear_prepack_fp16(weight, bias)
        else:
            raise RuntimeError("Unsupported dtype on dynamic quantized linear!")

    @torch.jit.export
    def _weight_bias(self):
        if self.dtype == torch.qint8:
            return torch.ops.quantized.linear_unpack(self._packed_params)
        elif self.dtype == torch.float16:
            return torch.ops.quantized.linear_unpack_fp16(self._packed_params)
        else:
            raise RuntimeError("Unsupported dtype on dynamic quantized linear!")

    def forward(self, x):
        return x

    # Version 1
    #   self
    #   |--- weight : Tensor
    #   |--- bias : Tensor
    #
    # Version 2
    #   self
    #   |--- weight : Tensor
    #   |--- bias : Tensor
    #   |--- dtype : torch.dtype
    #
    # Version 3
    #   self
    #   |--- _packed_params : (Tensor, Tensor) representing (weight, bias)
    #                         of LinearPackedParams
    #   |--- dtype : torch.dtype
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "dtype"] = self.dtype
        destination[prefix + "_packed_params"] = self._weight_bias()

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
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            self.dtype = torch.qint8
        else:
            self.dtype = state_dict[prefix + "dtype"]
            state_dict.pop(prefix + "dtype")

        if version is None or version < 3:
            self.set_weight_bias(
                state_dict[prefix + "weight"], state_dict[prefix + "bias"]
            )
            state_dict.pop(prefix + "weight")
            state_dict.pop(prefix + "bias")

        if version == 3:
            weight, bias = state_dict[prefix + "_packed_params"]
            state_dict.pop(prefix + "_packed_params")
            self.set_weight_bias(weight, bias)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            False,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def __repr__(self):
        return self._weight_bias().__repr__()


class Linear(WeightedQuantizedModule):
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

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> m = nn.quantized.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> # xdoctest: +SKIP
        >>> input = torch.quantize_per_tensor(input, 1.0, 0, torch.quint8)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    _version = 3
    _FLOAT_MODULE = (nn.Linear, nn.modules.linear.NonDynamicallyQuantizableLinear)

    def __init__(self, in_features, out_features, bias_=True, dtype=torch.qint8):
        super().__init__()
        # We don't muck around with buffers or attributes or anything here
        # to keep the module simple. *everything* is simply a Python attribute.
        # Serialization logic is explicitly handled in the below serialization and
        # deserialization modules
        self.in_features = in_features
        self.out_features = out_features
        bias = None
        if bias_:
            bias = torch.zeros(out_features, dtype=torch.float)

        if dtype == torch.qint8:
            qweight = torch._empty_affine_quantized(
                [out_features, in_features], scale=1, zero_point=0, dtype=torch.qint8
            )
        elif dtype == torch.float16:
            qweight = torch.zeros([out_features, in_features], dtype=torch.float)
        else:
            raise RuntimeError("Unsupported dtype specified for quantized Linear!")

        self._packed_params = LinearPackedParams(dtype)
        self._packed_params.set_weight_bias(qweight, bias)
        self.scale = 1.0
        self.zero_point = 0

    def _get_name(self):
        return "QuantizedLinear"

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, scale={self.scale}, "
            f"zero_point={self.zero_point}, qscheme={self.weight().qscheme()}"
        )

    def __repr__(self):
        return _hide_packed_params_repr(self, LinearPackedParams)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.quantized.linear(
            x, self._packed_params._packed_params, self.scale, self.zero_point
        )

    # ===== Serialization methods =====
    # The special consideration here is that we have to unpack the weights into their
    # regular QTensor form for serialization. Packed weights should not live
    # outside the process in which they were created, rather they should be derived
    # from the QTensor weight.
    #
    # Version 1
    #   self
    #   |--- scale : float
    #   |--- zero_point : int
    #   |--- weight : Tensor
    #   |--- bias : Tensor
    #
    # Version 2
    #   self
    #   |--- scale : float
    #   |--- zero_point : int
    #   |--- _packed_params : Module
    #        |--- weight : Tensor
    #        |--- bias : Tensor
    #
    # Version 3
    #   self
    #   |--- scale : float
    #   |--- zero_point : int
    #   |--- _packed_params : Module
    #        |--- _packed_params : (Tensor, Tensor) representing weight, bias
    #                              of LinearPackedParams C++ struct
    #
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "scale"] = torch.tensor(self.scale)
        destination[prefix + "zero_point"] = torch.tensor(self.zero_point)

    # ===== Deserialization methods =====
    # Counterpart to the serialization methods, we must pack the serialized QTensor
    # weight into its packed format for use by the FBGEMM ops.
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
        self.scale = float(state_dict[prefix + "scale"])
        state_dict.pop(prefix + "scale")

        self.zero_point = int(state_dict[prefix + "zero_point"])
        state_dict.pop(prefix + "zero_point")

        version = local_metadata.get("version", None)

        if version is None or version == 1:
            # We moved the parameters into a LinearPackedParameters submodule
            weight = state_dict.pop(prefix + "weight")
            bias = state_dict.pop(prefix + "bias")
            state_dict.update(
                {
                    prefix + "_packed_params.weight": weight,
                    prefix + "_packed_params.bias": bias,
                }
            )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            False,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    # Function rather than property to make sure that JIT serialization doesn't
    # register this as an attribute
    def _weight_bias(self):
        return self._packed_params._weight_bias()

    def weight(self):
        return self._weight_bias()[0]

    def bias(self):
        return self._weight_bias()[1]

    def set_weight_bias(self, w: torch.Tensor, b: Optional[torch.Tensor]) -> None:
        self._packed_params.set_weight_bias(w, b)

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        r"""Create a quantized module from an observed float module

        Args:
            mod (Module): a float module, either produced by torch.ao.quantization
                          utilities or provided by the user
            use_precomputed_fake_quant (bool): if True, the module will reuse min/max
                          values from the precomputed fake quant module.
        """
        if hasattr(mod, "weight_fake_quant"):
            if type_before_parametrizations(mod) == nniqat.LinearBn1d:
                mod.weight, mod.bias = fuse_linear_bn_weights(
                    mod.weight,
                    mod.bias,
                    mod.bn.running_mean,
                    mod.bn.running_var,
                    mod.bn.eps,
                    mod.bn.weight,
                    mod.bn.bias,
                )
            weight_post_process = mod.weight_fake_quant
            activation_post_process = mod.activation_post_process
        else:
            # This function does not participate in JIT, so it is OK to ignore
            # the type mismatch in assignment. Also, mypy has an issue with
            # iterables not being implemented, so we are ignoring those too.
            if not isinstance(cls._FLOAT_MODULE, Iterable):
                cls._FLOAT_MODULE = [cls._FLOAT_MODULE]  # type: ignore[assignment]
            supported_modules = ", ".join([float_mod.__name__ for float_mod in cls._FLOAT_MODULE])  # type: ignore[attr-defined]
            error_msg = f"nnq.{cls.__name__}.from_float only works for {supported_modules}, but got: {type(mod)}"
            assert type_before_parametrizations(mod) in cls._FLOAT_MODULE, error_msg.format()  # type: ignore[attr-defined]
            assert hasattr(
                mod, "qconfig"
            ), "Input float module must have qconfig defined"
            activation_post_process = mod.activation_post_process
            if type_before_parametrizations(mod) == nni.LinearReLU:
                mod = mod[0]
            weight_post_process = (
                mod.qconfig.weight()
                if not hasattr(mod, "weight_fake_quant")
                else mod.weight_fake_quant
            )

        if not use_precomputed_fake_quant:
            # Observer may not have been called yet
            # Observer might have been called in the previous stage via PTQ algorithm e.g. AdaRound
            weight_post_process(mod.weight)
        dtype = weight_post_process.dtype
        act_scale, act_zp = activation_post_process.calculate_qparams()
        assert dtype == torch.qint8, "Weight observer must have dtype torch.qint8"
        qweight = _quantize_weight(mod.weight.float(), weight_post_process)
        qlinear = cls(mod.in_features, mod.out_features, dtype=dtype)
        qlinear.set_weight_bias(qweight, mod.bias)
        qlinear.scale = float(act_scale)
        qlinear.zero_point = int(act_zp)
        return qlinear

    @classmethod
    def from_reference(cls, ref_qlinear, output_scale, output_zero_point):
        r"""Create a (fbgemm/qnnpack) quantized module from a reference quantized module

        Args:
            ref_qlinear (Module): a reference quantized linear module, either produced by torch.ao.quantization
                          utilities or provided by the user
            output_scale (float): scale for output Tensor
            output_zero_point (int): zero point for output Tensor
        """
        qlinear = cls(ref_qlinear.in_features, ref_qlinear.out_features)
        qweight = ref_qlinear.get_quantized_weight()
        qlinear.set_weight_bias(qweight, ref_qlinear.bias)

        qlinear.scale = float(output_scale)
        qlinear.zero_point = int(output_zero_point)
        return qlinear
