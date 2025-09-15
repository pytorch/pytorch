# mypy: allow-untyped-defs
from warnings import warn

import torch


__all__ = [
    "ReLU6",
    "Hardswish",
    "ELU",
    "LeakyReLU",
    "Sigmoid",
    "Softmax",
    "MultiheadAttention",
    "PReLU",
]


class ReLU6(torch.nn.ReLU):
    r"""Applies the element-wise function:

    :math:`\text{ReLU6}(x) = \min(\max(x_0, x), q(6))`, where :math:`x_0` is the
    zero_point, and :math:`q(6)` is the quantized representation of number 6.

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: ../scripts/activation_images/ReLU6.png

    Examples::

        >>> m = nn.quantized.ReLU6()
        >>> input = torch.randn(2)
        >>> # xdoctest: +SKIP
        >>> input = torch.quantize_per_tensor(input, 1.0, 0, dtype=torch.qint32)
        >>> output = m(input)
    """

    def __init__(self, inplace=False):
        super().__init__(inplace)
        self.inplace = inplace

    def forward(self, input):
        return torch.ops.quantized.relu6(input, self.inplace)

    def _get_name(self):
        return "QuantizedReLU6"

    @staticmethod
    def from_float(mod, use_precomputed_fake_quant=False):
        return ReLU6(mod.inplace)


class Hardswish(torch.nn.Hardswish):
    r"""This is the quantized version of :class:`~torch.nn.Hardswish`.

    Args:
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
    """

    def __init__(self, scale, zero_point, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.register_buffer("scale", torch.tensor(scale, **factory_kwargs))
        self.register_buffer("zero_point", torch.tensor(zero_point, **factory_kwargs))

    def forward(self, input):
        return torch.ops.quantized.hardswish(input, self.scale, self.zero_point)

    def _get_name(self):
        return "QuantizedHardswish"

    @staticmethod
    def from_float(mod, use_precomputed_fake_quant=False):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        return Hardswish(float(scale), int(zero_point))

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        return cls(float(scale), int(zero_point))


class ELU(torch.nn.ELU):
    r"""This is the quantized equivalent of :class:`~torch.nn.ELU`.

    Args:
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
        alpha: the alpha constant
    """

    def __init__(self, scale, zero_point, alpha=1.0):
        super().__init__(alpha)
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, input):
        return torch.ao.nn.quantized.functional.elu(
            input, self.scale, self.zero_point, self.alpha
        )

    def _get_name(self):
        return "QuantizedELU"

    @staticmethod
    def from_float(mod, use_precomputed_fake_quant=False):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        return ELU(float(scale), int(zero_point), mod.alpha)

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        return cls(float(scale), int(zero_point), mod.alpha)


class LeakyReLU(torch.nn.LeakyReLU):
    r"""This is the quantized equivalent of :class:`~torch.nn.LeakyReLU`.

    Args:
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
        negative_slope: Controls the angle of the negative slope. Default: 1e-2
    """

    def __init__(
        self,
        scale: float,
        zero_point: int,
        negative_slope: float = 1e-2,
        inplace: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(negative_slope, inplace)
        self.register_buffer("scale", torch.tensor(scale, **factory_kwargs))
        self.register_buffer("zero_point", torch.tensor(zero_point, **factory_kwargs))

    def forward(self, input):
        return torch.ops.quantized.leaky_relu(
            input, self.negative_slope, self.inplace, self.scale, self.zero_point
        )

    def _get_name(self):
        return "QuantizedLeakyReLU"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        return cls(float(scale), int(zero_point), mod.negative_slope, mod.inplace)

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        return cls(float(scale), int(zero_point), mod.negative_slope, mod.inplace)


class Sigmoid(torch.nn.Sigmoid):
    r"""This is the quantized equivalent of :class:`~torch.nn.Sigmoid`.

    Args:
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
    """

    def __init__(self, output_scale: float, output_zero_point: int):
        super().__init__()
        self.output_scale = output_scale
        self.output_zero_point = output_zero_point

    def forward(self, input):
        return torch.ops.quantized.sigmoid(
            input, self.output_scale, self.output_zero_point
        )

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        (
            output_scale,
            output_zero_point,
        ) = mod.activation_post_process.calculate_qparams()
        return cls(float(output_scale), int(output_zero_point))


class Softmax(torch.nn.Softmax):
    r"""This is the quantized version of :class:`~torch.nn.Softmax`.

    Args:
        dim: A dimension along which Softmax will be computed (so every slice along dim will sum to 1).
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
    """

    def __init__(self, dim=None, scale=1.0, zero_point=0):
        super().__init__()
        self.dim = dim
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, input):
        dim = self.dim
        if dim is None:
            stacklevel = 3
            # Note: adding the mypy ignore on _get_softmax_dim seems less bad
            # than making `_get_softmax_dim` an official API.
            dim = torch.nn.functional._get_softmax_dim(  # type: ignore[attr-defined]
                "softmax", input.dim(), stacklevel
            )
        return torch.ops.quantized.softmax(input, dim, self.scale, self.zero_point)

    def _get_name(self):
        return "QuantizedSoftmax"

    @staticmethod
    def from_float(mod, use_precomputed_fake_quant=False):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        return Softmax(mod.dim, float(scale), int(zero_point))

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        return cls(mod.dim, float(scale), int(zero_point))


class MultiheadAttention(torch.ao.nn.quantizable.MultiheadAttention):
    _FLOAT_MODULE = torch.ao.nn.quantizable.MultiheadAttention

    def _get_name(self):
        return "QuantizedMultiheadAttention"

    @classmethod
    def from_float(cls, other):
        # The whole flow is float -> observed -> quantized
        # This class does observed -> quantized only
        raise NotImplementedError(
            "It looks like you are trying to convert a "
            "non-observed MHA module. Please, see "
            "the examples on quantizable MHAs."
        )

    @classmethod
    def from_observed(cls, other):
        converted = torch.ao.quantization.convert(
            other,
            mapping=None,
            inplace=False,
            remove_qconfig=True,
            convert_custom_config_dict=None,
        )
        converted.__class__ = cls
        # Remove the parameters for the bias_k and bias_v to quantize them
        # TODO: This is a potential source of accuracy drop.
        #       quantized cat takes the scale and zp of the first
        #       element, which might lose the precision in the bias_k
        #       and the bias_v (which are cat'ed with k/v being first).
        if converted.bias_k is not None:
            bias_k = converted._parameters.pop("bias_k")
            sc, zp = torch._choose_qparams_per_tensor(bias_k, reduce_range=False)
            bias_k = torch.quantize_per_tensor(bias_k, sc, zp, torch.quint8)
            setattr(converted, "bias_k", bias_k)  # noqa: B010

        if converted.bias_v is not None:
            bias_v = converted._parameters.pop("bias_v")
            sc, zp = torch._choose_qparams_per_tensor(
                bias_k,  # type: ignore[possibly-undefined]
                reduce_range=False,
            )
            bias_v = torch.quantize_per_tensor(bias_v, sc, zp, torch.quint8)
            setattr(converted, "bias_v", bias_v)  # noqa: B010

        del converted.in_proj_weight
        del converted.in_proj_bias

        return converted


class PReLU(torch.nn.Module):
    r"""This is the quantized equivalent of :class:`~torch.nn.PReLU`.

    Args:
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
        num_parameters: number of parameters: 1, or the number of channels at input. Default: 1
    """

    def __init__(
        self, output_scale: float, output_zero_point: int, num_parameters: int = 1
    ) -> None:
        super().__init__()
        self.num_parameters = num_parameters
        self.scale = output_scale
        self.zero_point = output_zero_point
        w = torch.randn(num_parameters, dtype=torch.float)
        qw = torch.quantize_per_tensor(w, scale=1.0, zero_point=0, dtype=torch.quint8)
        self.set_weight(qw)

    def set_weight(self, w: torch.Tensor) -> None:
        self.weight = w

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.ops.quantized.prelu(
            input, self.weight, self.scale, self.zero_point
        )

    def _get_name(self):
        return "QuantizedPReLU"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        qprelu = cls(float(scale), int(zero_point), mod.num_parameters)
        float_wt = mod.weight.float()
        observer = mod.qconfig.weight()
        observer(float_wt)
        if observer.dtype != torch.quint8:
            warn(
                f"PReLU's weight observer should have dtype quint8 but got {observer.dtype}"
            )
        wt_scale, wt_zp = observer.calculate_qparams()
        qweight = torch.quantize_per_tensor(
            float_wt, float(wt_scale), int(wt_zp), torch.quint8
        )
        qprelu.set_weight(qweight)
        return qprelu

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        qprelu = cls(float(scale), int(zero_point), mod.num_parameters)
        float_wt = mod.weight.float()
        observer = mod.qconfig.weight()
        observer(float_wt)
        if observer.dtype != torch.quint8:
            warn(
                f"PReLU's weight observer should have dtype quint8 but got {observer.dtype}"
            )
        wt_scale, wt_zp = observer.calculate_qparams()
        qweight = torch.quantize_per_tensor(
            float_wt, float(wt_scale), int(wt_zp), torch.quint8
        )
        qprelu.set_weight(qweight)
        return qprelu
