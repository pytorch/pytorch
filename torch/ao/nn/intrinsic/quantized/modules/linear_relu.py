import torch
import torch.ao.nn.quantized as nnq
import torch.ao.nn.intrinsic as nni
from torch.ao.nn.quantized.modules.utils import _quantize_weight

class LinearReLU(nnq.Linear):
    r"""
    A LinearReLU module fused from Linear and ReLU modules

    We adopt the same interface as :class:`torch.ao.nn.quantized.Linear`.

    Attributes:
        Same as torch.ao.nn.quantized.Linear

    Examples::

        >>> # xdoctest: +SKIP
        >>> m = nn.intrinsic.LinearReLU(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    _FLOAT_MODULE = nni.LinearReLU

    def __init__(self, in_features, out_features, bias=True, dtype=torch.qint8):
        super().__init__(in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.quantized.linear_relu(
            x, self._packed_params._packed_params, self.scale, self.zero_point)

    def _get_name(self):
        return 'QuantizedLinearReLU'

    @classmethod
    def from_float(cls, mod):
        return super(LinearReLU, cls).from_float(mod)

    @classmethod
    def from_reference(cls, ref_linear_relu, output_scale, output_zero_point):
        return super().from_reference(ref_linear_relu[0], output_scale, output_zero_point)

class LinearLeakyReLU(nnq.Linear):
    r"""
    For onednn backend only
    A LinearLeakyReLU module fused from Linear and LeakyReLU modules
    We adopt the same interface as :class:`torch.nn.quantized.Linear`.
    Attributes:
        Same as torch.nn.quantized.Linear
        + negative_slope
    Examples::
        >>> m = nn.intrinsic.LinearLeakyReLU(20, 30, 0.01)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    _FLOAT_MODULE = nni.LinearLeakyReLU

    def __init__(self, in_features, out_features, negative_slope, bias=True, dtype=torch.qint8):
        super().__init__(in_features, out_features, bias, dtype)
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.quantized.linear_leaky_relu(
            x, self._packed_params._packed_params, self.scale, self.zero_point, self.negative_slope)

    def _get_name(self):
        return 'QuantizedLinearLeakyReLU'

    @classmethod
    def from_float(cls, mod):
        assert type(mod) == nni.LinearLeakyReLU, 'Input float module should be LinearLeakyReLU'
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        activation_post_process = mod.activation_post_process
        leaky_relu = mod[1]
        mod = mod[0]
        weight_post_process = mod.qconfig.weight()
        weight_post_process(mod.weight)
        dtype = weight_post_process.dtype
        act_scale, act_zp = activation_post_process.calculate_qparams()  # type: ignore[union-attr,operator]
        assert dtype == torch.qint8, 'Weight observer must have dtype torch.qint8'
        qweight = _quantize_weight(mod.weight.float(), weight_post_process)
        qlinear = cls(mod.in_features,
                      mod.out_features,
                      leaky_relu.negative_slope,
                      dtype=dtype)
        qlinear.set_weight_bias(qweight, mod.bias)
        qlinear.scale = float(act_scale)
        qlinear.zero_point = int(act_zp)
        return qlinear

    @classmethod
    def from_reference(cls, ref_mod, output_scale, output_zero_point):
        linear = ref_mod[0]
        leaky_relu = ref_mod[1]
        qlinear = cls(
            linear.in_features,
            linear.out_features,
            leaky_relu.negative_slope)
        qweight = linear.get_quantized_weight()
        qlinear.set_weight_bias(qweight, linear.bias)

        qlinear.scale = float(output_scale)
        qlinear.zero_point = int(output_zero_point)
        return qlinear
