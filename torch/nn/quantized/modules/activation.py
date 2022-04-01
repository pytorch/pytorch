import torch
import torch.nn.quantized.functional

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
        >>> input = torch.quantize_per_tensor(input, 1.0, 0, dtype=torch.qint32)
        >>> output = m(input)
    """
    def __init__(self, inplace=False):
        super(ReLU6, self).__init__(inplace)
        self.inplace = inplace

    def forward(self, input):
        return torch.ops.quantized.relu6(input, self.inplace)

    def _get_name(self):
        return 'QuantizedReLU6'

    @staticmethod
    def from_float(mod):
        return ReLU6(mod.inplace)

class Hardswish(torch.nn.Hardswish):
    r"""This is the quantized version of :class:`~torch.nn.Hardswish`.

    Args:
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
    """
    def __init__(self, scale, zero_point):
        super(Hardswish, self).__init__()
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, input):
        return torch.nn.quantized.functional.hardswish(
            input, scale=self.scale, zero_point=self.zero_point)

    def _get_name(self):
        return 'QuantizedHardswish'

    @staticmethod
    def from_float(mod):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        return Hardswish(float(scale), int(zero_point))

class ELU(torch.nn.ELU):
    r"""This is the quantized equivalent of :class:`~torch.nn.ELU`.

    Args:
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
        alpha: the alpha constant
    """
    def __init__(self, scale, zero_point, alpha=1.):
        super(ELU, self).__init__(alpha)
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, input):
        return torch.nn.quantized.functional.elu(
            input, self.scale, self.zero_point, self.alpha)

    def _get_name(self):
        return 'QuantizedELU'

    @staticmethod
    def from_float(mod):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        return ELU(float(scale), int(zero_point), mod.alpha)

class LeakyReLU(torch.nn.LeakyReLU):
    r"""This is the quantized equivalent of :class:`~torch.nn.LeakyReLU`.

    Args:
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
        negative_slope: Controls the angle of the negative slope. Default: 1e-2
    """
    def __init__(self, scale: float, zero_point: int, negative_slope: float = 1e-2,
                 inplace: bool = False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(negative_slope, inplace)
        self.register_buffer('scale', torch.tensor(scale, **factory_kwargs))
        self.register_buffer('zero_point', torch.tensor(zero_point, **factory_kwargs))

    def forward(self, input):
        return torch.ops.quantized.leaky_relu(
            input, self.negative_slope, self.inplace, self.scale, self.zero_point)

    def _get_name(self):
        return 'QuantizedLeakyReLU'

    @classmethod
    def from_float(cls, mod):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
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
        return torch.ops.quantized.sigmoid(input, self.output_scale, self.output_zero_point)

    @classmethod
    def from_float(cls, mod):
        output_scale, output_zero_point = mod.activation_post_process.calculate_qparams()
        return cls(float(output_scale), int(output_zero_point))
