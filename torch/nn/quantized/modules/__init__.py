# @lint-ignore-every PYTHON3COMPATIMPORTS

import torch
from torch.nn.modules.pooling import MaxPool2d

from .activation import ReLU, ReLU6
from .batchnorm import BatchNorm2d, BatchNorm3d
from .conv import Conv2d, Conv3d
from .linear import Linear

from .functional_modules import FloatFunctional, QFunctional


class Quantize(torch.nn.Module):
    r"""Quantizes an incoming tensor

    Args:
     `scale`: scale of the output Quantized Tensor
     `zero_point`: zero_point of output Quantized Tensor
     `dtype`: data type of output Quantized Tensor

    Attributes:
      `scale`, `zero_point`, `dtype`

    Examples::
        >>> t = torch.tensor([[1., -1.], [1., -1.]])
        >>> scale, zero_point, dtype = 1.0, 2, torch.qint8
        >>> qm = Quantize(scale, zero_point, dtype)
        >>> qt = qm(t)
        >>> print(qt)
        tensor([[ 1., -1.],
                [ 1., -1.]], size=(2, 2), dtype=torch.qint8, scale=1.0, zero_point=2)
    """

    def __init__(self, scale, zero_point, dtype):
        super(Quantize, self).__init__()
        self.register_buffer('scale', torch.tensor([scale]))
        self.register_buffer('zero_point', torch.tensor([zero_point], dtype=torch.long))
        self.dtype = dtype

    def forward(self, X):
        return torch.quantize_per_tensor(X, float(self.scale),
                                         int(self.zero_point), self.dtype)

    @staticmethod
    def from_float(mod):
        assert hasattr(mod, 'activation_post_process')
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        return Quantize(scale.float().item(), zero_point.long().item(), mod.activation_post_process.dtype)

    def extra_repr(self):
        return 'scale={}, zero_point={}, dtype={}'.format(self.scale, self.zero_point, self.dtype)


class DeQuantize(torch.nn.Module):
    r"""Dequantizes an incoming tensor

    Examples::
        >>> input = torch.tensor([[1., -1.], [1., -1.]])
        >>> scale, zero_point, dtype = 1.0, 2, torch.qint8
        >>> qm = Quantize(scale, zero_point, dtype)
        >>> quantized_input = qm(input)
        >>> dqm = DeQuantize()
        >>> dequantized = dqm(quantized_input)
        >>> print(dequantized)
        tensor([[ 1., -1.],
                [ 1., -1.]], dtype=torch.float32)
    """

    def __init__(self):
        super(DeQuantize, self).__init__()

    def forward(self, Xq):
        return Xq.dequantize()

    @staticmethod
    def from_float(mod):
        return DeQuantize()

__all__ = [
    'BatchNorm2d',
    'BatchNorm3d',
    'Conv2d',
    'Conv3d',
    'DeQuantize',
    'Linear',
    'MaxPool2d',
    'Quantize',
    'ReLU',
    'ReLU6',
    # Wrapper modules
    'FloatFunctional',
    'QFunctional',
]
