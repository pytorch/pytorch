# @lint-ignore-every PYTHON3COMPATIMPORTS

from torch.nn.modules.pooling import MaxPool2d

from .activation import ReLU
from .conv import Conv2d
from .linear import Linear
from .linear import Quantize
from .linear import DeQuantize

__all__ = [
    'Conv2d',
    'DeQuantize',
    'Linear',
    'MaxPool2d',
    'Quantize',
    'ReLU',
]

# Generated modules -- use torch/quantization/tools/make_modules to regenerate.
from ._generated import Add

__all__ += ['Add']
