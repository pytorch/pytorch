# @lint-ignore-every PYTHON3COMPATIMPORTS

from torch.nn.modules.pooling import MaxPool2d

from .activation import ReLU
from .conv import Conv2d
from .linear import Linear
from .linear import Quantize
from .linear import DeQuantize

from ._wrapped_modules import Add

__all__ = [
    'Conv2d',
    'DeQuantize',
    'Linear',
    'MaxPool2d',
    'Quantize',
    'ReLU',
    # Wrapped modules
    'Add'
]
