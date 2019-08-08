# @lint-ignore-every PYTHON3COMPATIMPORTS

from torch.nn.modules.pooling import MaxPool2d

from .activation import ReLU
from .conv import Conv2d
from .linear import Linear
from .linear import Quantize
from .linear import DeQuantize

from .wrapper_module import UnaryWrapper, BinaryWrapper
from .wrapper_module import QuantizedUnaryWrapper, QuantizedBinaryWrapper

__all__ = [
    'Conv2d',
    'DeQuantize',
    'Linear',
    'MaxPool2d',
    'Quantize',
    'ReLU',
    # Wrapper modules
    'UnaryWrapper',
    'BinaryWrapper',
    'QuantizedUnaryWrapper',
    'QuantizedBinaryWrapper',
]
