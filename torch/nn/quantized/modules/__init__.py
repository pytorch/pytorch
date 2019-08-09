# @lint-ignore-every PYTHON3COMPATIMPORTS

from torch.nn.modules.pooling import MaxPool2d

from .activation import ReLU
from .conv import Conv2d
from .linear import Linear
from .linear import Quantize
from .linear import DeQuantize

from .wrapper_module import make_wrapper
from .wrapper_module import Add, Cat
from .wrapper_module import QuantizedAdd, QuantizedCat

__all__ = [
    'Conv2d',
    'DeQuantize',
    'Linear',
    'MaxPool2d',
    'Quantize',
    'ReLU',
    # Wrapper modules
    'make_wrapper',
    'Add',
    'Cat',
    'QuantizedAdd',
    'QuantizedCat',
]
