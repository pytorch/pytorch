r"""
The quantized module contains the data structures and ops for implementation of
the quantized models.
"""

from torch.nn.quantized.functional import add_relu
from torch.nn.quantized.functional import relu

__all__ = [
    'add_relu',
    'relu'
]
