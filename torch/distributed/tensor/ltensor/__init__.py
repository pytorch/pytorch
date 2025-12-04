"""
LTensor (Local Tensor) module for variance tracking in distributed operations.

LTensor is a PyTorch tensor subclass that tracks variance (which mesh axes tensors
vary along) within local_map operations, enabling correct automatic differentiation
in distributed computations.
"""

from .ltensor import LTensor

__all__ = [
    "LTensor",
]
