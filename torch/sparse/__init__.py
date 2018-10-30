# The Tensor classes are added to this module by python_tensor.cpp
import torch

__all__ = [
    'addmm',
]


def addmm(mat, mat1, mat2, beta=1, alpha=1):
    r"""
    .. function:: torch.sparse.addmm(mat, mat1, mat2, beta=1, alpha=1) -> Tensor
    """
    return torch._sparse_addmm(mat, mat1, mat2, beta=beta, alpha=alpha)
