# The Tensor classes are added to this module by python_tensor.cpp
import torch

__all__ = [
    'addmm',
]


def addmm(mat, mat1, mat2, beta=1, alpha=1):
    r"""
    .. function:: torch.sparse.addmm(mat, mat1, mat2, beta=1, alpha=1) -> Tensor

    This function does exact same thing as :meth:`~Torch.addmm` in the forward,
    except that it supports backward for coalesced sparse matrix `mat1`.

    Args:
        mat (Tensor): a dense matrix to be added
        mat1 (Tensor): a sparse matrix to be multiplied
        mat2 (Tensor): a dense matrix be multiplied
        beta (Number, optional): multiplier for :attr:`mat` (:math:`\beta`)
        alpha (Number, optional): multiplier for :math:`mat1 @ mat2` (:math:`\alpha`)
    """
    return torch._sparse_addmm(mat, mat1, mat2, beta=beta, alpha=alpha)
