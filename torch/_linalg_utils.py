# mypy: allow-untyped-defs
"""Various linear algebra utility methods for internal use."""

from typing import Optional

import torch
from torch import Tensor


def is_sparse(A):
    """Check if tensor A is a sparse COO tensor. All other sparse storage formats (CSR, CSC, etc...) will return False."""
    if isinstance(A, torch.Tensor):
        return A.layout == torch.sparse_coo

    error_str = "expected Tensor"
    if not torch.jit.is_scripting():
        error_str += f" but got {type(A)}"
    raise TypeError(error_str)


def get_floating_dtype(A):
    """Return the floating point dtype of tensor A.

    Integer types map to float32.
    """
    dtype = A.dtype
    if dtype in (torch.float16, torch.float32, torch.float64):
        return dtype
    return torch.float32


def matmul(A: Optional[Tensor], B: Tensor) -> Tensor:
    """Multiply two matrices.

    If A is None, return B. A can be sparse or dense. B is always
    dense.
    """
    if A is None:
        return B
    if is_sparse(A):
        return torch.sparse.mm(A, B)
    return torch.matmul(A, B)


def bform(X: Tensor, A: Optional[Tensor], Y: Tensor) -> Tensor:
    """Return bilinear form of matrices: :math:`X^T A Y`."""
    return matmul(X.mT, matmul(A, Y))


def qform(A: Optional[Tensor], S: Tensor):
    """Return quadratic form :math:`S^T A S`."""
    return bform(S, A, S)


def basis(A):
    """Return orthogonal basis of A columns."""
    return torch.linalg.qr(A).Q


def symeig(A: Tensor, largest: Optional[bool] = False) -> tuple[Tensor, Tensor]:
    """Return eigenpairs of A with specified ordering."""
    if largest is None:
        largest = False
    E, Z = torch.linalg.eigh(A, UPLO="U")
    # assuming that E is ordered
    if largest:
        E = torch.flip(E, dims=(-1,))
        Z = torch.flip(Z, dims=(-1,))
    return E, Z
