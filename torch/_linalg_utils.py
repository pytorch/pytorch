"""Various linear algebra utility methods for internal use.

"""

from typing import Optional, Tuple

import torch
from torch import Tensor


def is_sparse(A):
    """Check if tensor A is a sparse tensor"""
    if isinstance(A, torch.Tensor):
        return A.layout == torch.sparse_coo

    error_str = "expected Tensor"
    if not torch.jit.is_scripting():
        error_str += " but got {}".format(type(A))
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


def conjugate(A):
    """Return conjugate of tensor A.

    .. note:: If A's dtype is not complex, A is returned.
    """
    if A.is_complex():
        return A.conj()
    return A


def transpose(A):
    """Return transpose of a matrix or batches of matrices."""
    ndim = len(A.shape)
    return A.transpose(ndim - 1, ndim - 2)


def transjugate(A):
    """Return transpose conjugate of a matrix or batches of matrices."""
    return conjugate(transpose(A))


def bform(X: Tensor, A: Optional[Tensor], Y: Tensor) -> Tensor:
    """Return bilinear form of matrices: :math:`X^T A Y`."""
    return matmul(transpose(X), matmul(A, Y))


def qform(A: Optional[Tensor], S: Tensor):
    """Return quadratic form :math:`S^T A S`."""
    return bform(S, A, S)


def basis(A):
    """Return orthogonal basis of A columns."""
    return torch.linalg.qr(A).Q


def symeig(A: Tensor, largest: Optional[bool] = False) -> Tuple[Tensor, Tensor]:
    """Return eigenpairs of A with specified ordering."""
    if largest is None:
        largest = False
    E, Z = torch.linalg.eigh(A, UPLO="U")
    # assuming that E is ordered
    if largest:
        E = torch.flip(E, dims=(-1,))
        Z = torch.flip(Z, dims=(-1,))
    return E, Z


# These functions were deprecated and removed
# This nice error message can be removed in version 1.13+
def matrix_rank(input, tol=None, symmetric=False, *, out=None) -> Tensor:
    raise RuntimeError(
        "This function was deprecated since version 1.9 and is now removed.\n"
        "Please use the `torch.linalg.matrix_rank` function instead. "
        "The parameter 'symmetric' was renamed in `torch.linalg.matrix_rank()` to 'hermitian'."
    )


def solve(input: Tensor, A: Tensor, *, out=None) -> Tuple[Tensor, Tensor]:
    raise RuntimeError(
        "This function was deprecated since version 1.9 and is now removed. "
        "`torch.solve` is deprecated in favor of `torch.linalg.solve`. "
        "`torch.linalg.solve` has its arguments reversed and does not return the LU factorization.\n\n"
        "To get the LU factorization see `torch.lu`, which can be used with `torch.lu_solve` or `torch.lu_unpack`.\n"
        "X = torch.solve(B, A).solution "
        "should be replaced with:\n"
        "X = torch.linalg.solve(A, B)"
    )


def lstsq(input: Tensor, A: Tensor, *, out=None) -> Tuple[Tensor, Tensor]:
    raise RuntimeError(
        "This function was deprecated since version 1.9 and is now removed. "
        "`torch.lstsq` is deprecated in favor of `torch.linalg.lstsq`.\n"
        "`torch.linalg.lstsq` has reversed arguments and does not return the QR decomposition in "
        "the returned tuple (although it returns other information about the problem).\n\n"
        "To get the QR decomposition consider using `torch.linalg.qr`.\n\n"
        "The returned solution in `torch.lstsq` stored the residuals of the solution in the "
        "last m - n columns of the returned value whenever m > n. In torch.linalg.lstsq, "
        "the residuals are in the field 'residuals' of the returned named tuple.\n\n"
        "The unpacking of the solution, as in\n"
        "X, _ = torch.lstsq(B, A).solution[:A.size(1)]\n"
        "should be replaced with:\n"
        "X = torch.linalg.lstsq(A, B).solution"
    )


def _symeig(
    input, eigenvectors=False, upper=True, *, out=None
) -> Tuple[Tensor, Tensor]:
    raise RuntimeError(
        "This function was deprecated since version 1.9 and is now removed. "
        "The default behavior has changed from using the upper triangular portion of the matrix by default "
        "to using the lower triangular portion.\n\n"
        "L, _ = torch.symeig(A, upper=upper) "
        "should be replaced with:\n"
        "L = torch.linalg.eigvalsh(A, UPLO='U' if upper else 'L')\n\n"
        "and\n\n"
        "L, V = torch.symeig(A, eigenvectors=True) "
        "should be replaced with:\n"
        "L, V = torch.linalg.eigh(A, UPLO='U' if upper else 'L')"
    )


def eig(
    self: Tensor, eigenvectors: bool = False, *, e=None, v=None
) -> Tuple[Tensor, Tensor]:
    raise RuntimeError(
        "This function was deprecated since version 1.9 and is now removed. "
        "`torch.linalg.eig` returns complex tensors of dtype `cfloat` or `cdouble` rather than real tensors "
        "mimicking complex tensors.\n\n"
        "L, _ = torch.eig(A) "
        "should be replaced with:\n"
        "L_complex = torch.linalg.eigvals(A)\n\n"
        "and\n\n"
        "L, V = torch.eig(A, eigenvectors=True) "
        "should be replaced with:\n"
        "L_complex, V_complex = torch.linalg.eig(A)"
    )
