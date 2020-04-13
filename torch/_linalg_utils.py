"""Various linear algebra utility methods for internal use.

"""

import torch


def is_sparse(A):
    """Check if tensor A is a sparse tensor"""
    if isinstance(A, torch.Tensor):
        return A.layout == torch.sparse_coo
    raise TypeError("expected Tensor but got %s" % (type(A).__name__))


def get_floating_dtype(A):
    """Return the floating point dtype of tensor A.

    Integer types map to float32.
    """
    dtype = A.dtype
    if dtype in (torch.float16, torch.float32, torch.float64):
        return dtype
    return torch.float32


def matmul(A, B):
    # type: (Optional[Tensor], Tensor) -> Tensor
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
    """Return transpose of a matrix or batches of matrices.
    """
    ndim = len(A.shape)
    return A.transpose(ndim - 1, ndim - 2)


def transjugate(A):
    """Return transpose conjugate of a matrix or batches of matrices.
    """
    return conjugate(transpose(A))


def bform(X, A, Y):
    # type: (Tensor, Optional[Tensor], Tensor) -> Tensor
    """Return bilinear form of matrices: :math:`X^T A Y`.
    """
    return matmul(transpose(X), matmul(A, Y))


def qform(A, S):
    # type: (Optional[Tensor], Tensor) -> Tensor
    """Return quadratic form :math:`S^T A S`.
    """
    return bform(S, A, S)


def basis(A):
    """Return orthogonal basis of A columns.
    """
    if A.is_cuda:
        # torch.orgqr is not available in CUDA
        Q, _ = torch.qr(A, some=True)
    else:
        Q = torch.orgqr(*torch.geqrf(A))
    return Q


def symeig(A, largest=False, eigenvectors=True):
    # type: (Tensor, Optional[bool], Optional[bool]) -> Tuple[Tensor, Tensor]
    """Return eigenpairs of A with specified ordering.
    """
    if largest is None:
        largest = False
    if eigenvectors is None:
        eigenvectors = True
    E, Z = torch.symeig(A, eigenvectors, True)
    # assuming that E is ordered
    if largest:
        E = torch.flip(E, dims=(-1,))
        Z = torch.flip(Z, dims=(-1,))
    return E, Z
