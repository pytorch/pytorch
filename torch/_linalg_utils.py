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
    """Multiply two matrices.

    If A is None, return B. A can be sparse or dense. B is always
    dense.
    """
    # type: (Optional[Tensor], Tensor) -> Tensor
    if A is None:
        return B
    if is_sparse(A):
        return torch.sparse.mm(A, B)
    return torch.matmul(A, B)


def conjugate(A):
    """Return conjugate of tensor A.

    .. note:: If A's dtype is not complex, A is returned.
    """
    if A.dtype in [torch.complex32, torch.complex64, torch.complex128]:    
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


def norm(A):
    """Return Frobenius norm of a real matrix.
    """
    if isinstance(A, torch.Tensor):
        try:
            return A.norm(dim=(-2, -1), p='fro')
        except RuntimeError:
            # e.g. conj is not available in CUDA
            return matmul(transpose(A), A).trace() ** 0.5
    else:
        raise TypeError("expected Tensor but got %s" % (type(A).__name__))


def bform(X, A, Y):
    """Return bilinear form of matrices: :math:`X^T A Y`.
    """
    return matmul(transpose(X), matmul(A, Y))


def qform(A, S):
    """Return quadratic form :math:`S^T A S`.
    """
    return bform(S, A, S)


def basis(A):
    """Return orthogonal basis of A columns.
    """
    try:
        Q = torch.orgqr(*torch.geqrf(A))
    except (RuntimeError, AttributeError):
        # torch.orgqr is not available in CUDA
        Q, _ = torch.qr(A, some=True)
    return Q


def symeig(A, largest=False, eigenvectors=True):
    """Return eigenpairs of A with specified ordering.
    """
    E, Z = torch.symeig(A, eigenvectors=eigenvectors)
    # assuming that E is ordered
    if largest:
        E = torch.flip(E, dims=(-1,))
        Z = torch.flip(Z, dims=(-1,))
    return E, Z
