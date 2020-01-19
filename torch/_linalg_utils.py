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
    """
    if isinstance(A, torch.Tensor):
        index = (0, ) * len(A.shape)
        return (A.__getitem__(index) * 1.0).dtype
    return A.dtype.type


def get_matmul(A):
    """Return matrix multiplication implementation.
    """
    if A is None:  # A is identity
        return lambda A, other: other
    if isinstance(A, torch.Tensor):
        if is_sparse(A):
            return torch.sparse.mm
        else:
            return torch.matmul
    else:
        import numpy
        import scipy.sparse
        if isinstance(A, scipy.sparse.coo.coo_matrix):
            return lambda A, other : A.dot(other)
        elif isinstance(A, numpy.ndarray):
            return numpy.dot

    raise NotImplementedError('get_matmul(<{} instance>)'
                              .format(type(A).__name__))


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
            return get_matmul(A)(transpose(A), A).trace() ** 0.5
    else:
        raise TypeError("expected Tensor but got %s" % (type(A).__name__))


def bform(X, A, Y):
    """Return bilinear form of matrices: :math:`X^T A Y`.
    """
    return get_matmul(X)(transpose(X), get_matmul(A)(A, Y))


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
