"""Various linear algebra utility methods for internal use.

"""

import torch


def uniform(low=0.0, high=1.0, size=None, dtype=None, device=None):
    """Returns a tensor filled with random numbers from a uniform
    distribution on the interval :math:`[low, high)`

    Arguments::

      size (int...): a sequence of integers defining the shape of the
        output tensor.  Can be a variable number of arguments or a
        collection like a list or tuple.

      dtype (:class:`torch.dtype`, optional): the desired data type of
        returned tensor.  Default: if ``None``, uses a global default
        (see :func:`torch.set_default_tensor_type`).

      device (:class:`torch.device`, optional): the desired device of
        returned tensor.  Default: if ``None``, uses the current
        device for the default tensor type (see
        :func:`torch.set_default_tensor_type`). :attr:`device` will be
        the CPU for CPU tensor types and the current CUDA device for
        CUDA tensor types.

    """
    attrs = dict(dtype=dtype, device=device)
    if size is None:
        r = low + (high - low) * torch.rand(1, **attrs)[0]
    else:
        r = low + (high - low) * torch.rand(*size, **attrs)
    if dtype in [torch.complex32, torch.complex64, torch.complex128]:
        if size is None:
            i = low + (high - low) * torch.rand(1, **attrs)[0]
        else:
            i = low + (high - low) * torch.rand(*size, **attrs)
        return r + 1j * i
    return r


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


def _identity_matmul(A, other):
    return other


def get_matmul(A):
    """Return matrix multiplication implementation.
    """
    if A is None:  # A is identity
        return _identity_matmul
    if isinstance(A, torch.Tensor):
        if is_sparse(A):
            return torch.sparse.mm
        else:
            return torch.matmul
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
