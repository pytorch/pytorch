"""Implement various linear algebra algorithms for low rank matrices.
"""

import torch


def is_sparse(A):
    """Check if tensor A is a sparse tensor"""
    if isinstance(A, torch.Tensor):
        return A.layout == torch.sparse_coo
    import scipy.sparse
    return isinstance(A, scipy.sparse.base.spmatrix)


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


def batch_transpose(A):
    """Return transpose of a matrix or batches of matrices.
    """
    return A.transpose(-1, -2)


def batch_transjugate(A):
    """Return transpose conjugate of a matrix or batches of matrices.
    """
    return conjugate(batch_transpose(A))


def get_approximate_basis(A, q, niter=2, M=None):
    """Return tensor :math:`Q` with q orthonormal columns such that
    :math:`Q Q^H A` approximates :math:`A`. If :math:`M` is specified,
    then `Q` is such that :math:`Q Q^H (A - M)` approximates :math:`A
    - M`.

    .. note:: The implementation is based on the Algorithm 4.4 from
              Halko et al, 2009.

    .. note:: For an adequate approximation of a k-rank matrix A,
              where k is not known in advance but could be estimated,
              the number of Q columns q can be choosen according to
              the following criteria: in general, :math:`k <= q <=
              min(2*k, m, n)`. For large low-rank matrices, take
              :math:`q = k + 5..10`.  If k is relatively small
              compared to :math:`min(m, n)`, choosing :math:`q = k +
              0..2` may be sufficient.

    .. note:: To obtain repeatable results, reset the seed for the
              pseudorandom number generator

    Arguments::
        A (Tensor): the input tensor of size :math:`(*, m, n)`

        q (int): the dimension of subspace spanned by Q columns.

        niter (int, optional): the number of subspace iterations to
                               conduct; niter must be a nonnegative
                               integer. In most cases, the default
                               value 2 is more than enough.

        M (Tensor, optional): the input tensor's mean of size
                              :math:`(*, 1, n)`.

    References::
        - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding
          structure with randomness: probabilistic algorithms for
          constructing approximate matrix decompositions,
          arXiv:0909.4061 [math.NA; math.PR], 2009 (available at
          `arXiv <http://arxiv.org/abs/0909.4061>`_).

    """
    m, n = A.shape[-2:]
    dtype = get_floating_dtype(A)
    matmul = get_matmul(A)

    R = uniform(low=-1.0, high=1.0, size=(n, q),
                dtype=dtype, device=A.device)

    A_H = batch_transjugate(A)
    if M is None:
        (Q, _) = matmul(A, R).qr()
        for i in range(niter):
            (Q, _) = matmul(A_H, Q).qr()
            (Q, _) = matmul(A, Q).qr()
    else:
        matmul2 = get_matmul(M)
        M_H = batch_transjugate(M)
        (Q, _) = (matmul(A, R) - matmul2(M, R)).qr()
        for i in range(niter):
            (Q, _) = (matmul(A_H, Q) - matmul2(M_H, Q)).qr()
            (Q, _) = (matmul(A, Q) - matmul2(M, Q)).qr()

    return Q


def svd(A, q=6, niter=2, M=None):
    """Return the singular value decomposition ``(U, S, V)`` of a matrix,
    batches of matrices, or a sparse matrix :math:`A` such that
    :math:`A \approx U diag(S) V^T`. In case :math:`M` is given, then
    SVD is computed for the matrix :math:`A - M`.

    .. note:: The implementation is based on the Algorithm 5.1 from
              Halko et al, 2009.

    .. note:: To obtain repeatable results, reset the seed for the
              pseudorandom number generator

    .. note:: The input is assumed to be a low-rank matrix.

    .. note:: In general, use the full-rank SVD implementation
              ``torch.svd`` for dense matrices due to its 10-fold
              higher performance characteristics. The low-rank SVD
              will be useful for huge sparse matrices that
              ``torch.svd`` cannot handle.

    Arguments::
        A (Tensor): the input tensor of size :math:`(*, m, n)`

        q (int, optional): a slightly overestimated rank of A.

        niter (int, optional): the number of subspace iterations to
                               conduct; niter must be a nonnegative
                               integer, and defaults to 2

        M (Tensor, optional): the input tensor's mean of size
                              :math:`(*, 1, n)`.

    References::
        - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding
          structure with randomness: probabilistic algorithms for
          constructing approximate matrix decompositions,
          arXiv:0909.4061 [math.NA; math.PR], 2009 (available at
          `arXiv <http://arxiv.org/abs/0909.4061>`_).

    """
    m, n = A.shape[-2:]
    matmul = get_matmul(A)
    if M is None:
        M_t = None
    else:
        matmul2 = get_matmul(M)
        M_t = batch_transpose(M)
    A_t = batch_transpose(A)

    # Algorithm 5.1 in Halko et al 2009, slightly modified to reduce
    # the number conjugate and transpose operations
    if m < n:
        # computing the SVD approximation of a transpose in order to
        # keep B shape minimal
        Q = get_approximate_basis(A_t, q, niter=niter, M=M_t)
        Q_c = conjugate(Q)
        if M is None:
            B_t = matmul(A, Q_c)
        else:
            B_t = matmul(A, Q_c) - matmul2(M, Q_c)
        U, S, V = torch.svd(B_t)
        V = Q.matmul(V)
    else:
        Q = get_approximate_basis(A, q, niter=niter, M=M)
        Q_c = conjugate(Q)
        if M is None:
            B = matmul(A_t, Q_c)
        else:
            B = matmul(A_t, Q_c) - matmul2(M_t, Q_c)
        U, S, V = torch.svd(batch_transpose(B))
        U = Q.matmul(U)

    return U, S, V


def pca(A, q=None, center=True, niter=2):
    r"""Performs Principal Component Analysis (PCA) on a low-rank matrix,
    batches of such matrices, or sparse matrix.

    This function returns a namedtuple ``(U, S, V)`` which is the
    nearly optimal approximation of a singular value decomposition of
    a centered matrix :attr:`A` such that :math:`A = U diag(S) V^T`.

    .. note:: Different from the standard SVD, the size of returned
              matrices depend on the specified rank and q
              values as follows:
                - U is m x q matrix
                - S is q-vector
                - V is n x q matrix

    .. note:: To obtain repeatable results, reset the seed for the
              pseudorandom number generator

    Arguments:

        A (Tensor): the input tensor of size :math:`(*, m, n)`

        q (int, optional): a slightly overestimated rank of A. By
                           default, q = min(6, m, n).

        center (bool, optional): if True, center the input tensor,
                                 otherwise, assume that the input is
                                 centered.

        niter (int, optional): the number of subspace iterations to
                               conduct; niter must be a nonnegative
                               integer, and defaults to 2

    References::

        - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding
          structure with randomness: probabilistic algorithms for
          constructing approximate matrix decompositions,
          arXiv:0909.4061 [math.NA; math.PR], 2009 (available at
          `arXiv <http://arxiv.org/abs/0909.4061>`_).

    """
    (m, n) = A.shape[-2:]

    if q is None:
        q = min(6, m, n)
    elif not (q >= 0 and q <= min(m, n)):
        raise ValueError('q(={}) must be non-negative integer'
                         ' and not greater than min(m, n)={}'
                         .format(q, min(m, n)))
    if not (niter >= 0):
        raise ValueError('niter(={}) must be non-negative integer'
                         .format(niter))

    dtype = get_floating_dtype(A)

    if not center:
        return svd(A, q, niter=niter)

    if is_sparse(A):
        if len(A.shape) != 2:
            raise ValueError('pca input is expected to be 2-dimensional tensor')
        c = torch.sparse.sum(A, dim=-2) / m
        # reshape c
        column_indices = c.indices()[0]
        indices = torch.zeros(2, len(column_indices),
                              dtype=column_indices.dtype,
                              device=column_indices.device)
        indices[0] = column_indices
        C_t = torch.sparse_coo_tensor(
            indices, c.values(), (n, 1), dtype=dtype, device=A.device)

        ones_m1_t = torch.ones(A.shape[:-2] + (1, m), dtype=dtype, device=A.device)
        M = batch_transpose(torch.sparse.mm(C_t, ones_m1_t))
        return svd(A, q, niter=niter, M=M)
    else:
        c = A.sum(axis=-2) / m
        C = c.reshape(A.shape[:-2] + (1, n))
        ones_m1 = torch.ones(A.shape[:-1] + (1, ), dtype=dtype, device=A.device)
        M = ones_m1.matmul(C)
        return svd(A - M, q, niter=niter)
