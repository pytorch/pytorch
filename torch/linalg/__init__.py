import sys

import torch
from torch._C import _add_docstr, _linalg  # type: ignore

Tensor = torch.Tensor

# Note: This not only adds doc strings for functions in the linalg namespace, but
# also connects the torch.linalg Python namespace to the torch._C._linalg builtins.

cholesky = _add_docstr(_linalg.linalg_cholesky, r"""
linalg.cholesky(input, *, out=None) -> Tensor

Computes the Cholesky decomposition of a Hermitian (or symmetric for real-valued matrices)
positive-definite matrix or the Cholesky decompositions for a batch of such matrices.
Each decomposition has the form:

.. math::

    \text{input} = LL^H

where :math:`L` is a lower-triangular matrix and :math:`L^H` is the conjugate transpose of :math:`L`,
which is just a transpose for the case of real-valued input matrices.
In code it translates to ``input = L @ L.t()` if :attr:`input` is real-valued and
``input = L @ L.conj().t()`` if :attr:`input` is complex-valued.
The batch of :math:`L` matrices is returned.

Supports real-valued and complex-valued inputs.

.. note:: If :attr:`input` is not a Hermitian positive-definite matrix, or if it's a batch of matrices
          and one or more of them is not a Hermitian positive-definite matrix, then a RuntimeError will be thrown.
          If :attr:`input` is a batch of matrices, then the error message will include the batch index
          of the first matrix that is not Hermitian positive-definite.

.. warning:: This function always checks whether :attr:`input` is a Hermitian positive-definite matrix
             using `info` argument to LAPACK/MAGMA call. For CUDA this causes cross-device memory synchronization.

Args:
    input (Tensor): the input tensor of size :math:`(*, n, n)` consisting of Hermitian positive-definite
                    :math:`n \times n` matrices, where `*` is zero or more batch dimensions.

Keyword args:
    out (Tensor, optional): The output tensor. Ignored if ``None``. Default: ``None``

Examples::

    >>> a = torch.randn(2, 2, dtype=torch.complex128)
    >>> a = torch.mm(a, a.t().conj())  # creates a Hermitian positive-definite matrix
    >>> l = torch.linalg.cholesky(a)
    >>> a
    tensor([[2.5266+0.0000j, 1.9586-2.0626j],
            [1.9586+2.0626j, 9.4160+0.0000j]], dtype=torch.complex128)
    >>> l
    tensor([[1.5895+0.0000j, 0.0000+0.0000j],
            [1.2322+1.2976j, 2.4928+0.0000j]], dtype=torch.complex128)
    >>> torch.mm(l, l.t().conj())
    tensor([[2.5266+0.0000j, 1.9586-2.0626j],
            [1.9586+2.0626j, 9.4160+0.0000j]], dtype=torch.complex128)

    >>> a = torch.randn(3, 2, 2, dtype=torch.float64)
    >>> a = torch.matmul(a, a.transpose(-2, -1))  # creates a symmetric positive-definite matrix
    >>> l = torch.linalg.cholesky(a)
    >>> a
    tensor([[[ 1.1629,  2.0237],
            [ 2.0237,  6.6593]],

            [[ 0.4187,  0.1830],
            [ 0.1830,  0.1018]],

            [[ 1.9348, -2.5744],
            [-2.5744,  4.6386]]], dtype=torch.float64)
    >>> l
    tensor([[[ 1.0784,  0.0000],
            [ 1.8766,  1.7713]],

            [[ 0.6471,  0.0000],
            [ 0.2829,  0.1477]],

            [[ 1.3910,  0.0000],
            [-1.8509,  1.1014]]], dtype=torch.float64)
    >>> torch.allclose(torch.matmul(l, l.transpose(-2, -1)), a)
    True
""")

det = _add_docstr(_linalg.linalg_det, r"""
linalg.det(input) -> Tensor

Alias of :func:`torch.det`.
""")

eigh = _add_docstr(_linalg.linalg_eigh, r"""
linalg.eigh(input, UPLO='L') -> tuple(Tensor, Tensor)

This function computes the eigenvalues and eigenvectors
of a complex Hermitian (or real symmetric) matrix, or batch of such matrices, :attr:`input`.
For a single matrix :attr:`input`, the tensor of eigenvalues :math:`w` and the tensor of eigenvectors :math:`V`
decompose the :attr:`input` such that :math:`\text{input} = V \text{diag}(w) V^H`,
where :math:`^H` is the conjugate transpose operation.

Since the matrix or matrices in :attr:`input` are assumed to be Hermitian, the imaginary part of their diagonals
is always treated as zero. When :attr:`UPLO` is "L", its default value, only the lower triangular part of
each matrix is used in the computation. When :attr:`UPLO` is "U" only the upper triangular part of each matrix is used.

Supports input of ``float``, ``double``, ``cfloat`` and ``cdouble`` data types.

See :func:`torch.linalg.eigvalsh` for a related function that computes only eigenvalues,
however that function is not differentiable.

.. note:: The eigenvalues of real symmetric or complex Hermitian matrices are always real.

.. note:: The eigenvectors of matrices are not unique, so any eigenvector multiplied by a constant remains
          a valid eigenvector. This function may compute different eigenvector representations on
          different device types. Usually the difference is only in the sign of the eigenvector.

.. note:: The eigenvalues/eigenvectors are computed using LAPACK/MAGMA routines ``_syevd`` and ``_heevd``.
          This function always checks whether the call to LAPACK/MAGMA is successful
          using ``info`` argument of ``_syevd``, ``_heevd`` and throws a RuntimeError if it isn't.
          On CUDA this causes a cross-device memory synchronization.

Args:
    input (Tensor): the Hermitian :math:`n \times n` matrix or the batch
                    of such matrices of size :math:`(*, n, n)` where `*` is one or more batch dimensions.
    UPLO ('L', 'U', optional): controls whether to use the upper-triangular or the lower-triangular part
                               of :attr:`input` in the computations. Default: ``'L'``

Returns:
    (Tensor, Tensor): A namedtuple (eigenvalues, eigenvectors) containing

        - **eigenvalues** (*Tensor*): Shape :math:`(*, m)`.
            The eigenvalues in ascending order.
        - **eigenvectors** (*Tensor*): Shape :math:`(*, m, m)`.
            The orthonormal eigenvectors of the ``input``.

Examples::

    >>> a = torch.randn(2, 2, dtype=torch.complex128)
    >>> a = a + a.t().conj()  # creates a Hermitian matrix
    >>> a
    tensor([[2.9228+0.0000j, 0.2029-0.0862j],
            [0.2029+0.0862j, 0.3464+0.0000j]], dtype=torch.complex128)
    >>> w, v = torch.linalg.eigh(a)
    >>> w
    tensor([0.3277, 2.9415], dtype=torch.float64)
    >>> v
    tensor([[-0.0846+-0.0000j, -0.9964+0.0000j],
            [ 0.9170+0.3898j, -0.0779-0.0331j]], dtype=torch.complex128)
    >>> torch.allclose(torch.matmul(v, torch.matmul(w.to(v.dtype).diag_embed(), v.t().conj())), a)
    True

    >>> a = torch.randn(3, 2, 2, dtype=torch.float64)
    >>> a = a + a.transpose(-2, -1)  # creates a symmetric matrix
    >>> w, v = torch.linalg.eigh(a)
    >>> torch.allclose(torch.matmul(v, torch.matmul(w.diag_embed(), v.transpose(-2, -1))), a)
    True
""")

eigvalsh = _add_docstr(_linalg.linalg_eigvalsh, r"""
linalg.eigvalsh(input, UPLO='L') -> Tensor

This function computes the eigenvalues of a complex Hermitian (or real symmetric) matrix,
or batch of such matrices, :attr:`input`. The eigenvalues are returned in ascending order.

Since the matrix or matrices in :attr:`input` are assumed to be Hermitian, the imaginary part of their diagonals
is always treated as zero. When :attr:`UPLO` is "L", its default value, only the lower triangular part of
each matrix is used in the computation. When :attr:`UPLO` is "U" only the upper triangular part of each matrix is used.

Supports input of ``float``, ``double``, ``cfloat`` and ``cdouble`` data types.

See :func:`torch.linalg.eigh` for a related function that computes both eigenvalues and eigenvectors.

.. note:: The eigenvalues of real symmetric or complex Hermitian matrices are always real.

.. note:: The eigenvalues/eigenvectors are computed using LAPACK/MAGMA routines ``_syevd`` and ``_heevd``.
          This function always checks whether the call to LAPACK/MAGMA is successful
          using ``info`` argument of ``_syevd``, ``_heevd`` and throws a RuntimeError if it isn't.
          On CUDA this causes a cross-device memory synchronization.

.. note:: This function doesn't support backpropagation, please use :func:`torch.linalg.eigh` instead,
          that also computes the eigenvectors.

Args:
    input (Tensor): the Hermitian :math:`n \times n` matrix or the batch
                    of such matrices of size :math:`(*, n, n)` where `*` is one or more batch dimensions.
    UPLO ('L', 'U', optional): controls whether to use the upper-triangular or the lower-triangular part
                               of :attr:`input` in the computations. Default: ``'L'``

Examples::

    >>> a = torch.randn(2, 2, dtype=torch.complex128)
    >>> a = a + a.t().conj()  # creates a Hermitian matrix
    >>> a
    tensor([[2.9228+0.0000j, 0.2029-0.0862j],
            [0.2029+0.0862j, 0.3464+0.0000j]], dtype=torch.complex128)
    >>> w = torch.linalg.eigvalsh(a)
    >>> w
    tensor([0.3277, 2.9415], dtype=torch.float64)

    >>> a = torch.randn(3, 2, 2, dtype=torch.float64)
    >>> a = a + a.transpose(-2, -1)  # creates a symmetric matrix
    >>> a
    tensor([[[ 2.8050, -0.3850],
            [-0.3850,  3.2376]],

            [[-1.0307, -2.7457],
            [-2.7457, -1.7517]],

            [[ 1.7166,  2.2207],
            [ 2.2207, -2.0898]]], dtype=torch.float64)
    >>> w = torch.linalg.eigvalsh(a)
    >>> w
    tensor([[ 2.5797,  3.4629],
            [-4.1605,  1.3780],
            [-3.1113,  2.7381]], dtype=torch.float64)
""")

matrix_rank = _add_docstr(_linalg.linalg_matrix_rank, r"""
matrix_rank(input, tol=None, hermitian=False) -> Tensor

Computes the numerical rank of a matrix :attr:`input`, or of each matrix in a batched :attr:`input`.
The matrix rank is computed as the number of singular values (or the absolute eigenvalues when :attr:`hermitian` is ``True``)
above the specified :attr:`tol` threshold.

If :attr:`tol` is not specified, :attr:`tol` is set to
``S.max(dim=-1) * max(input.shape[-2:]) * eps`` where ``S`` is the singular values
(or the absolute eigenvalues when :attr:`hermitian` is ``True``),
and ``eps`` is the epsilon value for the datatype of :attr:`input`.
The epsilon value can be obtained using ``eps`` attribute of :class:`torch.finfo`.

The method to compute the matrix rank is done using singular value decomposition (see :func:`torch.linalg.svd`) by default.
If :attr:`hermitian` is ``True``, then :attr:`input` is assumed to be Hermitian (symmetric if real-valued),
and the computation of the rank is done by obtaining the eigenvalues (see :func:`torch.linalg.eigvalsh`).

Supports input of ``float``, ``double``, ``cfloat`` and ``cdouble`` datatypes.

.. note:: When given inputs on a CUDA device, this function synchronizes that device with the CPU.

Args:
    input (Tensor): the input matrix of size :math:`(m, n)` or the batch of matrices of size :math:`(*, m, n)`
                    where `*` is one or more batch dimensions.
    tol (float, optional): the tolerance value. Default: ``None``
    hermitian(bool, optional): indicates whether :attr:`input` is Hermitian. Default: ``False``

Examples::

    >>> a = torch.eye(10)
    >>> torch.linalg.matrix_rank(a)
    tensor(10)
    >>> b = torch.eye(10)
    >>> b[0, 0] = 0
    >>> torch.linalg.matrix_rank(b)
    tensor(9)

    >>> a = torch.randn(4, 3, 2)
    >>> torch.linalg.matrix_rank(a)
    tensor([2, 2, 2, 2])

    >>> a = torch.randn(2, 4, 2, 3)
    >>> torch.linalg.matrix_rank(a)
    tensor([[2, 2, 2, 2],
            [2, 2, 2, 2]])

    >>> a = torch.randn(2, 4, 3, 3, dtype=torch.complex64)
    >>> torch.linalg.matrix_rank(a)
    tensor([[3, 3, 3, 3],
            [3, 3, 3, 3]])
    >>> torch.linalg.matrix_rank(a, hermitian=True)
    tensor([[3, 3, 3, 3],
            [3, 3, 3, 3]])
    >>> torch.linalg.matrix_rank(a, tol=1.0)
    tensor([[3, 2, 2, 2],
            [1, 2, 1, 2]])
    >>> torch.linalg.matrix_rank(a, tol=1.0, hermitian=True)
    tensor([[2, 2, 2, 1],
            [1, 2, 2, 2]])
""")

norm = _add_docstr(_linalg.linalg_norm, r"""
linalg.norm(input, ord=None, dim=None, keepdim=False, *, out=None, dtype=None) -> Tensor

Returns the matrix norm or vector norm of a given tensor.

This function can calculate one of eight different types of matrix norms, or one
of an infinite number of vector norms, depending on both the number of reduction
dimensions and the value of the `ord` parameter.

Args:
    input (Tensor): The input tensor. If dim is None, x must be 1-D or 2-D, unless :attr:`ord`
        is None. If both :attr:`dim` and :attr:`ord` are None, the 2-norm of the input flattened to 1-D
        will be returned. Its data type must be either a floating point or complex type. For complex
        inputs, the norm is calculated on of the absolute values of each element. If the input is
        complex and neither :attr:`dtype` nor :attr:`out` is specified, the result's data type will
        be the corresponding floating point type (e.g. float if :attr:`input` is complexfloat).

    ord (int, float, inf, -inf, 'fro', 'nuc', optional): The order of norm.
        inf refers to :attr:`float('inf')`, numpy's :attr:`inf` object, or any equivalent object.
        The following norms can be calculated:

        =====  ============================  ==========================
        ord    norm for matrices             norm for vectors
        =====  ============================  ==========================
        None   Frobenius norm                2-norm
        'fro'  Frobenius norm                -- not supported --
        'nuc'  nuclear norm                  -- not supported --
        inf    max(sum(abs(x), dim=1))       max(abs(x))
        -inf   min(sum(abs(x), dim=1))       min(abs(x))
        0      -- not supported --           sum(x != 0)
        1      max(sum(abs(x), dim=0))       as below
        -1     min(sum(abs(x), dim=0))       as below
        2      2-norm (largest sing. value)  as below
        -2     smallest singular value       as below
        other  -- not supported --           sum(abs(x)**ord)**(1./ord)
        =====  ============================  ==========================

        Default: ``None``

    dim (int, 2-tuple of ints, 2-list of ints, optional): If :attr:`dim` is an int,
        vector norm will be calculated over the specified dimension. If :attr:`dim`
        is a 2-tuple of ints, matrix norm will be calculated over the specified
        dimensions. If :attr:`dim` is None, matrix norm will be calculated
        when the input tensor has two dimensions, and vector norm will be
        calculated when the input tensor has one dimension. Default: ``None``

    keepdim (bool, optional): If set to True, the reduced dimensions are retained
        in the result as dimensions with size one. Default: ``False``

Keyword args:

    out (Tensor, optional): The output tensor. Ignored if ``None``. Default: ``None``

    dtype (:class:`torch.dtype`, optional): If specified, the input tensor is cast to
        :attr:`dtype` before performing the operation, and the returned tensor's type
        will be :attr:`dtype`. If this argument is used in conjunction with the
        :attr:`out` argument, the output tensor's type must match this argument or a
        RuntimeError will be raised. Default: ``None``

Examples::

    >>> import torch
    >>> from torch import linalg as LA
    >>> a = torch.arange(9, dtype=torch.float) - 4
    >>> a
    tensor([-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])
    >>> b = a.reshape((3, 3))
    >>> b
    tensor([[-4., -3., -2.],
            [-1.,  0.,  1.],
            [ 2.,  3.,  4.]])

    >>> LA.norm(a)
    tensor(7.7460)
    >>> LA.norm(b)
    tensor(7.7460)
    >>> LA.norm(b, 'fro')
    tensor(7.7460)
    >>> LA.norm(a, float('inf'))
    tensor(4.)
    >>> LA.norm(b, float('inf'))
    tensor(9.)
    >>> LA.norm(a, -float('inf'))
    tensor(0.)
    >>> LA.norm(b, -float('inf'))
    tensor(2.)

    >>> LA.norm(a, 1)
    tensor(20.)
    >>> LA.norm(b, 1)
    tensor(7.)
    >>> LA.norm(a, -1)
    tensor(0.)
    >>> LA.norm(b, -1)
    tensor(6.)
    >>> LA.norm(a, 2)
    tensor(7.7460)
    >>> LA.norm(b, 2)
    tensor(7.3485)

    >>> LA.norm(a, -2)
    tensor(0.)
    >>> LA.norm(b.double(), -2)
    tensor(1.8570e-16, dtype=torch.float64)
    >>> LA.norm(a, 3)
    tensor(5.8480)
    >>> LA.norm(a, -3)
    tensor(0.)

Using the :attr:`dim` argument to compute vector norms::

    >>> c = torch.tensor([[1., 2., 3.],
    ...                   [-1, 1, 4]])
    >>> LA.norm(c, dim=0)
    tensor([1.4142, 2.2361, 5.0000])
    >>> LA.norm(c, dim=1)
    tensor([3.7417, 4.2426])
    >>> LA.norm(c, ord=1, dim=1)
    tensor([6., 6.])

Using the :attr:`dim` argument to compute matrix norms::

    >>> m = torch.arange(8, dtype=torch.float).reshape(2, 2, 2)
    >>> LA.norm(m, dim=(1,2))
    tensor([ 3.7417, 11.2250])
    >>> LA.norm(m[0, :, :]), LA.norm(m[1, :, :])
    (tensor(3.7417), tensor(11.2250))
""")

cond = _add_docstr(_linalg.linalg_cond, r"""
linalg.cond(input, p=None, *, out=None) -> Tensor

Computes the condition number of a matrix :attr:`input`,
or of each matrix in a batched :attr:`input`, using the matrix norm defined by :attr:`p`.
For norms ``p = {'fro', 'nuc', inf, -inf, 1, -1}`` this is defined as the matrix norm of :attr:`input`
times the matrix norm of the inverse of :attr:`input`. And for norms ``p = {None, 2, -2}`` this is defined as
the ratio between the largest and smallest singular values.

This function supports ``float``, ``double``, ``cfloat`` and ``cdouble`` dtypes for :attr:`input`.
If the input is complex and neither :attr:`dtype` nor :attr:`out` is specified, the result's data type will
be the corresponding floating point type (e.g. float if :attr:`input` is complexfloat).

.. note:: For ``p = {None, 2, -2}`` the condition number is computed as the ratio between the largest
          and smallest singular values computed using :func:`torch.linalg.svd`.
          For these norms :attr:`input` may be a non-square matrix or batch of non-square matrices.
          For other norms, however, :attr:`input` must be a square matrix or a batch of square matrices,
          and if this requirement is not satisfied a RuntimeError will be thrown.

.. note:: For ``p = {'fro', 'nuc', inf, -inf, 1, -1}`` if :attr:`input` is a non-invertible matrix then
          a tensor containing infinity will be returned. If :attr:`input` is a batch of matrices and one
          or more of them is not invertible then a RuntimeError will be thrown.

.. note:: When given inputs on a CUDA device, this function synchronizes that device with the CPU.

Args:
    input (Tensor): the input matrix of size :math:`(m, n)` or the batch of matrices of size :math:`(*, m, n)`
                    where `*` is one or more batch dimensions.

    p (int, float, inf, -inf, 'fro', 'nuc', optional): the type of the matrix norm to use in the computations.
        The following norms are supported:

        =====  ============================
        p      norm for matrices
        =====  ============================
        None   ratio of the largest singular value to the smallest singular value
        'fro'  Frobenius norm
        'nuc'  nuclear norm
        inf    max(sum(abs(x), dim=1))
        -inf   min(sum(abs(x), dim=1))
        1      max(sum(abs(x), dim=0))
        -1     min(sum(abs(x), dim=0))
        2      ratio of the largest singular value to the smallest singular value
        -2     ratio of the smallest singular value to the largest singular value
        =====  ============================

        Default: ``None``

Keyword args:
    out (Tensor, optional): The output tensor. Ignored if ``None``. Default: ``None``

Examples::

    >>> from torch import linalg as LA
    >>> a = torch.tensor([[1., 0, -1], [0, 1, 0], [1, 0, 1]])
    >>> LA.cond(a)
    tensor(1.4142)
    >>> LA.cond(a, 'fro')
    tensor(3.1623)
    >>> LA.cond(a, 'nuc')
    tensor(9.2426)
    >>> LA.cond(a, np.inf)
    tensor(2.)
    >>> LA.cond(a, -np.inf)
    tensor(1.)
    >>> LA.cond(a, 1)
    tensor(2.)
    >>> LA.cond(a, -1)
    tensor(1.)
    >>> LA.cond(a, 2)
    tensor(1.4142)
    >>> LA.cond(a, -2)
    tensor(0.7071)

    >>> a = torch.randn(3, 4, 4)
    >>> LA.cond(a)
    tensor([ 4.4739, 76.5234, 10.8409])

    >>> a = torch.randn(3, 4, 4, dtype=torch.complex64)
    >>> LA.cond(a)
    tensor([ 5.9175, 48.4590,  5.6443])
    >>> LA.cond(a, 1)
    >>> tensor([ 11.6734+0.j, 105.1037+0.j,  10.1978+0.j])
""")

solve = _add_docstr(_linalg.linalg_solve, r"""
linalg.solve(input, other, *, out=None) -> Tensor

Computes the solution ``x`` to the matrix equation ``matmul(input, x) = other``
with a square matrix, or batches of such matrices, :attr:`input` and one or more right-hand side vectors :attr:`other`.
If :attr:`input` is batched and :attr:`other` is not, then :attr:`other` is broadcast
to have the same batch dimensions as :attr:`input`.
The resulting tensor has the same shape as the (possibly broadcast) :attr:`other`.

Supports input of ``float``, ``double``, ``cfloat`` and ``cdouble`` dtypes.

.. note:: If :attr:`input` is a non-square or non-invertible matrix, or a batch containing non-square matrices
          or one or more non-invertible matrices, then a RuntimeError will be thrown.
.. note:: When given inputs on a CUDA device, this function synchronizes that device with the CPU.

Args:
    input (Tensor): the square :math:`n \times n` matrix or the batch
                    of such matrices of size :math:`(*, n, n)` where ``*`` is one or more batch dimensions.
    other (Tensor): right-hand side tensor of shape :math:`(*, n)` or :math:`(*, n, k)`,
                    where :math:`k` is the number of right-hand side vectors.

Keyword args:
    out (Tensor, optional): The output tensor. Ignored if ``None``. Default: ``None``

Examples::

    >>> A = torch.eye(3)
    >>> b = torch.randn(3)
    >>> x = torch.linalg.solve(A, b)
    >>> torch.allclose(A @ x, b)
    True

Batched input::

    >>> A = torch.randn(2, 3, 3)
    >>> b = torch.randn(3, 1)
    >>> x = torch.linalg.solve(A, b)
    >>> torch.allclose(A @ x, b)
    True
    >>> b = torch.rand(3) # b is broadcast internally to (*A.shape[:-2], 3)
    >>> x = torch.linalg.solve(A, b)
    >>> x.shape
    torch.Size([2, 3])
    >>> Ax = A @ x.unsqueeze(-1)
    >>> torch.allclose(Ax, b.unsqueeze(-1).expand_as(Ax))
    True
""")

tensorinv = _add_docstr(_linalg.linalg_tensorinv, r"""
linalg.tensorinv(input, ind=2, *, out=None) -> Tensor

Computes a tensor ``input_inv`` such that ``tensordot(input_inv, input, ind) == I_n`` (inverse tensor equation),
where ``I_n`` is the n-dimensional identity tensor and ``n`` is equal to ``input.ndim``.
The resulting tensor ``input_inv`` has shape equal to ``input.shape[ind:] + input.shape[:ind]``.

Supports input of ``float``, ``double``, ``cfloat`` and ``cdouble`` data types.

.. note:: If :attr:`input` is not invertible or does not satisfy the requirement
          ``prod(input.shape[ind:]) == prod(input.shape[:ind])``,
          then a RuntimeError will be thrown.

.. note:: When :attr:`input` is a 2-dimensional tensor and ``ind=1``, this function computes the
          (multiplicative) inverse of :attr:`input`, equivalent to calling :func:`torch.inverse`.

Args:
    input (Tensor): A tensor to invert. Its shape must satisfy ``prod(input.shape[:ind]) == prod(input.shape[ind:])``.
    ind (int): A positive integer that describes the inverse tensor equation. See :func:`torch.tensordot` for details. Default: 2.

Keyword args:
    out (Tensor, optional): The output tensor. Ignored if ``None``. Default: ``None``

Examples::

    >>> a = torch.eye(4 * 6).reshape((4, 6, 8, 3))
    >>> ainv = torch.linalg.tensorinv(a, ind=2)
    >>> ainv.shape
    torch.Size([8, 3, 4, 6])
    >>> b = torch.randn(4, 6)
    >>> torch.allclose(torch.tensordot(ainv, b), torch.linalg.tensorsolve(a, b))
    True

    >>> a = torch.randn(4, 4)
    >>> a_tensorinv = torch.linalg.tensorinv(a, ind=1)
    >>> a_inv = torch.inverse(a)
    >>> torch.allclose(a_tensorinv, a_inv)
    True
""")

tensorsolve = _add_docstr(_linalg.linalg_tensorsolve, r"""
linalg.tensorsolve(input, other, dims=None, *, out=None) -> Tensor

Computes a tensor ``x`` such that ``tensordot(input, x, dims=x.ndim) = other``.
The resulting tensor ``x`` has the same shape as ``input[other.ndim:]``.

Supports real-valued and complex-valued inputs.

.. note:: If :attr:`input` does not satisfy the requirement
          ``prod(input.shape[other.ndim:]) == prod(input.shape[:other.ndim])``
          after (optionally) moving the dimensions using :attr:`dims`, then a RuntimeError will be thrown.

Args:
    input (Tensor): "left-hand-side" tensor, it must satisfy the requirement
                    ``prod(input.shape[other.ndim:]) == prod(input.shape[:other.ndim])``.
    other (Tensor): "right-hand-side" tensor of shape ``input.shape[other.ndim]``.
    dims (Tuple[int]): dimensions of :attr:`input` to be moved before the computation.
                       Equivalent to calling ``input = movedim(input, dims, range(len(dims) - input.ndim, 0))``.
                       If None (default), no dimensions are moved.

Keyword args:
    out (Tensor, optional): The output tensor. Ignored if ``None``. Default: ``None``

Examples::

    >>> a = torch.eye(2 * 3 * 4).reshape((2 * 3, 4, 2, 3, 4))
    >>> b = torch.randn(2 * 3, 4)
    >>> x = torch.linalg.tensorsolve(a, b)
    >>> x.shape
    torch.Size([2, 3, 4])
    >>> torch.allclose(torch.tensordot(a, x, dims=x.ndim), b)
    True

    >>> a = torch.randn(6, 4, 4, 3, 2)
    >>> b = torch.randn(4, 3, 2)
    >>> x = torch.linalg.tensorsolve(a, b, dims=(0, 2))
    >>> x.shape
    torch.Size([6, 4])
    >>> a = a.permute(1, 3, 4, 0, 2)
    >>> a.shape[b.ndim:]
    torch.Size([6, 4])
    >>> torch.allclose(torch.tensordot(a, x, dims=x.ndim), b, atol=1e-6)
    True
""")


qr = _add_docstr(_linalg.linalg_qr, r"""
qr(input, mode='reduced', *, out=None) -> (Tensor, Tensor)

Computes the QR decomposition of a matrix or a batch of matrices :attr:`input`,
and returns a namedtuple (Q, R) of tensors such that :math:`\text{input} = Q R`
with :math:`Q` being an orthogonal matrix or batch of orthogonal matrices and
:math:`R` being an upper triangular matrix or batch of upper triangular matrices.

Depending on the value of :attr:`mode` this function returns the reduced or
complete QR factorization. See below for a list of valid modes.

.. note::  **Differences with** ``numpy.linalg.qr``:

           * ``mode='raw'`` is not implemented

           * unlike ``numpy.linalg.qr``, this function always returns a
             tuple of two tensors. When ``mode='r'``, the `Q` tensor is an
             empty tensor.

.. note::
          Backpropagation is not supported for ``mode='r'``. Use ``mode='reduced'`` instead.

          If you plan to backpropagate through QR, note that the current backward implementation
          is only well-defined when the first :math:`\min(input.size(-1), input.size(-2))`
          columns of :attr:`input` are linearly independent.
          This behavior may change in the future.

.. note:: This function uses LAPACK for CPU inputs and MAGMA for CUDA inputs,
          and may produce different (valid) decompositions on different device types
          and different platforms, depending on the precise version of the
          underlying library.

Args:
    input (Tensor): the input tensor of size :math:`(*, m, n)` where `*` is zero or more
                batch dimensions consisting of matrices of dimension :math:`m \times n`.
    mode (str, optional): if `k = min(m, n)` then:

          * ``'reduced'`` : returns `(Q, R)` with dimensions (m, k), (k, n) (default)

          * ``'complete'``: returns `(Q, R)` with dimensions (m, m), (m, n)

          * ``'r'``: computes only `R`; returns `(Q, R)` where `Q` is empty and `R` has dimensions (k, n)

Keyword args:
    out (tuple, optional): tuple of `Q` and `R` tensors
                satisfying :code:`input = torch.matmul(Q, R)`.
                The dimensions of `Q` and `R` are :math:`(*, m, k)` and :math:`(*, k, n)`
                respectively, where :math:`k = \min(m, n)` if :attr:`mode` is `'reduced'` and
                :math:`k = m` if :attr:`mode` is `'complete'`.

Example::

    >>> a = torch.tensor([[12., -51, 4], [6, 167, -68], [-4, 24, -41]])
    >>> q, r = torch.linalg.qr(a)
    >>> q
    tensor([[-0.8571,  0.3943,  0.3314],
            [-0.4286, -0.9029, -0.0343],
            [ 0.2857, -0.1714,  0.9429]])
    >>> r
    tensor([[ -14.0000,  -21.0000,   14.0000],
            [   0.0000, -175.0000,   70.0000],
            [   0.0000,    0.0000,  -35.0000]])
    >>> torch.mm(q, r).round()
    tensor([[  12.,  -51.,    4.],
            [   6.,  167.,  -68.],
            [  -4.,   24.,  -41.]])
    >>> torch.mm(q.t(), q).round()
    tensor([[ 1.,  0.,  0.],
            [ 0.,  1., -0.],
            [ 0., -0.,  1.]])
    >>> a = torch.randn(3, 4, 5)
    >>> q, r = torch.linalg.qr(a, mode='complete')
    >>> torch.allclose(torch.matmul(q, r), a)
    True
    >>> torch.allclose(torch.matmul(q.transpose(-2, -1), q), torch.eye(5))
    True
""")
