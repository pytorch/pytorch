import torch
from torch._C import _add_docstr, _linalg  # type: ignore[attr-defined]

LinAlgError = torch._C._LinAlgError  # type: ignore[attr-defined]

Tensor = torch.Tensor

common_notes = {
    "experimental_warning": """This function is "experimental" and it may change in a future PyTorch release.""",
    "sync_note": "When inputs are on a CUDA device, this function synchronizes that device with the CPU.",
    "sync_note_ex": r"When the inputs are on a CUDA device, this function synchronizes only when :attr:`check_errors`\ `= True`.",
    "sync_note_has_ex": ("When inputs are on a CUDA device, this function synchronizes that device with the CPU. "
                         "For a version of this function that does not synchronize, see :func:`{}`.")
}


# Note: This not only adds doc strings for functions in the linalg namespace, but
# also connects the torch.linalg Python namespace to the torch._C._linalg builtins.

cross = _add_docstr(_linalg.linalg_cross, r"""
linalg.cross(input, other, *, dim=-1, out=None) -> Tensor


Computes the cross product of two 3-dimensional vectors.

Supports input of float, double, cfloat and cdouble dtypes. Also supports batches
of vectors, for which it computes the product along the dimension :attr:`dim`.
It broadcasts over the batch dimensions.

Args:
    input (Tensor): the first input tensor.
    other (Tensor): the second input tensor.
    dim  (int, optional): the dimension along which to take the cross-product. Default: `-1`.

Keyword args:
    out (Tensor, optional): the output tensor. Ignored if `None`. Default: `None`.

Example:
    >>> a = torch.randn(4, 3)
    >>> a
    tensor([[-0.3956,  1.1455,  1.6895],
            [-0.5849,  1.3672,  0.3599],
            [-1.1626,  0.7180, -0.0521],
            [-0.1339,  0.9902, -2.0225]])
    >>> b = torch.randn(4, 3)
    >>> b
    tensor([[-0.0257, -1.4725, -1.2251],
            [-1.1479, -0.7005, -1.9757],
            [-1.3904,  0.3726, -1.1836],
            [-0.9688, -0.7153,  0.2159]])
    >>> torch.linalg.cross(a, b)
    tensor([[ 1.0844, -0.5281,  0.6120],
            [-2.4490, -1.5687,  1.9792],
            [-0.8304, -1.3037,  0.5650],
            [-1.2329,  1.9883,  1.0551]])
    >>> a = torch.randn(1, 3)  # a is broadcast to match shape of b
    >>> a
    tensor([[-0.9941, -0.5132,  0.5681]])
    >>> torch.linalg.cross(a, b)
    tensor([[ 1.4653, -1.2325,  1.4507],
            [ 1.4119, -2.6163,  0.1073],
            [ 0.3957, -1.9666, -1.0840],
            [ 0.2956, -0.3357,  0.2139]])
""")

cholesky = _add_docstr(_linalg.linalg_cholesky, r"""
linalg.cholesky(A, *, upper=False, out=None) -> Tensor

Computes the Cholesky decomposition of a complex Hermitian or real symmetric positive-definite matrix.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
the **Cholesky decomposition** of a complex Hermitian or real symmetric positive-definite matrix
:math:`A \in \mathbb{K}^{n \times n}` is defined as

.. math::

    A = LL^{\text{H}}\mathrlap{\qquad L \in \mathbb{K}^{n \times n}}

where :math:`L` is a lower triangular matrix with real positive diagonal (even in the complex case) and
:math:`L^{\text{H}}` is the conjugate transpose when :math:`L` is complex, and the transpose when :math:`L` is real-valued.

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

""" + fr"""
.. note:: {common_notes["sync_note_has_ex"].format("torch.linalg.cholesky_ex")}
""" + r"""

.. seealso::

        :func:`torch.linalg.cholesky_ex` for a version of this operation that
        skips the (slow) error checking by default and instead returns the debug
        information. This makes it a faster way to check if a matrix is
        positive-definite.

        :func:`torch.linalg.eigh` for a different decomposition of a Hermitian matrix.
        The eigenvalue decomposition gives more information about the matrix but it
        slower to compute than the Cholesky decomposition.

Args:
    A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
                consisting of symmetric or Hermitian positive-definite matrices.

Keyword args:
    upper (bool, optional): whether to return an upper triangular matrix.
        The tensor returned with upper=True is the conjugate transpose of the tensor
        returned with upper=False.
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Raises:
    RuntimeError: if the :attr:`A` matrix or any matrix in a batched :attr:`A` is not Hermitian
                  (resp. symmetric) positive-definite. If :attr:`A` is a batch of matrices,
                  the error message will include the batch index of the first matrix that fails
                  to meet this condition.

Examples::

    >>> A = torch.randn(2, 2, dtype=torch.complex128)
    >>> A = A @ A.T.conj() + torch.eye(2) # creates a Hermitian positive-definite matrix
    >>> A
    tensor([[2.5266+0.0000j, 1.9586-2.0626j],
            [1.9586+2.0626j, 9.4160+0.0000j]], dtype=torch.complex128)
    >>> L = torch.linalg.cholesky(A)
    >>> L
    tensor([[1.5895+0.0000j, 0.0000+0.0000j],
            [1.2322+1.2976j, 2.4928+0.0000j]], dtype=torch.complex128)
    >>> torch.dist(L @ L.T.conj(), A)
    tensor(4.4692e-16, dtype=torch.float64)

    >>> A = torch.randn(3, 2, 2, dtype=torch.float64)
    >>> A = A @ A.mT + torch.eye(2)  # batch of symmetric positive-definite matrices
    >>> L = torch.linalg.cholesky(A)
    >>> torch.dist(L @ L.mT, A)
    tensor(5.8747e-16, dtype=torch.float64)
""")

cholesky_ex = _add_docstr(_linalg.linalg_cholesky_ex, r"""
linalg.cholesky_ex(A, *, upper=False, check_errors=False, out=None) -> (Tensor, Tensor)

Computes the Cholesky decomposition of a complex Hermitian or real
symmetric positive-definite matrix.

This function skips the (slow) error checking and error message construction
of :func:`torch.linalg.cholesky`, instead directly returning the LAPACK
error codes as part of a named tuple ``(L, info)``. This makes this function
a faster way to check if a matrix is positive-definite, and it provides an
opportunity to handle decomposition errors more gracefully or performantly
than :func:`torch.linalg.cholesky` does.

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

If :attr:`A` is not a Hermitian positive-definite matrix, or if it's a batch of matrices
and one or more of them is not a Hermitian positive-definite matrix,
then ``info`` stores a positive integer for the corresponding matrix.
The positive integer indicates the order of the leading minor that is not positive-definite,
and the decomposition could not be completed.
``info`` filled with zeros indicates that the decomposition was successful.
If ``check_errors=True`` and ``info`` contains positive integers, then a RuntimeError is thrown.

""" + fr"""
.. note:: {common_notes["sync_note_ex"]}

.. warning:: {common_notes["experimental_warning"]}
""" + r"""

.. seealso::
        :func:`torch.linalg.cholesky` is a NumPy compatible variant that always checks for errors.

Args:
    A (Tensor): the Hermitian `n \times n` matrix or the batch of such matrices of size
                    `(*, n, n)` where `*` is one or more batch dimensions.

Keyword args:
    upper (bool, optional): whether to return an upper triangular matrix.
        The tensor returned with upper=True is the conjugate transpose of the tensor
        returned with upper=False.
    check_errors (bool, optional): controls whether to check the content of ``infos``. Default: `False`.
    out (tuple, optional): tuple of two tensors to write the output to. Ignored if `None`. Default: `None`.

Examples::

    >>> A = torch.randn(2, 2, dtype=torch.complex128)
    >>> A = A @ A.t().conj()  # creates a Hermitian positive-definite matrix
    >>> L, info = torch.linalg.cholesky_ex(A)
    >>> A
    tensor([[ 2.3792+0.0000j, -0.9023+0.9831j],
            [-0.9023-0.9831j,  0.8757+0.0000j]], dtype=torch.complex128)
    >>> L
    tensor([[ 1.5425+0.0000j,  0.0000+0.0000j],
            [-0.5850-0.6374j,  0.3567+0.0000j]], dtype=torch.complex128)
    >>> info
    tensor(0, dtype=torch.int32)

""")

inv = _add_docstr(_linalg.linalg_inv, r"""
linalg.inv(A, *, out=None) -> Tensor

Computes the inverse of a square matrix if it exists.
Throws a `RuntimeError` if the matrix is not invertible.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
for a matrix :math:`A \in \mathbb{K}^{n \times n}`,
its **inverse matrix** :math:`A^{-1} \in \mathbb{K}^{n \times n}` (if it exists) is defined as

.. math::

    A^{-1}A = AA^{-1} = \mathrm{I}_n

where :math:`\mathrm{I}_n` is the `n`-dimensional identity matrix.

The inverse matrix exists if and only if :math:`A` is `invertible`_. In this case,
the inverse is unique.

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices
then the output has the same batch dimensions.

""" + fr"""
.. note:: {common_notes["sync_note_has_ex"].format("torch.linalg.inv_ex")}
""" + r"""

.. note::
    Consider using :func:`torch.linalg.solve` if possible for multiplying a matrix on the left by
    the inverse, as::

        linalg.solve(A, B) == linalg.inv(A) @ B  # When B is a matrix

    It is always preferred to use :func:`~solve` when possible, as it is faster and more
    numerically stable than computing the inverse explicitly.

.. seealso::

        :func:`torch.linalg.pinv` computes the pseudoinverse (Moore-Penrose inverse) of matrices
        of any shape.

        :func:`torch.linalg.solve` computes :attr:`A`\ `.inv() @ \ `:attr:`B` with a
        numerically stable algorithm.

Args:
    A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
                consisting of invertible matrices.

Keyword args:
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Raises:
    RuntimeError: if the matrix :attr:`A` or any matrix in the batch of matrices :attr:`A` is not invertible.

Examples::

    >>> A = torch.randn(4, 4)
    >>> Ainv = torch.linalg.inv(A)
    >>> torch.dist(A @ Ainv, torch.eye(4))
    tensor(1.1921e-07)

    >>> A = torch.randn(2, 3, 4, 4)  # Batch of matrices
    >>> Ainv = torch.linalg.inv(A)
    >>> torch.dist(A @ Ainv, torch.eye(4))
    tensor(1.9073e-06)

    >>> A = torch.randn(4, 4, dtype=torch.complex128)  # Complex matrix
    >>> Ainv = torch.linalg.inv(A)
    >>> torch.dist(A @ Ainv, torch.eye(4))
    tensor(7.5107e-16, dtype=torch.float64)

.. _invertible:
    https://en.wikipedia.org/wiki/Invertible_matrix#The_invertible_matrix_theorem
""")

solve_ex = _add_docstr(_linalg.linalg_solve_ex, r"""
linalg.solve_ex(A, B, *, left=True, check_errors=False, out=None) -> (Tensor, Tensor)

A version of :func:`~solve` that does not perform error checks unless :attr:`check_errors`\ `= True`.
It also returns the :attr:`info` tensor returned by `LAPACK's getrf`_.

""" + fr"""
.. note:: {common_notes["sync_note_ex"]}

.. warning:: {common_notes["experimental_warning"]}
""" + r"""

Args:
    A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions.

Keyword args:
    left (bool, optional): whether to solve the system :math:`AX=B` or :math:`XA = B`. Default: `True`.
    check_errors (bool, optional): controls whether to check the content of ``infos`` and raise
                                   an error if it is non-zero. Default: `False`.
    out (tuple, optional): tuple of two tensors to write the output to. Ignored if `None`. Default: `None`.

Returns:
    A named tuple `(result, info)`.

Examples::

    >>> A = torch.randn(3, 3)
    >>> Ainv, info = torch.linalg.solve_ex(A)
    >>> torch.dist(torch.linalg.inv(A), Ainv)
    tensor(0.)
    >>> info
    tensor(0, dtype=torch.int32)

.. _LAPACK's getrf:
    https://www.netlib.org/lapack/explore-html/dd/d9a/group__double_g_ecomputational_ga0019443faea08275ca60a734d0593e60.html
""")

inv_ex = _add_docstr(_linalg.linalg_inv_ex, r"""
linalg.inv_ex(A, *, check_errors=False, out=None) -> (Tensor, Tensor)

Computes the inverse of a square matrix if it is invertible.

Returns a namedtuple ``(inverse, info)``. ``inverse`` contains the result of
inverting :attr:`A` and ``info`` stores the LAPACK error codes.

If :attr:`A` is not an invertible matrix, or if it's a batch of matrices
and one or more of them is not an invertible matrix,
then ``info`` stores a positive integer for the corresponding matrix.
The positive integer indicates the diagonal element of the LU decomposition of
the input matrix that is exactly zero.
``info`` filled with zeros indicates that the inversion was successful.
If ``check_errors=True`` and ``info`` contains positive integers, then a RuntimeError is thrown.

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

""" + fr"""
.. note:: {common_notes["sync_note_ex"]}

.. warning:: {common_notes["experimental_warning"]}
""" + r"""

.. seealso::

        :func:`torch.linalg.inv` is a NumPy compatible variant that always checks for errors.

Args:
    A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
                    consisting of square matrices.
    check_errors (bool, optional): controls whether to check the content of ``info``. Default: `False`.

Keyword args:
    out (tuple, optional): tuple of two tensors to write the output to. Ignored if `None`. Default: `None`.

Examples::

    >>> A = torch.randn(3, 3)
    >>> Ainv, info = torch.linalg.inv_ex(A)
    >>> torch.dist(torch.linalg.inv(A), Ainv)
    tensor(0.)
    >>> info
    tensor(0, dtype=torch.int32)

""")

det = _add_docstr(_linalg.linalg_det, r"""
linalg.det(A, *, out=None) -> Tensor

Computes the determinant of a square matrix.

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

.. seealso::

        :func:`torch.linalg.slogdet` computes the sign and natural logarithm of the absolute
        value of the determinant of square matrices.

Args:
    A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions.

Keyword args:
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Examples::

    >>> A = torch.randn(3, 3)
    >>> torch.linalg.det(A)
    tensor(0.0934)

    >>> A = torch.randn(3, 2, 2)
    >>> torch.linalg.det(A)
    tensor([1.1990, 0.4099, 0.7386])
""")

slogdet = _add_docstr(_linalg.linalg_slogdet, r"""
linalg.slogdet(A, *, out=None) -> (Tensor, Tensor)

Computes the sign and natural logarithm of the absolute value of the determinant of a square matrix.

For complex :attr:`A`, it returns the sign and the natural logarithm of the modulus of the
determinant, that is, a logarithmic polar decomposition of the determinant.

The determinant can be recovered as `sign * exp(logabsdet)`.
When a matrix has a determinant of zero, it returns `(0, -inf)`.

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

.. seealso::

        :func:`torch.linalg.det` computes the determinant of square matrices.

Args:
    A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions.

Keyword args:
    out (tuple, optional): output tuple of two tensors. Ignored if `None`. Default: `None`.

Returns:
    A named tuple `(sign, logabsdet)`.

    `sign` will have the same dtype as :attr:`A`.

    `logabsdet` will always be real-valued, even when :attr:`A` is complex.

Examples::

    >>> A = torch.randn(3, 3)
    >>> A
    tensor([[ 0.0032, -0.2239, -1.1219],
            [-0.6690,  0.1161,  0.4053],
            [-1.6218, -0.9273, -0.0082]])
    >>> torch.linalg.det(A)
    tensor(-0.7576)
    >>> torch.logdet(A)
    tensor(nan)
    >>> torch.linalg.slogdet(A)
    torch.return_types.linalg_slogdet(sign=tensor(-1.), logabsdet=tensor(-0.2776))
""")

eig = _add_docstr(_linalg.linalg_eig, r"""
linalg.eig(A, *, out=None) -> (Tensor, Tensor)

Computes the eigenvalue decomposition of a square matrix if it exists.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
the **eigenvalue decomposition** of a square matrix
:math:`A \in \mathbb{K}^{n \times n}` (if it exists) is defined as

.. math::

    A = V \operatorname{diag}(\Lambda) V^{-1}\mathrlap{\qquad V \in \mathbb{C}^{n \times n}, \Lambda \in \mathbb{C}^n}

This decomposition exists if and only if :math:`A` is `diagonalizable`_.
This is the case when all its eigenvalues are different.

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

The returned eigenvalues are not guaranteed to be in any specific order.

.. note:: The eigenvalues and eigenvectors of a real matrix may be complex.

""" + fr"""
.. note:: {common_notes["sync_note"]}
""" + r"""

.. warning:: This function assumes that :attr:`A` is `diagonalizable`_ (for example, when all the
             eigenvalues are different). If it is not diagonalizable, the returned
             eigenvalues will be correct but :math:`A \neq V \operatorname{diag}(\Lambda)V^{-1}`.

.. warning:: The returned eigenvectors are normalized to have norm `1`.
             Even then, the eigenvectors of a matrix are not unique, nor are they continuous with respect to
             :attr:`A`. Due to this lack of uniqueness, different hardware and software may compute
             different eigenvectors.

             This non-uniqueness is caused by the fact that multiplying an eigenvector by
             by :math:`e^{i \phi}, \phi \in \mathbb{R}` produces another set of valid eigenvectors
             of the matrix.  For this reason, the loss function shall not depend on the phase of the
             eigenvectors, as this quantity is not well-defined.
             This is checked when computing the gradients of this function. As such,
             when inputs are on a CUDA device, the computation of the gradients
             of this function synchronizes that device with the CPU.


.. warning:: Gradients computed using the `eigenvectors` tensor will only be finite when
             :attr:`A` has distinct eigenvalues.
             Furthermore, if the distance between any two eigenvalues is close to zero,
             the gradient will be numerically unstable, as it depends on the eigenvalues
             :math:`\lambda_i` through the computation of
             :math:`\frac{1}{\min_{i \neq j} \lambda_i - \lambda_j}`.

.. seealso::

        :func:`torch.linalg.eigvals` computes only the eigenvalues.
        Unlike :func:`torch.linalg.eig`, the gradients of :func:`~eigvals` are always
        numerically stable.

        :func:`torch.linalg.eigh` for a (faster) function that computes the eigenvalue decomposition
        for Hermitian and symmetric matrices.

        :func:`torch.linalg.svd` for a function that computes another type of spectral
        decomposition that works on matrices of any shape.

        :func:`torch.linalg.qr` for another (much faster) decomposition that works on matrices of
        any shape.

Args:
    A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
                consisting of diagonalizable matrices.

Keyword args:
    out (tuple, optional): output tuple of two tensors. Ignored if `None`. Default: `None`.

Returns:
    A named tuple `(eigenvalues, eigenvectors)` which corresponds to :math:`\Lambda` and :math:`V` above.

    `eigenvalues` and `eigenvectors` will always be complex-valued, even when :attr:`A` is real. The eigenvectors
    will be given by the columns of `eigenvectors`.

Examples::

    >>> A = torch.randn(2, 2, dtype=torch.complex128)
    >>> A
    tensor([[ 0.9828+0.3889j, -0.4617+0.3010j],
            [ 0.1662-0.7435j, -0.6139+0.0562j]], dtype=torch.complex128)
    >>> L, V = torch.linalg.eig(A)
    >>> L
    tensor([ 1.1226+0.5738j, -0.7537-0.1286j], dtype=torch.complex128)
    >>> V
    tensor([[ 0.9218+0.0000j,  0.1882-0.2220j],
            [-0.0270-0.3867j,  0.9567+0.0000j]], dtype=torch.complex128)
    >>> torch.dist(V @ torch.diag(L) @ torch.linalg.inv(V), A)
    tensor(7.7119e-16, dtype=torch.float64)

    >>> A = torch.randn(3, 2, 2, dtype=torch.float64)
    >>> L, V = torch.linalg.eig(A)
    >>> torch.dist(V @ torch.diag_embed(L) @ torch.linalg.inv(V), A)
    tensor(3.2841e-16, dtype=torch.float64)

.. _diagonalizable:
    https://en.wikipedia.org/wiki/Diagonalizable_matrix#Definition
""")

eigvals = _add_docstr(_linalg.linalg_eigvals, r"""
linalg.eigvals(A, *, out=None) -> Tensor

Computes the eigenvalues of a square matrix.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
the **eigenvalues** of a square matrix :math:`A \in \mathbb{K}^{n \times n}` are defined
as the roots (counted with multiplicity) of the polynomial `p` of degree `n` given by

.. math::

    p(\lambda) = \operatorname{det}(A - \lambda \mathrm{I}_n)\mathrlap{\qquad \lambda \in \mathbb{C}}

where :math:`\mathrm{I}_n` is the `n`-dimensional identity matrix.

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

The returned eigenvalues are not guaranteed to be in any specific order.

.. note:: The eigenvalues of a real matrix may be complex, as the roots of a real polynomial may be complex.

          The eigenvalues of a matrix are always well-defined, even when the matrix is not diagonalizable.

""" + fr"""
.. note:: {common_notes["sync_note"]}
""" + r"""

.. seealso::

        :func:`torch.linalg.eig` computes the full eigenvalue decomposition.

Args:
    A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions.

Keyword args:
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Returns:
    A complex-valued tensor containing the eigenvalues even when :attr:`A` is real.

Examples::

    >>> A = torch.randn(2, 2, dtype=torch.complex128)
    >>> L = torch.linalg.eigvals(A)
    >>> L
    tensor([ 1.1226+0.5738j, -0.7537-0.1286j], dtype=torch.complex128)

    >>> torch.dist(L, torch.linalg.eig(A).eigenvalues)
    tensor(2.4576e-07)
""")

eigh = _add_docstr(_linalg.linalg_eigh, r"""
linalg.eigh(A, UPLO='L', *, out=None) -> (Tensor, Tensor)

Computes the eigenvalue decomposition of a complex Hermitian or real symmetric matrix.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
the **eigenvalue decomposition** of a complex Hermitian or real symmetric matrix
:math:`A \in \mathbb{K}^{n \times n}` is defined as

.. math::

    A = Q \operatorname{diag}(\Lambda) Q^{\text{H}}\mathrlap{\qquad Q \in \mathbb{K}^{n \times n}, \Lambda \in \mathbb{R}^n}

where :math:`Q^{\text{H}}` is the conjugate transpose when :math:`Q` is complex, and the transpose when :math:`Q` is real-valued.
:math:`Q` is orthogonal in the real case and unitary in the complex case.

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

:attr:`A` is assumed to be Hermitian (resp. symmetric), but this is not checked internally, instead:

- If :attr:`UPLO`\ `= 'L'` (default), only the lower triangular part of the matrix is used in the computation.
- If :attr:`UPLO`\ `= 'U'`, only the upper triangular part of the matrix is used.

The eigenvalues are returned in ascending order.

""" + fr"""
.. note:: {common_notes["sync_note"]}
""" + r"""

.. note:: The eigenvalues of real symmetric or complex Hermitian matrices are always real.

.. warning:: The eigenvectors of a symmetric matrix are not unique, nor are they continuous with
             respect to :attr:`A`. Due to this lack of uniqueness, different hardware and
             software may compute different eigenvectors.

             This non-uniqueness is caused by the fact that multiplying an eigenvector by
             `-1` in the real case or by :math:`e^{i \phi}, \phi \in \mathbb{R}` in the complex
             case produces another set of valid eigenvectors of the matrix.
             For this reason, the loss function shall not depend on the phase of the eigenvectors, as
             this quantity is not well-defined.
             This is checked for complex inputs when computing the gradients of this function. As such,
             when inputs are complex and are on a CUDA device, the computation of the gradients
             of this function synchronizes that device with the CPU.

.. warning:: Gradients computed using the `eigenvectors` tensor will only be finite when
             :attr:`A` has distinct eigenvalues.
             Furthermore, if the distance between any two eigenvalues is close to zero,
             the gradient will be numerically unstable, as it depends on the eigenvalues
             :math:`\lambda_i` through the computation of
             :math:`\frac{1}{\min_{i \neq j} \lambda_i - \lambda_j}`.

.. warning:: User may see pytorch crashes if running `eigh` on CUDA devices with CUDA versions before 12.1 update 1
             with large ill-conditioned matrices as inputs.
             Refer to :ref:`Linear Algebra Numerical Stability<Linear Algebra Stability>` for more details.
             If this is the case, user may (1) tune their matrix inputs to be less ill-conditioned,
             or (2) use :func:`torch.backends.cuda.preferred_linalg_library` to
             try other supported backends.

.. seealso::

        :func:`torch.linalg.eigvalsh` computes only the eigenvalues of a Hermitian matrix.
        Unlike :func:`torch.linalg.eigh`, the gradients of :func:`~eigvalsh` are always
        numerically stable.

        :func:`torch.linalg.cholesky` for a different decomposition of a Hermitian matrix.
        The Cholesky decomposition gives less information about the matrix but is much faster
        to compute than the eigenvalue decomposition.

        :func:`torch.linalg.eig` for a (slower) function that computes the eigenvalue decomposition
        of a not necessarily Hermitian square matrix.

        :func:`torch.linalg.svd` for a (slower) function that computes the more general SVD
        decomposition of matrices of any shape.

        :func:`torch.linalg.qr` for another (much faster) decomposition that works on general
        matrices.

Args:
    A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
                consisting of symmetric or Hermitian matrices.
    UPLO ('L', 'U', optional): controls whether to use the upper or lower triangular part
                               of :attr:`A` in the computations. Default: `'L'`.

Keyword args:
    out (tuple, optional): output tuple of two tensors. Ignored if `None`. Default: `None`.

Returns:
    A named tuple `(eigenvalues, eigenvectors)` which corresponds to :math:`\Lambda` and :math:`Q` above.

    `eigenvalues` will always be real-valued, even when :attr:`A` is complex.
    It will also be ordered in ascending order.

    `eigenvectors` will have the same dtype as :attr:`A` and will contain the eigenvectors as its columns.

Examples::
    >>> A = torch.randn(2, 2, dtype=torch.complex128)
    >>> A = A + A.T.conj()  # creates a Hermitian matrix
    >>> A
    tensor([[2.9228+0.0000j, 0.2029-0.0862j],
            [0.2029+0.0862j, 0.3464+0.0000j]], dtype=torch.complex128)
    >>> L, Q = torch.linalg.eigh(A)
    >>> L
    tensor([0.3277, 2.9415], dtype=torch.float64)
    >>> Q
    tensor([[-0.0846+-0.0000j, -0.9964+0.0000j],
            [ 0.9170+0.3898j, -0.0779-0.0331j]], dtype=torch.complex128)
    >>> torch.dist(Q @ torch.diag(L.cdouble()) @ Q.T.conj(), A)
    tensor(6.1062e-16, dtype=torch.float64)

    >>> A = torch.randn(3, 2, 2, dtype=torch.float64)
    >>> A = A + A.mT  # creates a batch of symmetric matrices
    >>> L, Q = torch.linalg.eigh(A)
    >>> torch.dist(Q @ torch.diag_embed(L) @ Q.mH, A)
    tensor(1.5423e-15, dtype=torch.float64)
""")

eigvalsh = _add_docstr(_linalg.linalg_eigvalsh, r"""
linalg.eigvalsh(A, UPLO='L', *, out=None) -> Tensor

Computes the eigenvalues of a complex Hermitian or real symmetric matrix.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
the **eigenvalues** of a complex Hermitian or real symmetric  matrix :math:`A \in \mathbb{K}^{n \times n}`
are defined as the roots (counted with multiplicity) of the polynomial `p` of degree `n` given by

.. math::

    p(\lambda) = \operatorname{det}(A - \lambda \mathrm{I}_n)\mathrlap{\qquad \lambda \in \mathbb{R}}

where :math:`\mathrm{I}_n` is the `n`-dimensional identity matrix.
The eigenvalues of a real symmetric or complex Hermitian matrix are always real.

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

The eigenvalues are returned in ascending order.

:attr:`A` is assumed to be Hermitian (resp. symmetric), but this is not checked internally, instead:

- If :attr:`UPLO`\ `= 'L'` (default), only the lower triangular part of the matrix is used in the computation.
- If :attr:`UPLO`\ `= 'U'`, only the upper triangular part of the matrix is used.

""" + fr"""
.. note:: {common_notes["sync_note"]}
""" + r"""

.. seealso::

        :func:`torch.linalg.eigh` computes the full eigenvalue decomposition.

Args:
    A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
                consisting of symmetric or Hermitian matrices.
    UPLO ('L', 'U', optional): controls whether to use the upper or lower triangular part
                               of :attr:`A` in the computations. Default: `'L'`.

Keyword args:
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Returns:
    A real-valued tensor containing the eigenvalues even when :attr:`A` is complex.
    The eigenvalues are returned in ascending order.

Examples::

    >>> A = torch.randn(2, 2, dtype=torch.complex128)
    >>> A = A + A.T.conj()  # creates a Hermitian matrix
    >>> A
    tensor([[2.9228+0.0000j, 0.2029-0.0862j],
            [0.2029+0.0862j, 0.3464+0.0000j]], dtype=torch.complex128)
    >>> torch.linalg.eigvalsh(A)
    tensor([0.3277, 2.9415], dtype=torch.float64)

    >>> A = torch.randn(3, 2, 2, dtype=torch.float64)
    >>> A = A + A.mT  # creates a batch of symmetric matrices
    >>> torch.linalg.eigvalsh(A)
    tensor([[ 2.5797,  3.4629],
            [-4.1605,  1.3780],
            [-3.1113,  2.7381]], dtype=torch.float64)
""")

householder_product = _add_docstr(_linalg.linalg_householder_product, r"""
householder_product(A, tau, *, out=None) -> Tensor

Computes the first `n` columns of a product of Householder matrices.

Let :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`, and
let :math:`A \in \mathbb{K}^{m \times n}` be a matrix with columns :math:`a_i \in \mathbb{K}^m`
for :math:`i=1,\ldots,m` with :math:`m \geq n`. Denote by :math:`b_i` the vector resulting from
zeroing out the first :math:`i-1` components of :math:`a_i` and setting to `1` the :math:`i`-th.
For a vector :math:`\tau \in \mathbb{K}^k` with :math:`k \leq n`, this function computes the
first :math:`n` columns of the matrix

.. math::

    H_1H_2 ... H_k \qquad\text{with}\qquad H_i = \mathrm{I}_m - \tau_i b_i b_i^{\text{H}}

where :math:`\mathrm{I}_m` is the `m`-dimensional identity matrix and :math:`b^{\text{H}}` is the
conjugate transpose when :math:`b` is complex, and the transpose when :math:`b` is real-valued.
The output matrix is the same size as the input matrix :attr:`A`.

See `Representation of Orthogonal or Unitary Matrices`_ for further details.

Supports inputs of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if the inputs are batches of matrices then
the output has the same batch dimensions.

.. seealso::

        :func:`torch.geqrf` can be used together with this function to form the `Q` from the
        :func:`~qr` decomposition.

        :func:`torch.ormqr` is a related function that computes the matrix multiplication
        of a product of Householder matrices with another matrix.
        However, that function is not supported by autograd.

.. warning::
    Gradient computations are only well-defined if :math:`\tau_i \neq \frac{1}{||a_i||^2}`.
    If this condition is not met, no error will be thrown, but the gradient produced may contain `NaN`.

Args:
    A (Tensor): tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.
    tau (Tensor): tensor of shape `(*, k)` where `*` is zero or more batch dimensions.

Keyword args:
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Raises:
    RuntimeError: if :attr:`A` doesn't satisfy the requirement `m >= n`,
                  or :attr:`tau` doesn't satisfy the requirement `n >= k`.

Examples::

    >>> A = torch.randn(2, 2)
    >>> h, tau = torch.geqrf(A)
    >>> Q = torch.linalg.householder_product(h, tau)
    >>> torch.dist(Q, torch.linalg.qr(A).Q)
    tensor(0.)

    >>> h = torch.randn(3, 2, 2, dtype=torch.complex128)
    >>> tau = torch.randn(3, 1, dtype=torch.complex128)
    >>> Q = torch.linalg.householder_product(h, tau)
    >>> Q
    tensor([[[ 1.8034+0.4184j,  0.2588-1.0174j],
            [-0.6853+0.7953j,  2.0790+0.5620j]],

            [[ 1.4581+1.6989j, -1.5360+0.1193j],
            [ 1.3877-0.6691j,  1.3512+1.3024j]],

            [[ 1.4766+0.5783j,  0.0361+0.6587j],
            [ 0.6396+0.1612j,  1.3693+0.4481j]]], dtype=torch.complex128)

.. _Representation of Orthogonal or Unitary Matrices:
    https://www.netlib.org/lapack/lug/node128.html
""")

ldl_factor = _add_docstr(_linalg.linalg_ldl_factor, r"""
linalg.ldl_factor(A, *, hermitian=False, out=None) -> (Tensor, Tensor)

Computes a compact representation of the LDL factorization of a Hermitian or symmetric (possibly indefinite) matrix.

When :attr:`A` is complex valued it can be Hermitian (:attr:`hermitian`\ `= True`)
or symmetric (:attr:`hermitian`\ `= False`).

The factorization is of the form the form :math:`A = L D L^T`.
If :attr:`hermitian` is `True` then transpose operation is the conjugate transpose.

:math:`L` (or :math:`U`) and :math:`D` are stored in compact form in ``LD``.
They follow the format specified by `LAPACK's sytrf`_ function.
These tensors may be used in :func:`torch.linalg.ldl_solve` to solve linear systems.

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

""" + fr"""
.. note:: {common_notes["sync_note_has_ex"].format("torch.linalg.ldl_factor_ex")}
""" + r"""

Args:
    A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
                consisting of symmetric or Hermitian matrices.

Keyword args:
    hermitian (bool, optional): whether to consider the input to be Hermitian or symmetric.
                                For real-valued matrices, this switch has no effect. Default: `False`.
    out (tuple, optional): tuple of two tensors to write the output to. Ignored if `None`. Default: `None`.

Returns:
    A named tuple `(LD, pivots)`.

Examples::

    >>> A = torch.randn(3, 3)
    >>> A = A @ A.mT # make symmetric
    >>> A
    tensor([[7.2079, 4.2414, 1.9428],
            [4.2414, 3.4554, 0.3264],
            [1.9428, 0.3264, 1.3823]])
    >>> LD, pivots = torch.linalg.ldl_factor(A)
    >>> LD
    tensor([[ 7.2079,  0.0000,  0.0000],
            [ 0.5884,  0.9595,  0.0000],
            [ 0.2695, -0.8513,  0.1633]])
    >>> pivots
    tensor([1, 2, 3], dtype=torch.int32)

.. _LAPACK's sytrf:
    https://www.netlib.org/lapack/explore-html/d3/db6/group__double_s_ycomputational_gad91bde1212277b3e909eb6af7f64858a.html
""")

ldl_factor_ex = _add_docstr(_linalg.linalg_ldl_factor_ex, r"""
linalg.ldl_factor_ex(A, *, hermitian=False, check_errors=False, out=None) -> (Tensor, Tensor, Tensor)

This is a version of :func:`~ldl_factor` that does not perform error checks unless :attr:`check_errors`\ `= True`.
It also returns the :attr:`info` tensor returned by `LAPACK's sytrf`_.
``info`` stores integer error codes from the backend library.
A positive integer indicates the diagonal element of :math:`D` that is zero.
Division by 0 will occur if the result is used for solving a system of linear equations.
``info`` filled with zeros indicates that the factorization was successful.
If ``check_errors=True`` and ``info`` contains positive integers, then a `RuntimeError` is thrown.

""" + fr"""
.. note:: {common_notes["sync_note_ex"]}

.. warning:: {common_notes["experimental_warning"]}
""" + r"""

Args:
    A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
                consisting of symmetric or Hermitian matrices.

Keyword args:
    hermitian (bool, optional): whether to consider the input to be Hermitian or symmetric.
                                For real-valued matrices, this switch has no effect. Default: `False`.
    check_errors (bool, optional): controls whether to check the content of ``info`` and raise
                                   an error if it is non-zero. Default: `False`.
    out (tuple, optional): tuple of three tensors to write the output to. Ignored if `None`. Default: `None`.

Returns:
    A named tuple `(LD, pivots, info)`.

Examples::

    >>> A = torch.randn(3, 3)
    >>> A = A @ A.mT # make symmetric
    >>> A
    tensor([[7.2079, 4.2414, 1.9428],
            [4.2414, 3.4554, 0.3264],
            [1.9428, 0.3264, 1.3823]])
    >>> LD, pivots, info = torch.linalg.ldl_factor_ex(A)
    >>> LD
    tensor([[ 7.2079,  0.0000,  0.0000],
            [ 0.5884,  0.9595,  0.0000],
            [ 0.2695, -0.8513,  0.1633]])
    >>> pivots
    tensor([1, 2, 3], dtype=torch.int32)
    >>> info
    tensor(0, dtype=torch.int32)

.. _LAPACK's sytrf:
    https://www.netlib.org/lapack/explore-html/d3/db6/group__double_s_ycomputational_gad91bde1212277b3e909eb6af7f64858a.html
""")

ldl_solve = _add_docstr(_linalg.linalg_ldl_solve, r"""
linalg.ldl_solve(LD, pivots, B, *, hermitian=False, out=None) -> Tensor

Computes the solution of a system of linear equations using the LDL factorization.

:attr:`LD` and :attr:`pivots` are the compact representation of the LDL factorization and
are expected to be computed by :func:`torch.linalg.ldl_factor_ex`.
:attr:`hermitian` argument to this function should be the same
as the corresponding arguments in :func:`torch.linalg.ldl_factor_ex`.

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

""" + fr"""
.. warning:: {common_notes["experimental_warning"]}
""" + r"""

Args:
    LD (Tensor): the `n \times n` matrix or the batch of such matrices of size
                      `(*, n, n)` where `*` is one or more batch dimensions.
    pivots (Tensor): the pivots corresponding to the LDL factorization of :attr:`LD`.
    B (Tensor): right-hand side tensor of shape `(*, n, k)`.

Keyword args:
    hermitian (bool, optional): whether to consider the decomposed matrix to be Hermitian or symmetric.
                                For real-valued matrices, this switch has no effect. Default: `False`.
    out (tuple, optional): output tensor. `B` may be passed as `out` and the result is computed in-place on `B`.
                           Ignored if `None`. Default: `None`.

Examples::

    >>> A = torch.randn(2, 3, 3)
    >>> A = A @ A.mT # make symmetric
    >>> LD, pivots, info = torch.linalg.ldl_factor_ex(A)
    >>> B = torch.randn(2, 3, 4)
    >>> X = torch.linalg.ldl_solve(LD, pivots, B)
    >>> torch.linalg.norm(A @ X - B)
    >>> tensor(0.0001)
""")

lstsq = _add_docstr(_linalg.linalg_lstsq, r"""
torch.linalg.lstsq(A, B, rcond=None, *, driver=None) -> (Tensor, Tensor, Tensor, Tensor)

Computes a solution to the least squares problem of a system of linear equations.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
the **least squares problem** for a linear system :math:`AX = B` with
:math:`A \in \mathbb{K}^{m \times n}, B \in \mathbb{K}^{m \times k}` is defined as

.. math::

    \min_{X \in \mathbb{K}^{n \times k}} \|AX - B\|_F

where :math:`\|-\|_F` denotes the Frobenius norm.

Supports inputs of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if the inputs are batches of matrices then
the output has the same batch dimensions.

:attr:`driver` chooses the backend function that will be used.
For CPU inputs the valid values are `'gels'`, `'gelsy'`, `'gelsd`, `'gelss'`.
To choose the best driver on CPU consider:

- If :attr:`A` is well-conditioned (its `condition number`_ is not too large), or you do not mind some precision loss.

  - For a general matrix: `'gelsy'` (QR with pivoting) (default)
  - If :attr:`A` is full-rank: `'gels'` (QR)

- If :attr:`A` is not well-conditioned.

  - `'gelsd'` (tridiagonal reduction and SVD)
  - But if you run into memory issues: `'gelss'` (full SVD).

For CUDA input, the only valid driver is `'gels'`, which assumes that :attr:`A` is full-rank.

See also the `full description of these drivers`_

:attr:`rcond` is used to determine the effective rank of the matrices in :attr:`A`
when :attr:`driver` is one of (`'gelsy'`, `'gelsd'`, `'gelss'`).
In this case, if :math:`\sigma_i` are the singular values of `A` in decreasing order,
:math:`\sigma_i` will be rounded down to zero if :math:`\sigma_i \leq \text{rcond} \cdot \sigma_1`.
If :attr:`rcond`\ `= None` (default), :attr:`rcond` is set to the machine precision of the dtype of :attr:`A` times `max(m, n)`.

This function returns the solution to the problem and some extra information in a named tuple of
four tensors `(solution, residuals, rank, singular_values)`. For inputs :attr:`A`, :attr:`B`
of shape `(*, m, n)`, `(*, m, k)` respectively, it contains

- `solution`: the least squares solution. It has shape `(*, n, k)`.
- `residuals`: the squared residuals of the solutions, that is, :math:`\|AX - B\|_F^2`.
  It has shape equal to the batch dimensions of :attr:`A`.
  It is computed when `m > n` and every matrix in :attr:`A` is full-rank,
  otherwise, it is an empty tensor.
  If :attr:`A` is a batch of matrices and any matrix in the batch is not full rank,
  then an empty tensor is returned. This behavior may change in a future PyTorch release.
- `rank`: tensor of ranks of the matrices in :attr:`A`.
  It has shape equal to the batch dimensions of :attr:`A`.
  It is computed when :attr:`driver` is one of (`'gelsy'`, `'gelsd'`, `'gelss'`),
  otherwise it is an empty tensor.
- `singular_values`: tensor of singular values of the matrices in :attr:`A`.
  It has shape `(*, min(m, n))`.
  It is computed when :attr:`driver` is one of (`'gelsd'`, `'gelss'`),
  otherwise it is an empty tensor.

.. note::
    This function computes `X = \ `:attr:`A`\ `.pinverse() @ \ `:attr:`B` in a faster and
    more numerically stable way than performing the computations separately.

.. warning::
    The default value of :attr:`rcond` may change in a future PyTorch release.
    It is therefore recommended to use a fixed value to avoid potential
    breaking changes.

Args:
    A (Tensor): lhs tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.
    B (Tensor): rhs tensor of shape `(*, m, k)` where `*` is zero or more batch dimensions.
    rcond (float, optional): used to determine the effective rank of :attr:`A`.
                             If :attr:`rcond`\ `= None`, :attr:`rcond` is set to the machine
                             precision of the dtype of :attr:`A` times `max(m, n)`. Default: `None`.

Keyword args:
    driver (str, optional): name of the LAPACK/MAGMA method to be used.
        If `None`, `'gelsy'` is used for CPU inputs and `'gels'` for CUDA inputs.
        Default: `None`.

Returns:
    A named tuple `(solution, residuals, rank, singular_values)`.

Examples::

    >>> A = torch.randn(1,3,3)
    >>> A
    tensor([[[-1.0838,  0.0225,  0.2275],
         [ 0.2438,  0.3844,  0.5499],
         [ 0.1175, -0.9102,  2.0870]]])
    >>> B = torch.randn(2,3,3)
    >>> B
    tensor([[[-0.6772,  0.7758,  0.5109],
         [-1.4382,  1.3769,  1.1818],
         [-0.3450,  0.0806,  0.3967]],
        [[-1.3994, -0.1521, -0.1473],
         [ 1.9194,  1.0458,  0.6705],
         [-1.1802, -0.9796,  1.4086]]])
    >>> X = torch.linalg.lstsq(A, B).solution # A is broadcasted to shape (2, 3, 3)
    >>> torch.dist(X, torch.linalg.pinv(A) @ B)
    tensor(1.5152e-06)

    >>> S = torch.linalg.lstsq(A, B, driver='gelsd').singular_values
    >>> torch.dist(S, torch.linalg.svdvals(A))
    tensor(2.3842e-07)

    >>> A[:, 0].zero_()  # Decrease the rank of A
    >>> rank = torch.linalg.lstsq(A, B).rank
    >>> rank
    tensor([2])

.. _condition number:
    https://pytorch.org/docs/main/linalg.html#torch.linalg.cond
.. _full description of these drivers:
    https://www.netlib.org/lapack/lug/node27.html
""")

matrix_power = _add_docstr(_linalg.linalg_matrix_power, r"""
matrix_power(A, n, *, out=None) -> Tensor

Computes the `n`-th power of a square matrix for an integer `n`.

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

If :attr:`n`\ `= 0`, it returns the identity matrix (or batch) of the same shape
as :attr:`A`. If :attr:`n` is negative, it returns the inverse of each matrix
(if invertible) raised to the power of `abs(n)`.

.. note::
    Consider using :func:`torch.linalg.solve` if possible for multiplying a matrix on the left by
    a negative power as, if :attr:`n`\ `> 0`::

        torch.linalg.solve(matrix_power(A, n), B) == matrix_power(A, -n)  @ B

    It is always preferred to use :func:`~solve` when possible, as it is faster and more
    numerically stable than computing :math:`A^{-n}` explicitly.

.. seealso::

        :func:`torch.linalg.solve` computes :attr:`A`\ `.inverse() @ \ `:attr:`B` with a
        numerically stable algorithm.

Args:
    A (Tensor): tensor of shape `(*, m, m)` where `*` is zero or more batch dimensions.
    n (int): the exponent.

Keyword args:
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Raises:
    RuntimeError: if :attr:`n`\ `< 0` and the matrix :attr:`A` or any matrix in the
                  batch of matrices :attr:`A` is not invertible.

Examples::

    >>> A = torch.randn(3, 3)
    >>> torch.linalg.matrix_power(A, 0)
    tensor([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])
    >>> torch.linalg.matrix_power(A, 3)
    tensor([[ 1.0756,  0.4980,  0.0100],
            [-1.6617,  1.4994, -1.9980],
            [-0.4509,  0.2731,  0.8001]])
    >>> torch.linalg.matrix_power(A.expand(2, -1, -1), -2)
    tensor([[[ 0.2640,  0.4571, -0.5511],
            [-1.0163,  0.3491, -1.5292],
            [-0.4899,  0.0822,  0.2773]],
            [[ 0.2640,  0.4571, -0.5511],
            [-1.0163,  0.3491, -1.5292],
            [-0.4899,  0.0822,  0.2773]]])
""")

matrix_rank = _add_docstr(_linalg.linalg_matrix_rank, r"""
linalg.matrix_rank(A, *, atol=None, rtol=None, hermitian=False, out=None) -> Tensor

Computes the numerical rank of a matrix.

The matrix rank is computed as the number of singular values
(or eigenvalues in absolute value when :attr:`hermitian`\ `= True`)
that are greater than :math:`\max(\text{atol}, \sigma_1 * \text{rtol})` threshold,
where :math:`\sigma_1` is the largest singular value (or eigenvalue).

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

If :attr:`hermitian`\ `= True`, :attr:`A` is assumed to be Hermitian if complex or
symmetric if real, but this is not checked internally. Instead, just the lower
triangular part of the matrix is used in the computations.

If :attr:`rtol` is not specified and :attr:`A` is a matrix of dimensions `(m, n)`,
the relative tolerance is set to be :math:`\text{rtol} = \max(m, n) \varepsilon`
and :math:`\varepsilon` is the epsilon value for the dtype of :attr:`A` (see :class:`.finfo`).
If :attr:`rtol` is not specified and :attr:`atol` is specified to be larger than zero then
:attr:`rtol` is set to zero.

If :attr:`atol` or :attr:`rtol` is a :class:`torch.Tensor`, its shape must be broadcastable to that
of the singular values of :attr:`A` as returned by :func:`torch.linalg.svdvals`.

.. note::
    This function has NumPy compatible variant `linalg.matrix_rank(A, tol, hermitian=False)`.
    However, use of the positional argument :attr:`tol` is deprecated in favor of :attr:`atol` and :attr:`rtol`.

""" + fr"""
.. note:: The matrix rank is computed using a singular value decomposition
          :func:`torch.linalg.svdvals` if :attr:`hermitian`\ `= False` (default) and the eigenvalue
          decomposition :func:`torch.linalg.eigvalsh` when :attr:`hermitian`\ `= True`.
          {common_notes["sync_note"]}
""" + r"""

Args:
    A (Tensor): tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.
    tol (float, Tensor, optional): [NumPy Compat] Alias for :attr:`atol`. Default: `None`.

Keyword args:
    atol (float, Tensor, optional): the absolute tolerance value. When `None` it's considered to be zero.
                                    Default: `None`.
    rtol (float, Tensor, optional): the relative tolerance value. See above for the value it takes when `None`.
                                    Default: `None`.
    hermitian(bool): indicates whether :attr:`A` is Hermitian if complex
                     or symmetric if real. Default: `False`.
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Examples::

    >>> A = torch.eye(10)
    >>> torch.linalg.matrix_rank(A)
    tensor(10)
    >>> B = torch.eye(10)
    >>> B[0, 0] = 0
    >>> torch.linalg.matrix_rank(B)
    tensor(9)

    >>> A = torch.randn(4, 3, 2)
    >>> torch.linalg.matrix_rank(A)
    tensor([2, 2, 2, 2])

    >>> A = torch.randn(2, 4, 2, 3)
    >>> torch.linalg.matrix_rank(A)
    tensor([[2, 2, 2, 2],
            [2, 2, 2, 2]])

    >>> A = torch.randn(2, 4, 3, 3, dtype=torch.complex64)
    >>> torch.linalg.matrix_rank(A)
    tensor([[3, 3, 3, 3],
            [3, 3, 3, 3]])
    >>> torch.linalg.matrix_rank(A, hermitian=True)
    tensor([[3, 3, 3, 3],
            [3, 3, 3, 3]])
    >>> torch.linalg.matrix_rank(A, atol=1.0, rtol=0.0)
    tensor([[3, 2, 2, 2],
            [1, 2, 1, 2]])
    >>> torch.linalg.matrix_rank(A, atol=1.0, rtol=0.0, hermitian=True)
    tensor([[2, 2, 2, 1],
            [1, 2, 2, 2]])
""")

norm = _add_docstr(_linalg.linalg_norm, r"""
linalg.norm(A, ord=None, dim=None, keepdim=False, *, out=None, dtype=None) -> Tensor

Computes a vector or matrix norm.

Supports input of float, double, cfloat and cdouble dtypes.

Whether this function computes a vector or matrix norm is determined as follows:

- If :attr:`dim` is an `int`, the vector norm will be computed.
- If :attr:`dim` is a `2`-`tuple`, the matrix norm will be computed.
- If :attr:`dim`\ `= None` and :attr:`ord`\ `= None`,
  :attr:`A` will be flattened to 1D and the `2`-norm of the resulting vector will be computed.
- If :attr:`dim`\ `= None` and :attr:`ord` `!= None`, :attr:`A` must be 1D or 2D.

:attr:`ord` defines the norm that is computed. The following norms are supported:

======================     =========================  ========================================================
:attr:`ord`                norm for matrices          norm for vectors
======================     =========================  ========================================================
`None` (default)           Frobenius norm             `2`-norm (see below)
`'fro'`                    Frobenius norm             -- not supported --
`'nuc'`                    nuclear norm               -- not supported --
`inf`                      `max(sum(abs(x), dim=1))`  `max(abs(x))`
`-inf`                     `min(sum(abs(x), dim=1))`  `min(abs(x))`
`0`                        -- not supported --        `sum(x != 0)`
`1`                        `max(sum(abs(x), dim=0))`  as below
`-1`                       `min(sum(abs(x), dim=0))`  as below
`2`                        largest singular value     as below
`-2`                       smallest singular value    as below
other `int` or `float`     -- not supported --        `sum(abs(x)^{ord})^{(1 / ord)}`
======================     =========================  ========================================================

where `inf` refers to `float('inf')`, NumPy's `inf` object, or any equivalent object.

.. seealso::

        :func:`torch.linalg.vector_norm` computes a vector norm.

        :func:`torch.linalg.matrix_norm` computes a matrix norm.

        The above functions are often clearer and more flexible than using :func:`torch.linalg.norm`.
        For example, `torch.linalg.norm(A, ord=1, dim=(0, 1))` always
        computes a matrix norm, but with `torch.linalg.vector_norm(A, ord=1, dim=(0, 1))` it is possible
        to compute a vector norm over the two dimensions.

Args:
    A (Tensor): tensor of shape `(*, n)` or `(*, m, n)` where `*` is zero or more batch dimensions
    ord (int, float, inf, -inf, 'fro', 'nuc', optional): order of norm. Default: `None`
    dim (int, Tuple[int], optional): dimensions over which to compute
        the vector or matrix norm. See above for the behavior when :attr:`dim`\ `= None`.
        Default: `None`
    keepdim (bool, optional): If set to `True`, the reduced dimensions are retained
        in the result as dimensions with size one. Default: `False`

Keyword args:
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.
    dtype (:class:`torch.dtype`, optional): If specified, the input tensor is cast to
        :attr:`dtype` before performing the operation, and the returned tensor's type
        will be :attr:`dtype`. Default: `None`

Returns:
    A real-valued tensor, even when :attr:`A` is complex.

Examples::

    >>> from torch import linalg as LA
    >>> a = torch.arange(9, dtype=torch.float) - 4
    >>> a
    tensor([-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])
    >>> B = a.reshape((3, 3))
    >>> B
    tensor([[-4., -3., -2.],
            [-1.,  0.,  1.],
            [ 2.,  3.,  4.]])

    >>> LA.norm(a)
    tensor(7.7460)
    >>> LA.norm(B)
    tensor(7.7460)
    >>> LA.norm(B, 'fro')
    tensor(7.7460)
    >>> LA.norm(a, float('inf'))
    tensor(4.)
    >>> LA.norm(B, float('inf'))
    tensor(9.)
    >>> LA.norm(a, -float('inf'))
    tensor(0.)
    >>> LA.norm(B, -float('inf'))
    tensor(2.)

    >>> LA.norm(a, 1)
    tensor(20.)
    >>> LA.norm(B, 1)
    tensor(7.)
    >>> LA.norm(a, -1)
    tensor(0.)
    >>> LA.norm(B, -1)
    tensor(6.)
    >>> LA.norm(a, 2)
    tensor(7.7460)
    >>> LA.norm(B, 2)
    tensor(7.3485)

    >>> LA.norm(a, -2)
    tensor(0.)
    >>> LA.norm(B.double(), -2)
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

    >>> A = torch.arange(8, dtype=torch.float).reshape(2, 2, 2)
    >>> LA.norm(A, dim=(1,2))
    tensor([ 3.7417, 11.2250])
    >>> LA.norm(A[0, :, :]), LA.norm(A[1, :, :])
    (tensor(3.7417), tensor(11.2250))
""")

vector_norm = _add_docstr(_linalg.linalg_vector_norm, r"""
linalg.vector_norm(x, ord=2, dim=None, keepdim=False, *, dtype=None, out=None) -> Tensor

Computes a vector norm.

If :attr:`x` is complex valued, it computes the norm of :attr:`x`\ `.abs()`

Supports input of float, double, cfloat and cdouble dtypes.

This function does not necessarily treat multidimensional :attr:`x` as a batch of
vectors, instead:

- If :attr:`dim`\ `= None`, :attr:`x` will be flattened before the norm is computed.
- If :attr:`dim` is an `int` or a `tuple`, the norm will be computed over these dimensions
  and the other dimensions will be treated as batch dimensions.

This behavior is for consistency with :func:`torch.linalg.norm`.

:attr:`ord` defines the vector norm that is computed. The following norms are supported:

======================   ===============================
:attr:`ord`              vector norm
======================   ===============================
`2` (default)            `2`-norm (see below)
`inf`                    `max(abs(x))`
`-inf`                   `min(abs(x))`
`0`                      `sum(x != 0)`
other `int` or `float`   `sum(abs(x)^{ord})^{(1 / ord)}`
======================   ===============================

where `inf` refers to `float('inf')`, NumPy's `inf` object, or any equivalent object.

:attr:`dtype` may be used to perform the computation in a more precise dtype.
It is semantically equivalent to calling ``linalg.vector_norm(x.to(dtype))``
but it is faster in some cases.

.. seealso::

        :func:`torch.linalg.matrix_norm` computes a matrix norm.

Args:
    x (Tensor): tensor, flattened by default, but this behavior can be
        controlled using :attr:`dim`.
    ord (int, float, inf, -inf, 'fro', 'nuc', optional): order of norm. Default: `2`
    dim (int, Tuple[int], optional): dimensions over which to compute
        the norm. See above for the behavior when :attr:`dim`\ `= None`.
        Default: `None`
    keepdim (bool, optional): If set to `True`, the reduced dimensions are retained
        in the result as dimensions with size one. Default: `False`

Keyword args:
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.
    dtype (:class:`torch.dtype`, optional): type used to perform the accumulation and the return.
        If specified, :attr:`x` is cast to :attr:`dtype` before performing the operation,
        and the returned tensor's type will be :attr:`dtype` if real and of its real counterpart if complex.
        :attr:`dtype` may be complex if :attr:`x` is complex, otherwise it must be real.
        :attr:`x` should be convertible without narrowing to :attr:`dtype`. Default: None

Returns:
    A real-valued tensor, even when :attr:`x` is complex.

Examples::

    >>> from torch import linalg as LA
    >>> a = torch.arange(9, dtype=torch.float) - 4
    >>> a
    tensor([-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])
    >>> B = a.reshape((3, 3))
    >>> B
    tensor([[-4., -3., -2.],
            [-1.,  0.,  1.],
            [ 2.,  3.,  4.]])
    >>> LA.vector_norm(a, ord=3.5)
    tensor(5.4345)
    >>> LA.vector_norm(B, ord=3.5)
    tensor(5.4345)
""")

matrix_norm = _add_docstr(_linalg.linalg_matrix_norm, r"""
linalg.matrix_norm(A, ord='fro', dim=(-2, -1), keepdim=False, *, dtype=None, out=None) -> Tensor

Computes a matrix norm.

If :attr:`A` is complex valued, it computes the norm of :attr:`A`\ `.abs()`

Support input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices: the norm will be computed over the
dimensions specified by the 2-tuple :attr:`dim` and the other dimensions will
be treated as batch dimensions. The output will have the same batch dimensions.

:attr:`ord` defines the matrix norm that is computed. The following norms are supported:

======================   ========================================================
:attr:`ord`              matrix norm
======================   ========================================================
`'fro'` (default)        Frobenius norm
`'nuc'`                  nuclear norm
`inf`                    `max(sum(abs(x), dim=1))`
`-inf`                   `min(sum(abs(x), dim=1))`
`1`                      `max(sum(abs(x), dim=0))`
`-1`                     `min(sum(abs(x), dim=0))`
`2`                      largest singular value
`-2`                     smallest singular value
======================   ========================================================

where `inf` refers to `float('inf')`, NumPy's `inf` object, or any equivalent object.

Args:
    A (Tensor): tensor with two or more dimensions. By default its
        shape is interpreted as `(*, m, n)` where `*` is zero or more
        batch dimensions, but this behavior can be controlled using :attr:`dim`.
    ord (int, inf, -inf, 'fro', 'nuc', optional): order of norm. Default: `'fro'`
    dim (Tuple[int, int], optional): dimensions over which to compute the norm. Default: `(-2, -1)`
    keepdim (bool, optional): If set to `True`, the reduced dimensions are retained
        in the result as dimensions with size one. Default: `False`

Keyword args:
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.
    dtype (:class:`torch.dtype`, optional): If specified, the input tensor is cast to
        :attr:`dtype` before performing the operation, and the returned tensor's type
        will be :attr:`dtype`. Default: `None`

Returns:
    A real-valued tensor, even when :attr:`A` is complex.

Examples::

    >>> from torch import linalg as LA
    >>> A = torch.arange(9, dtype=torch.float).reshape(3, 3)
    >>> A
    tensor([[0., 1., 2.],
            [3., 4., 5.],
            [6., 7., 8.]])
    >>> LA.matrix_norm(A)
    tensor(14.2829)
    >>> LA.matrix_norm(A, ord=-1)
    tensor(9.)
    >>> B = A.expand(2, -1, -1)
    >>> B
    tensor([[[0., 1., 2.],
            [3., 4., 5.],
            [6., 7., 8.]],

            [[0., 1., 2.],
            [3., 4., 5.],
            [6., 7., 8.]]])
    >>> LA.matrix_norm(B)
    tensor([14.2829, 14.2829])
    >>> LA.matrix_norm(B, dim=(0, 2))
    tensor([ 3.1623, 10.0000, 17.2627])
""")

matmul = _add_docstr(_linalg.linalg_matmul, r"""
linalg.matmul(input, other, *, out=None) -> Tensor

Alias for :func:`torch.matmul`
""")

diagonal = _add_docstr(_linalg.linalg_diagonal, r"""
linalg.diagonal(A, *, offset=0, dim1=-2, dim2=-1) -> Tensor

Alias for :func:`torch.diagonal` with defaults :attr:`dim1`\ `= -2`, :attr:`dim2`\ `= -1`.
""")

multi_dot = _add_docstr(_linalg.linalg_multi_dot, r"""
linalg.multi_dot(tensors, *, out=None)

Efficiently multiplies two or more matrices by reordering the multiplications so that
the fewest arithmetic operations are performed.

Supports inputs of float, double, cfloat and cdouble dtypes.
This function does not support batched inputs.

Every tensor in :attr:`tensors` must be 2D, except for the first and last which
may be 1D. If the first tensor is a 1D vector of shape `(n,)` it is treated as a row vector
of shape `(1, n)`, similarly if the last tensor is a 1D vector of shape `(n,)` it is treated
as a column vector of shape `(n, 1)`.

If the first and last tensors are matrices, the output will be a matrix.
However, if either is a 1D vector, then the output will be a 1D vector.

Differences with `numpy.linalg.multi_dot`:

- Unlike `numpy.linalg.multi_dot`, the first and last tensors must either be 1D or 2D
  whereas NumPy allows them to be nD

.. warning:: This function does not broadcast.

.. note:: This function is implemented by chaining :func:`torch.mm` calls after
          computing the optimal matrix multiplication order.

.. note:: The cost of multiplying two matrices with shapes `(a, b)` and `(b, c)` is
          `a * b * c`. Given matrices `A`, `B`, `C` with shapes `(10, 100)`,
          `(100, 5)`, `(5, 50)` respectively, we can calculate the cost of different
          multiplication orders as follows:

          .. math::

             \begin{align*}
             \operatorname{cost}((AB)C) &= 10 \times 100 \times 5 + 10 \times 5 \times 50 = 7500 \\
             \operatorname{cost}(A(BC)) &= 10 \times 100 \times 50 + 100 \times 5 \times 50 = 75000
             \end{align*}

          In this case, multiplying `A` and `B` first followed by `C` is 10 times faster.

Args:
    tensors (Sequence[Tensor]): two or more tensors to multiply. The first and last
        tensors may be 1D or 2D. Every other tensor must be 2D.

Keyword args:
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Examples::

    >>> from torch.linalg import multi_dot

    >>> multi_dot([torch.tensor([1, 2]), torch.tensor([2, 3])])
    tensor(8)
    >>> multi_dot([torch.tensor([[1, 2]]), torch.tensor([2, 3])])
    tensor([8])
    >>> multi_dot([torch.tensor([[1, 2]]), torch.tensor([[2], [3]])])
    tensor([[8]])

    >>> A = torch.arange(2 * 3).view(2, 3)
    >>> B = torch.arange(3 * 2).view(3, 2)
    >>> C = torch.arange(2 * 2).view(2, 2)
    >>> multi_dot((A, B, C))
    tensor([[ 26,  49],
            [ 80, 148]])
""")

svd = _add_docstr(_linalg.linalg_svd, r"""
linalg.svd(A, full_matrices=True, *, driver=None, out=None) -> (Tensor, Tensor, Tensor)

Computes the singular value decomposition (SVD) of a matrix.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
the **full SVD** of a matrix
:math:`A \in \mathbb{K}^{m \times n}`, if `k = min(m,n)`, is defined as

.. math::

    A = U \operatorname{diag}(S) V^{\text{H}}
    \mathrlap{\qquad U \in \mathbb{K}^{m \times m}, S \in \mathbb{R}^k, V \in \mathbb{K}^{n \times n}}

where :math:`\operatorname{diag}(S) \in \mathbb{K}^{m \times n}`,
:math:`V^{\text{H}}` is the conjugate transpose when :math:`V` is complex, and the transpose when :math:`V` is real-valued.
The matrices  :math:`U`, :math:`V` (and thus :math:`V^{\text{H}}`) are orthogonal in the real case, and unitary in the complex case.

When `m > n` (resp. `m < n`) we can drop the last `m - n` (resp. `n - m`) columns of `U` (resp. `V`) to form the **reduced SVD**:

.. math::

    A = U \operatorname{diag}(S) V^{\text{H}}
    \mathrlap{\qquad U \in \mathbb{K}^{m \times k}, S \in \mathbb{R}^k, V \in \mathbb{K}^{k \times n}}

where :math:`\operatorname{diag}(S) \in \mathbb{K}^{k \times k}`.
In this case, :math:`U` and :math:`V` also have orthonormal columns.

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

The returned decomposition is a named tuple `(U, S, Vh)`
which corresponds to :math:`U`, :math:`S`, :math:`V^{\text{H}}` above.

The singular values are returned in descending order.

The parameter :attr:`full_matrices` chooses between the full (default) and reduced SVD.

The :attr:`driver` kwarg may be used in CUDA with a cuSOLVER backend to choose the algorithm used to compute the SVD.
The choice of a driver is a trade-off between accuracy and speed.

- If :attr:`A` is well-conditioned (its `condition number`_ is not too large), or you do not mind some precision loss.

  - For a general matrix: `'gesvdj'` (Jacobi method)
  - If :attr:`A` is tall or wide (`m >> n` or `m << n`): `'gesvda'` (Approximate method)

- If :attr:`A` is not well-conditioned or precision is relevant: `'gesvd'` (QR based)

By default (:attr:`driver`\ `= None`), we call `'gesvdj'` and, if it fails, we fallback to `'gesvd'`.

Differences with `numpy.linalg.svd`:

- Unlike `numpy.linalg.svd`, this function always returns a tuple of three tensors
  and it doesn't support `compute_uv` argument.
  Please use :func:`torch.linalg.svdvals`, which computes only the singular values,
  instead of `compute_uv=False`.

.. note:: When :attr:`full_matrices`\ `= True`, the gradients with respect to `U[..., :, min(m, n):]`
          and `Vh[..., min(m, n):, :]` will be ignored, as those vectors can be arbitrary bases
          of the corresponding subspaces.

.. warning:: The returned tensors `U` and `V` are not unique, nor are they continuous with
             respect to :attr:`A`.
             Due to this lack of uniqueness, different hardware and software may compute
             different singular vectors.

             This non-uniqueness is caused by the fact that multiplying any pair of singular
             vectors :math:`u_k, v_k` by `-1` in the real case or by
             :math:`e^{i \phi}, \phi \in \mathbb{R}` in the complex case produces another two
             valid singular vectors of the matrix.
             For this reason, the loss function shall not depend on this :math:`e^{i \phi}` quantity,
             as it is not well-defined.
             This is checked for complex inputs when computing the gradients of this function. As such,
             when inputs are complex and are on a CUDA device, the computation of the gradients
             of this function synchronizes that device with the CPU.

.. warning:: Gradients computed using `U` or `Vh` will only be finite when
             :attr:`A` does not have repeated singular values. If :attr:`A` is rectangular,
             additionally, zero must also not be one of its singular values.
             Furthermore, if the distance between any two singular values is close to zero,
             the gradient will be numerically unstable, as it depends on the singular values
             :math:`\sigma_i` through the computation of
             :math:`\frac{1}{\min_{i \neq j} \sigma_i^2 - \sigma_j^2}`.
             In the rectangular case, the gradient will also be numerically unstable when
             :attr:`A` has small singular values, as it also depends on the computation of
             :math:`\frac{1}{\sigma_i}`.

.. seealso::

        :func:`torch.linalg.svdvals` computes only the singular values.
        Unlike :func:`torch.linalg.svd`, the gradients of :func:`~svdvals` are always
        numerically stable.

        :func:`torch.linalg.eig` for a function that computes another type of spectral
        decomposition of a matrix. The eigendecomposition works just on square matrices.

        :func:`torch.linalg.eigh` for a (faster) function that computes the eigenvalue decomposition
        for Hermitian and symmetric matrices.

        :func:`torch.linalg.qr` for another (much faster) decomposition that works on general
        matrices.

Args:
    A (Tensor): tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.
    full_matrices (bool, optional): controls whether to compute the full or reduced
                                    SVD, and consequently,
                                    the shape of the returned tensors
                                    `U` and `Vh`. Default: `True`.

Keyword args:
    driver (str, optional): name of the cuSOLVER method to be used. This keyword argument only works on CUDA inputs.
        Available options are: `None`, `gesvd`, `gesvdj`, and `gesvda`.
        Default: `None`.
    out (tuple, optional): output tuple of three tensors. Ignored if `None`.

Returns:
    A named tuple `(U, S, Vh)` which corresponds to :math:`U`, :math:`S`, :math:`V^{\text{H}}` above.

    `S` will always be real-valued, even when :attr:`A` is complex.
    It will also be ordered in descending order.

    `U` and `Vh` will have the same dtype as :attr:`A`. The left / right singular vectors will be given by
    the columns of `U` and the rows of `Vh` respectively.

Examples::

    >>> A = torch.randn(5, 3)
    >>> U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    >>> U.shape, S.shape, Vh.shape
    (torch.Size([5, 3]), torch.Size([3]), torch.Size([3, 3]))
    >>> torch.dist(A, U @ torch.diag(S) @ Vh)
    tensor(1.0486e-06)

    >>> U, S, Vh = torch.linalg.svd(A)
    >>> U.shape, S.shape, Vh.shape
    (torch.Size([5, 5]), torch.Size([3]), torch.Size([3, 3]))
    >>> torch.dist(A, U[:, :3] @ torch.diag(S) @ Vh)
    tensor(1.0486e-06)

    >>> A = torch.randn(7, 5, 3)
    >>> U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    >>> torch.dist(A, U @ torch.diag_embed(S) @ Vh)
    tensor(3.0957e-06)

.. _condition number:
    https://pytorch.org/docs/main/linalg.html#torch.linalg.cond
.. _the resulting vectors will span the same subspace:
    https://en.wikipedia.org/wiki/Singular_value_decomposition#Singular_values,_singular_vectors,_and_their_relation_to_the_SVD
""")

svdvals = _add_docstr(_linalg.linalg_svdvals, r"""
linalg.svdvals(A, *, driver=None, out=None) -> Tensor

Computes the singular values of a matrix.

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

The singular values are returned in descending order.

.. note:: This function is equivalent to NumPy's `linalg.svd(A, compute_uv=False)`.

""" + fr"""
.. note:: {common_notes["sync_note"]}
""" + r"""

.. seealso::

        :func:`torch.linalg.svd` computes the full singular value decomposition.

Args:
    A (Tensor): tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.

Keyword args:
    driver (str, optional): name of the cuSOLVER method to be used. This keyword argument only works on CUDA inputs.
        Available options are: `None`, `gesvd`, `gesvdj`, and `gesvda`.
        Check :func:`torch.linalg.svd` for details.
        Default: `None`.
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Returns:
    A real-valued tensor, even when :attr:`A` is complex.

Examples::

    >>> A = torch.randn(5, 3)
    >>> S = torch.linalg.svdvals(A)
    >>> S
    tensor([2.5139, 2.1087, 1.1066])

    >>> torch.dist(S, torch.linalg.svd(A, full_matrices=False).S)
    tensor(2.4576e-07)
""")

cond = _add_docstr(_linalg.linalg_cond, r"""
linalg.cond(A, p=None, *, out=None) -> Tensor

Computes the condition number of a matrix with respect to a matrix norm.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
the **condition number** :math:`\kappa` of a matrix
:math:`A \in \mathbb{K}^{n \times n}` is defined as

.. math::

    \kappa(A) = \|A\|_p\|A^{-1}\|_p

The condition number of :attr:`A` measures the numerical stability of the linear system `AX = B`
with respect to a matrix norm.

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

:attr:`p` defines the matrix norm that is computed. The following norms are supported:

=========    =================================
:attr:`p`    matrix norm
=========    =================================
`None`       `2`-norm (largest singular value)
`'fro'`      Frobenius norm
`'nuc'`      nuclear norm
`inf`        `max(sum(abs(x), dim=1))`
`-inf`       `min(sum(abs(x), dim=1))`
`1`          `max(sum(abs(x), dim=0))`
`-1`         `min(sum(abs(x), dim=0))`
`2`          largest singular value
`-2`         smallest singular value
=========    =================================

where `inf` refers to `float('inf')`, NumPy's `inf` object, or any equivalent object.

For :attr:`p` is one of `('fro', 'nuc', inf, -inf, 1, -1)`, this function uses
:func:`torch.linalg.norm` and :func:`torch.linalg.inv`.
As such, in this case, the matrix (or every matrix in the batch) :attr:`A` has to be square
and invertible.

For :attr:`p` in `(2, -2)`, this function can be computed in terms of the singular values
:math:`\sigma_1 \geq \ldots \geq \sigma_n`

.. math::

    \kappa_2(A) = \frac{\sigma_1}{\sigma_n}\qquad \kappa_{-2}(A) = \frac{\sigma_n}{\sigma_1}

In these cases, it is computed using :func:`torch.linalg.svdvals`. For these norms, the matrix
(or every matrix in the batch) :attr:`A` may have any shape.

.. note :: When inputs are on a CUDA device, this function synchronizes that device with the CPU
           if :attr:`p` is one of `('fro', 'nuc', inf, -inf, 1, -1)`.

.. seealso::

        :func:`torch.linalg.solve` for a function that solves linear systems of square matrices.

        :func:`torch.linalg.lstsq` for a function that solves linear systems of general matrices.

Args:
    A (Tensor): tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions
                    for :attr:`p` in `(2, -2)`, and of shape `(*, n, n)` where every matrix
                    is invertible for :attr:`p` in `('fro', 'nuc', inf, -inf, 1, -1)`.
    p (int, inf, -inf, 'fro', 'nuc', optional):
        the type of the matrix norm to use in the computations (see above). Default: `None`

Keyword args:
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Returns:
    A real-valued tensor, even when :attr:`A` is complex.

Raises:
    RuntimeError:
        if :attr:`p` is one of `('fro', 'nuc', inf, -inf, 1, -1)`
        and the :attr:`A` matrix or any matrix in the batch :attr:`A` is not square
        or invertible.

Examples::

    >>> A = torch.randn(3, 4, 4, dtype=torch.complex64)
    >>> torch.linalg.cond(A)
    >>> A = torch.tensor([[1., 0, -1], [0, 1, 0], [1, 0, 1]])
    >>> torch.linalg.cond(A)
    tensor([1.4142])
    >>> torch.linalg.cond(A, 'fro')
    tensor(3.1623)
    >>> torch.linalg.cond(A, 'nuc')
    tensor(9.2426)
    >>> torch.linalg.cond(A, float('inf'))
    tensor(2.)
    >>> torch.linalg.cond(A, float('-inf'))
    tensor(1.)
    >>> torch.linalg.cond(A, 1)
    tensor(2.)
    >>> torch.linalg.cond(A, -1)
    tensor(1.)
    >>> torch.linalg.cond(A, 2)
    tensor([1.4142])
    >>> torch.linalg.cond(A, -2)
    tensor([0.7071])

    >>> A = torch.randn(2, 3, 3)
    >>> torch.linalg.cond(A)
    tensor([[9.5917],
            [3.2538]])
    >>> A = torch.randn(2, 3, 3, dtype=torch.complex64)
    >>> torch.linalg.cond(A)
    tensor([[4.6245],
            [4.5671]])
""")

pinv = _add_docstr(_linalg.linalg_pinv, r"""
linalg.pinv(A, *, atol=None, rtol=None, hermitian=False, out=None) -> Tensor

Computes the pseudoinverse (Moore-Penrose inverse) of a matrix.

The pseudoinverse may be `defined algebraically`_
but it is more computationally convenient to understand it `through the SVD`_

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

If :attr:`hermitian`\ `= True`, :attr:`A` is assumed to be Hermitian if complex or
symmetric if real, but this is not checked internally. Instead, just the lower
triangular part of the matrix is used in the computations.

The singular values (or the norm of the eigenvalues when :attr:`hermitian`\ `= True`)
that are below :math:`\max(\text{atol}, \sigma_1 \cdot \text{rtol})` threshold are
treated as zero and discarded in the computation,
where :math:`\sigma_1` is the largest singular value (or eigenvalue).

If :attr:`rtol` is not specified and :attr:`A` is a matrix of dimensions `(m, n)`,
the relative tolerance is set to be :math:`\text{rtol} = \max(m, n) \varepsilon`
and :math:`\varepsilon` is the epsilon value for the dtype of :attr:`A` (see :class:`.finfo`).
If :attr:`rtol` is not specified and :attr:`atol` is specified to be larger than zero then
:attr:`rtol` is set to zero.

If :attr:`atol` or :attr:`rtol` is a :class:`torch.Tensor`, its shape must be broadcastable to that
of the singular values of :attr:`A` as returned by :func:`torch.linalg.svd`.

.. note:: This function uses :func:`torch.linalg.svd` if :attr:`hermitian`\ `= False` and
          :func:`torch.linalg.eigh` if :attr:`hermitian`\ `= True`.
          For CUDA inputs, this function synchronizes that device with the CPU.

.. note::
    Consider using :func:`torch.linalg.lstsq` if possible for multiplying a matrix on the left by
    the pseudoinverse, as::

        torch.linalg.lstsq(A, B).solution == A.pinv() @ B

    It is always preferred to use :func:`~lstsq` when possible, as it is faster and more
    numerically stable than computing the pseudoinverse explicitly.

.. note::
    This function has NumPy compatible variant `linalg.pinv(A, rcond, hermitian=False)`.
    However, use of the positional argument :attr:`rcond` is deprecated in favor of :attr:`rtol`.

.. warning::
    This function uses internally :func:`torch.linalg.svd` (or :func:`torch.linalg.eigh`
    when :attr:`hermitian`\ `= True`), so its derivative has the same problems as those of these
    functions. See the warnings in :func:`torch.linalg.svd` and :func:`torch.linalg.eigh` for
    more details.

.. seealso::

        :func:`torch.linalg.inv` computes the inverse of a square matrix.

        :func:`torch.linalg.lstsq` computes :attr:`A`\ `.pinv() @ \ `:attr:`B` with a
        numerically stable algorithm.

Args:
    A (Tensor): tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.
    rcond (float, Tensor, optional): [NumPy Compat]. Alias for :attr:`rtol`. Default: `None`.

Keyword args:
    atol (float, Tensor, optional): the absolute tolerance value. When `None` it's considered to be zero.
                                    Default: `None`.
    rtol (float, Tensor, optional): the relative tolerance value. See above for the value it takes when `None`.
                                    Default: `None`.
    hermitian(bool, optional): indicates whether :attr:`A` is Hermitian if complex
                               or symmetric if real. Default: `False`.
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Examples::

    >>> A = torch.randn(3, 5)
    >>> A
    tensor([[ 0.5495,  0.0979, -1.4092, -0.1128,  0.4132],
            [-1.1143, -0.3662,  0.3042,  1.6374, -0.9294],
            [-0.3269, -0.5745, -0.0382, -0.5922, -0.6759]])
    >>> torch.linalg.pinv(A)
    tensor([[ 0.0600, -0.1933, -0.2090],
            [-0.0903, -0.0817, -0.4752],
            [-0.7124, -0.1631, -0.2272],
            [ 0.1356,  0.3933, -0.5023],
            [-0.0308, -0.1725, -0.5216]])

    >>> A = torch.randn(2, 6, 3)
    >>> Apinv = torch.linalg.pinv(A)
    >>> torch.dist(Apinv @ A, torch.eye(3))
    tensor(8.5633e-07)

    >>> A = torch.randn(3, 3, dtype=torch.complex64)
    >>> A = A + A.T.conj()  # creates a Hermitian matrix
    >>> Apinv = torch.linalg.pinv(A, hermitian=True)
    >>> torch.dist(Apinv @ A, torch.eye(3))
    tensor(1.0830e-06)

.. _defined algebraically:
    https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse#Existence_and_uniqueness
.. _through the SVD:
    https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse#Singular_value_decomposition_(SVD)
""")

matrix_exp = _add_docstr(_linalg.linalg_matrix_exp, r"""
linalg.matrix_exp(A) -> Tensor

Computes the matrix exponential of a square matrix.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
this function computes the **matrix exponential** of :math:`A \in \mathbb{K}^{n \times n}`, which is defined as

.. math::
    \mathrm{matrix\_exp}(A) = \sum_{k=0}^\infty \frac{1}{k!}A^k \in \mathbb{K}^{n \times n}.

If the matrix :math:`A` has eigenvalues :math:`\lambda_i \in \mathbb{C}`,
the matrix :math:`\mathrm{matrix\_exp}(A)` has eigenvalues :math:`e^{\lambda_i} \in \mathbb{C}`.

Supports input of bfloat16, float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

Args:
    A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions.

Example::

    >>> A = torch.empty(2, 2, 2)
    >>> A[0, :, :] = torch.eye(2, 2)
    >>> A[1, :, :] = 2 * torch.eye(2, 2)
    >>> A
    tensor([[[1., 0.],
             [0., 1.]],

            [[2., 0.],
             [0., 2.]]])
    >>> torch.linalg.matrix_exp(A)
    tensor([[[2.7183, 0.0000],
             [0.0000, 2.7183]],

             [[7.3891, 0.0000],
              [0.0000, 7.3891]]])

    >>> import math
    >>> A = torch.tensor([[0, math.pi/3], [-math.pi/3, 0]]) # A is skew-symmetric
    >>> torch.linalg.matrix_exp(A) # matrix_exp(A) = [[cos(pi/3), sin(pi/3)], [-sin(pi/3), cos(pi/3)]]
    tensor([[ 0.5000,  0.8660],
            [-0.8660,  0.5000]])
""")


solve = _add_docstr(_linalg.linalg_solve, r"""
linalg.solve(A, B, *, left=True, out=None) -> Tensor

Computes the solution of a square system of linear equations with a unique solution.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
this function computes the solution :math:`X \in \mathbb{K}^{n \times k}` of the **linear system** associated to
:math:`A \in \mathbb{K}^{n \times n}, B \in \mathbb{K}^{n \times k}`, which is defined as

.. math:: AX = B

If :attr:`left`\ `= False`, this function returns the matrix :math:`X \in \mathbb{K}^{n \times k}` that solves the system

.. math::

    XA = B\mathrlap{\qquad A \in \mathbb{K}^{k \times k}, B \in \mathbb{K}^{n \times k}.}

This system of linear equations has one solution if and only if :math:`A` is `invertible`_.
This function assumes that :math:`A` is invertible.

Supports inputs of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if the inputs are batches of matrices then
the output has the same batch dimensions.

Letting `*` be zero or more batch dimensions,

- If :attr:`A` has shape `(*, n, n)` and :attr:`B` has shape `(*, n)` (a batch of vectors) or shape
  `(*, n, k)` (a batch of matrices or "multiple right-hand sides"), this function returns `X` of shape
  `(*, n)` or `(*, n, k)` respectively.
- Otherwise, if :attr:`A` has shape `(*, n, n)` and  :attr:`B` has shape `(n,)`  or `(n, k)`, :attr:`B`
  is broadcasted to have shape `(*, n)` or `(*, n, k)` respectively.
  This function then returns the solution of the resulting batch of systems of linear equations.

.. note::
    This function computes `X = \ `:attr:`A`\ `.inverse() @ \ `:attr:`B` in a faster and
    more numerically stable way than performing the computations separately.

.. note::
    It is possible to compute the solution of the system :math:`XA = B` by passing the inputs
    :attr:`A` and :attr:`B` transposed and transposing the output returned by this function.

.. note::
    :attr:`A` is allowed to be a non-batched `torch.sparse_csr_tensor`, but only with `left=True`.

""" + fr"""
.. note:: {common_notes["sync_note_has_ex"].format("torch.linalg.solve_ex")}
""" + r"""

.. seealso::

        :func:`torch.linalg.solve_triangular` computes the solution of a triangular system of linear
        equations with a unique solution.

Args:
    A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions.
    B (Tensor): right-hand side tensor of shape `(*, n)` or  `(*, n, k)` or `(n,)` or `(n, k)`
                according to the rules described above

Keyword args:
    left (bool, optional): whether to solve the system :math:`AX=B` or :math:`XA = B`. Default: `True`.
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Raises:
    RuntimeError: if the :attr:`A` matrix is not invertible or any matrix in a batched :attr:`A`
                  is not invertible.

Examples::

    >>> A = torch.randn(3, 3)
    >>> b = torch.randn(3)
    >>> x = torch.linalg.solve(A, b)
    >>> torch.allclose(A @ x, b)
    True
    >>> A = torch.randn(2, 3, 3)
    >>> B = torch.randn(2, 3, 4)
    >>> X = torch.linalg.solve(A, B)
    >>> X.shape
    torch.Size([2, 3, 4])
    >>> torch.allclose(A @ X, B)
    True

    >>> A = torch.randn(2, 3, 3)
    >>> b = torch.randn(3, 1)
    >>> x = torch.linalg.solve(A, b) # b is broadcasted to size (2, 3, 1)
    >>> x.shape
    torch.Size([2, 3, 1])
    >>> torch.allclose(A @ x, b)
    True
    >>> b = torch.randn(3)
    >>> x = torch.linalg.solve(A, b) # b is broadcasted to size (2, 3)
    >>> x.shape
    torch.Size([2, 3])
    >>> Ax = A @ x.unsqueeze(-1)
    >>> torch.allclose(Ax, b.unsqueeze(-1).expand_as(Ax))
    True

.. _invertible:
    https://en.wikipedia.org/wiki/Invertible_matrix#The_invertible_matrix_theorem
""")

solve_triangular = _add_docstr(_linalg.linalg_solve_triangular, r"""
linalg.solve_triangular(A, B, *, upper, left=True, unitriangular=False, out=None) -> Tensor

Computes the solution of a triangular system of linear equations with a unique solution.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
this function computes the solution :math:`X \in \mathbb{K}^{n \times k}` of the **linear system**
associated to the triangular matrix :math:`A \in \mathbb{K}^{n \times n}` without zeros on the diagonal
(that is, it is `invertible`_) and the rectangular matrix , :math:`B \in \mathbb{K}^{n \times k}`,
which is defined as

.. math:: AX = B

The argument :attr:`upper` signals whether :math:`A` is upper or lower triangular.

If :attr:`left`\ `= False`, this function returns the matrix :math:`X \in \mathbb{K}^{n \times k}` that
solves the system

.. math::

    XA = B\mathrlap{\qquad A \in \mathbb{K}^{k \times k}, B \in \mathbb{K}^{n \times k}.}

If :attr:`upper`\ `= True` (resp. `False`) just the upper (resp. lower) triangular half of :attr:`A`
will be accessed. The elements below the main diagonal will be considered to be zero and will not be accessed.

If :attr:`unitriangular`\ `= True`, the diagonal of :attr:`A` is assumed to be ones and will not be accessed.

The result may contain `NaN` s if the diagonal of :attr:`A` contains zeros or elements that
are very close to zero and :attr:`unitriangular`\ `= False` (default) or if the input matrix
has very small eigenvalues.

Supports inputs of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if the inputs are batches of matrices then
the output has the same batch dimensions.

.. seealso::

        :func:`torch.linalg.solve` computes the solution of a general square system of linear
        equations with a unique solution.

Args:
    A (Tensor): tensor of shape `(*, n, n)` (or `(*, k, k)` if :attr:`left`\ `= False`)
                where `*` is zero or more batch dimensions.
    B (Tensor): right-hand side tensor of shape `(*, n, k)`.

Keyword args:
    upper (bool): whether :attr:`A` is an upper or lower triangular matrix.
    left (bool, optional): whether to solve the system :math:`AX=B` or :math:`XA = B`. Default: `True`.
    unitriangular (bool, optional): if `True`, the diagonal elements of :attr:`A` are assumed to be
                                    all equal to `1`. Default: `False`.
    out (Tensor, optional): output tensor. `B` may be passed as `out` and the result is computed in-place on `B`.
                            Ignored if `None`. Default: `None`.

Examples::

    >>> A = torch.randn(3, 3).triu_()
    >>> B = torch.randn(3, 4)
    >>> X = torch.linalg.solve_triangular(A, B, upper=True)
    >>> torch.allclose(A @ X, B)
    True

    >>> A = torch.randn(2, 3, 3).tril_()
    >>> B = torch.randn(2, 3, 4)
    >>> X = torch.linalg.solve_triangular(A, B, upper=False)
    >>> torch.allclose(A @ X, B)
    True

    >>> A = torch.randn(2, 4, 4).tril_()
    >>> B = torch.randn(2, 3, 4)
    >>> X = torch.linalg.solve_triangular(A, B, upper=False, left=False)
    >>> torch.allclose(X @ A, B)
    True

.. _invertible:
    https://en.wikipedia.org/wiki/Invertible_matrix#The_invertible_matrix_theorem
""")

lu_factor = _add_docstr(_linalg.linalg_lu_factor, r"""
linalg.lu_factor(A, *, bool pivot=True, out=None) -> (Tensor, Tensor)

Computes a compact representation of the LU factorization with partial pivoting of a matrix.

This function computes a compact representation of the decomposition given by :func:`torch.linalg.lu`.
If the matrix is square, this representation may be used in :func:`torch.linalg.lu_solve`
to solve system of linear equations that share the matrix :attr:`A`.

The returned decomposition is represented as a named tuple `(LU, pivots)`.
The ``LU`` matrix has the same shape as the input matrix ``A``. Its upper and lower triangular
parts encode the non-constant elements of ``L`` and ``U`` of the LU decomposition of ``A``.

The returned permutation matrix is represented by a 1-indexed vector. `pivots[i] == j` represents
that in the `i`-th step of the algorithm, the `i`-th row was permuted with the `j-1`-th row.

On CUDA, one may use :attr:`pivot`\ `= False`. In this case, this function returns the LU
decomposition without pivoting if it exists.

Supports inputs of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if the inputs are batches of matrices then
the output has the same batch dimensions.

""" + fr"""
.. note:: {common_notes["sync_note_has_ex"].format("torch.linalg.lu_factor_ex")}
""" + r"""
.. warning:: The LU decomposition is almost never unique, as often there are different permutation
             matrices that can yield different LU decompositions.
             As such, different platforms, like SciPy, or inputs on different devices,
             may produce different valid decompositions.

             Gradient computations are only supported if the input matrix is full-rank.
             If this condition is not met, no error will be thrown, but the gradient may not be finite.
             This is because the LU decomposition with pivoting is not differentiable at these points.

.. seealso::

        :func:`torch.linalg.lu_solve` solves a system of linear equations given the output of this
        function provided the input matrix was square and invertible.

        :func:`torch.lu_unpack` unpacks the tensors returned by :func:`~lu_factor` into the three
        matrices `P, L, U` that form the decomposition.

        :func:`torch.linalg.lu` computes the LU decomposition with partial pivoting of a possibly
        non-square matrix. It is a composition of :func:`~lu_factor` and :func:`torch.lu_unpack`.

        :func:`torch.linalg.solve` solves a system of linear equations. It is a composition
        of :func:`~lu_factor` and :func:`~lu_solve`.

Args:
    A (Tensor): tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.

Keyword args:
    pivot (bool, optional): Whether to compute the LU decomposition with partial pivoting, or the regular LU
                            decomposition. :attr:`pivot`\ `= False` not supported on CPU. Default: `True`.
    out (tuple, optional): tuple of two tensors to write the output to. Ignored if `None`. Default: `None`.

Returns:
    A named tuple `(LU, pivots)`.

Raises:
    RuntimeError: if the :attr:`A` matrix is not invertible or any matrix in a batched :attr:`A`
                  is not invertible.

Examples::

    >>> A = torch.randn(2, 3, 3)
    >>> B1 = torch.randn(2, 3, 4)
    >>> B2 = torch.randn(2, 3, 7)
    >>> LU, pivots = torch.linalg.lu_factor(A)
    >>> X1 = torch.linalg.lu_solve(LU, pivots, B1)
    >>> X2 = torch.linalg.lu_solve(LU, pivots, B2)
    >>> torch.allclose(A @ X1, B1)
    True
    >>> torch.allclose(A @ X2, B2)
    True

.. _invertible:
    https://en.wikipedia.org/wiki/Invertible_matrix#The_invertible_matrix_theorem
""")

lu_factor_ex = _add_docstr(_linalg.linalg_lu_factor_ex, r"""
linalg.lu_factor_ex(A, *, pivot=True, check_errors=False, out=None) -> (Tensor, Tensor, Tensor)

This is a version of :func:`~lu_factor` that does not perform error checks unless :attr:`check_errors`\ `= True`.
It also returns the :attr:`info` tensor returned by `LAPACK's getrf`_.

""" + fr"""
.. note:: {common_notes["sync_note_ex"]}

.. warning:: {common_notes["experimental_warning"]}
""" + r"""

Args:
    A (Tensor): tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.

Keyword args:
    pivot (bool, optional): Whether to compute the LU decomposition with partial pivoting, or the regular LU
                            decomposition. :attr:`pivot`\ `= False` not supported on CPU. Default: `True`.
    check_errors (bool, optional): controls whether to check the content of ``infos`` and raise
                                   an error if it is non-zero. Default: `False`.
    out (tuple, optional): tuple of three tensors to write the output to. Ignored if `None`. Default: `None`.

Returns:
    A named tuple `(LU, pivots, info)`.

.. _LAPACK's getrf:
    https://www.netlib.org/lapack/explore-html/dd/d9a/group__double_g_ecomputational_ga0019443faea08275ca60a734d0593e60.html
""")

lu_solve = _add_docstr(_linalg.linalg_lu_solve, r"""
linalg.lu_solve(LU, pivots, B, *, left=True, adjoint=False, out=None) -> Tensor

Computes the solution of a square system of linear equations with a unique solution given an LU decomposition.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
this function computes the solution :math:`X \in \mathbb{K}^{n \times k}` of the **linear system** associated to
:math:`A \in \mathbb{K}^{n \times n}, B \in \mathbb{K}^{n \times k}`, which is defined as

.. math:: AX = B

where :math:`A` is given factorized as returned by :func:`~lu_factor`.

If :attr:`left`\ `= False`, this function returns the matrix :math:`X \in \mathbb{K}^{n \times k}` that solves the system

.. math::

    XA = B\mathrlap{\qquad A \in \mathbb{K}^{k \times k}, B \in \mathbb{K}^{n \times k}.}

If  :attr:`adjoint`\ `= True` (and :attr:`left`\ `= True`), given an LU factorization of :math:`A`
this function function returns the :math:`X \in \mathbb{K}^{n \times k}` that solves the system

.. math::

    A^{\text{H}}X = B\mathrlap{\qquad A \in \mathbb{K}^{k \times k}, B \in \mathbb{K}^{n \times k}.}

where :math:`A^{\text{H}}` is the conjugate transpose when :math:`A` is complex, and the
transpose when :math:`A` is real-valued. The :attr:`left`\ `= False` case is analogous.

Supports inputs of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if the inputs are batches of matrices then
the output has the same batch dimensions.

Args:
    LU (Tensor): tensor of shape `(*, n, n)` (or `(*, k, k)` if :attr:`left`\ `= True`)
                 where `*` is zero or more batch dimensions as returned by :func:`~lu_factor`.
    pivots (Tensor): tensor of shape `(*, n)` (or `(*, k)` if :attr:`left`\ `= True`)
                     where `*` is zero or more batch dimensions as returned by :func:`~lu_factor`.
    B (Tensor): right-hand side tensor of shape `(*, n, k)`.

Keyword args:
    left (bool, optional): whether to solve the system :math:`AX=B` or :math:`XA = B`. Default: `True`.
    adjoint (bool, optional): whether to solve the system :math:`AX=B` or :math:`A^{\text{H}}X = B`. Default: `False`.
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Examples::

    >>> A = torch.randn(3, 3)
    >>> LU, pivots = torch.linalg.lu_factor(A)
    >>> B = torch.randn(3, 2)
    >>> X = torch.linalg.lu_solve(LU, pivots, B)
    >>> torch.allclose(A @ X, B)
    True

    >>> B = torch.randn(3, 3, 2)   # Broadcasting rules apply: A is broadcasted
    >>> X = torch.linalg.lu_solve(LU, pivots, B)
    >>> torch.allclose(A @ X, B)
    True

    >>> B = torch.randn(3, 5, 3)
    >>> X = torch.linalg.lu_solve(LU, pivots, B, left=False)
    >>> torch.allclose(X @ A, B)
    True

    >>> B = torch.randn(3, 3, 4)   # Now solve for A^T
    >>> X = torch.linalg.lu_solve(LU, pivots, B, adjoint=True)
    >>> torch.allclose(A.mT @ X, B)
    True

.. _invertible:
    https://en.wikipedia.org/wiki/Invertible_matrix#The_invertible_matrix_theorem
""")

lu = _add_docstr(_linalg.linalg_lu, r"""
lu(A, *, pivot=True, out=None) -> (Tensor, Tensor, Tensor)

Computes the LU decomposition with partial pivoting of a matrix.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
the **LU decomposition with partial pivoting** of a matrix
:math:`A \in \mathbb{K}^{m \times n}` is defined as

.. math::

    A = PLU\mathrlap{\qquad P \in \mathbb{K}^{m \times m}, L \in \mathbb{K}^{m \times k}, U \in \mathbb{K}^{k \times n}}

where `k = min(m,n)`, :math:`P` is a `permutation matrix`_, :math:`L` is lower triangular with ones on the diagonal
and :math:`U` is upper triangular.

If :attr:`pivot`\ `= False` and :attr:`A` is on GPU, then the **LU decomposition without pivoting** is computed

.. math::

    A = LU\mathrlap{\qquad L \in \mathbb{K}^{m \times k}, U \in \mathbb{K}^{k \times n}}

When :attr:`pivot`\ `= False`, the returned matrix :attr:`P` will be empty.
The LU decomposition without pivoting `may not exist`_ if any of the principal minors of :attr:`A` is singular.
In this case, the output matrix may contain `inf` or `NaN`.

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

.. seealso::

        :func:`torch.linalg.solve` solves a system of linear equations using the LU decomposition
        with partial pivoting.

.. warning:: The LU decomposition is almost never unique, as often there are different permutation
             matrices that can yield different LU decompositions.
             As such, different platforms, like SciPy, or inputs on different devices,
             may produce different valid decompositions.

.. warning:: Gradient computations are only supported if the input matrix is full-rank.
             If this condition is not met, no error will be thrown, but the gradient
             may not be finite.
             This is because the LU decomposition with pivoting is not differentiable at these points.

Args:
    A (Tensor): tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.
    pivot (bool, optional): Controls whether to compute the LU decomposition with partial pivoting or
        no pivoting. Default: `True`.

Keyword args:
    out (tuple, optional): output tuple of three tensors. Ignored if `None`. Default: `None`.

Returns:
    A named tuple `(P, L, U)`.

Examples::

    >>> A = torch.randn(3, 2)
    >>> P, L, U = torch.linalg.lu(A)
    >>> P
    tensor([[0., 1., 0.],
            [0., 0., 1.],
            [1., 0., 0.]])
    >>> L
    tensor([[1.0000, 0.0000],
            [0.5007, 1.0000],
            [0.0633, 0.9755]])
    >>> U
    tensor([[0.3771, 0.0489],
            [0.0000, 0.9644]])
    >>> torch.dist(A, P @ L @ U)
    tensor(5.9605e-08)

    >>> A = torch.randn(2, 5, 7, device="cuda")
    >>> P, L, U = torch.linalg.lu(A, pivot=False)
    >>> P
    tensor([], device='cuda:0')
    >>> torch.dist(A, L @ U)
    tensor(1.0376e-06, device='cuda:0')

.. _permutation matrix:
    https://en.wikipedia.org/wiki/Permutation_matrix
.. _may not exist:
    https://en.wikipedia.org/wiki/LU_decomposition#Definitions
""")

tensorinv = _add_docstr(_linalg.linalg_tensorinv, r"""
linalg.tensorinv(A, ind=2, *, out=None) -> Tensor

Computes the multiplicative inverse of :func:`torch.tensordot`.

If `m` is the product of the first :attr:`ind` dimensions of :attr:`A` and `n` is the product of
the rest of the dimensions, this function expects `m` and `n` to be equal.
If this is the case, it computes a tensor `X` such that
`tensordot(\ `:attr:`A`\ `, X, \ `:attr:`ind`\ `)` is the identity matrix in dimension `m`.
`X` will have the shape of :attr:`A` but with the first :attr:`ind` dimensions pushed back to the end

.. code:: text

    X.shape == A.shape[ind:] + A.shape[:ind]

Supports input of float, double, cfloat and cdouble dtypes.

.. note:: When :attr:`A` is a `2`-dimensional tensor and :attr:`ind`\ `= 1`,
          this function computes the (multiplicative) inverse of :attr:`A`
          (see :func:`torch.linalg.inv`).

.. note::
    Consider using :func:`torch.linalg.tensorsolve` if possible for multiplying a tensor on the left
    by the tensor inverse, as::

        linalg.tensorsolve(A, B) == torch.tensordot(linalg.tensorinv(A), B)  # When B is a tensor with shape A.shape[:B.ndim]

    It is always preferred to use :func:`~tensorsolve` when possible, as it is faster and more
    numerically stable than computing the pseudoinverse explicitly.

.. seealso::

        :func:`torch.linalg.tensorsolve` computes
        `torch.tensordot(tensorinv(\ `:attr:`A`\ `), \ `:attr:`B`\ `)`.

Args:
    A (Tensor): tensor to invert. Its shape must satisfy
                    `prod(\ `:attr:`A`\ `.shape[:\ `:attr:`ind`\ `]) ==
                    prod(\ `:attr:`A`\ `.shape[\ `:attr:`ind`\ `:])`.
    ind (int): index at which to compute the inverse of :func:`torch.tensordot`. Default: `2`.

Keyword args:
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Raises:
    RuntimeError: if the reshaped :attr:`A` is not invertible or the product of the first
                  :attr:`ind` dimensions is not equal to the product of the rest.

Examples::

    >>> A = torch.eye(4 * 6).reshape((4, 6, 8, 3))
    >>> Ainv = torch.linalg.tensorinv(A, ind=2)
    >>> Ainv.shape
    torch.Size([8, 3, 4, 6])
    >>> B = torch.randn(4, 6)
    >>> torch.allclose(torch.tensordot(Ainv, B), torch.linalg.tensorsolve(A, B))
    True

    >>> A = torch.randn(4, 4)
    >>> Atensorinv = torch.linalg.tensorinv(A, ind=1)
    >>> Ainv = torch.linalg.inv(A)
    >>> torch.allclose(Atensorinv, Ainv)
    True
""")

tensorsolve = _add_docstr(_linalg.linalg_tensorsolve, r"""
linalg.tensorsolve(A, B, dims=None, *, out=None) -> Tensor

Computes the solution `X` to the system `torch.tensordot(A, X) = B`.

If `m` is the product of the first :attr:`B`\ `.ndim`  dimensions of :attr:`A` and
`n` is the product of the rest of the dimensions, this function expects `m` and `n` to be equal.

The returned tensor `x` satisfies
`tensordot(\ `:attr:`A`\ `, x, dims=x.ndim) == \ `:attr:`B`.
`x` has shape :attr:`A`\ `[B.ndim:]`.

If :attr:`dims` is specified, :attr:`A` will be reshaped as

.. code:: text

    A = movedim(A, dims, range(len(dims) - A.ndim + 1, 0))

Supports inputs of float, double, cfloat and cdouble dtypes.

.. seealso::

        :func:`torch.linalg.tensorinv` computes the multiplicative inverse of
        :func:`torch.tensordot`.

Args:
    A (Tensor): tensor to solve for. Its shape must satisfy
                    `prod(\ `:attr:`A`\ `.shape[:\ `:attr:`B`\ `.ndim]) ==
                    prod(\ `:attr:`A`\ `.shape[\ `:attr:`B`\ `.ndim:])`.
    B (Tensor): tensor of shape :attr:`A`\ `.shape[:\ `:attr:`B`\ `.ndim]`.
    dims (Tuple[int], optional): dimensions of :attr:`A` to be moved.
        If `None`, no dimensions are moved. Default: `None`.

Keyword args:
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Raises:
    RuntimeError: if the reshaped :attr:`A`\ `.view(m, m)` with `m` as above  is not
                  invertible or the product of the first :attr:`ind` dimensions is not equal
                  to the product of the rest of the dimensions.

Examples::

    >>> A = torch.eye(2 * 3 * 4).reshape((2 * 3, 4, 2, 3, 4))
    >>> B = torch.randn(2 * 3, 4)
    >>> X = torch.linalg.tensorsolve(A, B)
    >>> X.shape
    torch.Size([2, 3, 4])
    >>> torch.allclose(torch.tensordot(A, X, dims=X.ndim), B)
    True

    >>> A = torch.randn(6, 4, 4, 3, 2)
    >>> B = torch.randn(4, 3, 2)
    >>> X = torch.linalg.tensorsolve(A, B, dims=(0, 2))
    >>> X.shape
    torch.Size([6, 4])
    >>> A = A.permute(1, 3, 4, 0, 2)
    >>> A.shape[B.ndim:]
    torch.Size([6, 4])
    >>> torch.allclose(torch.tensordot(A, X, dims=X.ndim), B, atol=1e-6)
    True
""")

qr = _add_docstr(_linalg.linalg_qr, r"""
qr(A, mode='reduced', *, out=None) -> (Tensor, Tensor)

Computes the QR decomposition of a matrix.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
the **full QR decomposition** of a matrix
:math:`A \in \mathbb{K}^{m \times n}` is defined as

.. math::

    A = QR\mathrlap{\qquad Q \in \mathbb{K}^{m \times m}, R \in \mathbb{K}^{m \times n}}

where :math:`Q` is orthogonal in the real case and unitary in the complex case,
and :math:`R` is upper triangular with real diagonal (even in the complex case).

When `m > n` (tall matrix), as `R` is upper triangular, its last `m - n` rows are zero.
In this case, we can drop the last `m - n` columns of `Q` to form the
**reduced QR decomposition**:

.. math::

    A = QR\mathrlap{\qquad Q \in \mathbb{K}^{m \times n}, R \in \mathbb{K}^{n \times n}}

The reduced QR decomposition agrees with the full QR decomposition when `n >= m` (wide matrix).

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

The parameter :attr:`mode` chooses between the full and reduced QR decomposition.
If :attr:`A` has shape `(*, m, n)`, denoting `k = min(m, n)`

- :attr:`mode`\ `= 'reduced'` (default): Returns `(Q, R)` of shapes `(*, m, k)`, `(*, k, n)` respectively.
  It is always differentiable.
- :attr:`mode`\ `= 'complete'`: Returns `(Q, R)` of shapes `(*, m, m)`, `(*, m, n)` respectively.
  It is differentiable for `m <= n`.
- :attr:`mode`\ `= 'r'`: Computes only the reduced `R`. Returns `(Q, R)` with `Q` empty and `R` of shape `(*, k, n)`.
  It is never differentiable.

Differences with `numpy.linalg.qr`:

- :attr:`mode`\ `= 'raw'` is not implemented.
- Unlike `numpy.linalg.qr`, this function always returns a tuple of two tensors.
  When :attr:`mode`\ `= 'r'`, the `Q` tensor is an empty tensor.

.. warning:: The elements in the diagonal of `R` are not necessarily positive.
             As such, the returned QR decomposition is only unique up to the sign of the diagonal of `R`.
             Therefore, different platforms, like NumPy, or inputs on different devices,
             may produce different valid decompositions.

.. warning:: The QR decomposition is only well-defined if the first `k = min(m, n)` columns
             of every matrix in :attr:`A` are linearly independent.
             If this condition is not met, no error will be thrown, but the QR produced
             may be incorrect and its autodiff may fail or produce incorrect results.

Args:
    A (Tensor): tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.
    mode (str, optional): one of `'reduced'`, `'complete'`, `'r'`.
                          Controls the shape of the returned tensors. Default: `'reduced'`.

Keyword args:
    out (tuple, optional): output tuple of two tensors. Ignored if `None`. Default: `None`.

Returns:
    A named tuple `(Q, R)`.

Examples::

    >>> A = torch.tensor([[12., -51, 4], [6, 167, -68], [-4, 24, -41]])
    >>> Q, R = torch.linalg.qr(A)
    >>> Q
    tensor([[-0.8571,  0.3943,  0.3314],
            [-0.4286, -0.9029, -0.0343],
            [ 0.2857, -0.1714,  0.9429]])
    >>> R
    tensor([[ -14.0000,  -21.0000,   14.0000],
            [   0.0000, -175.0000,   70.0000],
            [   0.0000,    0.0000,  -35.0000]])
    >>> (Q @ R).round()
    tensor([[  12.,  -51.,    4.],
            [   6.,  167.,  -68.],
            [  -4.,   24.,  -41.]])
    >>> (Q.T @ Q).round()
    tensor([[ 1.,  0.,  0.],
            [ 0.,  1., -0.],
            [ 0., -0.,  1.]])
    >>> Q2, R2 = torch.linalg.qr(A, mode='r')
    >>> Q2
    tensor([])
    >>> torch.equal(R, R2)
    True
    >>> A = torch.randn(3, 4, 5)
    >>> Q, R = torch.linalg.qr(A, mode='complete')
    >>> torch.dist(Q @ R, A)
    tensor(1.6099e-06)
    >>> torch.dist(Q.mT @ Q, torch.eye(4))
    tensor(6.2158e-07)
""")

vander = _add_docstr(_linalg.linalg_vander, r"""
vander(x, N=None) -> Tensor

Generates a Vandermonde matrix.

Returns the Vandermonde matrix :math:`V`

.. math::

    V = \begin{pmatrix}
            1 & x_1 & x_1^2 & \dots & x_1^{N-1}\\
            1 & x_2 & x_2^2 & \dots & x_2^{N-1}\\
            1 & x_3 & x_3^2 & \dots & x_3^{N-1}\\
            \vdots & \vdots & \vdots & \ddots &\vdots \\
            1 & x_n & x_n^2 & \dots & x_n^{N-1}
        \end{pmatrix}.

for `N > 1`.
If :attr:`N`\ `= None`, then `N = x.size(-1)` so that the output is a square matrix.

Supports inputs of float, double, cfloat, cdouble, and integral dtypes.
Also supports batches of vectors, and if :attr:`x` is a batch of vectors then
the output has the same batch dimensions.

Differences with `numpy.vander`:

- Unlike `numpy.vander`, this function returns the powers of :attr:`x` in ascending order.
  To get them in the reverse order call ``linalg.vander(x, N).flip(-1)``.

Args:
    x (Tensor): tensor of shape `(*, n)` where `*` is zero or more batch dimensions
                consisting of vectors.

Keyword args:
    N (int, optional): Number of columns in the output. Default: `x.size(-1)`

Example::

    >>> x = torch.tensor([1, 2, 3, 5])
    >>> linalg.vander(x)
    tensor([[  1,   1,   1,   1],
            [  1,   2,   4,   8],
            [  1,   3,   9,  27],
            [  1,   5,  25, 125]])
    >>> linalg.vander(x, N=3)
    tensor([[ 1,  1,  1],
            [ 1,  2,  4],
            [ 1,  3,  9],
            [ 1,  5, 25]])
""")

vecdot = _add_docstr(_linalg.linalg_vecdot, r"""
linalg.vecdot(x, y, *, dim=-1, out=None) -> Tensor

Computes the dot product of two batches of vectors along a dimension.

In symbols, this function computes

.. math::

    \sum_{i=1}^n \overline{x_i}y_i.

over the dimension :attr:`dim` where :math:`\overline{x_i}` denotes the conjugate for complex
vectors, and it is the identity for real vectors.

Supports input of half, bfloat16, float, double, cfloat, cdouble and integral dtypes.
It also supports broadcasting.

Args:
    x (Tensor): first batch of vectors of shape `(*, n)`.
    y (Tensor): second batch of vectors of shape `(*, n)`.

Keyword args:
    dim (int): Dimension along which to compute the dot product. Default: `-1`.
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Examples::

    >>> v1 = torch.randn(3, 2)
    >>> v2 = torch.randn(3, 2)
    >>> linalg.vecdot(v1, v2)
    tensor([ 0.3223,  0.2815, -0.1944])
    >>> torch.vdot(v1[0], v2[0])
    tensor(0.3223)
""")
