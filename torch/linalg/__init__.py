# -*- coding: utf-8 -*-
import sys

import torch
from torch._C import _add_docstr, _linalg  # type: ignore[attr-defined]

Tensor = torch.Tensor

# Note: This not only adds doc strings for functions in the linalg namespace, but
# also connects the torch.linalg Python namespace to the torch._C._linalg builtins.

cholesky = _add_docstr(_linalg.linalg_cholesky, r"""
linalg.cholesky(input, *, out=None) -> Tensor

Computes the Cholesky decomposition of a complex Hermitian or real symmetric positive-definite matrix. Supports batched inputs.

For a complex Hermitian or real symmetric matrix :math:`A`, this is defined as

.. math::

    A = LL^{\text{H}}

where :math:`L` is a lower triangular matrix and
:math:`L^{\text{H}}` is the conjugate transpose when :math:`L` is complex, and the transpose when :math:`L` is real-valued.

If :attr:`input` is a batch of matrices, then the returned matrices are also batched with the
same batch dimensions.

Supports :attr:`input` of float, double, cfloat and cdouble dtypes.

.. note:: This function uses LAPACK's and MAGMA's `potrf` for CPU and CUDA inputs respectively.
          For CUDA inputs, this function synchronizes that device with the CPU.

.. seealso::

        :func:`torch.linalg.eigh` for a different decomposition of a Hermitian matrix.
        The eigenvalue decomposition gives more information about the matrix but it
        slower to compute than the Cholesky decomposition.

Args:
    input (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
                    consisting of symmetric or Hermitian positive-definite matrices.

Keyword args:
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Raises:
    RuntimeError: if the :attr:`input` matrix or any matrix in a batched :attr:`input` is not Hermitian
                  (resp. symmetric) positive-definite. If :attr:`input` is a batch of matrices,
                  the error message will include the batch index of the first matrix that fails
                  to meet this condition.

Examples::

    >>> a = torch.randn(2, 2, dtype=torch.complex128)
    >>> a = a @ a.t().conj()  + torch.eye(2) # creates a Hermitian positive-definite matrix
    >>> l = torch.linalg.cholesky(a)
    >>> a
    tensor([[2.5266+0.0000j, 1.9586-2.0626j],
            [1.9586+2.0626j, 9.4160+0.0000j]], dtype=torch.complex128)
    >>> l
    tensor([[1.5895+0.0000j, 0.0000+0.0000j],
            [1.2322+1.2976j, 2.4928+0.0000j]], dtype=torch.complex128)
    >>> l @ l.t().conj()
    tensor([[2.5266+0.0000j, 1.9586-2.0626j],
            [1.9586+2.0626j, 9.4160+0.0000j]], dtype=torch.complex128)

    >>> a = torch.randn(3, 2, 2, dtype=torch.float64)
    >>> a = a @ a.transpose(-2, -1) + torch.eye(2).squeeze(0)  # creates a batch of SPD matrices
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
    >>> torch.allclose(l @ l.transpose(-2, -1), a)
    True
""")

inv = _add_docstr(_linalg.linalg_inv, r"""
linalg.inv(input, *, out=None) -> Tensor

Computes the `inverse`_ of square matrix if it exists. Supports batched inputs

Supports :attr:`input` of float, double, cfloat and cdouble dtypes.

.. note:: This function is computed using LAPACK's `getrf` and `getri` for CPU inputs.
          For CUDA inputs, cuSOLVER's `getrf` and `getrs` as well as cuBLAS' `getrf`
          and `getri` are used if CUDA version >= 10.1.243, otherwise MAGMA's `getrf`
          and `getri` are used instead.
          For CUDA inputs, this function synchronizes that device with the CPU.

.. seealso::

        :func:`torch.linalg.pinv` computes the Moore-Penrose pseudoinverse of matrices of
        any shape.

        :func:torch.linalg.solve computes ``input.inv() @ other``.
        It is always prefered to use :func:`torch.linalg.solve` when possible, as it is
        faster and more stable than computing the inverse and then multiplying.

Args:
    input (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
                    consisting of invertible matrices.

Keyword args:
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Raises:
    RuntimeError: if the :attr:`input` matrix or any matrix in the batch :attr:`input` is not invertible

Examples::

    >>> x = torch.rand(4, 4)
    >>> y = torch.linalg.inv(x)
    >>> z = x @ y
    >>> z
    tensor([[ 1.0000, -0.0000, -0.0000,  0.0000],
            [ 0.0000,  1.0000,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  1.0000,  0.0000],
            [ 0.0000, -0.0000, -0.0000,  1.0000]])
    >>> torch.dist(z, torch.eye(4))
    tensor(1.1921e-07)

    >>> # Batched inverse example
    >>> x = torch.randn(2, 3, 4, 4)
    >>> y = torch.linalg.inv(x)
    >>> z = x @ y
    >>> torch.dist(z, torch.eye(4).expand_as(x))
    tensor(1.9073e-06)

    >>> x = torch.rand(4, 4, dtype=torch.cdouble)
    >>> y = torch.linalg.inv(x)
    >>> z = x @ y
    >>> torch.dist(z, torch.eye(4, dtype=torch.cdouble))
    tensor(7.5107e-16, dtype=torch.float64)

.. _inverse:
    https://en.wikipedia.org/wiki/Invertible_matrix
""")

det = _add_docstr(_linalg.linalg_det, r"""
linalg.det(input, *, out=None) -> Tensor

Computes the determinant of a square matrix. Supports batched inputs.

Supports :attr:`input` of float, double, cfloat and cdouble dtypes.

.. note:: This function is computed using :func:`torch.lu`.
          For CUDA inputs, this function synchronizes that device with the CPU.

.. note:: Backward through `det` internally uses :func:`torch.linalg.svd` when :attr:`input` is not
          invertible. In this case, double backward through `det` will be unstable when :attr:`input`
          doesn't have distinct singular values. See :func:`torch.linalg.svd` for more details.

.. seealso::

        :func:`torch.linalg.slogdet` computes the sign and logarithm of the norm of
        the determinant of square matrices.

Args:
    input (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions.

Keyword args:
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Example::

    >>> a = torch.randn(3, 3)
    >>> a
    tensor([[ 0.9478,  0.9158, -1.1295],
            [ 0.9701,  0.7346, -1.8044],
            [-0.2337,  0.0557,  0.6929]])
    >>> torch.linalg.det(a)
    tensor(0.0934)

    >>> out = torch.empty(0)
    >>> torch.linalg.det(a, out=out)
    tensor(0.0934)
    >>> out
    tensor(0.0934)

    >>> a = torch.randn(3, 2, 2)
    >>> a
    tensor([[[ 0.9254, -0.6213],
             [-0.5787,  1.6843]],

            [[ 0.3242, -0.9665],
             [ 0.4539, -0.0887]],

            [[ 1.1336, -0.4025],
             [-0.7089,  0.9032]]])
    >>> torch.linalg.det(a)
    tensor([1.1990, 0.4099, 0.7386])
""")

slogdet = _add_docstr(_linalg.linalg_slogdet, r"""
linalg.slogdet(input, *, out=None) -> (Tensor, Tensor)

Computes the sign and logarithm of the norm of the determinant of a square matrix. Supports batched inputs.

For a complex :attr:`input`, it returns the angle and the logarithm of the modulus of the
determinant, that is, a logarithmic polar decomposition of the determinant.

It returns a named tuple `(sign, logabsdet)`.
If :attr:`input` is a batch of matrices, then `sign`, `logabsdet` are also batched with the same
batch dimensions.

Supports :attr:`input` of float, double, cfloat and cdouble dtypes.
The output tensor `logabsdet` will always be real-valued, even when :attr:`input` is complex.
The output tensor `sign` will have the same dtype as :attr:`input`.


.. note:: This function is computed using :func:`torch.lu`.
          For CUDA inputs, this function synchronizes that device with the CPU.

.. note:: The determinant can be recovered as `sign * exp(logabsdet)`.

.. note:: When a matrix has a determinant of zero, it returns `(0, -inf)`.

.. seealso::

        :func:`torch.linalg.det` computes the determinant of square matrices.

Args:
    input (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions.

Keyword args:
    out (tuple, optional): output tuple of two tensors. Ignored if `None`. Default: `None`.

Example::

    >>> A = torch.randn(3, 3)
    >>> A
    tensor([[ 0.0032, -0.2239, -1.1219],
            [-0.6690,  0.1161,  0.4053],
            [-1.6218, -0.9273, -0.0082]])
    >>> torch.linalg.det(A)
    tensor(-0.7576)
    >>> torch.linalg.logdet(A)
    tensor(nan)
    >>> torch.linalg.slogdet(A)
    torch.return_types.linalg_slogdet(sign=tensor(-1.), logabsdet=tensor(-0.2776))
""")

eig = _add_docstr(_linalg.linalg_eig, r"""
linalg.eig(input, *, out=None) -> (Tensor, Tensor)

Computes the eigenvalue decomposition of a square matrix if it exists. Supports batched inputs.

For a square matrix :math:`A`, this is defined as

.. math::

    A = V \operatorname{diag}(L) V^{-1}

This decomposition exists if and only if :math:`A` is `diagonalizable`_. This is the case when all its eigenvalues are different.

The returned decomposition is a named tuple `(eigenvalues, eigenvectors)`,
which corresponds to :math:`L`, :math:`V` above. If :attr:`input` is a batch of matrices,
then `L`, `V` are also batched with the same batch dimensions.

Supports :attr:`input` of float, double, cfloat and cdouble dtypes.
The output tensors `eigenvalues` and `eigenvectors` will always be complex-valued, even when :attr:`input` is real.

.. note:: The eigenvalues and eigenvectors of a real matrix may be complex.

.. note:: This function is computed using LAPACK's and MAGMA's `geev` for CPU and CUDA inputs,
          respectively.
          For CUDA inputs, this function synchronizes that device with the CPU.

.. note:: The eigenvectors of a matrix are not unique, as multiplying an eigenvector by a
          non-zero number produces another set of eigenvectors for the matrix.
          As such, the returned eigenvectors are normalized to have norm
          `1` and largest real component.
          In particular, the eigenvectors are not continuous functions of the inputs.

.. warning:: This function assumes that :attr:`input` is `diagonalizable`_ (e.g. when all the
             eigenvalues are different). If it is not diagonalizable, the returned
             `eigenvalues` will be correct but

             .. code::

                input != eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.transpose(-2, -1)

.. warning:: Differentiation support for this function is not implemented yet.

.. seealso::

        :func:`torch.linalg.eigvals` computes only the eigenvalues.
        However, that function is not differentiable.

        :func:`torch.linalg.eigh` for a (faster) function that computes the eigenvalue decomposition
        for Hermitian and symmetric matrices.

        :func:`torch.linalg.svd` for a function that computes another type of spectral
        decomposition that works on matrices of any shape.

        :func:`torch.linalg.qr` for another (much faster) decomposition that works on matrices of
        any shape.

Args:
    input (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
                    consisting of diagonalizable matrices.

Keyword args:
    out (tuple, optional): output tuple of two tensors. Ignored if `None`. Default: `None`.

Examples::

    >>> a = torch.randn(2, 2, dtype=torch.complex128)
    >>> a
    tensor([[ 0.9828+0.3889j, -0.4617+0.3010j],
            [ 0.1662-0.7435j, -0.6139+0.0562j]], dtype=torch.complex128)
    >>> w, v = torch.linalg.eig(a)
    >>> w
    tensor([ 1.1226+0.5738j, -0.7537-0.1286j], dtype=torch.complex128)
    >>> v
    tensor([[ 0.9218+0.0000j,  0.1882-0.2220j],
            [-0.0270-0.3867j,  0.9567+0.0000j]], dtype=torch.complex128)
    >>> torch.allclose(torch.matmul(v, torch.matmul(w.diag_embed(), v.inverse())), a)
    True

    >>> a = torch.randn(3, 2, 2, dtype=torch.float64)
    >>> w, v = torch.linalg.eig(a)
    >>> torch.allclose(torch.matmul(v, torch.matmul(w.diag_embed(), v.inverse())).real, a)
    True

.. _diagonalizable:
    https://en.wikipedia.org/wiki/Diagonalizable_matrix#Definition
""")

eigvals = _add_docstr(_linalg.linalg_eigvals, r"""
linalg.eigvals(input, *, out=None) -> Tensor

Computes the eigenvalues of a square matrix. Supports batched inputs.

Supports :attr:`input` of float, double, cfloat and cdouble dtypes.
The output tensor will always be complex-valued, even when :attr:`input` is real.

.. note:: The eigenvalues of a real matrix may be complex.

.. note:: This function is computed using LAPACK's and MAGMA's `geev` for CPU and CUDA inputs,
          respectively.
          For CUDA inputs, this function synchronizes that device with the CPU.

.. note:: This function is not differentiable. Use :func:`torch.linalg.eig`
          instead, which also computes the eigenvectors.

Args:
    input (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions.

Keyword args:
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Examples::

    >>> a = torch.randn(2, 2, dtype=torch.complex128)
    >>> a
    tensor([[ 0.9828+0.3889j, -0.4617+0.3010j],
            [ 0.1662-0.7435j, -0.6139+0.0562j]], dtype=torch.complex128)
    >>> w = torch.linalg.eigvals(a)
    >>> w
    tensor([ 1.1226+0.5738j, -0.7537-0.1286j], dtype=torch.complex128)
""")

eigh = _add_docstr(_linalg.linalg_eigh, r"""
linalg.eigh(input, UPLO='L', *, out=None) -> (Tensor, Tensor)

Computes the eigenvalue decomposition of a complex Hermitian or real symmetric matrix. Supports batched inputs.

For a complex Hermitian or real symmetric matrix :math:`A`, this is defined as

.. math::

    A = V \operatorname{diag}(L) V^{\text{H}}

where :math:`V^{\text{H}}` is the conjugate transpose when :math:`V` is complex, and the transpose when :math:`V` is real-valued.
The matrix :math:`V` (and thus :math:`V^{\text{H}}`) is orthogonal in the real case,
and unitary in the complex case.

:attr:`input` is assumed to be Hermitian (resp. symmetric), but this is not checked internally.
When :attr:`UPLO` is `'L'` (default), only the lower triangular part of each matrix is used in the computation.
When :attr:`UPLO` is `'U'`, only the upper triangular part of each matrix is used.

The returned decomposition is represented as a namedtuple `(eigenvalues, eigenvectors)`,
which corresponds to :math:`L`, :math:`V` above. If :attr:`input` is a batch of matrices,
then `L`, `V` are also batched with the same batch dimensions.

The eigenvalues are returned in ascending order.

Supports :attr:`input` of float, double, cfloat and cdouble dtypes.
The output tensor `eigenvalues` will always be real-valued, even when :attr:`input` is complex.
The output tensor `eigenvectors` will have the same dtype as :attr:`input`.

.. note:: If the inputs are symmetric, this function is computed using LAPACK's and MAGMA's
          `syevd` for CPU and CUDA inputs,
          If the inputs are Hermitian, it is computed using LAPACK's and MAGMA's `heevd`.
          For CUDA inputs, this function synchronizes that device with the CPU.

.. note:: The eigenvalues of real symmetric or complex Hermitian matrices are always real.


.. warning:: The eigenvectors of a matrix are not unique. Any eigenvector may be multiplied by `-1`
             in the real case or by :math:`e^{i \phi}` for any :math:`\phi \in \mathbb{R}` in the
             complex case, and the resulting singular vectors will give a different
             eigendecomposition of the matrix.
             As such, the eigenvectors are not continuous functions of the inputs, and different
             platforms, like NumPy, or inputs on different devices, may produce different
             eigenvectors.

.. warning:: If `V` is used to compute gradients, the gradient will only be finite when
             :attr:`input` does not have repeated eigenvalues.
             Furthermore, if the distance between any two eigvalues is close to zero,
             the gradient will be numerically unstable, as it depends on the eigenvalues
             :math:`\lambda_i` through the computation of
             :math:`\frac{1}{\min_{i \neq j} \lambda_i - \lambda_j}`.

.. seealso::

        :func:`torch.linalg.eigvalsh` computes only the eigenvalues.
        However, that function is not differentiable.

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
    input (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
                    consisting of symmetric or Hermitian matrices.
    UPLO ('L', 'U', optional): controls whether to use the upper or lower triangular part
                               of :attr:`input` in the computations. Default: `'L'`.

Keyword args:
    out (tuple, optional): output tuple of two tensors. Ignored if `None`. Default: `None`.

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
linalg.eigvalsh(input, UPLO='L', *, out=None) -> Tensor

Computes the eigenvalues of a complex Hermitian or real symmetric matrix. Supports batched inputs.

The eigenvalues are returned in ascending order.

:attr:`input` is assumed to be Hermitian (resp. symmetric), but this is not checked internally.
When :attr:`UPLO` is `'L'` (default), only the lower triangular part of each matrix is used in the computation.
When :attr:`UPLO` is `'U'`, only the upper triangular part of each matrix is used.

Supports :attr:`input` of float, double, cfloat and cdouble dtypes.
The output tensor will always be real-valued, even when :attr:`input` is complex.

.. note:: This function is computed using LAPACK's and MAGMA's `syevd` / `heevd` for
          CPU and CUDA symmetric / Hermitian inputs respectively.
          For CUDA inputs, this function synchronizes that device with the CPU.

.. note:: The eigenvalues of real symmetric or complex Hermitian matrices are always real.

.. note:: This function is not differentiable. Please use :func:`torch.linalg.eigh`
          instead, which also computes the eigenvectors.

Args:
    input (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
                    consisting of symmetric or Hermitian matrices.
    UPLO ('L', 'U', optional): controls whether to use the upper or lower triangular part
                               of :attr:`input` in the computations. Default: `'L'`.

Keyword args:
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

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

householder_product = _add_docstr(_linalg.linalg_householder_product, r"""
householder_product(input, tau, *, out=None) -> Tensor

Computes the first `n` columns of a product of Householder matrices. Supports batched inputs.

Assume that :attr:`input` has shape `(*, m, n)` with `m >= n` and :attr:`tau` has shape
`(*, k)` with `k <= n` where `*` is zero or more batch dimensions. Denoting
:math:`v_i` `= \ `:attr:`input`\ `[..., :, i]` and :math:`\tau_i` `= \ `:attr:`tau`\ `[..., i]`,
for every element of the batch this function returns the first `n` columns of the matrix

.. math::

    H_1H_2 ... H_k \qquad\text{with}\qquad H_i = \mathrm{I}_m - \tau_i v_i v_i^{\text{H}}

where :math:`\mathrm{I}_m` is the `m`-dimensional identity matrix and
:math:`v^{\text{H}}` is the conjugate transpose when :math:`v` is complex, and the transpose when :math:`v` is real-valued.
The output has shape `(*, m, n)`.

Supports :attr:`input` of float, double, cfloat and cdouble dtypes.
:attr:`tau` should have the same dtype as :attr:`input`.

See `Representation of Orthogonal or Unitary Matrices`_ for further details.

.. note:: This function is computed using LAPACK's `orgqr` for CPU inputs,
          and cuSOLVER's `orgqr` for CUDA inputs if CUDA version >= 10.1.243.

.. note:: This function only uses the values strictly below the main diagonal of :attr:`input`.
          The other values are ignored.

.. seealso::

        :func:`torch.geqrf` is used together with this function to form the Q from the QR decomposition.

Args:
    input (Tensor): tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.
    tau (Tensor): tensor of shape `(*, k)` where `*` is zero or more batch dimensions.

Keyword args:
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Raises:
    RuntimeError: if :attr:`input` doesn't satisfy the requirement `m >= n`,
                  or :attr:`tau` doesn't satisfy the requirement `n >= r`

Examples::

    >>> a = torch.randn(2, 2)
    >>> h, tau = torch.geqrf(a)
    >>> q = torch.linalg.householder_product(h, tau)
    >>> torch.allclose(q, torch.linalg.qr(a)[0])
    True

    >>> h = torch.randn(3, 2, 2, dtype=torch.complex128)
    >>> tau = torch.randn(3, 1, dtype=torch.complex128)
    >>> q = torch.linalg.householder_product(h, tau)
    >>> q
    tensor([[[ 1.8034+0.4184j,  0.2588-1.0174j],
            [-0.6853+0.7953j,  2.0790+0.5620j]],

            [[ 1.4581+1.6989j, -1.5360+0.1193j],
            [ 1.3877-0.6691j,  1.3512+1.3024j]],

            [[ 1.4766+0.5783j,  0.0361+0.6587j],
            [ 0.6396+0.1612j,  1.3693+0.4481j]]], dtype=torch.complex128)

.. _Representation of Orthogonal or Unitary Matrices:
    https://www.netlib.org/lapack/lug/node128.html
""")

lstsq = _add_docstr(_linalg.linalg_lstsq, r"""
torch.linalg.lstsq(input, b, cond=None, *, driver=None) -> (Tensor, Tensor, Tensor, Tensor)

Computes the least squares solution of a system  of linear equations. Supports batched inputs.

Assume that :attr:`input`, :attr:`b` have shapes `(*, m, n)`, `(*, m, k)` where `*` is zero
or more batch dimensions. For every matrix :math:`A` in :attr:`input` and every set :math:`B`
in :attr:`b` of `k` vectors of dimension `m`, this function returns the solution to the problem

.. math::

    \min_{X \in \mathbb{K}^{n \times k}} \|AX - B\|_F

where :math:`\mathbb{K} = \mathbb{R}` or :math:`\mathbb{C}`, and :math:`\|-\|_F` denotes the Frobenius norm.
The output has shape `(*, m, k)`.

:attr:`driver` chooses the LAPACK/MAGMA driver that will be used.
For CPU inputs the valid values are `'gels'`, `'gelsy'`, `'gelsd`, `'gelss'`.
For CUDA input, the only valid driver is `'gels'`.
The driver `'gels'` assumes that the input is full-rank.
The drivers `'gelsy'`, `'gelsd'`, `'gelss'` handle rank-deficient inputs.
See the note below on how to choose the best driver. See also the `full description of these drivers`_

:attr:`cond` is used to determine the effective rank of the matrices in :attr:`input`
for rank-revealing drivers, that is, when :attr:`driver` is one of (`'gelsy'`, `'gelsd'`, `'gelss'`).
In this case, if :math:`\sigma_i` are the singular values of `A` in decreasing order, :math:`\sigma_i` will be rounded down to zero if
:math:`\sigma_i \leq \text{cond} \cdot \sigma_1`. If :attr:`cond` `= None` (default), :attr:`cond` is set to the machine precision of the dtype of :attr:`input`.

This function returns the solution to the problem and some extra information in a namedtuple of
four tensors `(solution, residuals, rank, singular_values)` containing:

- `solution`: the least squares solution
- `residuals`: the squared residuals of the solutions, that is :math:`\|AX - B\|_F^2`.
  It is computed when `m > n` and the matrix is full-rank,
  otherwise, it is an empty tensor.
- `rank`: tensor of ranks of the matrices in :attr:`input`
  It has shape equal to the batch dimensions of :attr:`input`.
  It is computed when :attr:`driver` is one of (`'gelsy'`, `'gelsd'`, `'gelss'`),
  otherwise it is an empty tensor.
- `singular_values`: tensor of singular values of the matrices in :attr:`input`.
  It has shape `(*, min(m, n))`.
  It is computed when :attr:`driver` is one of (`'gelsd'`, `'gelss'`),
  otherwise it is an empty tensor.

.. note::
    The solution to this problem can be computed in terms of the Moore-Penrose pseudoinverse of
    `A` as `A.pinv() @ B`.
    For this reason, this function is a faster and more stable way of computing `A.pinv() @ B`.

.. note::
    Choosing :attr:`driver`: if :attr:`input` is well-conditioned (its `condition number`_ is not
    too large) you should prefer `'gels'` (QR) if the output is full-rank and `'gelsy'`
    (QR with pivoting) otherwise.
    If :attr:`input` is not well-conditioned, you should almost always prefer `'gelsd'`
    (tridiagonal reduction and SVD), but try `'gelss'` (full SVD) if you run into memory issues.

.. note::
    The case when `m < n` is not supported on CUDA.

.. warning::
    The default value of :attr:`cond` may change in the future.
    It is therefore recommended to use a fixed value to avoid potential
    breaking changes.

Args:
    input (Tensor): lhs tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.
    b (Tensor): rhs tensor of shape `(*, m, k)` where `*` is zero or more batch dimensions.
    cond (float, optional): used to determine the effective rank of the input.
                            If :attr:`cond` `= None`, :attr:`cond` is set to the machine
                            of the dtype of :attr:`input`. Default: `None`.

Keyword args:
    driver (str, optional): name of the LAPACK/MAGMA method to be used.
        If `None`, `'gelsy'` is used for CPU inputs and `'gels'` for GPU inputs.
        Default: `None`.

Example::

    >>> a = torch.tensor([[10, 2, 3], [3, 10, 5], [5, 6, 12]], dtype=torch.float)
    >>> a.unsqueeze_(0)
    >>> b = torch.tensor([[[2, 5, 1], [3, 2, 1], [5, 1, 9]],
                          [[4, 2, 9], [2, 0, 3], [2, 5, 3]]], dtype=torch.float)
    >>> x = torch.linalg.lstsq(a, b).solution
    >>> torch.dist(x, a.pinverse() @ b)
    tensor(2.0862e-07)

    >>> sv = torch.linalg.lstsq(a, driver='gelsd').singular_values
    >>> torch.dist(sv, a.svd().S)
    tensor(5.7220e-06)

    >>> a[:, 0].zero_()
    >>> xx, rank, _ = torch.linalg.lstsq(a, b)
    >>> rank
    tensor([2])

.. _condition number:
    https://pytorch.org/docs/master/linalg.html#torch.linalg.cond
.. _full description of these drivers:
    https://www.netlib.org/lapack/lug/node27.html
""")

matrix_power = _add_docstr(_linalg.linalg_matrix_power, r"""
matrix_power(input, n, *, out=None) -> Tensor

Computes the :attr:`n`-th power of a matrix for an integer :attr:`n`. Supports batched inputs.

If :attr:`n` `= 0`, it returns the identity matrix (or batch) of the same shape
as :attr:`input`. If :attr:`n` is negative, it returns the inverse of each matrix
(if invertible) raised to the power of `abs(n)`.

Args:
    input (Tensor): tensor of shape `(*, m, m)` where `*` is zero or more batch dimensions.
    n (int): the exponent.

Keyword args:
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Example::

    >>> a = torch.randn(3, 3)
    >>> a
    tensor([[-0.2270,  0.6663, -1.3515],
            [-0.9838, -0.4002, -1.9313],
            [-0.7886, -0.0450,  0.0528]])
    >>> torch.linalg.matrix_power(a, 0)
    tensor([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])
    >>> torch.linalg.matrix_power(a, 3)
    tensor([[ 1.0756,  0.4980,  0.0100],
            [-1.6617,  1.4994, -1.9980],
            [-0.4509,  0.2731,  0.8001]])
    >>> torch.linalg.matrix_power(a.expand(2, -1, -1), -2)
    tensor([[[ 0.2640,  0.4571, -0.5511],
            [-1.0163,  0.3491, -1.5292],
            [-0.4899,  0.0822,  0.2773]],
            [[ 0.2640,  0.4571, -0.5511],
            [-1.0163,  0.3491, -1.5292],
            [-0.4899,  0.0822,  0.2773]]])
""")

matrix_rank = _add_docstr(_linalg.linalg_matrix_rank, r"""
matrix_rank(input, tol=None, hermitian=False, *, out=None) -> Tensor

Computes the numerical rank of a matrix. Supports batched inputs.

The matrix rank is computed as the number of singular values
(or eigenvalues in absolute value when :attr:`hermitian`\ `= True`)
that are greater than the specified :attr:`tol` threshold.

If :attr:`hermitian`\ `= True`, :attr:`input` is assumed to be Hermitian if complex or
symmetric if real.

If :attr:`tol` is not specified and :attr:`input` is a matrix of dimensions `(m, n)`,
the tolerance is set to be

.. math::

    \text{tol} = \sigma_1 \max(m, n) \varepsilon

where :math:`\sigma_1` is the largest singular value
(or eigenvalue in absolute value when :attr:`hermitian`\ `= True`), and
:math:`\varepsilon` is the epsilon value for the dtype of :attr:`input` (see :class:`torch.finfo`).
If :attr:`input` is a batch of matrices, :attr:`tol` is computed this way for every element of
the batch.

Supports :attr:`input` of float, double, cfloat and cdouble dtypes.

.. note:: The matrix rank is computed using singular value decomposition
          :func:`torch.linalg.svd` if :attr:`hermitian`\ `= False` (default) and the eigenvalue
          decomposition :func:`torch.linalg.eigvalsh` when :attr:`hermitian`\ `= True`.
          For CUDA inputs, this function synchronizes that device with the CPU.

Args:
    input (Tensor): the input matrix of shape `(m, n)` or the batch of matrices of shape `(*, m, n)`
                    where `*` is one or more batch dimensions.
    tol (float, Tensor, optional): the tolerance value. See above for the value it takes when `None`.
                                   Default: `None`.
    hermitian(bool, optional): indicates whether :attr:`input` is Hermitian (or symmetric if real).
                               Default: `False`.

Keyword args:
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

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

vector_norm = _add_docstr(_linalg.linalg_vector_norm, r"""
linalg.vector_norm(input, ord=None, dim=None, keepdim=False, *, dtype=None, out=None) -> Tensor

Computes the vector norm of a vector. Supports batched inputs.

If :attr:`input` is complex valued, it computes the norm of :attr:`input`\ `.abs()`

- If :attr:`dim`\ `= None`, the norm of the flattened :attr:`input` will be computed.
- If :attr:`dim` is an `int` or a `tuple`, the norm will be computed over these dimensions
  and the other dimensions will be treated as batch dimensions.

:attr:`ord` defines the vector norm that is computed. The following norms are supported:

======================   =======================================================
:attr:`ord`              vector norm
======================   =======================================================
`None` (default)         `2`-norm
`inf`                    `max(abs(x))`
`-inf`                   `min(abs(x))`
`0`                      `sum(x != 0)`
other `int` or `float`   `sum(abs(x)**\ `:attr:`ord`\ `)**(1./\ `:attr:`ord`\ `)`
======================   =======================================================

where `inf` refers to `float('inf')`, NumPy's `inf` object, or any equivalent object.

Supports :attr:`input` of float, double, cfloat and cdouble dtypes.
The output tensor will always be real-valued, even when :attr:`input` is complex.

Args:
    input (Tensor): a tensor of any shape.

    ord (int, float, inf, -inf, 'fro', 'nuc', optional): order of norm. Default: `None`

    dim (int, Tuple[int], optional): dimensions over which to compute
        the norm. See above for the behavior when :attr:`dim`\ `= None`.
        Default: `None`

    keepdim (bool, optional): If set to `True`, the reduced dimensions are retained
        in the result as dimensions with shape one. Default: `False`

Keyword args:
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

    dtype (:class:`torch.dtype`, optional): If specified, the input tensor is cast to
        :attr:`dtype` before performing the operation, and the returned tensor's type
        will be :attr:`dtype`. Default: `None`

Examples::

    >>> from torch import linalg as LA
    >>> a = torch.arange(9, dtype=torch.float) - 4
    >>> a
    tensor([-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])
    >>> b = a.reshape((3, 3))
    >>> b
    tensor([[-4., -3., -2.],
            [-1.,  0.,  1.],
            [ 2.,  3.,  4.]])
    >>> LA.vector_norm(a, ord=3.5)
    tensor(5.4345)
    >>> LA.vector_norm(b, ord=3.5)
    tensor(5.4345)
""")

multi_dot = _add_docstr(_linalg.linalg_multi_dot, r"""
linalg.multi_dot(tensors, *, out=None)

Efficiently multiplies two or more matrices by reordering the multiplications so that
the fewest arithmetic operations are performed.

Every tensor in :attr:`tensors` must be 2D, except for the first and last which
may be 1D. If the first tensor is a 1D vector of shape `n` it is treated as a row vector
of shape `(1, n)`, similarly if the last tensor is a 1D vector of shape `n` it is treated
as a column vector of shape `(n, 1)`.

If the first and last tensors are matrices, the output will be a matrix.
However, if either is a 1D vector, then the output will be a 1D vector.

.. warning:: This function does not broadcast.

.. note:: This function is implemented by chaining :func:`torch.mm` calls after
          computing the optimal matrix multiplication order.

.. note:: This function is similar to NumPy's `multi_dot` except that the first and last
          tensors must be either 1D or 2D whereas NumPy allows them to be nD.

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

    >>> a = torch.arange(2 * 3).view(2, 3)
    >>> b = torch.arange(3 * 2).view(3, 2)
    >>> c = torch.arange(2 * 2).view(2, 2)
    >>> multi_dot((a, b, c))
    tensor([[ 26,  49],
            [ 80, 148]])

    >>> multi_dot((a.to(torch.float), torch.empty(3, 0), torch.empty(0, 2)))
    tensor([[0., 0.],
            [0., 0.]])
""")

norm = _add_docstr(_linalg.linalg_norm, r"""
linalg.norm(input, ord=None, dim=None, keepdim=False, *, out=None, dtype=None) -> Tensor

Computes the vector norm of a vector or the matrix norm of a matrix. Supports batched inputs.

If :attr:`input` is complex valued, it computes the norm of :attr:`input`\ `.abs()`

- If :attr:`dim` is an `int`, the vector norm will be computed.
- If :attr:`dim` is a `2`-`tuple`, the matrix norm will be computed.
- If :attr:`dim`\ `= None` and :attr:`ord` `= None`, :attr:`input` is flattened to 1D and the `2`-norm of the resulting vector is returned.
- If :attr:`dim`\ `= None` and :attr:`ord` `!= None`, :attr:`input` must be 1D or 2D.

:attr:`ord` defines the norm that is computed. The following norms are supported:

======================     =========================  ========================================================
:attr:`ord`                norm for matrices          norm for vectors
======================     =========================  ========================================================
`None` (default)           Frobenius norm             `2`-norm
`'fro'`                    Frobenius norm             -- not supported --
`'nuc'`                    nuclear norm               -- not supported --
`inf`                      `max(sum(abs(x), dim=1))`  `max(abs(x))`
`-inf`                     `min(sum(abs(x), dim=1))`  `min(abs(x))`
`0`                        -- not supported --        `sum(x != 0)`
`1`                        `max(sum(abs(x), dim=0))`  as below
`-1`                       `min(sum(abs(x), dim=0))`  as below
`2`                        largest singular value     as below
`-2`                       smallest singular value    as below
other `int` or `float`     -- not supported --        `sum(abs(x)**\ `:attr:`ord`\ `)**(1./\ `:attr:`ord`\ `)`
======================     =========================  ========================================================

where `inf` refers to `float('inf')`, NumPy's `inf` object, or any equivalent object.

Supports :attr:`input` of float, double, cfloat and cdouble dtypes.
The output tensor will always be real-valued, even when :attr:`input` is complex.

Args:
    input (Tensor): the input tensor. The allowed shapes follow the rules described above.

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

Examples::

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

svd = _add_docstr(_linalg.linalg_svd, r"""
linalg.svd(input, full_matrices=True, compute_uv=True, *, out=None) -> (Tensor, Tensor, Tensor)

Computes the singular value decomposition of a matrix. Supports batched inputs.

For a matrix :math:`A`, this is defined as

.. math::

    A = U \operatorname{diag}(S) V^{\text{H}}

where :math:`V^{\text{H}}` is the conjugate transpose when :math:`V` is complex, and the transpose when :math:`V` is real-valued.
The matrices  :math:`U`, :math:`V` (and thus :math:`V^{\text{H}}`) are orthogonal in the real case,
and unitary in the complex case.

The returned decomposition is a named tuple `(U, S, Vh)`
which corresponds to :math:`U`, :math:`S`, :math:`V^{\text{H}}` above.
If :attr:`input` is a batch of matrices, then `U`, `S`, `Vh` are also batched with the same
batch dimensions.

The singular values are returned in descending order. If :attr:`input` is a batch of matrices,
then the singular values of each matrix in the batch are also returned in descending order.

If :attr:`full_matrices` `= False`, the method will return the reduced singular
value decomposition. In this case, if :attr:`input` has shape `(*, m, n)`,
the returned `U`, `Vh` will have shape `(*, m, min(m, n))` and `(*, min(m, n), n)`
respectively.

If :attr:`compute_uv` `= False`, the returned `U` and `Vh` will be empty
tensors with no elements and the same device as :attr:`input`. The
:attr:`full_matrices` argument has no effect when :attr:`compute_uv` `= False`.

Supports :attr:`input` of float, double, cfloat and cdouble dtypes.
The output tensor `S` will always be real-valued, even when :attr:`input` is complex.
The output tensors `U` and `Vh` will have the same dtype as :attr:`input`.

Differences with `numpy.linalg.svd`:

- Unlike `numpy.linalg.svd`, this function always returns a tuple of three tensors.
  When :attr:`compute_uv` `= False`, `U`, `Vh` are empty tensors.
  This behavior may change in a future PyTorch release.
  Please use :func:`torch.linalg.svdvals`, which computes only the singular values,
  instead of ``compute_uv=False``.

.. note:: The `S` tensor can only be used to compute gradients if :attr:`compute_uv` `= True`.

.. note:: When :attr:`full_matrices` `= True`, the gradients on `U[..., :, min(m, n):]`
          and `Vh[..., min(m, n):, :]` will be ignored in the backwards pass, as those vectors
          can be arbitrary bases of the corresponding subspaces.

.. note:: On CPU, this function uses LAPACK's `gesdd` instead of `gesvd` for speed.
          On CUDA, it uses cuSOLVER's `gesvdj` and `gesvdjBatched` on CUDA 10.1.243 and later,
          and MAGMA's `gesdd` on earlier versions of CUDA.

.. note:: The returned `U` will not be contiguous. The matrix (or batch of matrices) will
          be represented as a column-major matrix (i.e. Fortran-contiguous).

.. warning:: The singular vectors of a matrix are not unique. Any pair of singular
             vectors :math:`u_k, v_k` may be multiplied by `-1` when `U` and `V` are real-valued or
             by :math:`e^{i \phi}` for any :math:`\phi \in \mathbb{R}` when `U` and `V` are complex,
             and the resulting singular vectors will give a different SVD of the matrix.
             This non-uniqueness problem is even worse when the matrix has repeated singular values.
             In this case, one may multiply the associated singular vectors of `U` and `V` spanning
             the subspace by a rotation matrix and `the resulting vectors will span the same subspace`_.
             As such, `U` and `Vh` are not continuous functions of the inputs. Furthermore, different
             platforms, like NumPy, or inputs on different devices, may produce different
             `U` and `Vh` tensors.

.. warning:: If `U` or `Vh` are used to compute gradients, the gradient will only be finite when
             :attr:`input` does not have zero as a singular value or repeated singular values.
             Furthermore, if the distance between any two singular values is close to zero,
             the gradient will be numerically unstable, as it depends on the singular values
             :math:`\sigma_i` through the computation of
             :math:`\frac{1}{\min_{i \neq j} \sigma_i^2 - \sigma_j^2}`.
             The gradient will also be numerically unstable when :attr:`input` has small singular
             values, as it also depends on the computaiton of :math:`\frac{1}{\sigma_i}`.

.. seealso::

        :func:`torch.linalg.eig` for a function that computes another type of spectral
        decomposition of a matrix. The eigendecomposition works just on on square matrices.

        :func:`torch.linalg.eigh` for a (faster) function that computes the eigenvalue decomposition
        for Hermitian and symmetric matrices.

        :func:`torch.linalg.qr` for another (much faster) decomposition that works on general
        matrices.

Args:
    input (Tensor): tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.
    full_matrices (bool, optional): controls whether to compute the full or reduced decomposition, and
                                    consequently, the shape of returned `U` and `Vh`. Default: `True`.
    compute_uv (bool, optional): controls whether to compute `U` and `Vh`. Default: `True`.
    out (tuple, optional): output tuple of three tensors. Ignored if `None`.
                           If :attr:`compute_uv` `= False`, the 1st and 3rd arguments must be
                           tensors, but they are ignored. For example, you can pass
                           `(torch.tensor([]), out_S, torch.tensor([]))`. Default: `None`

Example::

    >>> a = torch.randn(5, 3)
    >>> a
    tensor([[-0.3357, -0.2987, -1.1096],
            [ 1.4894,  1.0016, -0.4572],
            [-1.9401,  0.7437,  2.0968],
            [ 0.1515,  1.3812,  1.5491],
            [-1.8489, -0.5907, -2.5673]])
    >>>
    >>> # reconstruction in the full_matrices=False case
    >>> u, s, vh = torch.linalg.svd(a, full_matrices=False)
    >>> u.shape, s.shape, vh.shape
    (torch.Size([5, 3]), torch.Size([3]), torch.Size([3, 3]))
    >>> torch.dist(a, u @ torch.diag(s) @ vh)
    tensor(1.0486e-06)
    >>>
    >>> # reconstruction in the full_matrices=True case
    >>> u, s, vh = torch.linalg.svd(a)
    >>> u.shape, s.shape, vh.shape
    (torch.Size([5, 5]), torch.Size([3]), torch.Size([3, 3]))
    >>> torch.dist(a, u[:, :3] @ torch.diag(s) @ vh)
    >>> torch.dist(a, u[:, :3] @ torch.diag(s) @ vh)
    tensor(1.0486e-06)
    >>>
    >>> # extra dimensions
    >>> a_big = torch.randn(7, 5, 3)
    >>> u, s, vh = torch.linalg.svd(a_big, full_matrices=False)
    >>> torch.dist(a_big, u @ torch.diag_embed(s) @ vh)
    tensor(3.0957e-06)

.. _the resulting vectors will span the same subspace:
    https://en.wikipedia.org/wiki/Singular_value_decomposition#Singular_values,_singular_vectors,_and_their_relation_to_the_SVD
""")

svdvals = _add_docstr(_linalg.linalg_svdvals, r"""
linalg.svdvals(input, *, out=None) -> Tensor

Computes the singular values of a matrix.

The singular values are returned in descending order. If :attr:`input` is a batch of matrices,
then the singular values of each matrix in the batch are returned in descending order.

Supports inputs of float, double, cfloat and cdouble dtypes.
Also supports batched inputs, and, if the input is batched, the output is batched with the same dimensions.

.. note:: This function is not differentiable. Please use :func:`torch.linalg.svd`
          instead, which also computes the singular vectors.

.. note:: This function is equivalent to NumPy's ``linalg.svd`` with ``compute_uv=False``.

.. note:: When given inputs on a CUDA device, this function synchronizes that device with the CPU.

Args:
    input (Tensor): the `m \times n` matrix or the batch of such matrices of size
                    `(*, m, n)` where `*` is one or more batch dimensions.

Keyword args:
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Example::

    >>> import torch
    >>> a = torch.randn(5, 3)
    >>> a
    tensor([[-1.3490, -0.1723,  0.7730],
            [-1.6118, -0.3385, -0.6490],
            [ 0.0908,  2.0704,  0.5647],
            [-0.6451,  0.1911,  0.7353],
            [ 0.5247,  0.5160,  0.5110]])
    >>> s = torch.linalg.svdvals(a)
    >>> s
    tensor([2.5139, 2.1087, 1.1066])
""")

cond = _add_docstr(_linalg.linalg_cond, r"""
linalg.cond(input, p=None, *, out=None) -> Tensor

Computes the condition number of a matrix with respect to a norm. Supports batched inputs.

The condition number :math:`\kappa` of a matrix :math:`A` is defined as

.. math::

    \kappa(A) = \|A\|_p\|A^{-1}\|_p

The condition number of `A` measures the numerical stability of the linear system `AX = B`
in some specific matrix norm.

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
As such, in this case, the matrix (or every matrix in the batch) :attr:`input` should be square
and invertible.

For :attr:`p` in `(2, -2)`, this function can be computed in terms of the singular values
:math:`\sigma_1 \geq \ldots \geq \sigma_n`

.. math::

    \kappa_2(A) = \frac{\sigma_1}{\sigma_n}\qquad \kappa_{-2}(A) = \frac{\sigma_n}{\sigma_1}

In these cases, it is computed using :func:`torch.linalg.svd`. For these norms, the matrix
(or every matrix in the batch) :attr:`input` may be rectangular

Supports :attr:`input` of float, double, cfloat and cdouble dtypes.
The output tensor will always be real-valued, even when :attr:`input` is complex.

.. note :: For CUDA inputs, this function synchronizes that device with the CPU if
           if :attr:`p` is one of `('fro', 'nuc', inf, -inf, 1, -1)`.

.. seealso::

        :func:`torch.linalg.solve` for a function that solves linear systems of square matrices.

        :func:`torch.linalg.lstsq` for a function that solves linear systems of general matrices.

Args:
    input (Tensor): tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions
                    for :attr:`p` in `(2, -2)`, and of shape `(*, n, n)` where every matrix
                    is invertible for :attr:`p` in `('fro', 'nuc', inf, -inf, 1, -1)`.
    p (int, inf, -inf, 'fro', 'nuc', optional):
        the type of the matrix norm to use in the computations (see above). Default: `None`

Keyword args:
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Raises:
    RuntimeError:
        if :attr:`p` is one of `('fro', 'nuc', inf, -inf, 1, -1)`
        and the :attr:`input` matrix or any matrix in the batch :attr:`input` is not square
        or invertible.


Examples::

    >>> a = torch.randn(3, 4, 4, dtype=torch.complex64)
    >>> torch.linalg.cond(a)
    >>> a = torch.tensor([[1., 0, -1], [0, 1, 0], [1, 0, 1]])
    >>> torch.linalg.cond(a)
    tensor([1.4142])
    >>> torch.linalg.cond(a, 'fro')
    tensor(3.1623)
    >>> torch.linalg.cond(a, 'nuc')
    tensor(9.2426)
    >>> torch.linalg.cond(a, float('inf'))
    tensor(2.)
    >>> torch.linalg.cond(a, float('-inf'))
    tensor(1.)
    >>> torch.linalg.cond(a, 1)
    tensor(2.)
    >>> torch.linalg.cond(a, -1)
    tensor(1.)
    >>> torch.linalg.cond(a, 2)
    tensor([1.4142])
    >>> torch.linalg.cond(a, -2)
    tensor([0.7071])

    >>> a = torch.randn(2, 3, 3)
    >>> a
    tensor([[[-0.9204,  1.1140,  1.2055],
            [ 0.3988, -0.2395, -0.7441],
            [-0.5160,  0.3115,  0.2619]],

            [[-2.2128,  0.9241,  2.1492],
            [-1.1277,  2.7604, -0.8760],
            [ 1.2159,  0.5960,  0.0498]]])
    >>> torch.linalg.cond(a)
    tensor([[9.5917],
            [3.2538]])

    >>> a = torch.randn(2, 3, 3, dtype=torch.complex64)
    >>> a
    tensor([[[-0.4671-0.2137j, -0.1334-0.9508j,  0.6252+0.1759j],
            [-0.3486-0.2991j, -0.1317+0.1252j,  0.3025-0.1604j],
            [-0.5634+0.8582j,  0.1118-0.4677j, -0.1121+0.7574j]],

            [[ 0.3964+0.2533j,  0.9385-0.6417j, -0.0283-0.8673j],
            [ 0.2635+0.2323j, -0.8929-1.1269j,  0.3332+0.0733j],
            [ 0.1151+0.1644j, -1.1163+0.3471j, -0.5870+0.1629j]]])
    >>> torch.linalg.cond(a)
    tensor([[4.6245],
            [4.5671]])
    >>> torch.linalg.cond(a, 1)
    tensor([9.2589, 9.3486])
""")

pinv = _add_docstr(_linalg.linalg_pinv, r"""
linalg.pinv(input, rcond=1e-15, hermitian=False, *, out=None) -> Tensor

Computes the Moore-Penrose pseudoinverse of a matrix. Supports batched inputs.

If :attr:`hermitian`\ `= True`, :attr:`input` should be a complex Hermitian or real symmetric
matrix or batch of matrices. In this case, it will just use the lower-triangular half of the matrix.

The singular values (or the norm of the eigenvalues when :attr:`hermitian`\ `= True`)
that are below the specified :attr:`rcond` threshold are treated as zero and discarded in the computation.

Supports :attr:`input` of float, double, cfloat and cdouble dtypes.


.. note:: This function uses :func:`torch.linalg.svd` if :attr:`hermitian`\ `= False` and
          :func:`torch.linalg.eigh` if :attr:`hermitian`\ `= True`.
          For CUDA inputs, this function synchronizes that device with the CPU.

.. seealso::

        :func:`torch.linalg.inv` computes the inverse of a square matrix.

        :func:torch.linalg.lstsq computes ``input.pinv() @ other``.
        It is always prefered to use :func:`torch.linalg.lstsq` when possible, as it is
        faster and more stable than computing the pseudoinverse and then multiplying.

Args:
    input (Tensor): tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.
    rcond (float or Tensor, optional): the tolerance value to determine when is a singular value zero
                                       If it is a :class:`torch.Tensor`, its shape must be
                                       broadcastable to that of the singular values of
                                       :attr:`input` as returned by :func:`torch.svd`.
                                       Default: `1e-15`.
    hermitian(bool, optional): indicates whether :attr:`input` is Hermitian if complex
                               or symmetric if real. Default: `False`.

Keyword args:
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Examples::

    >>> input = torch.randn(3, 5)
    >>> input
    tensor([[ 0.5495,  0.0979, -1.4092, -0.1128,  0.4132],
            [-1.1143, -0.3662,  0.3042,  1.6374, -0.9294],
            [-0.3269, -0.5745, -0.0382, -0.5922, -0.6759]])
    >>> torch.linalg.pinv(input)
    tensor([[ 0.0600, -0.1933, -0.2090],
            [-0.0903, -0.0817, -0.4752],
            [-0.7124, -0.1631, -0.2272],
            [ 0.1356,  0.3933, -0.5023],
            [-0.0308, -0.1725, -0.5216]])

    Batched linalg.pinv example
    >>> a = torch.randn(2, 6, 3)
    >>> b = torch.linalg.pinv(a)
    >>> torch.matmul(b, a)
    tensor([[[ 1.0000e+00,  1.6391e-07, -1.1548e-07],
            [ 8.3121e-08,  1.0000e+00, -2.7567e-07],
            [ 3.5390e-08,  1.4901e-08,  1.0000e+00]],

            [[ 1.0000e+00, -8.9407e-08,  2.9802e-08],
            [-2.2352e-07,  1.0000e+00,  1.1921e-07],
            [ 0.0000e+00,  8.9407e-08,  1.0000e+00]]])

    Hermitian input example
    >>> a = torch.randn(3, 3, dtype=torch.complex64)
    >>> a = a + a.t().conj()  # creates a Hermitian matrix
    >>> b = torch.linalg.pinv(a, hermitian=True)
    >>> torch.matmul(b, a)
    tensor([[ 1.0000e+00+0.0000e+00j, -1.1921e-07-2.3842e-07j,
            5.9605e-08-2.3842e-07j],
            [ 5.9605e-08+2.3842e-07j,  1.0000e+00+2.3842e-07j,
            -4.7684e-07+1.1921e-07j],
            [-1.1921e-07+0.0000e+00j, -2.3842e-07-2.9802e-07j,
            1.0000e+00-1.7897e-07j]])

    Non-default rcond example
    >>> rcond = 0.5
    >>> a = torch.randn(3, 3)
    >>> torch.linalg.pinv(a)
    tensor([[ 0.2971, -0.4280, -2.0111],
            [-0.0090,  0.6426, -0.1116],
            [-0.7832, -0.2465,  1.0994]])
    >>> torch.linalg.pinv(a, rcond)
    tensor([[-0.2672, -0.2351, -0.0539],
            [-0.0211,  0.6467, -0.0698],
            [-0.4400, -0.3638, -0.0910]])

    Matrix-wise rcond example
    >>> a = torch.randn(5, 6, 2, 3, 3)
    >>> rcond = torch.rand(2)  # different rcond values for each matrix in a[:, :, 0] and a[:, :, 1]
    >>> torch.linalg.pinv(a, rcond)
    >>> rcond = torch.randn(5, 6, 2) # different rcond value for each matrix in 'a'
    >>> torch.linalg.pinv(a, rcond)
""")

solve = _add_docstr(_linalg.linalg_solve, r"""
linalg.solve(input, other, *, out=None) -> Tensor

Computes the solution of a square system  of linear equations with unique solution. Supports batched inputs.

This method accepts both matrices and vectors as the right-hand side of the linear equation.
We will denote by `*` zero or more batch dimensions.

- If :attr:`input`, :attr:`other` have shapes `(*, n, n)`, `(*, n)`, for every matrix :math:`A`
  in :attr:`input` and vector :math:`b` in :attr:`other`, this function returns a vector :math:`x` such that

  .. math:: Ax = b

  The returned tensor have shape `(*, n)`.
- If :attr:`input`, :attr:`other` have shapes `(*, n, n)`, `(*, n, k)`, for every pair of matrices
  :math:`A` in :attr:`input` and :math:`B` in :attr:`other`, this function returns a matrix :math:`X` such that

  .. math:: AX = B

  The returned tensor have shape `(*, n, k)`.
- If none of the above apply and :attr:`input`, :attr:`other` have shapes `(*, n, n)`, `(n,)` (resp. `(n, k)`),
  the vector (resp. matrix) :attr:`other` is broadcasted to have shape `(*, n)` (resp. `(*, n, k)`).
  This function then  returns the solution of the resulting batch of systems of linear equations.

Supports :attr:`input` of float, double, cfloat and cdouble dtypes.
:attr:`other` should have the same dtype as :attr:`input`.

.. note:: The solution to these problems can be computed in terms of the inverse of `A`
          as `A.inv() @ b` and `A.inv() @ B` respectively.
          This function can be seen as faster and more stable way of computing these quantities.


.. note:: For CUDA inputs, this function synchronizes that device with the CPU.

Args:
    input (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions.
    other (Tensor): right-hand side tensor of shape `(*, n)` or  `(*, n, k)` or `(n,)` or `(n, k)`
                    according to the rules described above

Keyword args:
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Raises:
    RuntimeError: if the :attr:`input` matrix is not invertible or any matrix in a batched :attr:`input`
                  is not invertible

Examples::

    >>> A = torch.rand(3, 3)
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

Broadcasting::

    >>> A = torch.randn(2, 3, 3)
    >>> b = torch.randn(3, 1)
    >>> x = torch.linalg.solve(A, b) # b is broadcasted to size (2, 3, 1)
    >>> x.shape
    torch.Size([2, 3, 1])
    >>> torch.allclose(A @ x, b)
    True
    >>> b = torch.rand(3)
    >>> x = torch.linalg.solve(A, b) # b is broadcasted to size (2, 3)
    >>> x.shape
    torch.Size([2, 3])
    >>> Ax = A @ x.unsqueeze(-1)
    >>> torch.allclose(Ax, b.unsqueeze(-1).expand_as(Ax))
    True
""")

tensorinv = _add_docstr(_linalg.linalg_tensorinv, r"""
linalg.tensorinv(input, ind=2, *, out=None) -> Tensor

Computes the multiplicative inverse of :func:`torch.tensordot`.

If `m` is the product of the first :attr:`ind` dimensions of :attr:`input` and `n` is the product of
the rest of the dimensions, this function expects `m` and `n` to be equal.
If this is the case, it computes a tensor `x` such that
`tensordot(\ `:attr:`input`\ `, x, \ `:attr:`ind`\ `)` is the identity matrix in dimension `m`.
`x` will have the shape of :attr:`input` but with the first :attr:`ind` dimensions pushed back to the end

.. code:: text

    x.shape == input.shape[ind:] + input.shape[:ind]

Supports :attr:`input` of float, double, cfloat and cdouble dtypes.

.. note:: When :attr:`input` is a `2`-dimensional tensor and :attr:`ind`\ `=1`,
          this function computes the (multiplicative) inverse of :attr:`input`
          (see :func:`torch.linalg.inv`).

.. seealso::

        :func:`torch.linalg.tensorsolve` computes
        `torch.tensordot(tensorinv(\ `:attr:`input`\ `), \ `:attr:`other`\ `)`.
        It is always prefered to use :func:`~tensorsolve` when possible, as it is
        faster and more stable than using
        :func:`~tensorinv` followed by :func:`torch.tensordot`.

Args:
    input (Tensor): tensor to invert. Its shape must satisfy
                    `prod(\ `:attr:`input`\ `.shape[:\ `:attr:`ind`\ `]) ==
                    prod(\ `:attr:`input`\ `.shape[\ `:attr:`ind`\ `:])`.
    ind (int): index at which to compute the inverse of :func:`torch.tensordot`. Default: `2`.

Keyword args:
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Raises:
    RuntimeError: if the reshaped :attr:`input` is not invertible or the product of the first
                  :attr:`ind` dimensions is not equal to the product of the rest.

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

Computes the solution `X` to the system `torch.tensordot(input, X) = other`.

If `m` is the product of the first :attr:`other`\ `.ndim`  dimensions of :attr:`input` and
`n` is the product of the rest of the dimensions, this function expects `m` and `n` to be equal.

The returned tensor `x` satisfies
`tensordot(\ `:attr:`input`\ `, x, dims=x.ndim) == \ `:attr:`other`.
`x` has shape :attr:`input`\ `[other.ndim:]`.

If :attr:`dims` is specified, :attr:`input` will be reshaped as

.. code:: text

    input = movedim(input, dims, range(len(dims) - input.ndim + 1, 0))

Supports :attr:`input` of float, double, cfloat and cdouble dtypes.

Args:
    input (Tensor): tensor to solve for. Its shape must satisfy
                    `prod(\ `:attr:`input`\ `.shape[:\ `:attr:`other`\ `.ndim]) ==
                    prod(\ `:attr:`input`\ `.shape[\ `:attr:`other`\ `.ndim:])`.
    other (Tensor): tensor of shape :attr:`input`\ `.shape[\ `:attr:`other`\ `.ndim]`.
    dims (Tuple[int], optional): dimensions of :attr:`input` to be moved.
        If `None`, no dimensions are moved. Default: `None`.

.. seealso::

        :func:`torch.linalg.tensorinv` computes the multiplicative inverse of
        :func:`torch.tensordot`.

Keyword args:
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Raises:
    RuntimeError: if the reshaped :attr:`input`\ `.view(m, m)` with `m` as above  is not
                  invertible or the product of the first :attr:`ind` dimensions is not equal
                  to the product of the rest of the dimensions.

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

Computes the QR decomposition of a matrix. Supports batched inputs.

For a matrix :math:`A`, this is defined as

.. math::

    A = QR

where :math:`Q` is orthogonal in the real case and unitary in the complex case, and :math:`R` is upper triangular.

The returned decomposition is a named tuple `(Q, R)`.
If :attr:`input` is a batch of matrices, then `Q`, `R` are also batched with the same
batch dimensions.

The parameter :attr:`mode` controls the shape of the output. If :attr:`input` has shape
`(*, m, n)`, denoting `k = min(m, n)`

- :attr:`mode` `= 'reduced'` (default): Returns `(Q, R)` of shapes `(*, m, k)`, `(*, k, n)` respectively.
- :attr:`mode` `= 'complete'`: Returns `(Q, R)` of shapes `(*, m, m)`, `(*, m, n)` respectively.
- :attr:`mode` `= 'r'`: Computes only `R`. Returns `(Q, R)` with `Q` empty and `R` of shape `(*, k, n)`.

Differences with `numpy.linalg.qr`:

- :attr:`mode` `= 'raw'` is not implemented.
- Unlike `numpy.linalg.qr`, this function always returns a tuple of two tensors.
  When :attr:`mode` `= 'r'`, the `Q` tensor is an empty tensor.
  This behavior may change in a future PyTorch release.

.. note:: The elements in the diagonal of `R` are not necessarily positive.

.. note:: :attr:`mode` `= 'r'` does not support backpropagation. Use :attr:`mode` `= 'reduced'` instead.

.. note:: This function uses LAPACK and MAGMA for CPU and CUDA inputs respectively.

.. warning:: The QR decomposition is only locally unique when the input matrix has
             independent columns. If this is not the case, different platforms, like NumPy,
             or inputs on different devices, may produce different valid decompositions.

.. warning:: Backpropagation is only supported if the first `k = min(m, n)` columns
             of every matrix in :attr:`input` are linearly independent.
             If this condition is not met, no error will be thrown, but the gradient produced
             may be anything.
             This is because the QR decomposition is not differentiable at these points.

Args:
    input (Tensor): tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.
    mode (str, optional): one of `'reduced'`, `'complete'`, `'r'`.
                          Controls the shape of the output tensors. Default: `'reduced'`.

Keyword args:
    out (tuple, optional): output tuple of two tensors. Ignored if `None`. Default: `None`.

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
    >>> q2, r2 = torch.linalg.qr(a, mode='r')
    >>> q2
    tensor([])
    >>> torch.equal(r, r2)
    True
    >>> a = torch.randn(3, 4, 5)
    >>> q, r = torch.linalg.qr(a, mode='complete')
    >>> torch.allclose(torch.matmul(q, r), a, atol=1e-5)
    True
    >>> torch.allclose(torch.matmul(q.transpose(-2, -1), q), torch.eye(4), atol=1e-5)
    True
""")
