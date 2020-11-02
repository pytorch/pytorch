import sys

import torch
from torch._C import _add_docstr, _linalg  # type: ignore

import functools

Tensor = torch.Tensor

# Note: This not only adds doc strings for functions in the linalg namespace, but
# also connects the torch.linalg Python namespace to the torch._C._linalg builtins.

det = _add_docstr(_linalg.linalg_det, r"""
linalg.det(input) -> Tensor

Alias of :func:`torch.det`.
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
        will be returned.

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

_add_docstr(_linalg.linalg_eigh, r"""
linalg.eigh(input, UPLO='L') -> tuple(Tensor, Tensor)

This function returns eigenvalues and eigenvectors
of a complex Hermitian (conjugate symmetric) or real symmetric matrix :attr:`input`
represented by a namedtuple (eigenvalues, eigenvectors).

This function calculates all eigenvalues (and vectors) of :attr:`input`
such that :math:`\text{input} = V \text{diag}(e) V^H`.

Since the input matrix :attr:`input` is supposed to be Hermitian,
only the lower triangular portion is used by default
and the imaginary part of the diagonal will always be treated as zero.

.. note:: The eigenvalues of real symmetric or complex Hermitian matrices are always real.

.. note:: The eigenvalues/eigenvectors are computed using LAPACK routines ``_syevd``, ``_heevd``.

Args:
    input (Tensor): the input tensor of size :math:`(*, n, n)` where `*` is zero or more
                    batch dimensions consisting of Hermitian matrices.
    UPLO ('L', 'U', optional): controls whether to consider upper-triangular or lower-triangular part.
        Default: ``'L'``

Returns:
    (Tensor, Tensor): A namedtuple (eigenvalues, eigenvectors) containing

        - **eigenvalues** (*Tensor*): Shape :math:`(*, m)`.
            The eigenvalues in ascending order, each repeated according to its multiplicity.
        - **eigenvectors** (*Tensor*): Shape :math:`(*, m, m)`.
            The orthonormal eigenvectors of the ``input``.

Examples::

    >>> import torch
    >>> a = torch.randn(2, 2, dtype=torch.complex128)
    >>> a = a + a.t().conj()  # To make a Hermitian
    >>> a
    tensor([[2.9228+0.0000j, 0.2029-0.0862j],
            [0.2029+0.0862j, 0.3464+0.0000j]], dtype=torch.complex128)
    >>> w, v = torch.linalg.eigh(a)
    >>> w
    tensor([0.3277, 2.9415], dtype=torch.float64)
    >>> v
    tensor([[-0.0846+-0.0000j, -0.9964+0.0000j],
            [ 0.9170+0.3898j, -0.0779-0.0331j]], dtype=torch.complex128)
    >>> torch.allclose(torch.matmul(v, torch.matmul(w.to(v.dtype).diag_embed(), v.transpose(-2, -1).conj())), a)
    True
""")

_add_docstr(_linalg.linalg_eigvalsh, r"""
linalg.eigvalsh(input, UPLO='L') -> Tensor

This function returns eigenvalues of a complex Hermitian (conjugate symmetric)
or real symmetric matrix :attr:`input`.

.. note:: The eigenvalues of real symmetric or complex Hermitian matrices are always real.

.. note:: The eigenvalues are computed using LAPACK routines ``_syevd``, ``_heevd``.

Args:
    input (Tensor): the input tensor of size :math:`(*, n, n)` where `*` is zero or more
                    batch dimensions consisting of Hermitian matrices.
    UPLO ('L', 'U', optional): controls whether to consider upper-triangular or lower-triangular part.
        Default: ``'L'``

Returns:
    (Tensor): Shape :math:`(*, m)`. The eigenvalues in ascending order, each repeated according to its multiplicity.

Examples::

    >>> import torch
    >>> a = torch.randn(2, 2, dtype=torch.complex128)
    >>> a = a + a.t().conj()  # To make a Hermitian
    >>> a
    tensor([[2.9228+0.0000j, 0.2029-0.0862j],
            [0.2029+0.0862j, 0.3464+0.0000j]], dtype=torch.complex128)
    >>> w = torch.linalg.eigvalsh(a)
    >>> w
    tensor([0.3277, 2.9415], dtype=torch.float64)
""")

@functools.wraps(_linalg.linalg_eigh)
def eigh(a, UPLO="L"):
    return _linalg.linalg_eigh(a, UPLO)

@functools.wraps(_linalg.linalg_eigh)
def eigvalsh(a, UPLO="L"):
    return _linalg.linalg_eigvalsh(a, UPLO)

tensorsolve = _add_docstr(_linalg.linalg_tensorsolve, r"""
linalg.tensorsolve(input, other, dims=None, *, out=None) -> Tensor

Computes a tensor ``x`` such that ``tensordot(input, x, dims=x.ndim) = other``.
The resulting tensor ``x`` has the same shape as ``input[other.ndim:]``.

Supports real-valued and, only on the CPU, complex-valued inputs.

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
