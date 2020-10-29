import sys

import torch
from torch._C import _add_docstr, _linalg  # type: ignore

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


_add_docstr(_linalg.linalg_svd, "See ``linalg.svd``")

def svd(a, full_matrices=True, compute_uv=True, out=None):
    r"""
linalg.svd(input, full_matrices=True, compute_uv=True, out=None) -> (Tensor, Tensor, Tensor)

This function returns a namedtuple ``(U, S, Vh)`` which is the singular value
decomposition of a input real matrix or batches of real matrices :attr:`input` such that
:math:`input = U \times diag(S) \times V^H` (where :math:`V^H` is ``Vh``).

.. warning:: **Differences with** :meth:`~torch.svd`:

             * :attr:`full_matrices` is the opposite of
               :meth:`~torch.svd`'s :attr:`some`. Note that default value
               for both is ``True``, so the default behavior is effectively
               the opposite.

             * it returns ``Vh``, whereas :meth:`~torch.svd` returns
               ``V``. The result is that when using :meth:`~torch.svd` you
               need to manually transpose and conjugate ``V`` in order to
               reconstruct the original matrix.

             * If :attr:`compute_uv=False`, it returns ``None`` for ``U`` and
               ``V``, whereas :meth:`~torch.svd` returns zero-filled tensors.

             **Differences with** ``numpy.linalg.svd``:

             * if :attr:`compute_uv=False` it returns ``(None, S, None)``,
               whereas numpy returns ``S``.


The dtype of ``U`` and ``V`` is the same as the ``input`` matrix. The dtype of
``S`` is always real numbers, even if ``input`` is complex.

If :attr:`full_matrices` is ``False``, the method returns the reduced singular value decomposition
i.e., if the last two dimensions of :attr:`input` are ``m`` and ``n``, then the returned
`U` and `V` matrices will contain only :math:`min(n, m)` orthonormal columns.

If :attr:`compute_uv` is ``False``, the returned `U` and `V` will be None.:attr:`full_matrices` will
be ignored here.

.. note:: The singular values are returned in descending order. If :attr:`input` is a batch of matrices,
          then the singular values of each matrix in the batch is returned in descending order.

.. note:: The implementation of SVD on CPU uses the LAPACK routine `?gesdd` (a divide-and-conquer
          algorithm) instead of `?gesvd` for speed. Analogously, the SVD on GPU uses the MAGMA routine
          `gesdd` as well.

.. note:: Irrespective of the original strides, the returned matrix `U`
          will be transposed, i.e. with strides :code:`U.contiguous().transpose(-2, -1).stride()`

.. note:: Extra care needs to be taken when backward through `U` and `V`
          outputs. Such operation is really only stable when :attr:`input` is
          full rank with all distinct singular values. Otherwise, ``NaN`` can
          appear as the gradients are not properly defined. Also, notice that
          double backward will usually do an additional backward through `U` and
          `V` even if the original backward is only on `S`.

.. note:: When :attr:`full_matrices` = ``False``, the gradients on :code:`U[..., :, min(m, n):]`
          and :code:`V[..., :, min(m, n):]` will be ignored in backward as those vectors
          can be arbitrary bases of the subspaces.

.. note:: When :attr:`compute_uv` = ``False``, backward cannot be performed since `U` and `V`
          from the forward pass is required for the backward operation.

Args:
    input (Tensor): the input tensor of size :math:`(*, m, n)` where `*` is zero or more
                    batch dimensions consisting of :math:`m \times n` matrices.
    full_matrices (bool, optional): controls the shape of returned `U` and `V`
    compute_uv (bool, optional): option whether to compute `U` and `V` or not
    out (tuple, optional): the output tuple of tensors

Example::

    >>> import torch
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
    """
    if compute_uv:
        # easy case, same semantics as the C++ version of linalg_svd
        return _linalg.linalg_svd(a, full_matrices, compute_uv, out=out)

    # harder case: C++ returns (0-tensor, S, 0-tensor) but we want to return
    # (None, S, None) instead
    if out is not None:
        if type(out) != torch.Tensor:
            raise TypeError("linalg.svd: argument 'out' must be a Tensor when "
                            "compute_uv==False, not %s" % type(out).__name__)
        out = (torch.Tensor(), out, torch.Tensor())
    USV = _linalg.linalg_svd(a, full_matrices, compute_uv, out=out)
    # we want to return a value of type torch.return_types.linalg_svd
    # (which is a PyStructSequence). However, this type is not directly
    # exposed by pytorch, so we get a reference to it by calling type() on
    # USV.
    USV_TYPE = type(USV)
    return USV_TYPE((None, USV.S, None))
