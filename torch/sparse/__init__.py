# mypy: allow-untyped-defs
# The Tensor classes are added to this module by python_tensor.cpp
# A workaround to support both TorchScript and MyPy:
from typing import Any, Optional, TYPE_CHECKING, Union

import torch
from torch import Tensor
from torch._C import _add_docstr, _sparse  # type: ignore[attr-defined]

# Semi structured sparsity support
from .semi_structured import (
    SparseSemiStructuredTensor,
    SparseSemiStructuredTensorCUSPARSELT,
    SparseSemiStructuredTensorCUTLASS,
    to_sparse_semi_structured,
)


if TYPE_CHECKING:
    from torch.types import _dtype as DType

    DimOrDims = Optional[Union[int, tuple[int, ...], list[int]]]
else:
    # The JIT doesn't understand Union, nor torch.dtype here
    DType = int
    DimOrDims = Optional[tuple[int]]


__all__ = [
    "addmm",
    "check_sparse_tensor_invariants",
    "mm",
    "sum",
    "softmax",
    "solve",
    "log_softmax",
    "SparseSemiStructuredTensor",
    "SparseSemiStructuredTensorCUTLASS",
    "SparseSemiStructuredTensorCUSPARSELT",
    "to_sparse_semi_structured",
    "as_sparse_gradcheck",
]

addmm = _add_docstr(
    _sparse._sparse_addmm,
    r"""
sparse.addmm(mat, mat1, mat2, *, beta=1., alpha=1.) -> Tensor

This function does exact same thing as :func:`torch.addmm` in the forward,
except that it supports backward for sparse COO matrix :attr:`mat1`.
When :attr:`mat1` is a COO tensor it must have `sparse_dim = 2`.
When inputs are COO tensors, this function also supports backward for both inputs.

Supports both CSR and COO storage formats.

.. note::
    This function doesn't support computing derivatives with respect to CSR matrices.

Args:
    mat (Tensor): a dense matrix to be added
    mat1 (Tensor): a sparse matrix to be multiplied
    mat2 (Tensor): a dense matrix to be multiplied
    beta (Number, optional): multiplier for :attr:`mat` (:math:`\beta`)
    alpha (Number, optional): multiplier for :math:`mat1 @ mat2` (:math:`\alpha`)
""",
)


mm = _add_docstr(
    _sparse._sparse_mm,
    r"""
    Performs a matrix multiplication of the sparse matrix :attr:`mat1`
    and the (sparse or strided) matrix :attr:`mat2`. Similar to :func:`torch.mm`, if :attr:`mat1` is a
    :math:`(n \times m)` tensor, :attr:`mat2` is a :math:`(m \times p)` tensor, out will be a
    :math:`(n \times p)` tensor.
    When :attr:`mat1` is a COO tensor it must have `sparse_dim = 2`.
    When inputs are COO tensors, this function also supports backward for both inputs.

    Supports both CSR and COO storage formats.

.. note::
    This function doesn't support computing derivatives with respect to CSR matrices.

    This function also additionally accepts an optional :attr:`reduce` argument that allows
    specification of an optional reduction operation, mathematically performs the following operation:

.. math::

    z_{ij} = \bigoplus_{k = 0}^{K - 1} x_{ik} y_{kj}

where :math:`\bigoplus` defines the reduce operator. :attr:`reduce` is implemented only for
CSR storage format on CPU device.

Args:
    mat1 (Tensor): the first sparse matrix to be multiplied
    mat2 (Tensor): the second matrix to be multiplied, which could be sparse or dense
    reduce (str, optional): the reduction operation to apply for non-unique indices
        (:obj:`"sum"`, :obj:`"mean"`, :obj:`"amax"`, :obj:`"amin"`). Default :obj:`"sum"`.

Shape:
    The format of the output tensor of this function follows:
    - sparse x sparse -> sparse
    - sparse x dense -> dense

Example::

    >>> a = torch.tensor([[1., 0, 2], [0, 3, 0]]).to_sparse().requires_grad_()
    >>> a
    tensor(indices=tensor([[0, 0, 1],
                           [0, 2, 1]]),
           values=tensor([1., 2., 3.]),
           size=(2, 3), nnz=3, layout=torch.sparse_coo, requires_grad=True)
    >>> b = torch.tensor([[0, 1.], [2, 0], [0, 0]], requires_grad=True)
    >>> b
    tensor([[0., 1.],
            [2., 0.],
            [0., 0.]], requires_grad=True)
    >>> y = torch.sparse.mm(a, b)
    >>> y
    tensor([[0., 1.],
            [6., 0.]], grad_fn=<SparseAddmmBackward0>)
    >>> y.sum().backward()
    >>> a.grad
    tensor(indices=tensor([[0, 0, 1],
                           [0, 2, 1]]),
           values=tensor([1., 0., 2.]),
           size=(2, 3), nnz=3, layout=torch.sparse_coo)
    >>> c = a.detach().to_sparse_csr()
    >>> c
    tensor(crow_indices=tensor([0, 2, 3]),
           col_indices=tensor([0, 2, 1]),
           values=tensor([1., 2., 3.]), size=(2, 3), nnz=3,
           layout=torch.sparse_csr)
    >>> y1 = torch.sparse.mm(c, b, 'sum')
    >>> y1
    tensor([[0., 1.],
            [6., 0.]], grad_fn=<SparseMmReduceImplBackward0>)
    >>> y2 = torch.sparse.mm(c, b, 'max')
    >>> y2
    tensor([[0., 1.],
            [6., 0.]], grad_fn=<SparseMmReduceImplBackward0>)
""",
)


sampled_addmm = _add_docstr(
    _sparse.sparse_sampled_addmm,
    r"""
sparse.sampled_addmm(input, mat1, mat2, *, beta=1., alpha=1., out=None) -> Tensor

Performs a matrix multiplication of the dense matrices :attr:`mat1` and :attr:`mat2` at the locations
specified by the sparsity pattern of :attr:`input`. The matrix :attr:`input` is added to the final result.

Mathematically this performs the following operation:

.. math::

    \text{out} = \alpha\ (\text{mat1} \mathbin{@} \text{mat2})*\text{spy}(\text{input}) + \beta\ \text{input}

where :math:`\text{spy}(\text{input})` is the sparsity pattern matrix of :attr:`input`, :attr:`alpha`
and :attr:`beta` are the scaling factors.
:math:`\text{spy}(\text{input})` has value 1 at the positions where :attr:`input` has non-zero values, and 0 elsewhere.

.. note::
    :attr:`input` must be a sparse CSR tensor. :attr:`mat1` and :attr:`mat2` must be dense tensors.

Args:
    input (Tensor): a sparse CSR matrix of shape `(m, n)` to be added and used to compute
        the sampled matrix multiplication
    mat1 (Tensor): a dense matrix of shape `(m, k)` to be multiplied
    mat2 (Tensor): a dense matrix of shape `(k, n)` to be multiplied

Keyword args:
    beta (Number, optional): multiplier for :attr:`input` (:math:`\beta`)
    alpha (Number, optional): multiplier for :math:`mat1 @ mat2` (:math:`\alpha`)
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Examples::

    >>> input = torch.eye(3, device='cuda').to_sparse_csr()
    >>> mat1 = torch.randn(3, 5, device='cuda')
    >>> mat2 = torch.randn(5, 3, device='cuda')
    >>> torch.sparse.sampled_addmm(input, mat1, mat2)
    tensor(crow_indices=tensor([0, 1, 2, 3]),
        col_indices=tensor([0, 1, 2]),
        values=tensor([ 0.2847, -0.7805, -0.1900]), device='cuda:0',
        size=(3, 3), nnz=3, layout=torch.sparse_csr)
    >>> torch.sparse.sampled_addmm(input, mat1, mat2).to_dense()
    tensor([[ 0.2847,  0.0000,  0.0000],
        [ 0.0000, -0.7805,  0.0000],
        [ 0.0000,  0.0000, -0.1900]], device='cuda:0')
    >>> torch.sparse.sampled_addmm(input, mat1, mat2, beta=0.5, alpha=0.5)
    tensor(crow_indices=tensor([0, 1, 2, 3]),
        col_indices=tensor([0, 1, 2]),
        values=tensor([ 0.1423, -0.3903, -0.0950]), device='cuda:0',
        size=(3, 3), nnz=3, layout=torch.sparse_csr)
""",
)


def sum(input: Tensor, dim: DimOrDims = None, dtype: Optional[DType] = None) -> Tensor:
    r"""Return the sum of each row of the given sparse tensor.

    Returns the sum of each row of the sparse tensor :attr:`input` in the given
    dimensions :attr:`dim`. If :attr:`dim` is a list of dimensions,
    reduce over all of them. When sum over all ``sparse_dim``, this method
    returns a dense tensor instead of a sparse tensor.

    All summed :attr:`dim` are squeezed (see :func:`torch.squeeze`), resulting an output
    tensor having :attr:`dim` fewer dimensions than :attr:`input`.

    During backward, only gradients at ``nnz`` locations of :attr:`input`
    will propagate back. Note that the gradients of :attr:`input` is coalesced.

    Args:
        input (Tensor): the input sparse tensor
        dim (int or tuple of ints): a dimension or a list of dimensions to reduce. Default: reduce
            over all dims.
        dtype (:class:`torch.dtype`, optional): the desired data type of returned Tensor.
            Default: dtype of :attr:`input`.

    Example::

        >>> nnz = 3
        >>> dims = [5, 5, 2, 3]
        >>> I = torch.cat([torch.randint(0, dims[0], size=(nnz,)),
                           torch.randint(0, dims[1], size=(nnz,))], 0).reshape(2, nnz)
        >>> V = torch.randn(nnz, dims[2], dims[3])
        >>> size = torch.Size(dims)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> S = torch.sparse_coo_tensor(I, V, size)
        >>> S
        tensor(indices=tensor([[2, 0, 3],
                               [2, 4, 1]]),
               values=tensor([[[-0.6438, -1.6467,  1.4004],
                               [ 0.3411,  0.0918, -0.2312]],

                              [[ 0.5348,  0.0634, -2.0494],
                               [-0.7125, -1.0646,  2.1844]],

                              [[ 0.1276,  0.1874, -0.6334],
                               [-1.9682, -0.5340,  0.7483]]]),
               size=(5, 5, 2, 3), nnz=3, layout=torch.sparse_coo)

        # when sum over only part of sparse_dims, return a sparse tensor
        >>> torch.sparse.sum(S, [1, 3])
        tensor(indices=tensor([[0, 2, 3]]),
               values=tensor([[-1.4512,  0.4073],
                              [-0.8901,  0.2017],
                              [-0.3183, -1.7539]]),
               size=(5, 2), nnz=3, layout=torch.sparse_coo)

        # when sum over all sparse dim, return a dense tensor
        # with summed dims squeezed
        >>> torch.sparse.sum(S, [0, 1, 3])
        tensor([-2.6596, -1.1450])
    """
    if dtype is None:
        if dim is not None:
            return torch._sparse_sum(input, dim)
        else:
            return torch._sparse_sum(input)
    else:
        if dim is not None:
            return torch._sparse_sum(input, dim, dtype=dtype)
        else:
            return torch._sparse_sum(input, dtype=dtype)


softmax = _add_docstr(
    _sparse._sparse_softmax,
    r"""
sparse.softmax(input, dim, *, dtype=None) -> Tensor

Applies a softmax function.

Softmax is defined as:

:math:`\text{Softmax}(x_{i}) = \frac{exp(x_i)}{\sum_j exp(x_j)}`

where :math:`i, j` run over sparse tensor indices and unspecified
entries are ignores. This is equivalent to defining unspecified
entries as negative infinity so that :math:`exp(x_k) = 0` when the
entry with index :math:`k` has not specified.

It is applied to all slices along `dim`, and will re-scale them so
that the elements lie in the range `[0, 1]` and sum to 1.

Args:
    input (Tensor): input
    dim (int): A dimension along which softmax will be computed.
    dtype (:class:`torch.dtype`, optional): the desired data type
        of returned tensor.  If specified, the input tensor is
        casted to :attr:`dtype` before the operation is
        performed. This is useful for preventing data type
        overflows. Default: None
""",
)


spsolve = _add_docstr(
    _sparse._spsolve,
    r"""
sparse.spsolve(input, other, *, left=True) -> Tensor

Computes the solution of a square system of linear equations with
a unique solution. Its purpose is similar to :func:`torch.linalg.solve`,
except that the system is defined by a sparse CSR matrix with layout
`sparse_csr`.

Args:
    input (Tensor): a sparse CSR matrix of shape `(n, n)` representing the
        coefficients of the linear system.
    other (Tensor): a dense matrix of shape `(n, )` representing the right-hand
        side of the linear system.
    left (bool, optional): whether to solve the system for `input @ out = other`
        (default) or `out @ input = other`. Only `left=True` is supported.
""",
)

log_softmax = _add_docstr(
    _sparse._sparse_log_softmax,
    r"""
sparse.log_softmax(input, dim, *, dtype=None) -> Tensor

Applies a softmax function followed by logarithm.

See :class:`~torch.sparse.softmax` for more details.

Args:
    input (Tensor): input
    dim (int): A dimension along which softmax will be computed.
    dtype (:class:`torch.dtype`, optional): the desired data type
        of returned tensor.  If specified, the input tensor is
        casted to :attr:`dtype` before the operation is
        performed. This is useful for preventing data type
        overflows. Default: None
""",
)


spdiags = _add_docstr(
    _sparse._spdiags,
    r"""
sparse.spdiags(diagonals, offsets, shape, layout=None) -> Tensor

Creates a sparse 2D tensor by placing the values from rows of
:attr:`diagonals` along specified diagonals of the output

The :attr:`offsets` tensor controls which diagonals are set.

- If :attr:`offsets[i]` = 0, it is the main diagonal
- If :attr:`offsets[i]` < 0, it is below the main diagonal
- If :attr:`offsets[i]` > 0, it is above the main diagonal

The number of rows in :attr:`diagonals` must match the length of :attr:`offsets`,
and an offset may not be repeated.

Args:
    diagonals (Tensor): Matrix storing diagonals row-wise
    offsets (Tensor): The diagonals to be set, stored as a vector
    shape (2-tuple of ints): The desired shape of the result
Keyword args:
    layout (:class:`torch.layout`, optional): The desired layout of the
        returned tensor. ``torch.sparse_coo``, ``torch.sparse_csc`` and ``torch.sparse_csr``
        are supported. Default: ``torch.sparse_coo``

Examples:

Set the main and first two lower diagonals of a matrix::

    >>> diags = torch.arange(9).reshape(3, 3)
    >>> diags
    tensor([[0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]])
    >>> s = torch.sparse.spdiags(diags, torch.tensor([0, -1, -2]), (3, 3))
    >>> s
    tensor(indices=tensor([[0, 1, 2, 1, 2, 2],
                           [0, 1, 2, 0, 1, 0]]),
           values=tensor([0, 1, 2, 3, 4, 6]),
           size=(3, 3), nnz=6, layout=torch.sparse_coo)
    >>> s.to_dense()
    tensor([[0, 0, 0],
            [3, 1, 0],
            [6, 4, 2]])


Change the output layout::

    >>> diags = torch.arange(9).reshape(3, 3)
    >>> diags
    tensor([[0, 1, 2],[3, 4, 5], [6, 7, 8])
    >>> s = torch.sparse.spdiags(diags, torch.tensor([0, -1, -2]), (3, 3), layout=torch.sparse_csr)
    >>> s
    tensor(crow_indices=tensor([0, 1, 3, 6]),
           col_indices=tensor([0, 0, 1, 0, 1, 2]),
           values=tensor([0, 3, 1, 6, 4, 2]), size=(3, 3), nnz=6,
           layout=torch.sparse_csr)
    >>> s.to_dense()
    tensor([[0, 0, 0],
            [3, 1, 0],
            [6, 4, 2]])

Set partial diagonals of a large output::

    >>> diags = torch.tensor([[1, 2], [3, 4]])
    >>> offsets = torch.tensor([0, -1])
    >>> torch.sparse.spdiags(diags, offsets, (5, 5)).to_dense()
    tensor([[1, 0, 0, 0, 0],
            [3, 2, 0, 0, 0],
            [0, 4, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]])

.. note::

    When setting the values along a given diagonal the index into the diagonal
    and the index into the row of :attr:`diagonals` is taken as the
    column index in the output. This has the effect that when setting a diagonal
    with a positive offset `k` the first value along that diagonal will be
    the value in position `k` of the row of :attr:`diagonals`

Specifying a positive offset::

    >>> diags = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    >>> torch.sparse.spdiags(diags, torch.tensor([0, 1, 2]), (5, 5)).to_dense()
    tensor([[1, 2, 3, 0, 0],
            [0, 2, 3, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]])
""",
)


class check_sparse_tensor_invariants:
    """A tool to control checking sparse tensor invariants.

    The following options exists to manage sparsr tensor invariants
    checking in sparse tensor construction:

    1. Using a context manager:

       .. code:: python

           with torch.sparse.check_sparse_tensor_invariants():
               run_my_model()

    2. Using a procedural approach:

       .. code:: python

           prev_checks_enabled = torch.sparse.check_sparse_tensor_invariants.is_enabled()
           torch.sparse.check_sparse_tensor_invariants.enable()

           run_my_model()

           if not prev_checks_enabled:
               torch.sparse.check_sparse_tensor_invariants.disable()

    3. Using function decoration:

       .. code:: python

           @torch.sparse.check_sparse_tensor_invariants()
           def run_my_model():
               ...

           run_my_model()

    4. Using ``check_invariants`` keyword argument in sparse tensor constructor call.
       For example:

       >>> torch.sparse_csr_tensor([0, 1, 3], [0, 1], [1, 2], check_invariants=True)
       Traceback (most recent call last):
         File "<stdin>", line 1, in <module>
       RuntimeError: `crow_indices[..., -1] == nnz` is not satisfied.
    """

    @staticmethod
    def is_enabled():
        r"""Return True if the sparse tensor invariants checking is enabled.

        .. note::

            Use :func:`torch.sparse.check_sparse_tensor_invariants.enable` or
            :func:`torch.sparse.check_sparse_tensor_invariants.disable` to
            manage the state of the sparse tensor invariants checks.
        """
        return torch._C._check_sparse_tensor_invariants()

    @staticmethod
    def enable():
        r"""Enable sparse tensor invariants checking in sparse tensor constructors.

        .. note::

            By default, the sparse tensor invariants checks are disabled. Use
            :func:`torch.sparse.check_sparse_tensor_invariants.is_enabled` to
            retrieve the current state of sparse tensor invariants checking.

        .. note::

            The sparse tensor invariants check flag is effective to all sparse
            tensor constructors, both in Python and ATen.

        The flag can be locally overridden by the ``check_invariants``
        optional argument of the sparse tensor constructor functions.
        """
        torch._C._set_check_sparse_tensor_invariants(True)

    @staticmethod
    def disable():
        r"""Disable sparse tensor invariants checking in sparse tensor constructors.

        See :func:`torch.sparse.check_sparse_tensor_invariants.enable` for more information.
        """
        torch._C._set_check_sparse_tensor_invariants(False)

    # context manager support
    def __init__(self, enable=True):
        self.state = enable
        self.saved_state: Optional[bool] = None

    def __enter__(self):
        if self.saved_state is not None:
            raise RuntimeError(
                "This context manager instance is already activated."
                " Use a different context manager instance for context nesting."
            )
        self.saved_state = self.is_enabled()
        torch._C._set_check_sparse_tensor_invariants(self.state)

    def __exit__(self, type, value, traceback):
        assert self.saved_state is not None
        torch._C._set_check_sparse_tensor_invariants(self.saved_state)
        self.saved_state = None

    # decorator support
    def __call__(self, mth):
        def test_mth(*args, **kwargs):
            with type(self)(self.state):
                return mth(*args, **kwargs)

        return test_mth


def as_sparse_gradcheck(gradcheck):
    """Decorate function, to extend gradcheck for sparse tensors.

    Decorator for torch.autograd.gradcheck or its functools.partial
    variants that extends the gradcheck function with support to input
    functions that operate on or/and return sparse tensors.

    The specified gradcheck function itself is guaranteed to operate
    on strided tensors only.

    For example:

    >>> gradcheck = torch.sparse.as_sparse_gradcheck(torch.autograd.gradcheck)
    >>> x = torch.tensor([[0, 1], [2, 3]], dtype=torch.float64).to_sparse_coo().requires_grad_(True)
    >>> gradcheck(lambda x: x.to_sparse_csr(), x)
    True
    """

    def gradcheck_with_sparse_support(func, inputs, **kwargs):
        """
        Create gradcheck with support for sparse tensors.

        Same as :func:`torch.autograd.gradcheck` but with sparse tensors inputs and outputs support.
        """
        masked = kwargs.pop("masked", False)
        sparse_layouts = {
            torch.sparse_coo,
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        }
        sparse_compressed_layouts = {
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        }
        sparse_block_layouts = {torch.sparse_bsr, torch.sparse_bsc}
        STRIDED_REPRESENTATION = "__STRIDED_REPRESENTATION__"

        def convert_to_strided_representation(args):
            """Convert differentiable non-strided tensors to a representation containing differentiable strided tensors."""
            if not isinstance(args, (list, tuple)):
                args = (args,)
            new_args: list[Any] = []
            for obj in args:
                if (
                    isinstance(obj, torch.Tensor)
                    and obj.requires_grad
                    and obj.layout in sparse_layouts
                ):
                    d = dict(layout=obj.layout, shape=obj.shape)
                    if not masked:
                        # Materialize unspecified elements with zero values
                        batch_dim = obj.ndim - obj.dense_dim() - obj.sparse_dim()
                        blocksize = (
                            obj.values().shape[batch_dim + 1 : batch_dim + 3]
                            if obj.layout in sparse_block_layouts
                            else None
                        )
                        full_mask = torch.ones(
                            obj.shape, device=obj.device, dtype=torch.bool
                        ).to_sparse(
                            layout=obj.layout,
                            blocksize=blocksize,
                            dense_dim=obj.dense_dim(),
                        )
                        obj = obj.to_dense().sparse_mask(full_mask)
                    if obj.layout is torch.sparse_coo:
                        d.update(
                            indices=obj._indices(), is_coalesced=obj.is_coalesced()
                        )
                        values = obj._values()
                    elif obj.layout in {torch.sparse_csr, torch.sparse_bsr}:
                        d.update(
                            compressed_indices=obj.crow_indices(),
                            plain_indices=obj.col_indices(),
                        )
                        values = obj.values()
                    else:
                        d.update(
                            compressed_indices=obj.ccol_indices(),
                            plain_indices=obj.row_indices(),
                        )
                        values = obj.values()
                    new_args.extend(
                        (STRIDED_REPRESENTATION, d, values.requires_grad_(True))
                    )
                else:
                    new_args.append(obj)
            return tuple(new_args)

        def restore_from_strided_representation(args):
            """Restore non-strided differentiable tensosr from their strided representations."""
            new_args = []
            args = list(args)
            while args:
                a = args.pop(0)
                if a == STRIDED_REPRESENTATION:
                    d, values = args.pop(0), args.pop(0)
                    if d["layout"] is torch.sparse_coo:
                        a = torch.sparse_coo_tensor(
                            d["indices"],
                            values,
                            size=d["shape"],
                            is_coalesced=d["is_coalesced"],
                        )
                    elif d["layout"] in sparse_compressed_layouts:
                        a = torch.sparse_compressed_tensor(
                            d["compressed_indices"],
                            d["plain_indices"],
                            values,
                            size=d["shape"],
                            layout=d["layout"],
                        )
                    else:
                        raise NotImplementedError(
                            f'conversion of {d["layout"]} strided representation to tensor'
                        )
                new_args.append(a)
            return tuple(new_args)

        def func_wrapper(*args, **kwargs):
            restored_args = restore_from_strided_representation(args)

            # convert differentiable output sparse tensors to strided
            # tensors:
            outputs = func(*restored_args, **kwargs)

            strided_outputs = (
                tuple(outputs) if isinstance(outputs, (list, tuple)) else (outputs,)
            )
            strided_outputs = tuple(
                (
                    o.to_dense(masked_grad=masked)
                    if isinstance(o, torch.Tensor)
                    and o.requires_grad
                    and o.layout in sparse_layouts
                    else o
                )
                for o in strided_outputs
            )

            return (
                strided_outputs
                if isinstance(outputs, (list, tuple))
                else strided_outputs[0]
            )

        args = (func_wrapper, convert_to_strided_representation(inputs))

        return gradcheck(*args, **kwargs)

    return gradcheck_with_sparse_support
