# The Tensor classes are added to this module by python_tensor.cpp
import torch

__all__ = [
    'addmm',
    'mm',
    'sum',
    'add',
    'sub',
]


def addmm(mat, mat1, mat2, beta=1, alpha=1):
    r"""
    This function does exact same thing as :func:`torch.addmm` in the forward,
    except that it supports backward for sparse matrix :attr:`mat1`. :attr:`mat1`
    need to have `sparse_dim = 2`. Note that the gradients of :attr:`mat1` is a
    coalesced sparse tensor.

    Args:
        mat (Tensor): a dense matrix to be added
        mat1 (SparseTensor): a sparse matrix to be multiplied
        mat2 (Tensor): a dense matrix be multiplied
        beta (Number, optional): multiplier for :attr:`mat` (:math:`\beta`)
        alpha (Number, optional): multiplier for :math:`mat1 @ mat2` (:math:`\alpha`)
    """
    return torch._sparse_addmm(mat, mat1, mat2, beta=beta, alpha=alpha)


def mm(mat1, mat2):
    r"""
    Performs a matrix multiplication of the sparse matrix :attr:`mat1`
    and dense matrix :attr:`mat2`. Similar to :func:`torch.mm`, If :attr:`mat1` is a
    :math:`(n \times m)` tensor, :attr:`mat2` is a :math:`(m \times p)` tensor, out will be a
    :math:`(n \times p)` dense tensor. :attr:`mat1` need to have `sparse_dim = 2`.
    This function also supports backward for both matrices. Note that the gradients of
    :attr:`mat1` is a coalesced sparse tensor.

    Args:
        mat1 (SparseTensor): the first sparse matrix to be multiplied
        mat2 (Tensor): the second dense matrix to be multiplied

    Example::

        >>> a = torch.randn(2, 3).to_sparse().requires_grad_(True)
        >>> a
        tensor(indices=tensor([[0, 0, 0, 1, 1, 1],
                               [0, 1, 2, 0, 1, 2]]),
               values=tensor([ 1.5901,  0.0183, -0.6146,  1.8061, -0.0112,  0.6302]),
               size=(2, 3), nnz=6, layout=torch.sparse_coo, requires_grad=True)

        >>> b = torch.randn(3, 2, requires_grad=True)
        >>> b
        tensor([[-0.6479,  0.7874],
                [-1.2056,  0.5641],
                [-1.1716, -0.9923]], requires_grad=True)

        >>> y = torch.sparse.mm(a, b)
        >>> y
        tensor([[-0.3323,  1.8723],
                [-1.8951,  0.7904]], grad_fn=<SparseAddmmBackward>)
        >>> y.sum().backward()
        >>> a.grad
        tensor(indices=tensor([[0, 0, 0, 1, 1, 1],
                               [0, 1, 2, 0, 1, 2]]),
               values=tensor([ 0.1394, -0.6415, -2.1639,  0.1394, -0.6415, -2.1639]),
               size=(2, 3), nnz=6, layout=torch.sparse_coo)
    """
    return torch._sparse_mm(mat1, mat2)


def sum(input, dim=None, dtype=None):
    r"""
    Returns the sum of each row of SparseTensor :attr:`input` in the given
    dimensions :attr:`dim`. If :attr::`dim` is a list of dimensions,
    reduce over all of them. When sum over all ``sparse_dim``, this method
    returns a Tensor instead of SparseTensor.

    All summed :attr:`dim` are squeezed (see :func:`torch.squeeze`), resulting an output
    tensor having :attr::`dim` fewer dimensions than :attr:`input`.

    During backward, only gradients at ``nnz`` locations of :attr:`input`
    will propagate back. Note that the gradients of :attr:`input` is coalesced.

    Args:
        input (Tensor): the input SparseTensor
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

        # when sum over only part of sparse_dims, return a SparseTensor
        >>> torch.sparse.sum(S, [1, 3])
        tensor(indices=tensor([[0, 2, 3]]),
               values=tensor([[-1.4512,  0.4073],
                              [-0.8901,  0.2017],
                              [-0.3183, -1.7539]]),
               size=(5, 2), nnz=3, layout=torch.sparse_coo)

        # when sum over all sparse dim, return a dense Tensor
        # with summed dims squeezed
        >>> torch.sparse.sum(S, [0, 1, 3])
        tensor([-2.6596, -1.1450])
    """
    if dtype is None:
        if dim:
            return torch._sparse_sum(input, dim)
        else:
            return torch._sparse_sum(input)
    else:
        if dim:
            return torch._sparse_sum(input, dim, dtype=dtype)
        else:
            return torch._sparse_sum(input, dtype=dtype)


def add(input, other):
    r"""
    Element of the SparseTensor :attr:`other` is added to corresponding
    locations of SparseTensor :attr:`input`. The number of dims must be
    the same between two inputs tensors. This operation supports broadcasting
    on sparse dims, and requires dense dims to be the same between two inputs.
    For a sparse dim ``i``, if ``sizes(i)`` differs, then ``input.sizes(i) > 1``
    and ``other.sizes(i) == 1``.

    Backward supports for :attr:`input` but not :attr:`other`. Note that
    the gradients of :attr:`input` is coalesced.

    Args:
        input (SparseTensor): the first input SparseTensor
        other (SparseTensor or a scalar): the value to add to input

    Example::
        >>> nnz = 4
        >>> dims1 = [2, 2]
        >>> I = torch.cat([torch.randint(0, dims1[0], size=(nnz,)),
                           torch.randint(0, dims1[1], size=(nnz,)),], 0).reshape(2, nnz)
        >>> V = torch.rand(nnz)
        >>> S1 = torch.sparse_coo_tensor(I, V, dims1)
        >>> S1
        tensor(indices=tensor([[0, 0, 0, 1],
                               [0, 1, 0, 0]]),
               values=tensor([0.7437, 0.2626, 0.9034, 0.0787]),
               size=(2, 2), nnz=4, layout=torch.sparse_coo)

        >>> dims2 = [2, 1]
        >>> I = torch.cat([torch.randint(0, dims2[0], size=(nnz,)),
                           torch.randint(0, dims2[1], size=(nnz,)),], 0).reshape(2, nnz)
        >>> V = torch.rand(nnz)
        >>> S2 = torch.sparse_coo_tensor(I, V, dims2)
        >>> S2
        tensor(indices=tensor([[1, 0, 1, 1],
                               [0, 0, 0, 0]]),
               values=tensor([0.8652, 0.6640, 0.2394, 0.8972]),
               size=(2, 1), nnz=4, layout=torch.sparse_coo)

        >>> S3 = torch.sparse.add(S1, S2)
        >>> S3
        tensor(indices=tensor([[0, 0, 1],
                               [0, 1, 0]]),
               values=tensor([2.3111, 0.9266, 2.0804]),
               size=(2, 2), nnz=3, layout=torch.sparse_coo)

        # add a scalar
        >>> I = torch.cat([torch.randint(0, dims1[0], size=(nnz,)),
                           torch.randint(0, dims1[1], size=(nnz,)),], 0).reshape(2, nnz)
        >>> V = torch.rand(nnz)
        >>> S4 = torch.sparse_coo_tensor(I, V, dims1)
        >>> S4
        tensor(indices=tensor([[0, 1, 1, 1],
                               [1, 1, 0, 0]]),
               values=tensor([0.0945, 0.3280, 0.7648, 0.8265]),
               size=(2, 2), nnz=4, layout=torch.sparse_coo)

        >>> torch.sparse.add(S1, 1)
            tensor(indices=tensor([[0, 1, 1],
                                   [1, 0, 1]]),
                   values=tensor([1.0945, 2.5913, 1.3280]),
                   size=(2, 2), nnz=3, layout=torch.sparse_coo)
    """
    return torch._sparse_add(input, other, alpha=1)


def sub(input, other):
    r"""
    Element of the SparseTensor :attr:`other` is subtracted from corresponding
    locations of SparseTensor :attr:`input`. The number of dims must be
    the same between two inputs tensors. This operation supports broadcasting
    on sparse dims, and requires dense dims to be the same between two inputs.
    For a sparse dim ``i``, if ``sizes(i)`` differs, then ``input.sizes(i) > 1``
    and ``other.sizes(i) == 1``.

    Backward supports for :attr:`input` but not :attr:`other`. Note that
    the gradients of :attr:`input` is coalesced.

    Args:
        input (SparseTensor): the first input SparseTensor
        other (SparseTensor or a scalar): the value to add to input
    """
    return torch._sparse_add(input, other, alpha=-1)
