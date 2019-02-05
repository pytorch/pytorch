# The Tensor classes are added to this module by python_tensor.cpp
import torch

__all__ = [
    'addmm',
    'mm',
    'sum',
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

        >>> a = torch.arange(6).view(2, 3).float().to_sparse().requires_grad_(True)
        >>> a
        tensor(indices=tensor([[0, 0, 1, 1, 1],
                               [1, 2, 0, 1, 2]]),
               values=tensor([1., 2., 3., 4., 5.]),
               size=(2, 3), nnz=5, layout=torch.sparse_coo, requires_grad=True)

        >>> b = torch.arange(6).view(3, 2).float()
        >>> b
        tensor([[0., 1.],
                [2., 3.],
                [4., 5.]]...)

        >>> y = torch.sparse.mm(a, b)
        >>> y
        tensor([[10., 13.],
                [28., 40.]], grad_fn=<SparseAddmmBackward>)
        >>> y.sum().backward()
        >>> a.grad
        tensor(indices=tensor([[0, 0, 1, 1, 1],
                               [1, 2, 0, 1, 2]]),
               values=tensor([5., 9., 1., 5., 9.]),
               size=(2, 3), nnz=5, layout=torch.sparse_coo)
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

        >>> torch.manual_seed(0)
        >>> nnz = 3
        >>> dims = [5, 5, 2, 3]
        >>> I = torch.cat([torch.randint(0, dims[0], size=(nnz,)),
        >>>                torch.randint(0, dims[1], size=(nnz,))], 0).reshape(2, nnz)
        >>> V = torch.randn(nnz, dims[2], dims[3])
        >>> size = torch.Size(dims)
        >>> S = torch.sparse_coo_tensor(I, V, size)
        >>> S
        tensor(indices=tensor([[4, 4, 3],
                               [0, 3, 4]]),
               values=tensor([[[-0.3160, -2.1152,  0.4681],
                               [-0.1577,  1.4437,  0.2660]],
                              [[ 0.1665,  0.8744, -0.1435],
                               [-0.1116,  0.9318,  1.2590]],
                              [[ 2.0050,  0.0537,  0.6181],
                               [-0.4128, -0.8411, -2.3160]]]),
               size=(5, 5, 2, 3), nnz=3, layout=torch.sparse_coo)

        >>> # when sum over only part of sparse_dims, return a SparseTensor
        >>> torch.sparse.sum(S, dim=[1, 2])
        tensor(indices=tensor([[3, 4]]),
               values=tensor([[ 1.5922, -0.7873, -1.6980],
                              [-0.4189,  1.1346,  1.8497]]),
               size=(5, 3), nnz=2, layout=torch.sparse_coo)

        >>> # when sum over all sparse dim, return a dense Tensor
        >>> # with summed dims squeezed
        >>> torch.sparse.sum(S, [0, 1, 3])
        tensor([..., ...])
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
