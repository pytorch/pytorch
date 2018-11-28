# The Tensor classes are added to this module by python_tensor.cpp
import torch

__all__ = [
    'addmm',
    'sum',
]


def addmm(mat, mat1, mat2, beta=1, alpha=1):
    r"""
    This function does exact same thing as :func:`torch.addmm` in the forward,
    except that it supports backward for coalesced sparse matrix `mat1`.

    Args:
        mat (Tensor): a dense matrix to be added
        mat1 (Tensor): a sparse matrix to be multiplied
        mat2 (Tensor): a dense matrix be multiplied
        beta (Number, optional): multiplier for :attr:`mat` (:math:`\beta`)
        alpha (Number, optional): multiplier for :math:`mat1 @ mat2` (:math:`\alpha`)
    """
    return torch._sparse_addmm(mat, mat1, mat2, beta=beta, alpha=alpha)


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
