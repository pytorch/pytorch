# The Tensor classes are added to this module by python_tensor.cpp
import torch

__all__ = [
    'addmm',
    'sum',
]


def addmm(mat, mat1, mat2, beta=1, alpha=1):
    r"""
    This function does exact same thing as :meth:`~Torch.addmm` in the forward,
    except that it supports backward for coalesced sparse matrix `mat1`.

    Args:
        mat (Tensor): a dense matrix to be added
        mat1 (Tensor): a sparse matrix to be multiplied
        mat2 (Tensor): a dense matrix be multiplied
        beta (Number, optional): multiplier for :attr:`mat` (:math:`\beta`)
        alpha (Number, optional): multiplier for :math:`mat1 @ mat2` (:math:`\alpha`)
    """
    return torch._sparse_addmm(mat, mat1, mat2, beta=beta, alpha=alpha)

def sum(input, dim=None, keepdim=False, dtype=None):
    r"""
    .. function:: torch.sparse.sum(input, dim=None, keepdim=False, dtype=None) -> SparseTensor / Tensor

    Returns the sum of each row of the :attr:`input` SparseTensor in the given
    dimension :attr:`dim`. If :attr::`dim` is a list of dimensions,
    reduce over all of them. When sum over all ``sparse_dim``,
    this method returns a Tensor instead of SparseTensor.

    If :attr:`keepdim` is ``True``, the output tensor has the same :attr::`dim`
    as :attr:`input`. Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`),
    resulting in the output tensor having :attr::`dim` fewer dimension than :attr:`input`.
    For example, if an :attr:`input` SparseTensor has ``sparse_dim = 2`` and ``dense_dim = 2``,
    when sum over all ``sparse_dim`` with ``keepdim = True``, this method outputs a dense
    Tensor with ``dim = 4``.

    During backward, only gradients at ``nnz`` locations of :attr:`input` SparseTensor will propagate back.

    Args:
        input (Tensor): the input SparseTensor
        dim (int or tuple of ints): the dimension or list of dimensions to reduce. Default: reduce
            over all dims.
        keepdim (bool): whether the output tensor has :attr:`dim` retained or not. Default: ``False``
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
        tensor(indices=tensor([[1, 4, 1],
                               [1, 2, 4]]),
               values=tensor([[[ 0.8848, -1.3714,  0.7392],
                               [-0.6338,  0.0638, -1.3513]],

                              [[-0.1462, -0.7156, -0.6462],
                               [-0.2636, -3.0326, -0.4032]],

                              [[ 0.8911, -1.1238,  0.8311],
                               [ 0.7017, -1.9306, -0.5571]]]),
               size=(5, 5, 2, 3), nnz=3, layout=torch.sparse_coo)

        # when sum over some sparse_dim with keepdim = True,
        # a SparseTensor with indices Tensor that has summed dims filled with all zeros
        >>> torch.sparse_sum(S, [1, 3], True)
        tensor(indices=tensor([[0, 1, 4],
                               [0, 0, 0]]),
               values=tensor([[[-1.0981],
                               [-1.6233]],

                              [[ 2.2258],
                               [-1.7578]],

                              [[-3.4382],
                               [-0.6544]]]),
               size=(5, 1, 2, 1), nnz=3, layout=torch.sparse_coo)

        # when sum over all sparse dim with keepdim = True,
        # a dense tensor of the same number of dims is returned
        >>> torch.sparse_sum(S, [0, 1, 3], True)
        tensor([[[[-0.9137],
                  [-0.5834]]]])

        # when sum over all sparse dim with keepdim = False,
        # a dense tensor with summed dims squeezed is returned
        >>> torch.sparse_sum(S, [0, 1, 3], False)
        tensor([-0.9997, -1.2197])
    """
    if dtype is None:
        if dim:
            return torch._sparse_sum(input, dim, keepdim)
        else:
            return torch._sparse_sum(input)
    else:
        if dim:
            return torch._sparse_sum(input, dim, keepdim, dtype=dtype)
        else:
            return torch._sparse_sum(input, dtype=dtype)
