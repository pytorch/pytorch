import torch


def dropout(input, p=0.5, training=True):
    r"""
    During training, randomly zeroes values of the :attr:`input` SparseTensor
    with probability :attr:`p` using samples from a Bernoulli distribution.
    Values that are not zeroed will be scaled up by ``1 / (1 - p)``. Autograd
    is also supported. Note that the gradients of :attr:`input` is coalesced.

    See :func:`torch.nn.functional.dropout`

    Args:
        input (SparseTensor): the input SparseTensor
        p (float): probability of a value to be zeroed. Default: 0.5
        training (bool): apply dropout if ``True``. Default: ``True``

    Example::

        >>> S = torch.randn(2, 3).to_sparse()
        >>> S
        tensor(indices=tensor([[0, 0, 0, 1, 1, 1],
                               [0, 1, 2, 0, 1, 2]]),
               values=tensor([-0.9493,  0.5793, -2.0528,  0.3294, -0.3047, -0.6994]),
               size=(2, 3), nnz=6, layout=torch.sparse_coo)

        >>> S_out = torch.sparse.dropout(S, 0.5)
        >>> S_out
        tensor(indices=tensor([[0, 0, 0, 1, 1, 1],
                               [0, 1, 2, 0, 1, 2]]),
               values=tensor([-1.8987,  1.1585, -0.0000,  0.6588, -0.6094, -1.3988]),
               size=(2, 3), nnz=6, layout=torch.sparse_coo)
    """
    return torch._sparse_dropout(input, p, training)[0]
