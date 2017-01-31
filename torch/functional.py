import torch
from ._utils import _range


def split(tensor, split_size, dim=0):
    """Splits the tensor into equally sized chunks (if possible).

    Last chunk will be smaller if the tensor size along a given dimension
    is not divisible by ``split_size``.

    Arguments:
        tensor (Tensor): tensor to split.
        split_size (int): size of a single chunk.
        dim (int): dimension along which to split the tensor.
    """
    if dim < 0:
        dim += tensor.dim()
    dim_size = tensor.size(dim)
    num_splits = (dim_size + split_size - 1) // split_size
    last_split_size = split_size - (split_size * num_splits - dim_size)

    def get_split_size(i):
        return split_size if i < num_splits - 1 else last_split_size
    return tuple(tensor.narrow(int(dim), int(i * split_size), int(get_split_size(i))) for i
                 in _range(0, num_splits))


def chunk(tensor, chunks, dim=0):
    """Splits a tensor into a number of chunks along a given dimension.

    Arguments:
        tensor (Tensor): tensor to split.
        chunks (int): number of chunks to return.
        dim (int): dimension along which to split the tensor.
    """
    if dim < 0:
        dim += tensor.dim()
    split_size = (tensor.size(dim) + chunks - 1) // chunks
    return split(tensor, split_size, dim)


def stack(sequence, dim=0):
    """Concatenates sequence of tensors along a new dimension.

    All tensors need to be of the same size.

    Arguments:
        sqequence (Sequence): sequence of tensors to concatenate.
        dim (int): dimension to insert. Has to be between 0 and the number
            of dimensions of concatenated tensors (inclusive).
    """
    if len(sequence) == 0:
        raise TypeError("stack expects a non-empty sequence of tensors")
    if dim < 0:
        dim += sequence[0].dim()
    return torch.cat(list(t.unsqueeze(dim) for t in sequence), dim)


def unbind(tensor, dim=0):
    """Removes a tensor dimension.

    Returns a tuple of all slices along a given dimension, already without it.

    Arguments:
        tensor (Tensor): tensor to unbind.
        dim (int): dimension to remove.
    """
    return tuple(tensor.select(dim, i) for i in _range(tensor.size(dim)))
