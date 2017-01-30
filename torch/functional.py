import torch
from ._utils import _range


def split(tensor, split_size, dim=0):
    if dim < 0:
        dim += tensor.dim()
    dim_size = tensor.size(dim)
    num_splits = (dim_size + split_size - 1) // split_size
    last_split_size = split_size - (split_size * num_splits - dim_size)

    def get_split_size(i):
        return split_size if i < num_splits - 1 else last_split_size
    return tuple(tensor.narrow(int(dim), int(i * split_size), int(get_split_size(i))) for i
                 in _range(0, num_splits))


def chunk(tensor, n_chunks, dim=0):
    if dim < 0:
        dim += tensor.dim()
    split_size = (tensor.size(dim) + n_chunks - 1) // n_chunks
    return split(tensor, split_size, dim)


def stack(sequence, dim=0):
    if len(sequence) == 0:
        raise TypeError("stack expects a non-empty sequence of tensors")
    if dim < 0:
        dim += sequence[0].dim()
    return torch.cat(list(t.unsqueeze(dim) for t in sequence), dim)
