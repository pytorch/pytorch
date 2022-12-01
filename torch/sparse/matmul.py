import torch
from torch_sparse import SparseTensor
from torch_sparse.matmul import spspmm
from torch_geometric.utils import spmm

@torch.jit._overload  # noqa: F811
def matmul(src, other, reduce):  # noqa: F811
    # type: (SparseTensor, torch.Tensor, str) -> torch.Tensor
    pass


@torch.jit._overload  # noqa: F811
def matmul(src, other, reduce):  # noqa: F811
    # type: (SparseTensor, SparseTensor, str) -> SparseTensor
    pass


def matmul(src, other, reduce="sum"):  # noqa: F811
    if isinstance(other, torch.Tensor):
        return spmm(src, other, reduce)
    elif isinstance(other, SparseTensor):
        return spspmm(src, other, reduce)
    raise ValueError

SparseTensor.matmul = lambda self, other, reduce="sum": matmul(
    self, other, reduce)
SparseTensor.__matmul__ = lambda self, other: matmul(self, other, 'sum')
