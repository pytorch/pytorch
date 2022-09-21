import torch
from typing import Tuple
from torch_sparse.tensor import SparseTensor
import torch_sparse.matmul as _sparse

def spmm_sum(src: SparseTensor, other: torch.Tensor) -> torch.Tensor:
    
    rowptr, col, value = src.csr()

    if value is not None:
        value = value.to(other.dtype)

    if value is not None and value.requires_grad:
        row = src.storage.row()

    if other.requires_grad:
        row = src.storage.row()
        csr2csc = src.storage.csr2csc()
        colptr = src.storage.colptr()

    return torch.spmm_sum(rowptr, col, value, other)


def spmm_add(src: SparseTensor, other: torch.Tensor) -> torch.Tensor:
    return spmm_sum(src, other)

def spmm_mean(src: SparseTensor, other: torch.Tensor) -> torch.Tensor:
    return _sparse.spmm_mean(src, other)

def spmm_min(src: SparseTensor,
             other: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return _sparse.spmm_min(src, other)


def spmm_max(src: SparseTensor,
             other: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return _sparse.spmm_max(src, other)


def spmm(src: SparseTensor, other: torch.Tensor,
         reduce: str = "sum") -> torch.Tensor:
    if reduce == 'sum' or reduce == 'add':
        return spmm_sum(src, other)
    elif reduce == 'mean':
        return spmm_mean(src, other)
    elif reduce == 'min':
        return spmm_min(src, other)[0]
    elif reduce == 'max':
        return spmm_max(src, other)[0]
    else:
        raise ValueError


def spspmm_sum(src: SparseTensor, other: SparseTensor) -> SparseTensor:
    return _sparse.spspmm_sum(src, other)


def spspmm_add(src: SparseTensor, other: SparseTensor) -> SparseTensor:
    return _sparse.spspmm_add(src, other)


def spspmm(src: SparseTensor, other: SparseTensor,
           reduce: str = "sum") -> SparseTensor:
   return _sparse.spspmm(src, other, reduce)


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


SparseTensor.spmm = lambda self, other, reduce="sum": spmm(self, other, reduce)
SparseTensor.spspmm = lambda self, other, reduce="sum": spspmm(
    self, other, reduce)
SparseTensor.matmul = lambda self, other, reduce="sum": matmul(
    self, other, reduce)
SparseTensor.__matmul__ = lambda self, other: matmul(self, other, 'sum')
