import warnings

from abc import ABC, ABCMeta, abstractclassmethod, abstractmethod
from collections import namedtuple
from typing import Any, Optional, Type, TypeVar, Union

import torch

__all__ = [
    "SparseSemiStructuredTensor",
    "to_sparse_semi_structured",
]

_SEMI_STRUCTURED_SPARSE_CONFIG = namedtuple(
    "_SEMI_STRUCTURED_SPARSE_CONFIG",
    "sparse_min_rows sparse_min_cols dense_min_rows dense_min_cols",
)


class SparseSemiStructuredMeta(ABCMeta, torch._C._TensorMeta):
    pass


class SparseSemiStructuredTensor(ABC):
    _FORCE_CUTLASS = True
    _AUTOPAD_DENSE_INPUT = True
    _DTYPE_SHAPE_CONSTRAINTS = {}
    _PROTOTYPE_WARNING_SHOWN = False

    def __repr__(self) -> str:  # type: ignore[override]
        """Return string representation of SparseSemiStructuredTensor

        Returns:
            str: String representation

        Raises:
            None
        """
        assert hasattr(self, "shape")
        assert hasattr(self, "transposed")
        return (
            f"{self.__class__.__name__}(shape={self.shape}, "
            f"transposed={self.transposed})"
        )

    @staticmethod
    def _show_warning() -> None:
        if not SparseSemiStructuredTensor._PROTOTYPE_WARNING_SHOWN:
            warnings.warn(
                (
                    "The PyTorch API of SparseSemiStructuredTensor is in prototype stage "
                    "and will change in the near future. Please open a Github issue "
                    "for features requests and see our documentation on the torch.sparse "
                    "module for further information about the project."
                ),
                UserWarning,
            )
            SparseSemiStructuredTensor._PROTOTYPE_WARNING_SHOWN = True

    @classmethod
    def _validate_device_dim_dtype_shape(cls, original_tensor) -> None:
        # check device
        if not original_tensor.is_cuda:
            raise RuntimeError(
                f"Error original_tensor.device= {original_tensor.device} is not supported! "
                "Only CUDA tensors are currently supported."
            )

        # check dim
        if original_tensor.dim() != 2:
            raise RuntimeError(
                f"Error original_tensor.dim = {original_tensor.dim()} is not supported! "
                "Only 2d tensors are currently supported."
            )

        # check contiguous
        if not original_tensor.is_contiguous():
            raise RuntimeError(
                "Error original_tensor is not contiguous!"
                "Only contiguous tensors are currently supported."
            )

        # check dtype
        if original_tensor.dtype not in cls._DTYPE_SHAPE_CONSTRAINTS:
            raise RuntimeError(
                f"Error original_tensor.dtype {original_tensor.dtype} is not a supported dtype! "
                "dtype must be one of: {cls._DTYPE_SHAPE_CONSTRAINTS}"
            )

        # check shape
        m, n = original_tensor.shape
        min_rows = cls._DTYPE_SHAPE_CONSTRAINTS[original_tensor.dtype].sparse_min_rows
        min_cols = cls._DTYPE_SHAPE_CONSTRAINTS[original_tensor.dtype].sparse_min_cols
        if m < min_rows or m % min_rows or n < min_cols or n % min_cols:
            # TODO in the future we can add in padding to support sparse dimensions that aren't perfect multiples
            raise RuntimeError(
                f"Error original_tensor.shape {original_tensor.shape} is not supported! "
                f"Both dimensions must be larger or equal than and a multiple of ({min_rows}, {min_cols})"
            )

    @classmethod
    def _pad_dense_input(cls, dense_input: torch.Tensor) -> torch.Tensor:
        """
        Calculates padding for dense tensor and pads tensor if necessary.
        If padding is not required, this function returns the original tensor.
        """
        # only 2d matmul
        assert dense_input.dim() == 2

        # check shape
        m, n = dense_input.shape
        min_rows = cls._DTYPE_SHAPE_CONSTRAINTS[dense_input.dtype].dense_min_rows
        min_cols = cls._DTYPE_SHAPE_CONSTRAINTS[dense_input.dtype].dense_min_cols

        # calculate padding
        to_pad_m = -m % min_rows if m < min_rows or m % min_rows else 0
        to_pad_n = -n % min_cols if n < min_cols or n % min_rows else 0
        if to_pad_m or to_pad_n:
            return torch.nn.functional.pad(dense_input, (0, to_pad_n, 0, to_pad_m))
        else:
            return dense_input

    @classmethod
    @abstractmethod
    def from_dense(cls, original_tensor) -> SparseSemiStructuredMeta:
        pass

    @abstractmethod
    def to_dense(self) -> torch.Tensor:
        pass  # type: ignore[empty-body]


def to_sparse_semi_structured(
    original_tensor: torch.Tensor,
) -> Any:
    """
    This function converts a dense tensor into a sparse semi-structured tensor.
    It will return either
        1. a SparseSemiStructuredTensor if the input tensor is already in the correct format
        2. a regular SparseTensor if the input tensor was not in the correct format

    This function will check to ensure the dense tensor has the right dtype, size, dims, and device.
    We currently only support semi-structured sparse tensors for 2d CUDA tensors.
    Additionally, your tensor must be a positive multiple of a block size given the dtype

    - torch.float16  (r, c) must be >= and a multiple of 64
    - torch.int8     (r, c) must be >= and a multiple of 128

    Args:
        original_tensor (Tensor): the dense tensor to convert

    Returns:
        SparseSemiStructuredTensor: A sparse semi-structured tensor created from the given original_tensor

    Raises:
        None

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> A = torch.Tensor([0, 0, 1, 1]).tile((128, 32)).half().cuda()
        tensor([[0., 0., 1.,  ..., 0., 1., 1.],
                [0., 0., 1.,  ..., 0., 1., 1.],
                [0., 0., 1.,  ..., 0., 1., 1.],
                ...,
                [0., 0., 1.,  ..., 0., 1., 1.],
                [0., 0., 1.,  ..., 0., 1., 1.],
                [0., 0., 1.,  ..., 0., 1., 1.]], device='cuda:0', dtype=torch.float16)
        >>> A_sparse = to_sparse_semi_structured(A)
        SparseSemiStructuredTensor(shape=torch.Size([128, 128]), transposed=False, values=tensor([[1., 1., 1.,  ..., 1., 1., 1.],
                [1., 1., 1.,  ..., 1., 1., 1.],
                [1., 1., 1.,  ..., 1., 1., 1.],
                ...,
                [1., 1., 1.,  ..., 1., 1., 1.],
                [1., 1., 1.,  ..., 1., 1., 1.],
                [1., 1., 1.,  ..., 1., 1., 1.]], device='cuda:0', dtype=torch.float16),
            metadata=tensor([[-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                ...,
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370]], device='cuda:0',
       dtype=torch.int16))
    """
    from torch.sparse import (
        SparseSemiStructuredTensorCUSPARSELT,
        SparseSemiStructuredTensorCUTLASS,
    )

    if SparseSemiStructuredTensor._FORCE_CUTLASS:
        return SparseSemiStructuredTensorCUTLASS.from_dense(original_tensor)
    else:
        return SparseSemiStructuredTensorCUSPARSELT.from_dense(original_tensor)
