import warnings
from collections import namedtuple
from typing import Any, Optional, Union, TypeVar, Type
from abc import ABC, abstractmethod, ABCMeta

import torch

__all__ = [
    "SparseSemiStructuredTensor",
    "SparseSemiStructuredMeta"
]

SemiStructuredType = TypeVar("SemiStructuredTensorType", bound="SparseSemiStructuredTensor")

class SparseSemiStructuredMeta(ABCMeta, type(torch.Tensor)):
    pass

class SparseSemiStructuredTensor(ABC):
    _FORCE_CUTLASS = True
    _PROTOTYPE_WARNING_SHOWN = False

    def __repr__(self) -> str:  # type: ignore[override]
        """Return string representation of SparseSemiStructuredTensor

        Returns:
            str: String representation

        Raises:
            None
        """
        return (
            f"{self.__class__.__name__}(shape={self.shape}, "
            f"transposed={self.transposed}, "
            f"values={self.values()}, "
            f"metadata={self.indices()})"
        )

    @staticmethod
    def _show_warning():
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
    def _validate_device_dim_dtype_shape(cls, original_tensor):
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
                f"Error original_tensor is not contiguous!"
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
    def _pad_dense_input(cls, dense_input : torch.Tensor) -> torch.Tensor:
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

        to_pad_m = -m % min_rows if m < min_rows or m % min_rows else 0
        to_pad_n = -n % min_cols if n < min_cols or n % min_rows else 0
        if to_pad_m or to_pad_n:
            return torch.nn.functional.pad(dense_input, (0, to_pad_n, 0, to_pad_m))
        else:
            return dense_input

    @classmethod
    @abstractmethod
    def from_dense(cls, original_tensor) -> SemiStructuredType:
        pass

    @abstractmethod
    def to_dense(self) -> torch.Tensor:
        pass
