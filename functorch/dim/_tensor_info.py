from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TensorInfo:
    tensor: torch.Tensor
    levels: list[DimEntry]
    has_device: bool
    batchedtensor: torch.Tensor

    def ndim(self) -> int:
        from ._dim_entry import ndim_of_levels

        return ndim_of_levels(self.levels)

    def __bool__(self) -> bool:
        return self.tensor is not None

    @staticmethod
    def create(h, ensure_batched: bool = True, ensure_present: bool = True):
        from . import Dim, DimEntry, Tensor

        if Tensor.check_exact(h):
            # functorch Tensor with first-class dimensions
            return TensorInfo(
                h._get_tensor(),
                h._get_levels(),
                h._get_has_device(),
                h._get_batchtensor() if ensure_batched else None,
            )
        elif Dim.check_exact(h):
            return TensorInfo(
                h._get_range(),
                [DimEntry(h)],
                False,
                h._get_batchtensor() if ensure_batched else None,
            )
        elif isinstance(h, torch.Tensor):
            # Plain torch tensor - create positional levels
            levels = []
            for i in range(-h.dim(), 0):
                levels.append(DimEntry(i))
            return TensorInfo(h, levels, True, h)
        else:
            if ensure_present:
                raise ValueError("expected a tensor object")
            return TensorInfo(None, [], False, None)
