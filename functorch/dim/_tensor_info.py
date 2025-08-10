from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from ._dim_entry import DimEntry


@dataclass
class TensorInfo:
    tensor: Optional[torch.Tensor]
    levels: list[DimEntry]
    has_device: bool
    batchedtensor: Optional[torch.Tensor]

    def __post_init__(self) -> None:
        from ._dim_entry import DimEntry

        assert all(isinstance(l, DimEntry) for l in self.levels)

    def ndim(self) -> int:
        from ._dim_entry import ndim_of_levels

        return ndim_of_levels(self.levels)

    def __bool__(self) -> bool:
        return self.tensor is not None

    @staticmethod
    def create(
        h: Any, ensure_batched: bool = True, ensure_present: bool = True
    ) -> TensorInfo:
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
            # For Dim objects, only get range/batchtensor if needed and dimension is bound
            tensor = h._get_range() if h.is_bound else None
            batchtensor = (
                h._get_batchtensor() if ensure_batched and h.is_bound else None
            )
            return TensorInfo(
                tensor,
                [DimEntry(h)],
                False,
                batchtensor,
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
