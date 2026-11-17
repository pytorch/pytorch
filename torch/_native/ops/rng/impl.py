r"""Register supported CUDA uniform and normal overrides."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ... import cutedsl_utils as cu
from ...utils import capability as cap
from ...utils.lazy import LazyModule
from ...utils.scalar import f32
from ...utils.tensor import can_use_32bit_indexing


if TYPE_CHECKING:
    from . import cutedsl_kernels as CK
else:
    CK = LazyModule("torch._native.ops.rng.cutedsl_kernels")


_DTYPES = (torch.float32,)

_ARCH_MAJORS = cu.POINTWISE_ARCH_MAJORS


def _serveable(self: torch.Tensor, generator: torch.Generator | None) -> bool:
    # Explicit generators use ATen.
    return (
        generator is None
        and isinstance(self, torch.Tensor)
        and self.dtype in _DTYPES
        and self.is_contiguous()
        and self.numel() > 0
        and can_use_32bit_indexing(self)
        and 1 + (self.numel() - 1) * self.element_size() <= 2**31 - 1
        and not self.is_neg()
        and not self.is_conj()
        and not cap.is_traced(self)
        and cap.device_ok(self, _ARCH_MAJORS)
        and cap.on_current_device(self)
    )


def _uniform_cond(
    self: torch.Tensor,
    from_: float = 0.0,
    to: float = 1.0,
    *,
    generator: torch.Generator | None = None,
) -> bool:
    if not _serveable(self, generator):
        return False
    limit = torch.finfo(self.dtype).max
    return -limit <= from_ <= to <= limit and to - from_ <= limit


def _uniform_impl(
    self: torch.Tensor,
    from_: float = 0.0,
    to: float = 1.0,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    # Map a rounded upper endpoint back to from_, matching ATen.
    from_, to = f32(from_), f32(to)
    return CK.fill_random(self, "uniform", to - from_, from_, to)


def _normal_cond(
    self: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    *,
    generator: torch.Generator | None = None,
) -> bool:
    if not (std >= 0.0):
        return False
    return _serveable(self, generator)


def _normal_impl(
    self: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    return CK.fill_random(self, "normal", float(std), float(mean))


def register_rng_overrides() -> None:
    cu.register_op_override(
        "aten", "uniform_", "CUDA", cond=_uniform_cond, impl=_uniform_impl
    )
    cu.register_op_override(
        "aten", "normal_", "CUDA", cond=_normal_cond, impl=_normal_impl
    )
