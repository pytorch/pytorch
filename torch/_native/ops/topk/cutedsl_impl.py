"""CuTeDSL override registrations for ``aten::topk``.

One fused CTA-per-row radix-select + bitonic-sort kernel
(``cutedsl_kernels.py``). Runs four radix byte passes over shared memory
histograms (thread-0 picks the winning bin each pass), gathers the
selected elements, then sorts the K survivors with a cooperative bitonic
pass. Two specialisations of phase 2 / phase 3:
  * Deterministic (default under ``torch.use_deterministic_algorithms``):
    block-wide prefix-sum gather + lex ``(ord, -idx)`` sort. Bit-exact
    match to aten on values and indices.
  * Non-deterministic: smem atomic-counter gather + ord-only sort.
    Faster (~5-10%); indices may differ across runs on threshold ties.

Eligibility (see ``_cond``):
  - fp32 input, CUDA, not COW
  - ``largest=True``, ``sorted=True``
  - reducing over the last axis, ``self`` contiguous (2D flatten is a view)
  - ``k`` in ``{64, 128, 256, 512, 1024}`` (must be a power of 2 for the
    bitonic sort; K == block threads for the cooperative sort)
  - ``N >= MIN_N_MULT[k] * k`` and ``N % 4 == 0`` (perf gate: 2*k for
    k<=256, 8*k for k=512, 32*k for k=1024; plus 128-bit vector loads)
  - row count at least one full wave of SMs (perf gate)

Anything else falls through to aten.
"""

import functools
import math

import torch

from ... import cutedsl_utils as cu
from ._common import (
    any_cow,
    flatten_last_dim,
    last_dim_row_major_ok,
    unflatten_last_dim,
)


_SUPPORTED_KS: frozenset[int] = frozenset({64, 128, 256, 512, 1024})

# Per-K minimum N below which aten wins on B200. At low N the radix
# sweep (4 passes + gather) has too many passes over too little data
# to beat aten's launch-bound multi-pass kernel. K=512 needs ~8*K,
# K=1024 needs ~32*K because its smem footprint drops occupancy.
# Smaller K fits the 2*K rule tuned earlier.
_MIN_N_MULTIPLIER: dict[int, int] = {
    64: 2,
    128: 2,
    256: 2,
    512: 8,
    1024: 32,
}


@functools.cache
def _min_rows_for_full_wave(device_idx: int) -> int:
    """Row threshold below which the one-CTA-per-row kernel underutilises
    the GPU. A full wave is SM-count CTAs; below that, aten's multi-CTA
    kernel gets more parallelism out of the same rows and wins."""
    return torch.cuda.get_device_properties(device_idx).multi_processor_count


def _eligible(
    self: torch.Tensor, k: int, dim: int, largest: bool, sorted_: bool
) -> bool:
    if not self.is_cuda or self.dtype != torch.float32:
        return False
    if any_cow(self):
        return False
    if not largest or not sorted_:
        return False
    if k not in _SUPPORTED_KS:
        return False
    if not last_dim_row_major_ok(self, dim):
        return False
    N = self.shape[-1] if self.ndim >= 1 else 0
    # Perf gate: per-K minimum N. Below these thresholds aten is faster.
    if N < _MIN_N_MULTIPLIER[k] * k or (N % 4) != 0:
        return False
    # Performance gate: reject shapes where aten is faster. One CTA per
    # row means row_count < SM_count leaves the GPU underutilised.
    M = math.prod(self.shape[:-1]) if self.ndim >= 1 else 0
    if M < _min_rows_for_full_wave(self.device.index or 0):
        return False
    return True


def _cond(
    self: torch.Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
    *args,
    **kwargs,
) -> bool:
    return _eligible(self, int(k), int(dim), bool(largest), bool(sorted))


def _out_cond(
    self: torch.Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
    *,
    values: torch.Tensor,
    indices: torch.Tensor,
) -> bool:
    if not _cond(self, k, dim, largest, sorted):
        return False
    if any_cow(values, indices):
        return False
    expected_shape = self.shape[:-1] + (k,)
    if values.dtype != torch.float32 or values.shape != expected_shape:
        return False
    if indices.dtype != torch.int64 or indices.shape != expected_shape:
        return False
    return True


def _run(self: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    from .cutedsl_kernels import topk_radix

    self_2d = flatten_last_dim(self)
    # Pick the deterministic kernel under torch.use_deterministic_algorithms
    # (kept off by default for perf; the non-det kernel still produces
    # correct top-K values but indices may differ on ties).
    deterministic = torch.are_deterministic_algorithms_enabled()
    values_2d, indices_2d = topk_radix(self_2d, k, deterministic=deterministic)
    return unflatten_last_dim(values_2d, indices_2d, self, k)


def _impl(
    self: torch.Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
    *args,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _run(self, int(k))


def _out_impl(
    self: torch.Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
    *,
    values: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    v, i = _run(self, int(k))
    values.copy_(v)
    indices.copy_(i)
    return values, indices


def register_to_dispatch() -> None:
    for op_symbol, cond, impl in (
        ("topk", _cond, _impl),
        ("topk.values", _out_cond, _out_impl),
    ):
        cu.register_op_override(
            "aten",
            op_symbol,
            "CUDA",
            cond=cond,
            impl=impl,
            allow_multiple_override=True,
        )
