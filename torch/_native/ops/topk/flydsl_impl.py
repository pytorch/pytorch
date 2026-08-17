"""FlyDSL override registrations for ``aten::topk``.

Two kernels, picked by (K, N) - see ``flydsl_kernels.py``:

  * register: K in ``_REGISTER_KS``, N pow2 in ``_REGISTER_N_BOUNDS``. Keys
    are ``(ord << 32) | ~idx``, so ties come out ``(value desc, idx asc)``.
    Reproducible on its own, so ``_run`` picks it regardless of
    ``torch.use_deterministic_algorithms``; only the gather invariant is
    checked against aten, not the tie order.
  * radix: (K, N) in a row of ``_RADIX_GATE_RANGES``, K padded up to a
    power of 2 for the bitonic network. Deterministic mode gathers in
    input order and matches aten on values and indices; otherwise indices
    on threshold ties vary across runs.

``_cond`` also requires fp32 on ``_SUPPORTED_ARCHES``, largest+sorted, a
contiguous last-axis reduction, one CU-wave of rows, and ``M * N *
itemsize`` inside the 32-bit span an AMD buffer descriptor addresses. The
N bounds are closed intervals - both kernels lose to aten again at large
N; anything outside falls through.

``self`` is read through ``ConstTensorWrapper`` so a COW input dispatches
without materialising; ``out=`` is written, so ``_out_cond`` declines COW.
"""

from __future__ import annotations

import functools

import torch

from ... import flydsl_utils as fu
from ._common import (
    any_cow,
    flatten_last_dim,
    last_dim_row_major_ok,
    unflatten_last_dim,
)


_RUNTIME_AVAILABLE: bool = fu.runtime_available()
_SUPPORTED_ARCHES = ("gfx950",)
_REGISTER_KS: frozenset[int] = frozenset({2, 4, 8, 16})

# One (min, max) N range shared by every K in _REGISTER_KS, tuned on MI355.
_REGISTER_N_BOUNDS: tuple[int, int] = (1024, 8192)

# Per-K-range (min, max) N ranges tuned on MI355.
_RADIX_GATE_RANGES = (
    ((64, 256), (8192, 32768)),
    ((257, 383), (16384, 32768)),
    ((384, 831), (32768, 131072)),
    ((832, 1024), (32768, 262144)),
)


def _is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def _radix_n_range(k: int) -> tuple[int, int] | None:
    for (k_min, k_max), n_range in _RADIX_GATE_RANGES:
        if k_min <= k <= k_max:
            return n_range
    return None


def _kernel_for(k: int, n: int) -> str | None:
    register_min, register_max = _REGISTER_N_BOUNDS
    if k in _REGISTER_KS and _is_pow2(n) and register_min <= n <= register_max:
        return "register"
    radix_range = _radix_n_range(k)
    if radix_range is not None and radix_range[0] <= n <= radix_range[1]:
        return "radix"
    return None


@functools.cache
def _is_supported_arch(device_index: int) -> bool:
    arch = fu._resolve_rocm_arch(device_index)
    return arch is not None and arch.split(":", 1)[0] in _SUPPORTED_ARCHES


def _fits_int32_buffer_span(rows_m: int, n: int, itemsize: int) -> bool:
    int32_max = (1 << 31) - 1
    buffer_bytes_max = (1 << 32) - 1
    return (
        0 < rows_m <= int32_max
        and 0 < n <= int32_max
        and rows_m * n * itemsize <= buffer_bytes_max
    )


@functools.cache
def _min_rows_for_full_wave(device_idx: int) -> int:
    return torch.cuda.get_device_properties(device_idx).multi_processor_count


def _eligible(
    self: torch.Tensor, k: int, dim: int, largest: bool, sorted_: bool
) -> bool:
    if not _RUNTIME_AVAILABLE:
        return False
    if torch.version.hip is None:
        return False
    if not self.is_cuda or self.dtype != torch.float32:
        return False
    device_index = self.device.index
    if device_index is None or not _is_supported_arch(device_index):
        return False
    if not largest or not sorted_:
        return False
    if not last_dim_row_major_ok(self, dim):
        return False
    if self.numel() == 0:
        return False
    N = self.shape[-1] if self.ndim >= 1 else 0
    M = self.numel() // N if N else 0
    if not _fits_int32_buffer_span(M, N, self.element_size()):
        return False
    if M < _min_rows_for_full_wave(device_index):
        return False
    return _kernel_for(k, N) is not None


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
    expected_shape = self.shape[:-1] + (int(k),)
    if values.dtype != torch.float32 or values.shape != expected_shape:
        return False
    if indices.dtype != torch.int64 or indices.shape != expected_shape:
        return False
    if values.device != self.device or indices.device != self.device:
        return False
    return True


def _run(self: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    from .flydsl_kernels import topk_radix, topk_register

    self_2d = flatten_last_dim(self)
    kernel = _kernel_for(k, self_2d.shape[-1])
    if kernel == "register":
        values_2d, indices_2d = topk_register(self_2d, k)
        return unflatten_last_dim(values_2d, indices_2d, self, k)
    deterministic = torch.are_deterministic_algorithms_enabled()
    values_2d, indices_2d = topk_radix(self_2d, k, deterministic=deterministic)
    return unflatten_last_dim(values_2d, indices_2d, self, k)


def _flatten_topk_out(out: torch.Tensor, k: int) -> torch.Tensor:
    if out.ndim == 2:
        return out
    if out.ndim == 1:
        return out.view(1, k)
    return out.view(-1, k)


def _run_out(
    self: torch.Tensor, k: int, values: torch.Tensor, indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    from .flydsl_kernels import topk_radix_out, topk_register_out

    if (
        not values.is_contiguous()
        or not indices.is_contiguous()
        or torch._C._overlaps(self, values)
        or torch._C._overlaps(self, indices)
    ):
        v, i = _run(self, k)
        values.copy_(v)
        indices.copy_(i)
        return values, indices

    self_2d = flatten_last_dim(self)
    values_2d = _flatten_topk_out(values, k)
    indices_2d = _flatten_topk_out(indices, k)
    kernel = _kernel_for(k, self_2d.shape[-1])
    if kernel == "register":
        topk_register_out(self_2d, k, values_2d, indices_2d)
    else:
        deterministic = torch.are_deterministic_algorithms_enabled()
        topk_radix_out(self_2d, k, values_2d, indices_2d, deterministic=deterministic)
    return values, indices


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
    return _run_out(self, int(k), values, indices)


def register_to_dispatch() -> None:
    for op_symbol, cond, impl in (
        ("topk", _cond, _impl),
        ("topk.values", _out_cond, _out_impl),
    ):
        fu.register_op_override(
            "aten",
            op_symbol,
            "CUDA",
            cond=cond,
            impl=impl,
            allow_multiple_override=True,
        )
