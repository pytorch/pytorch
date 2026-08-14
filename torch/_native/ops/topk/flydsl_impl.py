"""FlyDSL override registrations for ``aten::topk``.

This is a conservative FlyDSL backend for topk.  The kernel currently handles
small fp32, last-dimension, largest+sorted cases; other shapes fall through to
the existing backends or aten.
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

# Per-K register ranges tuned on MI355.  K=32 loses to aten in the measured
# range.  A separate row-count gate below handles GPU underutilization.
_REGISTER_N_RANGE: tuple[int, int] = (1024, 8192)

# Per-K radix ranges tuned on MI355.
# A separate row-count gate below handles GPU underutilization.
# K>1024 exceeds the device workgroup size limit.
_RADIX_GATE_RANGES = (
    ((64, 256), (8192, 32768)),
    ((257, 383), (16384, 32768)),
    ((384, 831), (32768, 131072)),
    ((832, 1024), (32768, 262144)),
)
_TOPK_KERNELS = None


def _is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def _register_wins(k: int, n: int) -> bool:
    n_min, n_max = _REGISTER_N_RANGE
    return n_min <= n <= n_max


def _radix_n_range(k: int) -> tuple[int, int] | None:
    for (k_min, k_max), n_range in _RADIX_GATE_RANGES:
        if k_min <= k <= k_max:
            return n_range
    return None


def _radix_wins(k: int, n: int) -> bool:
    n_range = _radix_n_range(k)
    if n_range is None:
        return False
    n_min, n_max = n_range
    return n_min <= n <= n_max


def _kernel_for(k: int, n: int) -> str | None:
    if (k in _REGISTER_KS) and _is_pow2(n) and _register_wins(k, n):
        return "register"
    if _radix_wins(k, n):
        return "radix"
    return None


@functools.cache
def _is_supported_arch(device_index: int) -> bool:
    arch = fu._resolve_rocm_arch(device_index)
    return arch is not None and arch.split(":", 1)[0] in _SUPPORTED_ARCHES


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
    if M < _min_rows_for_full_wave(device_index):
        return False
    return _kernel_for(k, N) is not None


def _get_topk_kernels():
    global _TOPK_KERNELS
    if _TOPK_KERNELS is None:
        from .flydsl_kernels import (
            RadixSelectTopK,
            RadixSelectTopKOut,
            RegisterTopK,
            RegisterTopKOut,
        )

        _TOPK_KERNELS = (
            RadixSelectTopK,
            RadixSelectTopKOut,
            RegisterTopK,
            RegisterTopKOut,
        )
    return _TOPK_KERNELS


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
    RadixSelectTopK, _, RegisterTopK, _ = _get_topk_kernels()

    self_2d = flatten_last_dim(self)
    kernel = _kernel_for(k, self_2d.shape[-1])
    if kernel == "register":
        values_2d, indices_2d = RegisterTopK(self_2d, k)
        return unflatten_last_dim(values_2d, indices_2d, self, k)
    deterministic = torch.are_deterministic_algorithms_enabled()
    values_2d, indices_2d = RadixSelectTopK(self_2d, k, deterministic=deterministic)
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
    _, RadixSelectTopKOut, _, RegisterTopKOut = _get_topk_kernels()

    if not values.is_contiguous() or not indices.is_contiguous():
        v, i = _run(self, k)
        values.copy_(v)
        indices.copy_(i)
        return values, indices

    self_2d = flatten_last_dim(self)
    values_2d = _flatten_topk_out(values, k)
    indices_2d = _flatten_topk_out(indices, k)
    kernel = _kernel_for(k, self_2d.shape[-1])
    if kernel == "register":
        RegisterTopKOut(self_2d, k, values_2d, indices_2d)
    else:
        deterministic = torch.are_deterministic_algorithms_enabled()
        RadixSelectTopKOut(
            self_2d, k, values_2d, indices_2d, deterministic=deterministic
        )
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
