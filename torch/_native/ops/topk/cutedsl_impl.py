"""CuTeDSL override registrations for ``aten::topk``.

Two kernels, picked by dtype, capability, K, and N - see ``aot.py``:

  * Register-resident (small K, small N): K in {16, 32}, N a power of 2
    in a per-K range (see ``_REGISTER_N_RANGE``). Each warp sorts one
    row entirely in registers and writes only K outputs to gmem.
    Bit-exact to aten on values; indices on ties land in
    ``(value desc, idx asc)`` order which doesn't match aten's small-K
    CUDA kernel - same gather invariant though.

  * Fused radix-select (larger K): K in {64, 128, 256, 512, 1024}.
    Four radix byte passes over smem histograms, then a cooperative
    bitonic sort of the K survivors. Two phase-2/phase-3 specialisations:
      - Deterministic (under ``torch.use_deterministic_algorithms``):
        block-wide prefix-sum gather + lex ``(ord, -idx)`` sort.
        Bit-exact match to aten on values and indices.
      - Non-deterministic: smem atomic-counter gather + ord-only sort.
        Faster (~5-10%); indices may differ across runs on threshold ties.

Common eligibility (see ``_cond``):
  - fp32/bf16 input, CUDA on SM90 or newer
  - ``largest=True``, ``sorted=True``
  - reducing over the last axis, ``self`` contiguous (2D flatten is a view)
  - row count at least one full wave of SMs (perf gate)

Per-kernel additional eligibility:
  - register: fp32, K in {16, 32}, N in the measured exact-N set
  - radix: fp32/bf16, K in {64, 128, 256, 512, 1024}, N above the
    dtype/capability threshold, and ``N % 4 == 0`` (128-bit vector loads)

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
from .aot import _kernel_for, _RADIX_KS as _AOT_RADIX_KS, _RADIX_MIN_N, _REGISTER_NS


_RADIX_KS: frozenset[int] = frozenset(_AOT_RADIX_KS)
_REGISTER_KS: frozenset[int] = frozenset(_REGISTER_NS)
_SUPPORTED_KS: frozenset[int] = _RADIX_KS | _REGISTER_KS
_REGISTER_N_RANGE: dict[int, tuple[int, int]] = {
    k: (min(ns), max(ns)) for k, ns in _REGISTER_NS.items()
}
# Kept as a public test/tuning aid for the original SM100 fp32 policy.
_RADIX_MIN_N_MULTIPLIER: dict[int, int] = {
    k: n // k for k, n in _RADIX_MIN_N["float32"][10].items()
}
_DTYPE_NAMES = {
    torch.float32: "float32",
    torch.bfloat16: "bfloat16",
}


@functools.cache
def _device_major(device: int) -> int:
    return torch.cuda.get_device_capability(device)[0]


@functools.cache
def _min_rows_for_full_wave(device_idx: int) -> int:
    """Row threshold below which the one-CTA-per-row kernel underutilises
    the GPU. A full wave is SM-count CTAs; below that, aten's multi-CTA
    kernel gets more parallelism out of the same rows and wins."""
    return torch.cuda.get_device_properties(device_idx).multi_processor_count


def _eligible(
    self: torch.Tensor, k: int, dim: int, largest: bool, sorted_: bool
) -> bool:
    if not self.is_cuda or self.dtype not in _DTYPE_NAMES:
        return False
    device = self.device.index or 0
    major = _device_major(device)
    if major < 9:
        return False
    if not largest or not sorted_:
        return False
    if not last_dim_row_major_ok(self, dim):
        return False
    if self.const_data_ptr() % (4 * self.element_size()):  # type: ignore[attr-defined]
        return False
    N = self.shape[-1] if self.ndim >= 1 else 0
    if _kernel_for(_DTYPE_NAMES[self.dtype], N, k, major) is None:
        return False
    # Performance gate: reject shapes where aten is faster. One CTA per
    # row (radix) or one warp per row (register) - either way row_count
    # below SM_count leaves the GPU underutilised.
    M = math.prod(self.shape[:-1]) if self.ndim >= 1 else 0
    if M < _min_rows_for_full_wave(device):
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
    if values.dtype != self.dtype or values.shape != expected_shape:
        return False
    if indices.dtype != torch.int64 or indices.shape != expected_shape:
        return False
    return True


def _run(self: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    from .cutedsl_kernels import topk_radix, topk_register

    self_2d = flatten_last_dim(self)
    N = self_2d.shape[-1]
    major = _device_major(self.device.index or 0)
    kernel = _kernel_for(_DTYPE_NAMES[self.dtype], N, k, major)

    def _launch() -> tuple[torch.Tensor, torch.Tensor]:
        if kernel == "register":
            return topk_register(self_2d, k)
        # Pick the deterministic kernel under torch.use_deterministic_algorithms
        # (kept off by default for perf; the non-det kernel still produces
        # correct top-K values but indices may differ on ties).
        deterministic = torch.are_deterministic_algorithms_enabled()
        return topk_radix(self_2d, k, deterministic=deterministic)

    # The Python-native dispatch path does not get an automatic CUDA device
    # guard before launching the kernel (unlike the generated C++ ATen path),
    # so the CuTeDSL kernel runs on the current device's stream. Guard the
    # launch when the input is not already on the current device, while leaving
    # the common already-current path direct to avoid the context-manager
    # overhead. See #187983 for the same fix on the bmm override.
    device = self.get_device()
    if device == torch.cuda.current_device():
        values_2d, indices_2d = _launch()
    else:
        with torch.cuda.device(device):
            values_2d, indices_2d = _launch()
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
