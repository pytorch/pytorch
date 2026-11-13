"""FlyDSL override registrations for ``aten::topk``.

Two kernels, picked by (K, N) - see ``flydsl_kernels.py``:

  * register: K in ``_REGISTER_KS``, N pow2 in ``_REGISTER_N_BOUNDS``. Keys
    are ``(ord << 32) | ~idx``, so the smaller index wins equal ordinals in
    either ordering direction. Reproducible on its own, so ``_run`` picks it
    regardless of ``torch.use_deterministic_algorithms``; only the gather
    invariant is checked against aten, not the tie order.
  * radix: (K, N) in a row of ``_RADIX_GATE_RANGES``, K padded up to a
    power of 2 for the bitonic network. For finite inputs, deterministic
    mode gathers in input order and matches aten on values and indices;
    otherwise indices on threshold ties vary across runs. All NaNs share
    one ordinal, so distinct NaN payload/index choices can differ from aten.

``_cond`` also requires a supported dtype on ``_SUPPORTED_ARCHES``, sorted
output, a contiguous last-axis reduction, enough rows for the kernel/dtype
performance gate, and the input (``M * N * itemsize``), values
(``M * K * itemsize``), and indices (``M * K * 8``) buffers inside the 32-bit
span an AMD buffer descriptor addresses. Values are not checked separately
because every supported dtype is at most 4 bytes, so the indices bound is
tighter. The N bounds are closed intervals - both kernels lose to aten again
at large N; anything outside falls through.

``self`` is read through ``ConstTensorWrapper`` so a COW input dispatches
without materialising; writable ``out=`` buffers materialise through
``data_ptr()`` before launch.
"""

from __future__ import annotations

import functools

import torch

from ... import flydsl_utils as fu
from ._common import flatten_last_dim, last_dim_row_major_ok, unflatten_last_dim


_SUPPORTED_ARCHES = ("gfx950",)
_SUPPORTED_DTYPES = (
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.int32,
    torch.int16,
    torch.int8,
)
_REGISTER_KS: frozenset[int] = frozenset({2, 4, 8, 16})

# Per-dtype inclusive N ranges, tuned on MI355.
_REGISTER_N_BOUNDS = {
    torch.float32: (1024, 32768),
    torch.float16: (4096, 4096),
    torch.bfloat16: (4096, 4096),
    torch.int32: (1024, 16384),
    torch.int16: (4096, 4096),
    torch.int8: (8192, 32768),
}
_REGISTER_MIN_CU_WAVES = {
    torch.float32: 4,
    torch.float16: 16,
    torch.bfloat16: 16,
    torch.int32: 4,
    torch.int16: 16,
    torch.int8: 16,
}
_RADIX_MIN_CU_WAVES = 4

# Per-K-range, per-dtype inclusive N ranges, tuned on MI355.
_RADIX_GATE_RANGES = (
    (
        (64, 256),
        {
            torch.float32: (8192, 65536),
            torch.float16: (32768, 131072),
            torch.bfloat16: (32768, 131072),
            torch.int32: (32768, 65536),
            torch.int16: (32768, 131072),
            torch.int8: (32768, 131072),
        },
    ),
    (
        (257, 512),
        {
            torch.float32: (16384, 65536),
            torch.float16: (16384, 262144),
            torch.bfloat16: (16384, 262144),
            torch.int32: (16384, 65536),
            torch.int16: (16384, 262144),
            torch.int8: (65536, 262144),
        },
    ),
    (
        (513, 1024),
        dict.fromkeys(_SUPPORTED_DTYPES, (32768, 524288)),
    ),
)


def _fits_topk_buffer_span(rows_m: int, n: int, k: int, itemsize: int) -> bool:
    if not fu._fits_int32_buffer_span(rows_m, n, itemsize):
        return False
    return rows_m * k * 8 <= (1 << 32) - 1


def _is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def _radix_n_range(
    k: int, dtype: torch.dtype = torch.float32
) -> tuple[int, int] | None:
    for (k_min, k_max), dtype_ranges in _RADIX_GATE_RANGES:
        if k_min <= k <= k_max:
            return dtype_ranges.get(dtype)
    return None


def _kernel_for(k: int, n: int, dtype: torch.dtype = torch.float32) -> str | None:
    register_range = _REGISTER_N_BOUNDS.get(dtype)
    if register_range is not None:
        register_min, register_max = register_range
        if k in _REGISTER_KS and _is_pow2(n) and register_min <= n <= register_max:
            return "register"
    radix_range = _radix_n_range(k, dtype)
    if radix_range is not None and radix_range[0] <= n <= radix_range[1]:
        return "radix"
    return None


@functools.cache
def _min_rows_for_full_wave(device_idx: int) -> int:
    return torch.cuda.get_device_properties(device_idx).multi_processor_count


def _eligible(
    self: torch.Tensor, k: int, dim: int, largest: bool, sorted_: bool
) -> bool:
    if not self.is_cuda or self.dtype not in _SUPPORTED_DTYPES:
        return False
    device_index = self.device.index
    if device_index is None or not fu._is_supported_arch(
        device_index, _SUPPORTED_ARCHES
    ):
        return False
    if not sorted_:
        return False
    if not last_dim_row_major_ok(self, dim):
        return False
    if self.numel() == 0:
        return False
    N = self.shape[-1] if self.ndim >= 1 else 0
    M = self.numel() // N if N else 0
    if not _fits_topk_buffer_span(M, N, k, self.element_size()):
        return False
    kernel = _kernel_for(k, N, self.dtype)
    if kernel is None:
        return False
    # Amortize launch and per-row selection work across enough CU-waves.
    min_rows = _min_rows_for_full_wave(device_index)
    min_cu_waves = (
        _RADIX_MIN_CU_WAVES if kernel == "radix" else _REGISTER_MIN_CU_WAVES[self.dtype]
    )
    if M < min_rows * min_cu_waves:
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
    expected_shape = self.shape[:-1] + (int(k),)
    if values.dtype != self.dtype or values.shape != expected_shape:
        return False
    if indices.dtype != torch.int64 or indices.shape != expected_shape:
        return False
    if values.device != self.device or indices.device != self.device:
        return False
    return True


def _run(
    self: torch.Tensor, k: int, largest: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    from .flydsl_kernels import topk_radix, topk_register

    self_2d = flatten_last_dim(self)
    kernel = _kernel_for(k, self_2d.shape[-1], self.dtype)

    def _launch() -> tuple[torch.Tensor, torch.Tensor]:
        if kernel == "register":
            return topk_register(self_2d, k, largest=largest)
        deterministic = torch.are_deterministic_algorithms_enabled()
        return topk_radix(self_2d, k, deterministic=deterministic, largest=largest)

    # Python-native dispatch does not install a CUDA device guard before
    # launching (unlike generated ATen); see cutedsl_impl.py / #187983.
    device = self.get_device()
    if device == torch.cuda.current_device():
        values_2d, indices_2d = _launch()
    else:
        with torch.cuda.device(device):
            values_2d, indices_2d = _launch()
    return unflatten_last_dim(values_2d, indices_2d, self, k)


def _flatten_topk_out(out: torch.Tensor, k: int) -> torch.Tensor:
    if out.ndim == 2:
        return out
    return out.view(-1, k)


def _run_out(
    self: torch.Tensor,
    k: int,
    largest: bool,
    values: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from .flydsl_kernels import topk_radix_out, topk_register_out

    def _launch_out() -> tuple[torch.Tensor, torch.Tensor]:
        if (
            not values.is_contiguous()
            or not indices.is_contiguous()
            or torch._C._overlaps(self, values)
            or torch._C._overlaps(self, indices)
        ):
            v, i = _run(self, k, largest)
            values.copy_(v)
            indices.copy_(i)
            return values, indices

        self_2d = flatten_last_dim(self)
        values_2d = _flatten_topk_out(values, k)
        indices_2d = _flatten_topk_out(indices, k)
        kernel = _kernel_for(k, self_2d.shape[-1])
        if kernel == "register":
            topk_register_out(self_2d, k, values_2d, indices_2d, largest=largest)
        else:
            deterministic = torch.are_deterministic_algorithms_enabled()
            topk_radix_out(
                self_2d,
                k,
                values_2d,
                indices_2d,
                deterministic=deterministic,
                largest=largest,
            )
        return values, indices

    device = self.get_device()
    if device == torch.cuda.current_device():
        return _launch_out()
    with torch.cuda.device(device):
        return _launch_out()


def _impl(
    self: torch.Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
    *args,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _run(self, int(k), bool(largest))


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
    return _run_out(self, int(k), bool(largest), values, indices)


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
