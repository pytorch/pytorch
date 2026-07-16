# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Vendored FlyDSL plain RMSNorm forward kernel and PyTorch wrapper.

The device code is derived from ROCm/FlyDSL kernels/norm/rmsnorm_kernel.py at
commit a85595136c647b2ac4532be43ad6e37beaedc085. Only the plain RMSNorm
forward path needed by ATen is included; quantized and fused-add variants
remain out of scope.
"""

# mypy: allow-untyped-defs

from __future__ import annotations

import math

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.vector import ReductionOp
from flydsl.runtime.device import get_rocm_arch

from torch._native.flydsl_cache import jit_cache

from .flydsl_kernel_utils import dtype_to_elem_type
from .flydsl_rmsnorm_common import BLOCK_THREADS, EPS, VEC_WIDTH, WARP_SIZE
from .flydsl_rmsnorm_common import load_scalar as _load_scalar
from .flydsl_rmsnorm_common import load_vec as _load_vec
from .flydsl_rmsnorm_common import (
    make_single_reduction_storage as _make_single_reduction_storage,
)
from .flydsl_rmsnorm_common import store_scalar as _store_scalar
from .flydsl_rmsnorm_common import store_vec as _store_vec
from .flydsl_rmsnorm_common import to_elem_scalar as _to_elem_scalar
from .flydsl_rmsnorm_common import to_elem_vec as _to_elem_vec


_SUPPORTED_DTYPES: dict[torch.dtype, str] = {
    torch.float32: "f32",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
}
_COMPILE_BACKEND_NAME = flyc.compile_backend_name()
_ROCM_ARCH_BY_DEVICE: dict[int, str] = {}


def _dtype_str(dtype: torch.dtype) -> str:
    try:
        return _SUPPORTED_DTYPES[dtype]
    except KeyError as exc:
        raise TypeError(f"unsupported RMSNorm dtype for FlyDSL: {dtype}") from exc


def _normalized_shape_1d(normalized_shape) -> int | None:
    if isinstance(normalized_shape, int):
        return normalized_shape
    if isinstance(normalized_shape, (tuple, list, torch.Size)):
        if len(normalized_shape) != 1:
            return None
        return int(normalized_shape[0])
    return int(normalized_shape)


def _compile_key_arch(device_index: int) -> str:
    arch = _ROCM_ARCH_BY_DEVICE.get(device_index)
    if arch is None:
        arch = str(get_rocm_arch())
        _ROCM_ARCH_BY_DEVICE[device_index] = arch
    return arch


def _forward_block_threads(n: int) -> int:
    if n >= 24576:
        return 1024
    if n >= 12288:
        return 512
    return BLOCK_THREADS


def build_rmsnorm_module(
    N: int,
    dtype_str: str,
    store_rstd: bool = False,
    eps: float = EPS,
):
    if N <= 2048:
        return _build_rmsnorm_large_m_small_n_module(N, dtype_str, store_rstd, eps)

    arch = get_rocm_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")

    block_threads = _forward_block_threads(N)
    elem_bits = 32 if dtype_str == "f32" else 16
    vec_width = 4 if dtype_str == "f32" else VEC_WIDTH
    tile_cols = block_threads * vec_width
    RED_SLOTS = max(1, (block_threads + WARP_SIZE - 1) // WARP_SIZE)

    SharedStorage = _make_single_reduction_storage(RED_SLOTS)

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def rmsnorm_kernel(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Rstd: fx.Tensor,
        Output: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = dtype_to_elem_type(dtype_str)
        fm_fast = arith.FastMathFlags.fast
        eps_c = eps
        n_float = float(N)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_red = lds.s_red.view(fx.make_layout(RED_SLOTS, 1))

        if const_expr(store_rstd):
            Rstd_buf = fx.rocdl.make_buffer_tensor(Rstd)
            rstd_div = fx.logical_divide(Rstd_buf, fx.make_layout(1, 1))
            rstd_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

        def wave_reduce_add(x):
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << _sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        def block_reduce_add(val):
            if const_expr(RED_SLOTS == 1):
                return wave_reduce_add(val)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE

            w = wave_reduce_add(val)

            if lane == 0:
                fx.memref_store(w, s_red, wave)
            gpu.barrier()

            if wave == 0:
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, 0)
                v = fx.memref_load(s_red, lane_safe)
                ww = in_range.select(v, 0.0)
                ww = wave_reduce_add(ww)

                if lane == 0:
                    fx.memref_store(ww, s_red, 0)
            gpu.barrier()

            return fx.memref_load(s_red, 0)

        # ==================================================================
        # Fast path: N is a multiple of tile_cols
        # ==================================================================
        if const_expr(N >= tile_cols and N % tile_cols == 0):
            num_tiles = N // tile_cols
            # Layout API: buffer-backed tensors with tiled access.
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)

            row_in = fx.slice(Input_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))

            in_div = fx.logical_divide(row_in, fx.make_layout(vec_width, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(vec_width, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(vec_width, 1))

            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)

            c_zero_f = fx.Float32(0.0)
            thread_sumsq = c_zero_f
            in_local = []

            # Pass 1: load + cache + sumsq
            for tile_i in range_constexpr(num_tiles):
                idx = tid + tile_i * block_threads
                vec = _load_vec(copy_atom, vec_width, elem_dtype, in_div, idx)
                in_local.append(vec)
                x = vec.to(fx.Float32)

                x2 = x * x
                red2 = x2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sumsq = thread_sumsq + red2

            sum_sq = block_reduce_add(thread_sumsq)
            mean_sq = sum_sq / n_float
            ms_eps = mean_sq + eps_c
            rrms = fmath.rsqrt(ms_eps, fastmath=fm_fast)

            if const_expr(store_rstd):
                if tid == 0:
                    _store_scalar(
                        rstd_copy_atom,
                        fx.Float32,
                        fx.Float32,
                        rstd_div,
                        bid,
                        rrms,
                    )

            # Pass 2: normalize + gamma + store (reuse cached input)
            for tile_i in range_constexpr(num_tiles):
                idx = tid + tile_i * block_threads

                g = _load_vec(
                    copy_atom, vec_width, elem_dtype, gamma_div, idx
                ).to(fx.Float32)
                x = in_local[tile_i].to(fx.Float32)

                y = (x * rrms) * g
                out_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, y)

                out_idx = tid + tile_i * block_threads
                _store_vec(copy_atom, vec_width, elem_dtype, out_e, out_div, out_idx)

        else:
            # ==============================================================
            # Generic path: 128-bit vector body plus scalar tail.
            # ==============================================================
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)

            row_in = fx.slice(Input_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))

            generic_vec_width = 4 if dtype_str == "f32" else VEC_WIDTH
            full_vecs = N // generic_vec_width
            vec_steps = (full_vecs + block_threads - 1) // block_threads
            scalar_tail_start = full_vecs * generic_vec_width
            scalar_tail_elems = N - scalar_tail_start

            copy_atom_v = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
            copy_atom_s = fx.make_copy_atom(
                fx.rocdl.BufferCopy16b()
                if elem_bits <= 16
                else fx.rocdl.BufferCopy32b(),
                elem_bits,
            )

            in_div = fx.logical_divide(row_in, fx.make_layout(generic_vec_width, 1))
            out_vec_div = fx.logical_divide(row_out, fx.make_layout(generic_vec_width, 1))
            gamma_vec_div = fx.logical_divide(Gamma_buf, fx.make_layout(generic_vec_width, 1))
            row_div = fx.logical_divide(row_in, fx.make_layout(1, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(1, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(1, 1))

            c_zero_f = fx.Float32(0.0)
            thread_sumsq = c_zero_f
            in_local = []
            tail_x = c_zero_f

            for step in range_constexpr(vec_steps):
                vec_idx = tid + step * block_threads
                is_valid = vec_idx < full_vecs
                vec_idx_safe = is_valid.select(vec_idx, 0)
                vec = _load_vec(copy_atom_v, generic_vec_width, elem_dtype, in_div, vec_idx_safe)
                in_local.append(vec)
                x = vec.to(fx.Float32)
                x2 = x * x
                red2 = x2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                red2_safe = is_valid.select(red2, c_zero_f)
                thread_sumsq = thread_sumsq + red2_safe

            if const_expr(scalar_tail_elems > 0):
                tail_valid = tid < scalar_tail_elems
                tail_idx = scalar_tail_start + tid
                tail_idx_safe = tail_valid.select(tail_idx, 0)
                tail_x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, tail_idx_safe)
                tail_x = tail_x_e if dtype_str == "f32" else tail_x_e.to(fx.Float32)
                tail_x2 = tail_x * tail_x
                thread_sumsq = thread_sumsq + tail_valid.select(tail_x2, c_zero_f)

            sum_sq = block_reduce_add(thread_sumsq)
            mean_sq = sum_sq / n_float
            ms_eps = mean_sq + eps_c
            rrms = fmath.rsqrt(ms_eps, fastmath=fm_fast)

            if const_expr(store_rstd):
                if tid == 0:
                    _store_scalar(
                        rstd_copy_atom,
                        fx.Float32,
                        fx.Float32,
                        rstd_div,
                        bid,
                        rrms,
                    )

            for step in range_constexpr(vec_steps):
                vec_idx = tid + step * block_threads
                if vec_idx < full_vecs:
                    g = _load_vec(
                        copy_atom_v, generic_vec_width, elem_dtype, gamma_vec_div, vec_idx
                    ).to(fx.Float32)
                    x = in_local[step].to(fx.Float32)
                    y = (x * rrms) * g
                    out_e = _to_elem_vec(
                        dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, y
                    )
                    _store_vec(
                        copy_atom_v, generic_vec_width, elem_dtype, out_e, out_vec_div, vec_idx
                    )

            if const_expr(scalar_tail_elems > 0):
                tail_valid = tid < scalar_tail_elems
                tail_idx = scalar_tail_start + tid
                if tail_valid:
                    g_e = _load_scalar(copy_atom_s, elem_dtype, gamma_div, tail_idx)
                    g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                    y = (tail_x * rrms) * g
                    y_e = _to_elem_scalar(dtype_str, elem_dtype, y)
                    _store_scalar(
                        copy_atom_s, elem_dtype, elem_dtype, out_div, tail_idx, y_e
                    )

    if store_rstd:

        @flyc.jit
        def launch_rmsnorm(
            Input: fx.Tensor,
            Gamma: fx.Tensor,
            Output: fx.Tensor,
            Rstd: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            launcher = rmsnorm_kernel(Input, Gamma, Rstd, Output)
            launcher.launch(
                grid=(m_in, 1, 1),
                block=(block_threads, 1, 1),
                stream=stream,
            )

        return launch_rmsnorm

    @flyc.jit
    def launch_rmsnorm(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Output: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        # store_rstd=False path: the Rstd slot is an unused placeholder here, so
        # we pass Gamma to fill the argument (it is never dereferenced in-kernel).
        launcher = rmsnorm_kernel(Input, Gamma, Gamma, Output)
        launcher.launch(
            grid=(m_in, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_rmsnorm


def _build_rmsnorm_large_m_small_n_module(
    N: int,
    dtype_str: str,
    store_rstd: bool = False,
    eps: float = EPS,
):
    BLOCK_N = 1 << (N - 1).bit_length()
    BLOCK_M = max(min(16384 // BLOCK_N, 32), 8)
    THREADS_PER_ROW = min(WARP_SIZE, 1024 // BLOCK_M)
    BLOCK_THREADS_SPECIAL = BLOCK_M * THREADS_PER_ROW
    elem_bits = 32 if dtype_str == "f32" else 16

    @flyc.kernel(known_block_size=[BLOCK_THREADS_SPECIAL, 1, 1])
    def rmsnorm_large_m_small_n_kernel(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Rstd: fx.Tensor,
        Output: fx.Tensor,
        MIn: fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        lane = tid % THREADS_PER_ROW
        row_local = tid // THREADS_PER_ROW
        row = bid * fx.Int32(BLOCK_M) + row_local

        if row < MIn:
            elem_dtype = dtype_to_elem_type(dtype_str)
            fm_fast = arith.FastMathFlags.fast
            eps_c = eps
            n_float = float(N)

            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            if const_expr(store_rstd):
                Rstd_buf = fx.rocdl.make_buffer_tensor(Rstd)
                rstd_div = fx.logical_divide(Rstd_buf, fx.make_layout(1, 1))
                rstd_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

            row_in = fx.slice(Input_buf, (row, None))
            row_out = fx.slice(Output_buf, (row, None))

            copy_atom_s = fx.make_copy_atom(
                fx.rocdl.BufferCopy16b()
                if elem_bits <= 16
                else fx.rocdl.BufferCopy32b(),
                elem_bits,
            )

            row_div = fx.logical_divide(row_in, fx.make_layout(1, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(1, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(1, 1))

            def group_reduce_add(x):
                w = x
                for _sh_exp in range_constexpr(int(math.log2(THREADS_PER_ROW))):
                    off = THREADS_PER_ROW // (2 << _sh_exp)
                    peer = w.shuffle_xor(off, fx.Int32(THREADS_PER_ROW))
                    w = w.addf(peer, fastmath=fm_fast)
                return w

            c_zero_f = fx.Float32(0.0)
            thread_sumsq = c_zero_f

            for base_idx_int in range_constexpr(0, BLOCK_N, THREADS_PER_ROW):
                idx = lane + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, idx_safe)
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                x2 = x * x
                thread_sumsq = thread_sumsq + is_valid.select(x2, c_zero_f)

            sum_sq = group_reduce_add(thread_sumsq)
            mean_sq = sum_sq / n_float
            ms_eps = mean_sq + eps_c
            rrms = fmath.rsqrt(ms_eps, fastmath=fm_fast)

            if const_expr(store_rstd):
                if lane == 0:
                    _store_scalar(
                        rstd_copy_atom,
                        fx.Float32,
                        fx.Float32,
                        rstd_div,
                        row,
                        rrms,
                    )

            for base_idx_int in range_constexpr(0, BLOCK_N, THREADS_PER_ROW):
                idx = lane + base_idx_int
                if idx < N:
                    x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, idx)
                    g_e = _load_scalar(copy_atom_s, elem_dtype, gamma_div, idx)
                    x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                    g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                    y = (x * rrms) * g
                    y_e = _to_elem_scalar(dtype_str, elem_dtype, y)
                    _store_scalar(
                        copy_atom_s, elem_dtype, elem_dtype, out_div, idx, y_e
                    )

    if store_rstd:

        @flyc.jit
        def launch_rmsnorm_large_m_small_n(
            Input: fx.Tensor,
            Gamma: fx.Tensor,
            Output: fx.Tensor,
            Rstd: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            launcher = rmsnorm_large_m_small_n_kernel(Input, Gamma, Rstd, Output, m_in)
            launcher.launch(
                grid=((m_in + fx.Int32(BLOCK_M - 1)) // fx.Int32(BLOCK_M), 1, 1),
                block=(BLOCK_THREADS_SPECIAL, 1, 1),
                stream=stream,
            )

        return launch_rmsnorm_large_m_small_n

    @flyc.jit
    def launch_rmsnorm_large_m_small_n(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Output: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        # store_rstd=False path: the Rstd slot is an unused placeholder here, so
        # we pass Gamma to fill the argument (it is never dereferenced in-kernel).
        launcher = rmsnorm_large_m_small_n_kernel(Input, Gamma, Gamma, Output, m_in)
        launcher.launch(
            grid=((m_in + fx.Int32(BLOCK_M - 1)) // fx.Int32(BLOCK_M), 1, 1),
            block=(BLOCK_THREADS_SPECIAL, 1, 1),
            stream=stream,
        )

    return launch_rmsnorm_large_m_small_n


def _make_compile_arg(tensor: torch.Tensor):
    """Make only the row dimension dynamic so one kernel supports many M values."""

    return flyc.from_torch_tensor(tensor).mark_shape_dynamic(0)


@jit_cache
def _compile_rmsnorm_fwd(
    n: int,
    dtype: str,
    eps: float,
    arch: str,
    backend: str,
    device_index: int,
    *,
    compile_args,
) -> flyc.CompiledFunction:
    # arch/backend/device_index are explicit cache keys. flyc.compile binds the
    # resulting launcher to the active device/context, so cross-device reuse is
    # unsafe even when two GPUs share the same architecture.
    del arch, backend, device_index
    input_2d, weight, output_2d, rstd, rows_m, stream = compile_args
    launch = build_rmsnorm_module(n, dtype, store_rstd=True, eps=eps)
    return flyc.compile(
        launch,
        _make_compile_arg(input_2d),
        flyc.from_torch_tensor(weight),
        _make_compile_arg(output_2d),
        _make_compile_arg(rstd),
        rows_m,
        stream,
    )


def rmsnorm_fwd(
    input: torch.Tensor,
    normalized_shape,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run FlyDSL forward and return the ATen output/rstd pair."""
    n = _normalized_shape_1d(normalized_shape)
    if n is None:
        raise ValueError("FlyDSL RMSNorm currently requires one normalized dimension")

    rows_m = input.numel() // n
    input_shape = input.shape

    with torch.cuda.device(input.device):
        is_2d = input.ndim == 2
        input_2d = input if is_2d else input.reshape(rows_m, n)
        output_2d = torch.empty_like(input_2d)
        rstd_flat = torch.empty(rows_m, device=input.device, dtype=torch.float32)

        stream = torch.cuda.current_stream()
        device_index = input.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()

        compiled = _compile_rmsnorm_fwd(
            n,
            _dtype_str(input.dtype),
            float(eps),
            _compile_key_arch(device_index),
            _COMPILE_BACKEND_NAME,
            device_index,
            compile_args=(
                input_2d,
                weight,
                output_2d,
                rstd_flat,
                rows_m,
                stream,
            ),
        )

        compiled(input_2d, weight, output_2d, rstd_flat, rows_m, stream)

    if is_2d:
        result = output_2d, rstd_flat.view((rows_m, 1))
    else:
        stat_shape = (*input_shape[:-1], 1)
        result = output_2d.view(input_shape), rstd_flat.view(stat_shape)
    return result


def clear_rmsnorm_caches() -> None:
    """Clear native-op-level compile caches (used by tests/benchmarks)."""

    _compile_rmsnorm_fwd.cache_clear()


def rmsnorm_cache_info() -> dict[str, object]:
    """Return forward cache statistics for diagnostics."""

    return {
        "fwd": _compile_rmsnorm_fwd.cache_info(),
    }