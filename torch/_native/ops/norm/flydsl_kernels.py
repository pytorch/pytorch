# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Vendored FlyDSL plain RMSNorm FWD/BWD kernels and PyTorch wrappers.

The device code is derived from ROCm/FlyDSL kernels/norm/rmsnorm_kernel.py
and rmsnorm_bwd_kernel.py at commit
a85595136c647b2ac4532be43ad6e37beaedc085. Only the plain RMSNorm path
needed by ATen is included; quantized and fused-add variants remain out of
scope.
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
from .flydsl_rmsnorm_bwd_kernel import build_rmsnorm_bwd_module
from .flydsl_rmsnorm_common import BLOCK_THREADS, EPS, VEC_WIDTH, WARP_SIZE
from .flydsl_rmsnorm_common import load_scalar as _load_scalar
from .flydsl_rmsnorm_common import load_vec as _load_vec
from .flydsl_rmsnorm_common import (
    make_reduction_storage as _make_reduction_storage,
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


def _dtype_str(dtype: torch.dtype) -> str:
    try:
        return _SUPPORTED_DTYPES[dtype]
    except KeyError as exc:
        raise TypeError(f"unsupported RMSNorm dtype for FlyDSL: {dtype}") from exc


def _canonical_normalized_shape(normalized_shape) -> tuple[int, ...]:
    if isinstance(normalized_shape, torch.Size):
        return tuple(int(x) for x in normalized_shape)
    if isinstance(normalized_shape, (tuple, list)):
        return tuple(int(x) for x in normalized_shape)
    return (int(normalized_shape),)


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

    tile_cols = BLOCK_THREADS * VEC_WIDTH
    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    elem_bits = 32 if dtype_str == "f32" else 16

    SharedStorage = _make_reduction_storage(RED_SLOTS)

    @flyc.kernel
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
        s_red2 = lds.s_red2.view(fx.make_layout(RED_SLOTS, 1))

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
            dummy = fx.Float32(0.0)
            r0, _ = block_reduce_add2(val, dummy)
            return r0

        def block_reduce_add2(val0, val1):
            if const_expr(RED_SLOTS == 1):
                return wave_reduce_add(val0), wave_reduce_add(val1)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE

            w0 = wave_reduce_add(val0)
            w1 = wave_reduce_add(val1)

            if lane == 0:
                fx.memref_store(w0, s_red, wave)
                fx.memref_store(w1, s_red2, wave)
            gpu.barrier()

            if wave == 0:
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, 0)
                v0 = fx.memref_load(s_red, lane_safe)
                v1 = fx.memref_load(s_red2, lane_safe)
                ww0 = in_range.select(v0, 0.0)
                ww1 = in_range.select(v1, 0.0)
                ww0 = wave_reduce_add(ww0)
                ww1 = wave_reduce_add(ww1)

                if lane == 0:
                    fx.memref_store(ww0, s_red, 0)
                    fx.memref_store(ww1, s_red2, 0)
            gpu.barrier()

            return fx.memref_load(s_red, 0), fx.memref_load(s_red2, 0)

        # ==================================================================
        # Fast path: N is a multiple of tile_cols
        # ==================================================================
        if const_expr(N >= tile_cols and N % tile_cols == 0 and elem_bits <= 16):
            num_tiles = N // tile_cols
            # Layout API: buffer-backed tensors with tiled access.
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)

            row_in = fx.slice(Input_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))

            in_div = fx.logical_divide(row_in, fx.make_layout(VEC_WIDTH, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(VEC_WIDTH, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(VEC_WIDTH, 1))

            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)

            c_zero_f = fx.Float32(0.0)
            thread_sumsq = c_zero_f
            thread_dummy = c_zero_f
            in_local = []

            # Pass 1: load + cache + sumsq
            for tile_i in range_constexpr(num_tiles):
                idx = tid + tile_i * BLOCK_THREADS
                vec = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, in_div, idx)
                in_local.append(vec)
                x = vec.to(fx.Float32)

                x2 = x * x
                red2 = x2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sumsq = thread_sumsq + red2

            _, sum_sq = block_reduce_add2(thread_dummy, thread_sumsq)
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
                idx = tid + tile_i * BLOCK_THREADS

                g = _load_vec(
                    copy_atom, VEC_WIDTH, elem_dtype, gamma_div, idx
                ).to(fx.Float32)
                x = in_local[tile_i].to(fx.Float32)

                y = (x * rrms) * g
                out_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, y)

                out_idx = tid + tile_i * BLOCK_THREADS
                _store_vec(copy_atom, VEC_WIDTH, elem_dtype, out_e, out_div, out_idx)

        else:
            # ==============================================================
            # Generic path: scalar 2-pass for arbitrary N
            # ==============================================================
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)

            row_in = fx.slice(Input_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))

            copy_atom_s = fx.make_copy_atom(
                fx.rocdl.BufferCopy16b()
                if elem_bits <= 16
                else fx.rocdl.BufferCopy32b(),
                elem_bits,
            )

            row_div = fx.logical_divide(row_in, fx.make_layout(1, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(1, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(1, 1))

            c_zero_f = fx.Float32(0.0)
            thread_sumsq = c_zero_f

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, idx_safe)
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                x2 = x * x
                x2_safe = is_valid.select(x2, c_zero_f)
                thread_sumsq = thread_sumsq + x2_safe

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

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                if idx < N:
                    x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, idx)
                    g_e = _load_scalar(copy_atom_s, elem_dtype, gamma_div, idx)
                    x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                    g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                    norm = x * rrms
                    y = norm * g
                    y_e = _to_elem_scalar(dtype_str, elem_dtype, y)
                    _store_scalar(
                        copy_atom_s, elem_dtype, elem_dtype, out_div, idx, y_e
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
                block=(BLOCK_THREADS, 1, 1),
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
            block=(BLOCK_THREADS, 1, 1),
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


@jit_cache
def _compile_rmsnorm_bwd(
    n: int,
    dtype: str,
    arch: str,
    backend: str,
    device_index: int,
    *,
    compile_args,
) -> flyc.CompiledFunction:
    del arch, backend, device_index
    input_2d, weight, grad_2d, rstd, grad_input, grad_weight, rows_m, stream = (
        compile_args
    )
    launch = build_rmsnorm_bwd_module(n, dtype)
    return flyc.compile(
        launch,
        _make_compile_arg(input_2d),
        flyc.from_torch_tensor(weight),
        _make_compile_arg(grad_2d),
        _make_compile_arg(rstd),
        _make_compile_arg(grad_input),
        flyc.from_torch_tensor(grad_weight),
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

    shape = _canonical_normalized_shape(normalized_shape)
    if len(shape) != 1:
        raise ValueError("FlyDSL RMSNorm currently requires one normalized dimension")

    n = shape[0]
    rows_m = input.numel() // n
    input_shape = input.shape
    output = torch.empty_like(input)

    with torch.cuda.device(input.device):
        input_2d = input.reshape(rows_m, n)
        output_2d = output.reshape(rows_m, n)
        rstd_flat = torch.empty(rows_m, device=input.device, dtype=torch.float32)
        stream = torch.cuda.current_stream(input.device)
        device_index = input.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()

        compiled = _compile_rmsnorm_fwd(
            n,
            _dtype_str(input.dtype),
            float(eps),
            str(get_rocm_arch()),
            flyc.compile_backend_name(),
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

    stat_shape = list(input_shape[:-1]) + [1]
    return output, rstd_flat.view(stat_shape)


def rmsnorm_bwd(
    grad_out: torch.Tensor,
    input: torch.Tensor,
    normalized_shape,
    rstd: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run FlyDSL backward and return grad_input and grad_weight."""

    shape = _canonical_normalized_shape(normalized_shape)
    if len(shape) != 1:
        raise ValueError("FlyDSL RMSNorm currently requires one normalized dimension")

    n = shape[0]
    rows_m = input.numel() // n

    with torch.cuda.device(input.device):
        input_2d = input.reshape(rows_m, n)
        grad_2d = grad_out.reshape(rows_m, n)
        rstd_flat = rstd.reshape(rows_m).contiguous()
        grad_input_2d = torch.empty_like(input_2d)

        # The kernel atomically accumulates dweight in fp32. flyc.compile may
        # execute the kernel once while tracing, so this buffer is deliberately
        # cleared after compilation and immediately before the measured launch.
        grad_weight_fp32 = torch.zeros(n, device=input.device, dtype=torch.float32)
        stream = torch.cuda.current_stream(input.device)
        device_index = input.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()

        compiled = _compile_rmsnorm_bwd(
            n,
            _dtype_str(input.dtype),
            str(get_rocm_arch()),
            flyc.compile_backend_name(),
            device_index,
            compile_args=(
                input_2d,
                weight,
                grad_2d,
                rstd_flat,
                grad_input_2d,
                grad_weight_fp32,
                rows_m,
                stream,
            ),
        )
        grad_weight_fp32.zero_()
        compiled(
            input_2d,
            weight,
            grad_2d,
            rstd_flat,
            grad_input_2d,
            grad_weight_fp32,
            rows_m,
            stream,
        )

    grad_input = grad_input_2d.reshape(input.shape)
    grad_weight = grad_weight_fp32.to(weight.dtype).reshape(weight.shape)
    return grad_input, grad_weight


def clear_rmsnorm_caches() -> None:
    """Clear both native-op-level compile caches (used by tests/benchmarks)."""

    _compile_rmsnorm_fwd.cache_clear()
    _compile_rmsnorm_bwd.cache_clear()


def rmsnorm_cache_info() -> dict[str, object]:
    """Return forward/backward cache statistics for diagnostics."""

    return {
        "fwd": _compile_rmsnorm_fwd.cache_info(),
        "bwd": _compile_rmsnorm_bwd.cache_info(),
    }
