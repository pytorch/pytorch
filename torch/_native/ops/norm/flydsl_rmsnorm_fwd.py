# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Vendored FlyDSL plain RMSNorm forward kernel and PyTorch wrapper."""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, math as fmath, range_constexpr, ReductionOp
from flydsl.runtime.device import is_rdna_arch

import torch
from torch._native.flydsl_utils import _resolve_rocm_arch
from torch._native.instrumentation import instrumented_flydsl_cache
from torch._native.ops.norm.flydsl_rmsnorm_impl import _normalized_shape_1d


_SUPPORTED_DTYPES: dict[torch.dtype, str] = {
    torch.float32: "f32",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
}
_COMPILE_BACKEND_NAME = flyc.compile_backend_name()
BLOCK_THREADS = 256
VEC_WIDTH = 8


def get_warp_size(arch: str) -> int:
    """Return wave64 for CDNA GPUs and wave32 for RDNA GPUs."""
    return 32 if is_rdna_arch(arch) else 64


def _make_single_reduction_storage(red_slots: int):
    """Shared storage for one block-reduction accumulator."""

    @fx.struct
    class SharedStorage:
        s_red: fx.Array[fx.Float32, red_slots, 16]

    return SharedStorage


def dtype_to_elem_type(dtype_str: str):
    """Map the three supported PyTorch dtype strings to FlyDSL types."""
    if dtype_str == "f32":
        return fx.Float32
    if dtype_str == "f16":
        return fx.Float16
    if dtype_str == "bf16":
        return fx.BFloat16
    raise ValueError(
        f"unsupported dtype: {dtype_str!r} (expected 'f32', 'f16', or 'bf16')"
    )


def _load_vec(copy_atom, vec_width, elem_dtype, div_tensor, idx):
    r = fx.make_rmem_tensor(vec_width, elem_dtype)
    fx.copy_atom_call(copy_atom, div_tensor[None, idx], r)
    return r.load()


def _store_vec(copy_atom, vec_width, elem_dtype, val, div_tensor, idx):
    r = fx.make_rmem_tensor(vec_width, elem_dtype)
    r.store(val)
    fx.copy_atom_call(copy_atom, r, div_tensor[None, idx])


def _to_elem(dtype_str: str, elem_dtype, y):
    if const_expr(dtype_str == "f32"):
        return y
    return y.to(elem_dtype)


def _dtype_str(dtype: torch.dtype) -> str:
    try:
        return _SUPPORTED_DTYPES[dtype]
    except KeyError as exc:
        raise TypeError(f"unsupported RMSNorm dtype for FlyDSL: {dtype}") from exc


def _forward_block_threads(n: int) -> int:
    if n >= 24576:
        return 1024
    if n >= 12288:
        return 512
    return BLOCK_THREADS


def build_rmsnorm_module(
    N: int,
    dtype_str: str,
    arch: str,
):
    # Baked into the kernel below, so it must come from the arch this module is
    # compiled for -- not from whichever device happened to be current at
    # import time. A wave64 reduction on a wave32 device is silently wrong.
    WARP_SIZE = get_warp_size(arch)

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
        Output: fx.Tensor,
        Rstd: fx.Tensor,
        Eps: fx.Float32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = dtype_to_elem_type(dtype_str)
        eps_c = Eps
        n_float = float(N)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_red = lds.s_red.view(fx.make_layout(RED_SLOTS, 1))

        def wave_reduce_add(x):
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << _sh_exp)
                w += gpu.shuffle_xor(w, off, WARP_SIZE)
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
                v = s_red[lane_safe]
                ww = in_range.select(v, 0.0)
                ww = wave_reduce_add(ww)
                if lane == 0:
                    fx.memref_store(ww, s_red, 0)
            gpu.barrier()
            return s_red[0]

        # ==================================================================
        # Fast path: N is a multiple of tile_cols
        # ==================================================================
        if const_expr(N >= tile_cols and N % tile_cols == 0):
            num_tiles = N // tile_cols
            # Layout API: buffer-backed tensors with tiled access.
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)

            row_in = Input_buf[bid, None]
            row_out = Output_buf[bid, None]

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
                red2 = x2.reduce(ReductionOp.ADD)
                thread_sumsq = thread_sumsq + red2

            sum_sq = block_reduce_add(thread_sumsq)
            mean_sq = sum_sq / n_float
            ms_eps = mean_sq + eps_c
            rrms = fmath.rsqrt(ms_eps)

            # The fused ATen contract returns this value for backward.
            if tid == 0:
                Rstd[bid] = rrms

            # Pass 2: normalize + gamma + store (reuse cached input)
            for tile_i in range_constexpr(num_tiles):
                idx = tid + tile_i * block_threads

                g = _load_vec(copy_atom, vec_width, elem_dtype, gamma_div, idx).to(
                    fx.Float32
                )
                x = in_local[tile_i].to(fx.Float32)

                y = (x * rrms) * g
                out_e = _to_elem(dtype_str, elem_dtype, y)

                out_idx = tid + tile_i * block_threads
                _store_vec(copy_atom, vec_width, elem_dtype, out_e, out_div, out_idx)

        else:
            # ==============================================================
            # Generic path: 128-bit vector body plus scalar tail.
            # ==============================================================
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)

            row_in = Input_buf[bid, None]
            row_out = Output_buf[bid, None]

            generic_vec_width = 4 if dtype_str == "f32" else VEC_WIDTH
            full_vecs = N // generic_vec_width
            vec_steps = (full_vecs + block_threads - 1) // block_threads
            scalar_tail_start = full_vecs * generic_vec_width
            scalar_tail_elems = N - scalar_tail_start

            copy_atom_v = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
            in_div = fx.logical_divide(row_in, fx.make_layout(generic_vec_width, 1))
            out_vec_div = fx.logical_divide(
                row_out, fx.make_layout(generic_vec_width, 1)
            )
            gamma_vec_div = fx.logical_divide(
                Gamma_buf, fx.make_layout(generic_vec_width, 1)
            )

            c_zero_f = fx.Float32(0.0)
            thread_sumsq = c_zero_f
            in_local = []
            tail_x = c_zero_f

            for step in range_constexpr(vec_steps):
                vec_idx = tid + step * block_threads
                is_valid = vec_idx < full_vecs
                vec_idx_safe = is_valid.select(vec_idx, 0)
                vec = _load_vec(
                    copy_atom_v, generic_vec_width, elem_dtype, in_div, vec_idx_safe
                )
                in_local.append(vec)
                x = vec.to(fx.Float32)
                x2 = x * x
                red2 = x2.reduce(ReductionOp.ADD)
                red2_safe = is_valid.select(red2, c_zero_f)
                thread_sumsq = thread_sumsq + red2_safe

            if const_expr(scalar_tail_elems > 0):
                tail_valid = tid < scalar_tail_elems
                tail_idx = scalar_tail_start + tid
                tail_x_e = row_in[tail_valid.select(tail_idx, 0)]
                tail_x = tail_x_e if dtype_str == "f32" else tail_x_e.to(fx.Float32)
                thread_sumsq = thread_sumsq + tail_valid.select(
                    tail_x * tail_x, c_zero_f
                )

            sum_sq = block_reduce_add(thread_sumsq)
            mean_sq = sum_sq / n_float
            ms_eps = mean_sq + eps_c
            rrms = fmath.rsqrt(ms_eps)

            # The fused ATen contract returns this value for backward.
            if tid == 0:
                Rstd[bid] = rrms

            for step in range_constexpr(vec_steps):
                vec_idx = tid + step * block_threads
                if vec_idx < full_vecs:
                    g = _load_vec(
                        copy_atom_v,
                        generic_vec_width,
                        elem_dtype,
                        gamma_vec_div,
                        vec_idx,
                    ).to(fx.Float32)
                    x = in_local[step].to(fx.Float32)
                    y = (x * rrms) * g
                    out_e = _to_elem(dtype_str, elem_dtype, y)
                    _store_vec(
                        copy_atom_v,
                        generic_vec_width,
                        elem_dtype,
                        out_e,
                        out_vec_div,
                        vec_idx,
                    )

            if const_expr(scalar_tail_elems > 0):
                tail_valid = tid < scalar_tail_elems
                tail_idx = scalar_tail_start + tid
                if tail_valid:
                    g_e = Gamma_buf[tail_idx]
                    g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                    y = (tail_x * rrms) * g
                    y_e = _to_elem(dtype_str, elem_dtype, y)
                    row_out[tail_idx] = elem_dtype(y_e)

    @flyc.jit
    def launch_rmsnorm(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Output: fx.Tensor,
        Rstd: fx.Tensor,
        m_in: fx.Int32,
        eps: fx.Float32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = rmsnorm_kernel(Input, Gamma, Output, Rstd, eps)
        launcher.launch(
            grid=(m_in, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_rmsnorm


def _make_compile_arg(tensor: torch.Tensor):
    """Make only the row dimension dynamic so one kernel supports many M values."""
    return flyc.from_torch_tensor(tensor).mark_shape_dynamic(0)


@instrumented_flydsl_cache(
    "aten::_fused_rms_norm",
    key_fn=lambda n, dtype, arch, *a, **k: f"fwd N={n} {dtype} {arch}",
)
def _compile_rmsnorm_fwd(
    n: int,
    dtype: str,
    arch: str,
    backend: str,
    device_index: int,
    *,
    compile_args,
) -> flyc.CompiledFunction:
    # backend/device_index are cache keys only. flyc.compile binds the resulting
    # launcher to the active device/context, so cross-device reuse is unsafe
    # even when two GPUs share the same architecture.
    del backend, device_index
    input_2d, weight, output_2d, rstd, rows_m, eps, stream = compile_args
    launch = build_rmsnorm_module(n, dtype, arch)
    return flyc.compile[{"fast_fp_math": True}](
        launch,
        _make_compile_arg(input_2d),
        flyc.from_torch_tensor(weight),
        _make_compile_arg(output_2d),
        _make_compile_arg(rstd),
        rows_m,
        eps,
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
            _resolve_rocm_arch(device_index),
            _COMPILE_BACKEND_NAME,
            device_index,
            compile_args=(
                input_2d,
                weight,
                output_2d,
                rstd_flat,
                rows_m,
                float(eps),
                stream,
            ),
        )

        compiled(input_2d, weight, output_2d, rstd_flat, rows_m, float(eps), stream)

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
