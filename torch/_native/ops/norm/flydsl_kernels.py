"""FlyDSL RMSNorm kernels used by the native RMSNorm override."""

# mypy: allow-untyped-defs

from __future__ import annotations

import math

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.vector import ReductionOp, full
from flydsl.runtime.device import get_rocm_arch, is_rdna_arch

from torch._native.flydsl_cache import jit_cache


EPS = 1e-5
BLOCK_THREADS = 256
VEC_WIDTH = 8

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


def _dtype_to_elem_type(dtype_str: str):
    if dtype_str == "f32":
        return fx.Float32
    if dtype_str == "f16":
        return fx.Float16
    if dtype_str == "bf16":
        return fx.BFloat16
    raise ValueError(f"unsupported dtype: {dtype_str!r} (expected 'f32', 'f16', or 'bf16')")


def _get_warp_size(arch=None) -> int:
    if arch is None:
        arch = get_rocm_arch()
    return 32 if is_rdna_arch(arch) else 64


def _make_reduction_storage(red_slots: int):
    @fx.struct
    class SharedStorage:
        s_red: fx.Array[fx.Float32, red_slots, 16]
        s_red2: fx.Array[fx.Float32, red_slots, 16]

    return SharedStorage


def _load_scalar(copy_atom, elem_dtype, divided_tensor, index):
    view = fx.slice(divided_tensor, (None, index))
    r = fx.make_rmem_tensor(1, elem_dtype)
    fx.copy_atom_call(copy_atom, view, r)
    return fx.memref_load_vec(r)[0]


def _store_scalar(copy_atom, elem_dtype, store_dtype, divided_tensor, index, val):
    r = fx.make_rmem_tensor(1, elem_dtype)
    ts = full(1, store_dtype(val), store_dtype)
    fx.memref_store_vec(ts, r)
    view = fx.slice(divided_tensor, (None, index))
    fx.copy_atom_call(copy_atom, r, view)


def _load_vec(copy_atom, vec_width, elem_dtype, div_tensor, idx):
    r = fx.make_rmem_tensor(vec_width, elem_dtype)
    fx.copy_atom_call(copy_atom, fx.slice(div_tensor, (None, idx)), r)
    return fx.memref_load_vec(r)


def _store_vec(copy_atom, vec_width, elem_dtype, val, div_tensor, idx):
    r = fx.make_rmem_tensor(vec_width, elem_dtype)
    fx.memref_store_vec(val, r)
    fx.copy_atom_call(copy_atom, r, fx.slice(div_tensor, (None, idx)))


def _to_elem_scalar(dtype_str: str, elem_dtype, y):
    if const_expr(dtype_str == "f32"):
        return y
    return y.to(elem_dtype)


def _to_elem_vec(dtype_str: str, elem_dtype, use_hw_cvt_bf16: bool, y):
    if const_expr(dtype_str == "bf16"):
        if const_expr(use_hw_cvt_bf16):
            return y.to(elem_dtype)
        u = y.bitcast(fx.Uint32)
        upper = u >> 16
        lsb = upper & 1
        bias = lsb + 0x7FFF
        u_round = y.bitcast(fx.Uint32) + bias
        bf16_bits = u_round >> 16
        even = bf16_bits.shuffle(bf16_bits, [0, 2, 4, 6])
        odd = bf16_bits.shuffle(bf16_bits, [1, 3, 5, 7])
        odd_sh = odd << 16
        packed = even | odd_sh
        return packed.bitcast(elem_dtype)
    if const_expr(dtype_str == "f32"):
        return y
    return y.to(elem_dtype)


def _build_rmsnorm_module(n: int, dtype_str: str):
    if n <= 2048:
        return _build_rmsnorm_large_m_small_n_module(n, dtype_str)

    arch = get_rocm_arch()
    use_hw_cvt_pk_bf16_f32 = arch == "gfx950" or str(arch).startswith("gfx95")
    warp_size = _get_warp_size(arch)

    tile_cols = BLOCK_THREADS * VEC_WIDTH
    red_slots = max(1, (BLOCK_THREADS + warp_size - 1) // warp_size)
    elem_bits = 32 if dtype_str == "f32" else 16
    shared_storage = _make_reduction_storage(red_slots)

    @flyc.kernel
    def rmsnorm_kernel(
        input: fx.Tensor,
        gamma: fx.Tensor,
        _unused: fx.Tensor,
        output: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = _dtype_to_elem_type(dtype_str)
        fm_fast = arith.FastMathFlags.fast
        n_float = float(n)

        lds = fx.SharedAllocator().allocate(shared_storage).peek()
        s_red = lds.s_red.view(fx.make_layout(red_slots, 1))
        s_red2 = lds.s_red2.view(fx.make_layout(red_slots, 1))

        def wave_reduce_add(x):
            w = x
            for sh_exp in range_constexpr(int(math.log2(warp_size))):
                off = warp_size // (2 << sh_exp)
                peer = w.shuffle_xor(off, warp_size)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        def block_reduce_add(val):
            dummy = fx.Float32(0.0)
            r0, _ = block_reduce_add2(val, dummy)
            return r0

        def block_reduce_add2(val0, val1):
            if const_expr(red_slots == 1):
                return wave_reduce_add(val0), wave_reduce_add(val1)

            lane = tid % warp_size
            wave = tid // warp_size
            w0 = wave_reduce_add(val0)
            w1 = wave_reduce_add(val1)

            if lane == 0:
                fx.memref_store(w0, s_red, wave)
                fx.memref_store(w1, s_red2, wave)
            gpu.barrier()

            if wave == 0:
                in_range = lane < red_slots
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

        if const_expr(n >= tile_cols and n % tile_cols == 0 and elem_bits <= 16):
            num_tiles = n // tile_cols
            input_buf = fx.rocdl.make_buffer_tensor(input)
            output_buf = fx.rocdl.make_buffer_tensor(output)
            gamma_buf = fx.rocdl.make_buffer_tensor(gamma)

            row_in = fx.slice(input_buf, (bid, None))
            row_out = fx.slice(output_buf, (bid, None))
            in_div = fx.logical_divide(row_in, fx.make_layout(VEC_WIDTH, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(VEC_WIDTH, 1))
            gamma_div = fx.logical_divide(gamma_buf, fx.make_layout(VEC_WIDTH, 1))
            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)

            thread_sumsq = fx.Float32(0.0)
            thread_dummy = fx.Float32(0.0)
            in_local = []
            for tile_i in range_constexpr(num_tiles):
                idx = tid + tile_i * BLOCK_THREADS
                vec = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, in_div, idx)
                in_local.append(vec)
                x = vec.to(fx.Float32)
                thread_sumsq = thread_sumsq + (x * x).reduce(ReductionOp.ADD, fastmath=fm_fast)

            _, sum_sq = block_reduce_add2(thread_dummy, thread_sumsq)
            rrms = fmath.rsqrt((sum_sq / n_float) + EPS, fastmath=fm_fast)

            for tile_i in range_constexpr(num_tiles):
                idx = tid + tile_i * BLOCK_THREADS
                g = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, gamma_div, idx).to(fx.Float32)
                x = in_local[tile_i].to(fx.Float32)
                y = (x * rrms) * g
                out_e = _to_elem_vec(dtype_str, elem_dtype, use_hw_cvt_pk_bf16_f32, y)
                _store_vec(copy_atom, VEC_WIDTH, elem_dtype, out_e, out_div, idx)
        else:
            input_buf = fx.rocdl.make_buffer_tensor(input)
            output_buf = fx.rocdl.make_buffer_tensor(output)
            gamma_buf = fx.rocdl.make_buffer_tensor(gamma)

            row_in = fx.slice(input_buf, (bid, None))
            row_out = fx.slice(output_buf, (bid, None))
            copy_atom_s = fx.make_copy_atom(
                fx.rocdl.BufferCopy16b() if elem_bits <= 16 else fx.rocdl.BufferCopy32b(),
                elem_bits,
            )
            row_div = fx.logical_divide(row_in, fx.make_layout(1, 1))
            gamma_div = fx.logical_divide(gamma_buf, fx.make_layout(1, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(1, 1))

            thread_sumsq = fx.Float32(0.0)
            for base_idx_int in range_constexpr(0, n, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < n
                idx_safe = is_valid.select(idx, 0)
                x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, idx_safe)
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                thread_sumsq = thread_sumsq + is_valid.select(x * x, fx.Float32(0.0))

            sum_sq = block_reduce_add(thread_sumsq)
            rrms = fmath.rsqrt((sum_sq / n_float) + EPS, fastmath=fm_fast)

            for base_idx_int in range_constexpr(0, n, BLOCK_THREADS):
                idx = tid + base_idx_int
                if idx < n:
                    x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, idx)
                    g_e = _load_scalar(copy_atom_s, elem_dtype, gamma_div, idx)
                    x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                    g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                    y_e = _to_elem_scalar(dtype_str, elem_dtype, (x * rrms) * g)
                    _store_scalar(copy_atom_s, elem_dtype, elem_dtype, out_div, idx, y_e)

    @flyc.jit
    def launch_rmsnorm(
        input: fx.Tensor,
        gamma: fx.Tensor,
        output: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = rmsnorm_kernel(input, gamma, gamma, output)
        launcher.launch(
            grid=(m_in, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_rmsnorm


def _build_rmsnorm_large_m_small_n_module(n: int, dtype_str: str):
    warp_size = _get_warp_size()
    block_n = 1 << (n - 1).bit_length()
    block_m = max(min(16384 // block_n, 32), 8)
    threads_per_row = min(warp_size, 1024 // block_m)
    block_threads = block_m * threads_per_row
    elem_bits = 32 if dtype_str == "f32" else 16

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def rmsnorm_large_m_small_n_kernel(
        input: fx.Tensor,
        gamma: fx.Tensor,
        _unused: fx.Tensor,
        output: fx.Tensor,
        m_in: fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        lane = tid % threads_per_row
        row_local = tid // threads_per_row
        row = bid * fx.Int32(block_m) + row_local

        if row < m_in:
            elem_dtype = _dtype_to_elem_type(dtype_str)
            fm_fast = arith.FastMathFlags.fast
            n_float = float(n)

            input_buf = fx.rocdl.make_buffer_tensor(input)
            gamma_buf = fx.rocdl.make_buffer_tensor(gamma)
            output_buf = fx.rocdl.make_buffer_tensor(output)
            row_in = fx.slice(input_buf, (row, None))
            row_out = fx.slice(output_buf, (row, None))
            copy_atom_s = fx.make_copy_atom(
                fx.rocdl.BufferCopy16b() if elem_bits <= 16 else fx.rocdl.BufferCopy32b(),
                elem_bits,
            )
            row_div = fx.logical_divide(row_in, fx.make_layout(1, 1))
            gamma_div = fx.logical_divide(gamma_buf, fx.make_layout(1, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(1, 1))

            def group_reduce_add(x):
                w = x
                for sh_exp in range_constexpr(int(math.log2(threads_per_row))):
                    off = threads_per_row // (2 << sh_exp)
                    peer = w.shuffle_xor(off, fx.Int32(threads_per_row))
                    w = w.addf(peer, fastmath=fm_fast)
                return w

            thread_sumsq = fx.Float32(0.0)
            for base_idx_int in range_constexpr(0, block_n, threads_per_row):
                idx = lane + base_idx_int
                is_valid = idx < n
                idx_safe = is_valid.select(idx, 0)
                x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, idx_safe)
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                thread_sumsq = thread_sumsq + is_valid.select(x * x, fx.Float32(0.0))

            sum_sq = group_reduce_add(thread_sumsq)
            rrms = fmath.rsqrt((sum_sq / n_float) + EPS, fastmath=fm_fast)

            for base_idx_int in range_constexpr(0, block_n, threads_per_row):
                idx = lane + base_idx_int
                if idx < n:
                    x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, idx)
                    g_e = _load_scalar(copy_atom_s, elem_dtype, gamma_div, idx)
                    x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                    g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                    y_e = _to_elem_scalar(dtype_str, elem_dtype, (x * rrms) * g)
                    _store_scalar(copy_atom_s, elem_dtype, elem_dtype, out_div, idx, y_e)

    @flyc.jit
    def launch_rmsnorm_large_m_small_n(
        input: fx.Tensor,
        gamma: fx.Tensor,
        output: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = rmsnorm_large_m_small_n_kernel(input, gamma, gamma, output, m_in)
        launcher.launch(
            grid=((m_in + fx.Int32(block_m - 1)) // fx.Int32(block_m), 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_rmsnorm_large_m_small_n


def _make_compile_arg(tensor: torch.Tensor):
    return flyc.from_torch_tensor(tensor).mark_shape_dynamic(0)


@jit_cache
def _compile_rmsnorm(
    n: int,
    dtype: str,
    arch: str,
    backend: str,
    *,
    compile_args,
) -> flyc.CompiledFunction:
    input_2d, weight, output_2d, rows_m, stream = compile_args
    launch = _build_rmsnorm_module(n, dtype)
    return flyc.compile(
        launch,
        _make_compile_arg(input_2d),
        flyc.from_torch_tensor(weight),
        _make_compile_arg(output_2d),
        rows_m,
        stream,
    )


def rmsnorm(
    input: torch.Tensor,
    normalized_shape,
    weight: torch.Tensor | None = None,
    eps: float | None = None,
) -> torch.Tensor:
    shape = _canonical_normalized_shape(normalized_shape)
    n = shape[0]
    rows_m = input.numel() // n
    output = torch.empty_like(input)

    with torch.cuda.device(input.device):
        input_2d = input.reshape(rows_m, n)
        output_2d = output.reshape(rows_m, n)
        stream = torch.cuda.current_stream(input.device)
        compiled = _compile_rmsnorm(
            n,
            _dtype_str(input.dtype),
            str(get_rocm_arch()),
            flyc.compile_backend_name(),
            compile_args=(input_2d, weight, output_2d, rows_m, stream),
        )
        compiled(input_2d, weight, output_2d, rows_m, stream)

    return output


def clear_rmsnorm_cache() -> None:
    _compile_rmsnorm.cache_clear()


def rmsnorm_cache_info():
    return _compile_rmsnorm.cache_info()
