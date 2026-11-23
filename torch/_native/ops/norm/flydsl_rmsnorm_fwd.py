"""FlyDSL plain RMSNorm forward kernel and its PyTorch wrapper.

The kernel builder below is specialized per (N, dtype, arch) and compiled on
first dispatch; flydsl_rmsnorm_impl.py owns the dispatcher predicate that
decides when it is worth using over ATen.
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, math as fmath, range_constexpr
from flydsl.runtime.device import is_rdna_arch

import torch
from torch._native.flydsl.compile_args import make_compile_arg, read_only_tensor
from torch._native.flydsl_utils import _resolve_rocm_arch
from torch._native.instrumentation import instrumented_flydsl_cache
from torch._native.ops.norm.flydsl_rmsnorm_utils import (
    normalized_shape_1d,
    SUPPORTED_DTYPES,
)


FLYDSL_DTYPE_CONFIGS = {
    "f32": (fx.Float32, 32),
    "f16": (fx.Float16, 16),
    "bf16": (fx.BFloat16, 16),
}


def _get_warp_size(arch: str) -> int:
    """Return wave64 for CDNA GPUs and wave32 for RDNA GPUs."""
    return 32 if is_rdna_arch(arch) else 64


def _make_single_reduction_storage(reduction_slots: int):
    """Shared storage for one block-reduction accumulator."""

    @fx.struct
    class SharedStorage:
        reduction_buffer: fx.Array[fx.Float32, reduction_slots, 16]

    return SharedStorage


def _dtype_config(dtype_str: str):
    try:
        return FLYDSL_DTYPE_CONFIGS[dtype_str]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {dtype_str!r}") from exc


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


def _to_f32(dtype_str: str, v):
    if const_expr(dtype_str == "f32"):
        return v
    return v.to(fx.Float32)


def _dtype_str(dtype: torch.dtype) -> str:
    try:
        return SUPPORTED_DTYPES[dtype]
    except KeyError as exc:
        raise TypeError(f"unsupported RMSNorm dtype for FlyDSL: {dtype}") from exc


def _forward_block_threads(n: int) -> int:
    if n >= 24576:
        return 1024
    if n >= 12288:
        return 512
    return 256


def _build_rmsnorm_module(
    N: int,
    dtype_str: str,
    arch: str,
):
    WARP_SIZE = _get_warp_size(arch)

    block_threads = _forward_block_threads(N)
    _, elem_bits = _dtype_config(dtype_str)
    vec_width = 128 // elem_bits
    reduction_slots = (block_threads + WARP_SIZE - 1) // WARP_SIZE
    # block_reduce_add's second stage reads slots through one masked wave.
    if reduction_slots > WARP_SIZE:
        raise AssertionError(
            f"block_reduce_add cannot combine {reduction_slots} partial sums "
            f"through one {WARP_SIZE}-lane wave"
        )

    SharedStorage = _make_single_reduction_storage(reduction_slots)

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

        elem_dtype = _dtype_config(dtype_str)[0]
        eps_c = Eps
        n_float = float(N)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        reduction_buffer = lds.reduction_buffer.view(fx.make_layout(reduction_slots, 1))

        def wave_reduce_add(x):
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << _sh_exp)
                w += gpu.shuffle_xor(w, off, WARP_SIZE)
            return w

        def block_reduce_add(val):
            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE
            w = wave_reduce_add(val)
            if lane == 0:
                fx.memref_store(w, reduction_buffer, wave)
            gpu.barrier()
            if wave == 0:
                in_range = lane < reduction_slots
                lane_safe = in_range.select(lane, 0)
                v = reduction_buffer[lane_safe]
                ww = in_range.select(v, 0.0)
                ww = wave_reduce_add(ww)
                if lane == 0:
                    fx.memref_store(ww, reduction_buffer, 0)
            gpu.barrier()
            return reduction_buffer[0]

        Input_buf = fx.rocdl.make_buffer_tensor(Input)
        Output_buf = fx.rocdl.make_buffer_tensor(Output)
        Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)

        row_in = Input_buf[bid, None]
        row_out = Output_buf[bid, None]

        full_vecs = N // vec_width
        vec_steps = (full_vecs + block_threads - 1) // block_threads
        scalar_tail_start = full_vecs * vec_width
        scalar_tail_elems = N - scalar_tail_start

        copy_atom_v = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
        in_div = fx.logical_divide(row_in, fx.make_layout(vec_width, 1))
        out_vec_div = fx.logical_divide(row_out, fx.make_layout(vec_width, 1))
        gamma_vec_div = fx.logical_divide(Gamma_buf, fx.make_layout(vec_width, 1))

        c_zero_f = fx.Float32(0.0)
        thread_sumsq = c_zero_f
        in_local = []
        tail_x = c_zero_f

        for step in range_constexpr(vec_steps):
            vec_idx = tid + step * block_threads
            is_valid = vec_idx < full_vecs
            vec_idx_safe = is_valid.select(vec_idx, 0)
            vec = _load_vec(copy_atom_v, vec_width, elem_dtype, in_div, vec_idx_safe)
            in_local.append(vec)
            x = vec.to(fx.Float32)
            # Keep rstd aligned with ATen's scalar lane accumulation for backward.
            for elem_i in range_constexpr(vec_width):
                x_elem = x[elem_i]
                x_sumsq = is_valid.select(x_elem * x_elem, c_zero_f)
                thread_sumsq = thread_sumsq + x_sumsq

        if const_expr(scalar_tail_elems > 0):
            tail_valid = tid < scalar_tail_elems
            tail_idx = scalar_tail_start + tid
            tail_x_e = row_in[tail_valid.select(tail_idx, 0)]
            tail_x = _to_f32(dtype_str, tail_x_e)
            thread_sumsq = thread_sumsq + tail_valid.select(tail_x * tail_x, c_zero_f)

        sum_sq = block_reduce_add(thread_sumsq)
        mean_sq = sum_sq / n_float
        ms_eps = mean_sq + eps_c
        rrms = fmath.rsqrt(ms_eps, fastmath="fast")

        # The fused ATen contract returns this value for backward.
        if tid == 0:
            Rstd[bid] = rrms

        for step in range_constexpr(vec_steps):
            vec_idx = tid + step * block_threads
            if vec_idx < full_vecs:
                g = _load_vec(
                    copy_atom_v,
                    vec_width,
                    elem_dtype,
                    gamma_vec_div,
                    vec_idx,
                ).to(fx.Float32)
                x = in_local[step].to(fx.Float32)
                y = (x * rrms) * g
                out_e = _to_elem(dtype_str, elem_dtype, y)
                _store_vec(
                    copy_atom_v,
                    vec_width,
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
                g = _to_f32(dtype_str, g_e)
                y = (tail_x * rrms) * g
                row_out[tail_idx] = _to_elem(dtype_str, elem_dtype, y)

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


@instrumented_flydsl_cache(
    "aten::_fused_rms_norm",
    key_fn=lambda n, dtype, arch, backend, device_index, *a, **k: (
        f"fwd N={n} {dtype} {arch} backend={backend} device={device_index}"
    ),
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
    # These are explicit cache keys: FlyDSL reads the backend from its environment
    # and the device from the active HIP context. It reuses compiled artifacts
    # across devices, but the returned callable contains context-local
    # module/function handles and must be cached per device.
    del backend, device_index
    input_2d, weight, output_2d, rstd, rows_m, eps, stream = compile_args
    launch = _build_rmsnorm_module(n, dtype, arch)
    return flyc.compile(
        launch,
        make_compile_arg(input_2d, read_only=True),
        flyc.from_torch_tensor(read_only_tensor(weight)),
        make_compile_arg(output_2d),
        make_compile_arg(rstd),
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
    n = normalized_shape_1d(normalized_shape)
    if n is None:
        raise ValueError("FlyDSL RMSNorm currently requires one normalized dimension")

    device_index = input.device.index
    arch: str = _resolve_rocm_arch(device_index)  # pyrefly: ignore[bad-assignment]

    rows_m = input.numel() // n
    input_shape = input.shape

    with torch.cuda.device(input.device):
        is_2d = input.ndim == 2
        input_2d = input if is_2d else input.reshape(rows_m, n)
        output_2d = torch.empty_like(input_2d)
        rstd_flat = torch.empty(rows_m, device=input.device, dtype=torch.float32)

        stream = torch.cuda.current_stream()

        compiled = _compile_rmsnorm_fwd(
            n,
            _dtype_str(input.dtype),
            arch,
            flyc.compile_backend_name(),
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

        compiled(
            read_only_tensor(input_2d),
            read_only_tensor(weight),
            output_2d,
            rstd_flat,
            rows_m,
            float(eps),
            stream,
        )

    if is_2d:
        result = output_2d, rstd_flat.view((rows_m, 1))
    else:
        stat_shape = (*input_shape[:-1], 1)
        result = output_2d.view(input_shape), rstd_flat.view(stat_shape)
    return result


def clear_rmsnorm_caches() -> None:
    """Clear native-op-level compile caches (used by tests/benchmarks)."""

    # flydsl_jit_cache attaches cache_clear/cache_info at runtime and
    # instrument_flydsl_compile forwards them, so neither is on the declared
    # Callable type.
    _compile_rmsnorm_fwd.cache_clear()  # pyrefly: ignore[missing-attribute]


def rmsnorm_cache_info() -> dict[str, object]:
    """Return forward cache statistics for diagnostics."""

    return {
        "fwd": _compile_rmsnorm_fwd.cache_info(),  # pyrefly: ignore[missing-attribute]
    }
