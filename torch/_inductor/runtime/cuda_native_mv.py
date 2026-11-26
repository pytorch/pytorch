# mypy: allow-untyped-defs
import re

import triton
import triton.language as tl

import torch

from .triton_helpers import (
    cuda_native_complex_mv_value,
    cuda_native_mv_mma_value,
    cuda_native_mv_value,
)


@triton.jit
def gemv(
    X,
    V,
    P,
    O,
    M,
    K,
    S0,
    S1,
    VS,
    CTAS,
    THREADS: tl.constexpr,
    ACTIVE_THREADS,
    INNER_CHUNK,
    INNER_STRIDE,
    KIND: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    FINAL: tl.constexpr,
    COMPLEX: tl.constexpr,
    SWAP,
    PRODUCT,
    PRE_SCALE,
    K_SLICE=0,
):
    r"""Compute a GEMV partial or final output using native reduction metadata."""
    row = tl.program_id(0).to(tl.int64) + tl.arange(0, 1)
    cta = tl.program_id(1)
    if COMPLEX:
        real, imag = cuda_native_complex_mv_value(
            X,
            V,
            P,
            row,
            cta,
            K,
            S0,
            S1,
            VS,
            CTAS,
            THREADS,
            ACTIVE_THREADS,
            INNER_CHUNK,
            INNER_STRIDE,
            KIND,
            ACC_TYPE,
            FINAL,
            CTAS,
            SWAP,
            PRODUCT,
            PRE_SCALE,
        )
        if FINAL:
            tl.store(O + 2 * row, real)
            tl.store(O + 2 * row + 1, imag)
        else:
            tl.store(P + 2 * (row * CTAS + cta), real)
            tl.store(P + 2 * (row * CTAS + cta) + 1, imag)
    else:
        value = cuda_native_mv_value(
            X,
            V,
            P,
            row,
            cta,
            K,
            S0,
            S1,
            VS,
            CTAS,
            THREADS,
            ACTIVE_THREADS,
            INNER_CHUNK,
            INNER_STRIDE,
            KIND,
            ACC_TYPE,
            FINAL,
            CTAS,
        )
        if FINAL:
            tl.store(O + row, value)
        else:
            tl.store(P + row * CTAS + cta, value)


@triton.jit
def gemv_mma(
    X,
    V,
    P,
    M,
    K,
    S0,
    S1,
    VS,
    CTAS,
    K_SLICE,
):
    row = tl.program_id(0) * 16 + tl.arange(0, 16)
    cta = tl.program_id(1)
    value = cuda_native_mv_mma_value(X, V, cta, M, K, S0, S1, VS, K_SLICE)
    tl.store(P + row.to(tl.int64) * CTAS + cta, value, row < M)


def cuda_native_mv(x, v, *, out):
    r"""Launch generated GEMV kernels using the actual runtime size and alignment."""
    m, k = x.shape
    s0, s1 = x.stride()
    vs = v.stride(0)
    if m == 0:
        return out
    if k == 0:
        parts = torch.view_as_real(out) if out.is_complex() else out
        zero[(triton.cdiv(parts.numel(), 256),)](parts, parts.numel(), num_warps=4)
        return out
    native_matrix = (s0 == 1 and s1 >= max(1, m)) or (s1 == 1 and s0 >= max(1, k))
    x = _materialize(x, contiguous=not native_matrix)
    v = _materialize(v, contiguous=vs == 0)
    s0, s1 = x.stride()
    vs = v.stride(0)
    transposed = not (s0 == 1 and s1 >= max(1, m))
    lda = s0 if transposed else s1
    with torch.cuda.device(x.device):
        plan = torch._C._cuda_getGemvLaunchPlan(
            x.dtype,
            k if transposed else m,
            m if transposed else k,
            lda,
            vs,
            transposed,
            x.data_ptr() & 15,
            v.data_ptr() & 15,
        )
    options = _native_options(plan, x.dtype, m, k, s0, s1, vs)
    acc = (
        torch.float64 if x.dtype in (torch.float64, torch.complex128) else torch.float32
    )
    options["ACC_TYPE"] = tl.float64 if acc == torch.float64 else tl.float32
    complex_dtype = x.is_complex()
    p = torch.empty(
        (m, options["CTAS"], 2) if complex_dtype else (m, options["CTAS"]),
        device=x.device,
        dtype=acc,
    )
    if complex_dtype:
        x = torch.view_as_real(x)
        v = torch.view_as_real(v)
        o = torch.view_as_real(out)
    else:
        o = out
    if options["KIND"] in (4, 5):
        gemv_mma[(triton.cdiv(m, 16), options["CTAS"])](
            x,
            v,
            out if options["KIND"] == 4 else p,
            **{k: options[k] for k in ("M", "K", "S0", "S1", "VS", "CTAS", "K_SLICE")},
            num_warps=4,
            enable_fp_fusion=False,
        )
        if options["KIND"] == 4:
            return out
        options["KIND"] = 3
    else:
        gemv[(m, options["CTAS"])](
            x,
            v,
            p,
            o,
            **options,
            FINAL=False,
            COMPLEX=complex_dtype,
            num_warps=4,
            enable_fp_fusion=False,
            enable_precise_libdevice=True,
        )
    gemv[(m,)](
        x,
        v,
        p,
        o,
        **options,
        FINAL=True,
        COMPLEX=complex_dtype,
        num_warps=4,
        enable_fp_fusion=False,
        enable_precise_libdevice=True,
    )
    return out


@triton.jit
def zero(OUT, N):
    index = tl.program_id(0) * 256 + tl.arange(0, 256)
    tl.store(OUT + index, 0, index < N)


@triton.jit
def _copy_kernel(
    SOURCE,
    DEST,
    N,
    K,
    S0,
    S1,
    D0,
    D1,
    COMPONENTS: tl.constexpr,
    NEGATIVE,
    CONJUGATE,
):
    index = tl.program_id(0).to(tl.int64) * 256 + tl.arange(0, 256)
    element = index // COMPONENTS
    component = index % COMPONENTS
    row = element // K
    column = element % K
    source = COMPONENTS * (row * S0 + column * S1) + component
    dest = COMPONENTS * (row * D0 + column * D1) + component
    value = tl.load(SOURCE + source, index < N, 0)
    if NEGATIVE:
        value = -value
    if CONJUGATE:
        value = tl.where(component == 1, -value, value)
    tl.store(DEST + dest, value, index < N)


def _materialize(value, *, contiguous):
    negative, conjugate = value.is_neg(), value.is_conj()
    if not (contiguous or negative or conjugate):
        return value
    result = (
        torch.empty(value.shape, dtype=value.dtype, device=value.device)
        if contiguous
        else torch.empty_like(value, memory_format=torch.preserve_format)
    )
    physical = torch._neg_view(value) if negative else value
    physical = physical.conj() if conjugate else physical
    components = 2 if value.is_complex() else 1
    source = torch.view_as_real(physical) if value.is_complex() else physical
    dest = torch.view_as_real(result) if value.is_complex() else result
    count = value.numel() * components
    _copy_kernel[(triton.cdiv(count, 256),)](
        source,
        dest,
        count,
        value.shape[-1],
        value.stride(0) if value.ndim == 2 else 0,
        value.stride(-1),
        result.stride(0) if result.ndim == 2 else 0,
        result.stride(-1),
        components,
        negative,
        conjugate,
        num_warps=4,
        enable_precise_libdevice=True,
    )
    return result


def _native_options(plan, dtype, m, k, s0, s1, vs):
    r"""Translate the native GEMV launch plan into Triton reduction metadata."""
    name, grid, block, _, partition = next(p for p in plan if "reduce" not in p[0])
    ctas, threads, kind = 1, 1, 0
    tile_k = 1
    if "dot_kernel" in name:
        kind, ctas, threads = 2, grid[0], block[0]
    elif name.startswith("nvjet_"):
        kind = 4
        tiles = re.search(r"nvjet_[^_]+_(\d+)x(\d+)_(\d+)x(\d+)_", name)
        if tiles is None:
            raise NotImplementedError(f"Strict CUDA mv tensor-core geometry for {name}")
        tile_m = max(int(tiles[1]), int(tiles[2]))
        tile_k = int(tiles[3])
        if any("splitKreduce" in p[0] for p in plan):
            kind = 5
            ctas = grid[0] * grid[1] * grid[2] // ((m + tile_m - 1) // tile_m)
    elif any("splitKreduce" in p[0] for p in plan):
        kind, ctas, threads = 3, grid[2], block[1]
    elif "gemvx" in name:
        flags = [int(v) for v in re.findall(r"Lb([01])E", name)]
        threads = block[0] if flags[2] else block[1]

    elif "gemv2N_kernel" in name:
        threads = 1
    elif "gemvNSP_kernel" in name:
        threads = block[1]
        partition = (4, 1, 0)
    elif k == 1:
        threads = 1
    else:
        raise NotImplementedError(f"Strict CUDA mv reduction order for {name}")
    acc_dtype = (
        torch.float64 if dtype in (torch.float64, torch.complex128) else torch.float32
    )
    options = dict(
        M=m,
        K=k,
        S0=s0,
        S1=s1,
        VS=vs,
        CTAS=ctas,
        THREADS=1 << (threads - 1).bit_length(),
        ACTIVE_THREADS=threads,
        INNER_CHUNK=partition[0],
        INNER_STRIDE=partition[1],
        KIND=kind,
        ACC_TYPE="tl.float64" if acc_dtype == torch.float64 else "tl.float32",
    )
    if kind in (4, 5):
        options.update(K_SLICE=((k + ctas - 1) // ctas + tile_k - 1) // tile_k * tile_k)
    options.update(
        SWAP=not bool(partition[2]) if kind == 2 else bool(partition[2]),
        PRODUCT="gemmk1_kernel" in name,
        PRE_SCALE="gemv2N_kernel" in name,
    )
    return options
