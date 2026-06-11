# Copyright (c) 2026, Tri Dao.
"""PyTorch-friendly interface for the SM100 MXFP8 blockscaled GEMM.

Shape / layout conventions (matches torch.matmul, torch._scaled_mm, cuBLAS):
  A:       (M, K)     or (L, M, K)       dtype float8_e4m3fn, K-contiguous (row-major)
  B:       (K, N)     or (L, K, N)       dtype float8_e4m3fn, K-contiguous (col-major)
  A_scale: (M, K/32)  or (L, M, K/32)    dtype float8_e8m0fnu, K-contiguous
  B_scale: (K/32, N)  or (L, K/32, N)    dtype float8_e8m0fnu, K-contiguous
  out:     (M, N)     or (L, M, N)       dtype bfloat16/float16, contiguous

"K-contiguous" means stride 1 on the K axis. This matches how torchao/cuBLAS
use `torch._scaled_mm(a, b.t(), ...)`:
  - you store a weight as nn.Linear-style `W` of shape `(N, K)` row-major
  - you pass `W.mT` (a zero-copy view of shape (K, N) with K-contig) as B
The interface applies `.mT` internally to reach the `(N, K) K-major` layout
the quack kernel consumes. No data is copied.
"""

from functools import lru_cache
from collections.abc import Callable
from typing import Optional, Tuple

import torch
from torch import Tensor

import cutlass

from .blockscaled_gemm_utils import (
    ceil_div,
    compile_blockscaled_gemm_tvm_ffi,
    pack_scale_2d_to_blocked_contig,
    scale_blocked_for_cublas,
    scale_view_for_kernel,
)
from .gemm_default_epi import GemmDefaultSm100
from .mx_utils import to_mx

_MXFP8_SF_VEC_SIZE = 32
_NVFP4_SF_VEC_SIZE = 16
_TORCH_TO_CUTLASS_D = {
    torch.bfloat16: cutlass.BFloat16,
    torch.float16: cutlass.Float16,
    torch.float32: cutlass.Float32,
}
_TORCH_TO_CUTLASS_AB = {
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    torch.float4_e2m1fn_x2: cutlass.Float4E2M1FN,
}
_EPILOGUE_ARG_KIND_TO_CODE = {"tile": 1, "row": 2, "col": 3}


def _epilogue_arg_kind_codes(epilogue_arg_kinds: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(_EPILOGUE_ARG_KIND_TO_CODE[kind] for kind in epilogue_arg_kinds)


def _blockscaled_format(
    a_dtype: torch.dtype, scale_dtype: torch.dtype
) -> tuple[int, cutlass.Numeric, cutlass.Numeric]:
    if a_dtype == torch.float8_e4m3fn and scale_dtype == torch.float8_e8m0fnu:
        return _MXFP8_SF_VEC_SIZE, cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU
    if a_dtype == torch.float4_e2m1fn_x2 and scale_dtype == torch.float8_e4m3fn:
        return _NVFP4_SF_VEC_SIZE, cutlass.Float4E2M1FN, cutlass.Float8E4M3FN
    raise AssertionError(f"unsupported blockscaled dtype pair: {a_dtype}, {scale_dtype}")


def _logical_k(tensor: Tensor) -> int:
    return tensor.shape[-1] * 2 if tensor.dtype == torch.float4_e2m1fn_x2 else tensor.shape[-1]


def _packed_k(logical_k: int, dtype: torch.dtype) -> int:
    return logical_k // 2 if dtype == torch.float4_e2m1fn_x2 else logical_k


def _fake_operand(dev, l, mn, logical_k, dtype):
    return torch.empty(
        l, mn, _packed_k(logical_k, dtype), dtype=dtype, device=dev
    ).permute(1, 2, 0)


def _default_tiler_cluster(m: int, n: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Pick a reasonable default (mma_tiler_mn, cluster_shape_mn)."""
    if m >= 512 and n >= 128:
        return (256, 128), (2, 1)
    return (128, 128), (1, 1)


@lru_cache(maxsize=64)
def _compile_cached(
    m: int,
    n: int,
    k: int,
    l: int,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    out_torch_dtype,
    ab_torch_dtype,
    sf_torch_dtype,
    ab_dtype_cutlass,
    sf_dtype_cutlass,
    sf_vec_size,
):
    """Compile kernel for a given (shape, dtype, tiler, cluster) and cache it."""
    dev = torch.device("cuda")
    rm = ceil_div(m, 128)
    rn = ceil_div(n, 128)
    rk = ceil_div(k // sf_vec_size, 4)
    fake_mA = _fake_operand(dev, l, m, k, ab_torch_dtype)
    fake_mB = _fake_operand(dev, l, n, k, ab_torch_dtype)
    fake_mD = torch.empty(l, m, n, dtype=out_torch_dtype, device=dev).permute(1, 2, 0)
    fake_sc_A = torch.empty(l, rm, rk, 512, dtype=sf_torch_dtype, device=dev)
    fake_sc_B = torch.empty(l, rn, rk, 512, dtype=sf_torch_dtype, device=dev)
    fake_mSFA = scale_view_for_kernel(fake_sc_A, m, k // sf_vec_size, l)
    fake_mSFB = scale_view_for_kernel(fake_sc_B, n, k // sf_vec_size, l)
    return compile_blockscaled_gemm_tvm_ffi(
        ab_dtype_cutlass,
        sf_dtype_cutlass,
        sf_vec_size,
        _TORCH_TO_CUTLASS_D[out_torch_dtype],
        mma_tiler_mn,
        cluster_shape_mn,
        fake_mA,
        fake_mB,
        fake_mD,
        fake_mSFA,
        fake_mSFB,
    )


@lru_cache(maxsize=64)
def _compile_epilogue_cached(
    m: int,
    n: int,
    k: int,
    l: int,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    out_torch_dtype,
    ab_torch_dtype,
    sf_torch_dtype,
    ab_dtype_cutlass,
    sf_dtype_cutlass,
    sf_vec_size,
    tensor_epilogue_fn: Callable,
    tensor_epilogue_key: str,
    tensor_epilogue_arg_kinds: Tuple[int, ...] = (),
    tensor_epilogue_rowvec_dtypes: Tuple[torch.dtype, ...] = (),
    tensor_epilogue_colvec_dtypes: Tuple[torch.dtype, ...] = (),
    tensor_epilogue_tile_dtypes: Tuple[torch.dtype, ...] = (),
):
    dev = torch.device("cuda")
    rm = ceil_div(m, 128)
    rn = ceil_div(n, 128)
    rk = ceil_div(k // sf_vec_size, 4)
    fake_mA = _fake_operand(dev, l, m, k, ab_torch_dtype)
    fake_mB = _fake_operand(dev, l, n, k, ab_torch_dtype)
    fake_mD = torch.empty(l, m, n, dtype=out_torch_dtype, device=dev).permute(1, 2, 0)
    fake_sc_A = torch.empty(l, rm, rk, 512, dtype=sf_torch_dtype, device=dev)
    fake_sc_B = torch.empty(l, rn, rk, 512, dtype=sf_torch_dtype, device=dev)
    fake_mSFA = scale_view_for_kernel(fake_sc_A, m, k // sf_vec_size, l)
    fake_mSFB = scale_view_for_kernel(fake_sc_B, n, k // sf_vec_size, l)
    fake_row_auxes = tuple(
        torch.empty(l, n, dtype=dtype, device=dev)
        for dtype in tensor_epilogue_rowvec_dtypes
    )
    fake_col_auxes = tuple(
        torch.empty(l, m, dtype=dtype, device=dev)
        for dtype in tensor_epilogue_colvec_dtypes
    )
    fake_tile_auxes = tuple(
        torch.empty(l, m, n, dtype=dtype, device=dev).permute(1, 2, 0)
        for dtype in tensor_epilogue_tile_dtypes
    )
    return compile_blockscaled_gemm_tvm_ffi(
        ab_dtype_cutlass,
        sf_dtype_cutlass,
        sf_vec_size,
        _TORCH_TO_CUTLASS_D[out_torch_dtype],
        mma_tiler_mn,
        cluster_shape_mn,
        fake_mA,
        fake_mB,
        fake_mD,
        fake_mSFA,
        fake_mSFB,
        tensor_epilogue_fn=tensor_epilogue_fn,
        tensor_epilogue_key=tensor_epilogue_key,
        tensor_epilogue_uses_c=bool(tensor_epilogue_arg_kinds),
        tensor_epilogue_arg_kinds=tensor_epilogue_arg_kinds,
        tensor_epilogue_rowvec_biases=fake_row_auxes,
        tensor_epilogue_colvec_biases=fake_col_auxes,
        tensor_epilogue_tile_biases=fake_tile_auxes,
    )


def _as_3d(x: Tensor, ndim_in: int) -> Tensor:
    """Add a leading batch dim if input is 2D. Returns a view."""
    if ndim_in == 2:
        return x.unsqueeze(0)
    return x


def _blocked_scale_1d_view(scale: Tensor, mn: int, sf_k: int, l: int) -> Tensor:
    rm = ceil_div(mn, 128)
    rk = ceil_div(sf_k, 4)
    assert scale.numel() == l * rm * rk * 512, (
        f"blocked scale size: expected {l * rm * rk * 512}, got {scale.numel()}"
    )
    return scale.contiguous().view(l, rm, rk, 512)


def _to_kernel_layout(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
) -> Tuple[int, int, int, int, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool]:
    """Normalize shapes/strides, validate, and repack scales. Returns
    (m, n, k, l, mA_mkl, mB_nkl, sc_contig_A, sc_contig_B, sfa_view, sfb_view, was_2d).

    A: (M,K) or (L,M,K) K-contig.  B: (K,N) or (L,K,N) K-contig.
    A_scale: (M,K/32) or (L,M,K/32) K-contig.  B_scale: (K/32,N) or (L,K/32,N) K-contig.
    """
    assert A.dtype in _TORCH_TO_CUTLASS_AB, f"unsupported A dtype: {A.dtype}"
    assert B.dtype == A.dtype, f"B dtype must match A dtype, got {B.dtype} vs {A.dtype}"
    assert A_scale.dtype == B_scale.dtype
    sf_vec_size, ab_dtype_cutlass, sf_dtype_cutlass = _blockscaled_format(
        A.dtype, A_scale.dtype
    )
    was_2d = A.dim() == 2
    # Flip B from (K,N) to (N,K) via .mT (zero-copy). User's B K-contig → .mT K-contig.
    A3 = _as_3d(A, A.dim())
    B3 = _as_3d(B, B.dim()).mT
    l, m, packed_k = A3.shape
    l2, n, packed_k2 = B3.shape
    k = _logical_k(A3)
    k2 = _logical_k(B3)
    assert l == l2, f"batch mismatch: A={l}, B={l2}"
    assert k == k2, f"K mismatch: A K={k}, B K={k2}"
    assert packed_k == packed_k2
    assert k % sf_vec_size == 0, f"K ({k}) must be divisible by {sf_vec_size}"
    assert A3.stride(-1) == 1, "A must be K-contiguous (stride 1 on K)"
    assert B3.stride(-1) == 1, (
        "B must be K-contiguous on its K axis (pass .mT of an (N,K) row-major tensor)"
    )
    sf_k = k // sf_vec_size
    if A_scale.dim() == 1:
        sc_contig_A = _blocked_scale_1d_view(A_scale, m, sf_k, l)
    else:
        as3 = _as_3d(A_scale, A_scale.dim())
        assert as3.stride(-1) == 1, "A_scale must be K-contiguous"
        assert as3.shape == (l, m, sf_k), (
            f"A_scale shape: expected (l={l},m={m},sf_k={sf_k}) K-contig, got {tuple(as3.shape)}"
        )
        sc_contig_A = pack_scale_2d_to_blocked_contig(as3.contiguous())

    if B_scale.dim() == 1:
        sc_contig_B = _blocked_scale_1d_view(B_scale, n, sf_k, l)
    else:
        bs3 = _as_3d(B_scale, B_scale.dim()).mT
        assert bs3.stride(-1) == 1, (
            "B_scale must be K-contiguous on its K axis (pass .mT of an (N, K/32) row-major tensor)"
        )
        assert bs3.shape == (l, n, sf_k), (
            f"B_scale shape: expected .mT of (l={l},sf_k={sf_k},n={n}) -> ({l},{n},{sf_k}), got {tuple(bs3.shape)}"
        )
        sc_contig_B = pack_scale_2d_to_blocked_contig(bs3.contiguous())

    # Force row-major contiguous for packer/kernel consumption.
    # A3 / B3 are views — .contiguous() materializes (l,m,k) / (l,n,k) row-major.
    A3_c = A3.contiguous()
    B3_c = B3.contiguous()
    # (l, m, k) -> (m, k, l) K-major view (no copy; strides (k, 1, m*k))
    mA_mkl = A3_c.permute(1, 2, 0)
    mB_nkl = B3_c.permute(1, 2, 0)
    sfa_view = scale_view_for_kernel(sc_contig_A, m, sf_k, l)
    sfb_view = scale_view_for_kernel(sc_contig_B, n, sf_k, l)
    return (
        m,
        n,
        k,
        l,
        mA_mkl,
        mB_nkl,
        sc_contig_A,
        sc_contig_B,
        sfa_view,
        sfb_view,
        was_2d,
        sf_vec_size,
        ab_dtype_cutlass,
        sf_dtype_cutlass,
    )


def mxfp8_gemm_out(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    out: Tensor,
    *,
    mma_tiler_mn: Optional[Tuple[int, int]] = None,
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
) -> None:
    """MXFP8 blockscaled GEMM with pre-allocated output. See module doc for shape conventions."""
    (
        m,
        n,
        k,
        l,
        mA,
        mB,
        _scA,
        _scB,
        sfa,
        sfb,
        was_2d,
        sf_vec_size,
        ab_dtype_cutlass,
        sf_dtype_cutlass,
    ) = _to_kernel_layout(A, B, A_scale, B_scale)
    out_dtype = out.dtype
    assert out_dtype in _TORCH_TO_CUTLASS_D, f"unsupported out dtype: {out_dtype}"
    expected_out_shape = (m, n) if was_2d else (l, m, n)
    assert tuple(out.shape) == expected_out_shape, (
        f"out shape {tuple(out.shape)} != expected {expected_out_shape}"
    )
    assert out.is_contiguous(), "out must be contiguous"
    # View caller's contiguous (M,N) or (L,M,N) as (M,N,L) N-major strided view, no copy.
    out_3d = out.unsqueeze(0) if was_2d else out  # (l, m, n)
    mD = out_3d.permute(1, 2, 0)  # (m, n, l), strides (n, 1, m*n)
    if mma_tiler_mn is None or cluster_shape_mn is None:
        tlr, clu = _default_tiler_cluster(m, n)
        mma_tiler_mn = mma_tiler_mn or tlr
        cluster_shape_mn = cluster_shape_mn or clu
    if not GemmDefaultSm100.can_implement_blockscaled(
        ab_dtype_cutlass,
        sf_dtype_cutlass,
        sf_vec_size,
        _TORCH_TO_CUTLASS_D[out_dtype],
        mma_tiler_mn,
        cluster_shape_mn,
        m,
        n,
        k,
        l,
        "k",
        "k",
        "n",
    ):
        raise ValueError(
            f"unsupported config: m={m}, n={n}, k={k}, l={l}, "
            f"tiler={mma_tiler_mn}, cluster={cluster_shape_mn}"
        )
    runner = _compile_cached(
        m,
        n,
        k,
        l,
        mma_tiler_mn,
        cluster_shape_mn,
        out_dtype,
        A.dtype,
        A_scale.dtype,
        ab_dtype_cutlass,
        sf_dtype_cutlass,
        sf_vec_size,
    )
    runner(mA, mB, mD, sfa, sfb)


def mxfp8_scaled_mm_epilogue(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    epilogue_fn: Callable,
    epilogue_key: str,
    out_dtype: torch.dtype = torch.bfloat16,
    *,
    mma_tiler_mn: Optional[Tuple[int, int]] = None,
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
    epilogue_arg_kinds: tuple[str, ...] = (),
    epilogue_rowvec_biases: tuple[Tensor, ...] = (),
    epilogue_colvec_biases: tuple[Tensor, ...] = (),
    epilogue_tile_biases: tuple[Tensor, ...] = (),
) -> Tensor:
    (
        m,
        n,
        k,
        l,
        mA,
        mB,
        _scA,
        _scB,
        sfa,
        sfb,
        was_2d,
        sf_vec_size,
        ab_dtype_cutlass,
        sf_dtype_cutlass,
    ) = _to_kernel_layout(A, B, A_scale, B_scale)
    out_shape = (m, n) if was_2d else (l, m, n)
    out = torch.empty(out_shape, dtype=out_dtype, device=A.device)
    mD = (out.unsqueeze(0) if was_2d else out).permute(1, 2, 0)
    if mma_tiler_mn is None or cluster_shape_mn is None:
        if (
            ab_dtype_cutlass == cutlass.Float4E2M1FN
            and sf_dtype_cutlass == cutlass.Float8E4M3FN
        ):
            tlr, clu = (128, 192), (1, 1)
        else:
            tlr, clu = _default_tiler_cluster(m, n)
        mma_tiler_mn = mma_tiler_mn or tlr
        cluster_shape_mn = cluster_shape_mn or clu
    if not GemmDefaultSm100.can_implement_blockscaled(
        ab_dtype_cutlass,
        sf_dtype_cutlass,
        sf_vec_size,
        _TORCH_TO_CUTLASS_D[out_dtype],
        mma_tiler_mn,
        cluster_shape_mn,
        m,
        n,
        k,
        l,
        "k",
        "k",
        "n",
    ):
        raise ValueError(
            f"unsupported config: m={m}, n={n}, k={k}, l={l}, "
            f"tiler={mma_tiler_mn}, cluster={cluster_shape_mn}"
        )
    row_auxes = tuple(
        tensor.unsqueeze(0) if tensor.ndim == 1 else tensor
        for tensor in epilogue_rowvec_biases
    )
    col_auxes = tuple(
        tensor.unsqueeze(0) if tensor.ndim == 1 else tensor
        for tensor in epilogue_colvec_biases
    )
    tile_auxes = tuple(
        (tensor.unsqueeze(0) if tensor.ndim == 2 else tensor).permute(1, 2, 0)
        for tensor in epilogue_tile_biases
    )
    runner = _compile_epilogue_cached(
        m,
        n,
        k,
        l,
        mma_tiler_mn,
        cluster_shape_mn,
        out_dtype,
        A.dtype,
        A_scale.dtype,
        ab_dtype_cutlass,
        sf_dtype_cutlass,
        sf_vec_size,
        epilogue_fn,
        epilogue_key,
        tuple({"tile": 1, "row": 2, "col": 3}[kind] for kind in epilogue_arg_kinds),
        tuple(tensor.dtype for tensor in row_auxes),
        tuple(tensor.dtype for tensor in col_auxes),
        tuple(tensor.dtype for tensor in tile_auxes),
    )
    runner(mA, mB, mD, sfa, sfb, row_auxes, col_auxes, tile_auxes)
    return out


def _mxfp8_varlen_m_scales_to_kernel_layout(
    A_scale: Tensor,
    B_scale: Tensor,
    total_m: int,
    n: int,
    k: int,
    offs: Tensor,
) -> tuple[Tensor, Tensor]:
    if A_scale.dim() == 4 and B_scale.dim() == 4:
        return A_scale, B_scale
    sf_k = k // _MXFP8_SF_VEC_SIZE
    rk = ceil_div(sf_k, 4)
    groups = offs.numel()
    total_padded_rm = ceil_div(total_m, 128) + groups - 1
    A_kernel_scale = torch.zeros(
        1,
        total_padded_rm,
        rk,
        512,
        dtype=A_scale.dtype,
        device=A_scale.device,
    )
    offset = 0
    flat_A_scale = A_scale.contiguous().view(-1)
    prev_end = 0
    for group_idx, end in enumerate(offs.detach().cpu().tolist()):
        group_m = end - prev_end
        src_rm = ceil_div(group_m, 128)
        chunk_size = src_rm * rk * 512
        dst_rm = prev_end // 128 + group_idx
        if chunk_size:
            A_kernel_scale[0, dst_rm : dst_rm + src_rm] = flat_A_scale[
                offset : offset + chunk_size
            ].view(src_rm, rk, 512)
        offset += chunk_size
        prev_end = end
    rn = ceil_div(n, 128)
    B_kernel_scale = B_scale.contiguous().view(groups, rn, rk, 512)
    return A_kernel_scale, B_kernel_scale


def _mxfp8_varlen_k_scales_to_kernel_layout(
    A_scale: Tensor,
    B_scale: Tensor,
    m: int,
    n: int,
    total_k: int,
    offs: Tensor,
) -> tuple[Tensor, Tensor]:
    if A_scale.dim() == 4 and B_scale.dim() == 4:
        return A_scale, B_scale
    groups = offs.numel()
    total_padded_rk = ceil_div(total_k, 128) + groups - 1
    rm = ceil_div(m, 128)
    rn = ceil_div(n, 128)
    A_kernel_scale = torch.zeros(
        1,
        rm,
        total_padded_rk,
        512,
        dtype=A_scale.dtype,
        device=A_scale.device,
    )
    B_kernel_scale = torch.zeros(
        1,
        rn,
        total_padded_rk,
        512,
        dtype=B_scale.dtype,
        device=B_scale.device,
    )
    flat_A_scale = A_scale.contiguous().view(-1)
    flat_B_scale = B_scale.contiguous().view(-1)
    offset_A = 0
    offset_B = 0
    prev_end = 0
    for group_idx, end in enumerate(offs.detach().cpu().tolist()):
        group_k = end - prev_end
        rk = ceil_div(group_k // _MXFP8_SF_VEC_SIZE, 4)
        dst_rk = prev_end // 128 + group_idx
        a_chunk_size = rm * rk * 512
        b_chunk_size = rn * rk * 512
        if rk:
            A_kernel_scale[0, :, dst_rk : dst_rk + rk] = flat_A_scale[
                offset_A : offset_A + a_chunk_size
            ].view(rm, rk, 512)
            B_kernel_scale[0, :, dst_rk : dst_rk + rk] = flat_B_scale[
                offset_B : offset_B + b_chunk_size
            ].view(rn, rk, 512)
        offset_A += a_chunk_size
        offset_B += b_chunk_size
        prev_end = end
    return A_kernel_scale, B_kernel_scale


def mxfp8_varlen_m_scaled_mm_epilogue(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    offs: Tensor,
    epilogue_fn: Callable,
    epilogue_key: str,
    out_dtype: torch.dtype = torch.bfloat16,
    *,
    mma_tiler_mn: Optional[Tuple[int, int]] = None,
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
    epilogue_args: tuple[Tensor, ...] = (),
    epilogue_arg_kinds: tuple[str, ...] = (),
    epilogue_rowvec_biases: tuple[Tensor, ...] = (),
    epilogue_colvec_biases: tuple[Tensor, ...] = (),
    tuned: bool = False,
    epilogue_source: str | None = None,
) -> Tensor:
    assert A.dim() == 2, f"varlen-M A must be (total_m, k), got {tuple(A.shape)}"
    assert B.dim() == 3, f"varlen-M B must be (n, k, groups), got {tuple(B.shape)}"
    assert offs.dtype is torch.int32, f"offs must be int32, got {offs.dtype}"
    assert A.dtype == torch.float8_e4m3fn and B.dtype == torch.float8_e4m3fn
    assert A_scale.dtype == torch.float8_e8m0fnu and B_scale.dtype == torch.float8_e8m0fnu
    total_m, k = A.shape
    n, k_b, groups = B.shape
    assert k == k_b
    assert offs.numel() == groups
    A_scale, B_scale = _mxfp8_varlen_m_scales_to_kernel_layout(
        A_scale, B_scale, total_m, n, k, offs
    )
    out = torch.empty(total_m, n, dtype=out_dtype, device=A.device)
    if mma_tiler_mn is None or cluster_shape_mn is None:
        mma_tiler_mn = mma_tiler_mn or (128, 128)
        cluster_shape_mn = cluster_shape_mn or (1, 1)
    if epilogue_args:
        epilogue_rowvec_biases = tuple(
            tensor for tensor, kind in zip(epilogue_args, epilogue_arg_kinds) if kind == "row"
        )
        epilogue_colvec_biases = tuple(
            tensor for tensor, kind in zip(epilogue_args, epilogue_arg_kinds) if kind == "col"
        )
        epilogue_arg_kinds = tuple(
            kind for kind in epilogue_arg_kinds if kind in ("row", "col")
        )
    row_auxes = tuple(
        tensor.expand(groups, n).contiguous() if tensor.ndim == 1 else tensor
        for tensor in epilogue_rowvec_biases
    )
    col_auxes = epilogue_colvec_biases
    runner = compile_blockscaled_gemm_tvm_ffi(
        cutlass.Float8E4M3FN,
        cutlass.Float8E8M0FNU,
        _MXFP8_SF_VEC_SIZE,
        _TORCH_TO_CUTLASS_D[out_dtype],
        mma_tiler_mn,
        cluster_shape_mn,
        A,
        B,
        out,
        A_scale,
        B_scale,
        varlen_m=True,
        tensor_epilogue_fn=epilogue_fn,
        tensor_epilogue_key=epilogue_key,
        tensor_epilogue_uses_c=bool(epilogue_arg_kinds),
        tensor_epilogue_arg_kinds=_epilogue_arg_kind_codes(epilogue_arg_kinds),
        tensor_epilogue_rowvec_biases=row_auxes,
        tensor_epilogue_colvec_biases=col_auxes,
    )
    cu_seqlens = torch.cat((offs.new_zeros(1), offs))
    runner(
        A,
        B,
        out,
        A_scale,
        B_scale,
        row_auxes=row_auxes,
        col_auxes=col_auxes,
        cu_seqlens=cu_seqlens,
    )
    return out


def mxfp8_varlen_k_scaled_mm_epilogue(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    offs: Tensor,
    epilogue_fn: Callable,
    epilogue_key: str,
    out_dtype: torch.dtype = torch.bfloat16,
    *,
    mma_tiler_mn: Optional[Tuple[int, int]] = None,
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
    epilogue_args: tuple[Tensor, ...] = (),
    epilogue_arg_kinds: tuple[str, ...] = (),
    epilogue_rowvec_biases: tuple[Tensor, ...] = (),
    epilogue_colvec_biases: tuple[Tensor, ...] = (),
    tuned: bool = False,
    epilogue_source: str | None = None,
) -> Tensor:
    assert A.dim() == 2 and B.dim() == 2
    assert offs.dtype is torch.int32, f"offs must be int32, got {offs.dtype}"
    assert A.dtype == torch.float8_e4m3fn and B.dtype == torch.float8_e4m3fn
    assert A_scale.dtype == torch.float8_e8m0fnu and B_scale.dtype == torch.float8_e8m0fnu
    m, total_k = A.shape
    n, total_k_b = B.shape
    assert total_k == total_k_b
    groups = offs.numel()
    A_scale, B_scale = _mxfp8_varlen_k_scales_to_kernel_layout(
        A_scale, B_scale, m, n, total_k, offs
    )
    out = torch.empty(groups, m, n, dtype=out_dtype, device=A.device).permute(1, 2, 0)
    if mma_tiler_mn is None or cluster_shape_mn is None:
        mma_tiler_mn = mma_tiler_mn or (128, 128)
        cluster_shape_mn = cluster_shape_mn or (1, 1)
    if epilogue_args:
        epilogue_rowvec_biases = tuple(
            tensor for tensor, kind in zip(epilogue_args, epilogue_arg_kinds) if kind == "row"
        )
        epilogue_colvec_biases = tuple(
            tensor for tensor, kind in zip(epilogue_args, epilogue_arg_kinds) if kind == "col"
        )
        epilogue_arg_kinds = tuple(
            kind for kind in epilogue_arg_kinds if kind in ("row", "col")
        )
    runner = compile_blockscaled_gemm_tvm_ffi(
        cutlass.Float8E4M3FN,
        cutlass.Float8E8M0FNU,
        _MXFP8_SF_VEC_SIZE,
        _TORCH_TO_CUTLASS_D[out_dtype],
        mma_tiler_mn,
        cluster_shape_mn,
        A,
        B,
        out,
        A_scale,
        B_scale,
        varlen_k=True,
        tensor_epilogue_fn=epilogue_fn,
        tensor_epilogue_key=epilogue_key,
        tensor_epilogue_uses_c=bool(epilogue_arg_kinds),
        tensor_epilogue_arg_kinds=_epilogue_arg_kind_codes(epilogue_arg_kinds),
        tensor_epilogue_rowvec_biases=epilogue_rowvec_biases,
        tensor_epilogue_colvec_biases=epilogue_colvec_biases,
    )
    cu_seqlens = torch.cat((offs.new_zeros(1), offs))
    runner(
        A,
        B,
        out,
        A_scale,
        B_scale,
        row_auxes=epilogue_rowvec_biases,
        col_auxes=epilogue_colvec_biases,
        cu_seqlens=cu_seqlens,
    )
    return out.permute(2, 0, 1).contiguous()


def mxfp8_gemm(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    out: Optional[Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    *,
    mma_tiler_mn: Optional[Tuple[int, int]] = None,
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
) -> Tensor:
    """MXFP8 blockscaled GEMM. Allocates output if not provided."""
    if out is None:
        # A: (M,K) or (L,M,K); B: (K,N) or (L,K,N); out: (M,N) or (L,M,N)
        if A.dim() == 2:
            out_shape = (A.shape[0], B.shape[1])
        else:
            out_shape = (A.shape[0], A.shape[1], B.shape[2])
        out = torch.empty(out_shape, dtype=out_dtype, device=A.device)
    mxfp8_gemm_out(
        A,
        B,
        A_scale,
        B_scale,
        out,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
    )
    return out


def mxfp8_quantize(x: Tensor) -> Tuple[Tensor, Tensor]:
    """Quantize a (..., K) bf16/fp32 tensor to MXFP8. Returns (qdata, scale_2d)
    in torchao-convention layout. Last dim (K) must be divisible by 32."""
    assert x.shape[-1] % _MXFP8_SF_VEC_SIZE == 0, (
        f"last dim ({x.shape[-1]}) must be divisible by {_MXFP8_SF_VEC_SIZE}"
    )
    return to_mx(x.contiguous(), _MXFP8_SF_VEC_SIZE)


def mxfp8_gemm_quantize(
    A: Tensor,
    B: Tensor,
    out: Optional[Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    *,
    mma_tiler_mn: Optional[Tuple[int, int]] = None,
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
) -> Tensor:
    """High-level: quantize bf16 A, B_as_NK to MXFP8, then run C = A @ B_as_NK.mT.
    Inputs: A=(M,K)/(L,M,K), B_as_NK=(N,K)/(L,N,K) bf16/fp32. Quantization
    scales along the last (K) dim. Returned output has shape (M,N)/(L,M,N)."""
    A_q, A_sc = mxfp8_quantize(A)
    B_q, B_sc = mxfp8_quantize(B)
    # B_q, B_sc are (..., N, K) / (..., N, K/32). Flip to (..., K, N) / (..., K/32, N)
    # K-contig zero-copy views to match the interface convention.
    return mxfp8_gemm(
        A_q,
        B_q.mT,
        A_sc,
        B_sc.mT,
        out=out,
        out_dtype=out_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
    )


def mxfp8_gemm_cublas(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Reference path via torch._scaled_mm. Requires l=1 (or 2D inputs)."""
    (
        m,
        n,
        k,
        l,
        _mA,
        _mB,
        sc_A,
        sc_B,
        _sfa,
        _sfb,
        was_2d,
        sf_vec_size,
        _ab_dtype_cutlass,
        _sf_dtype_cutlass,
    ) = _to_kernel_layout(A, B, A_scale, B_scale)
    assert l == 1, "torch._scaled_mm MXFP8 path is 2D only; pass 2D inputs or l=1"
    # torch._scaled_mm: A=(M,K) row-major, B=(K,N) col-major (both K-contig) -- same layout user gave us.
    a2d = A if A.dim() == 2 else A.squeeze(0)
    b2d = B if B.dim() == 2 else B.squeeze(0)
    sca = scale_blocked_for_cublas(sc_A, m, k // sf_vec_size, 0)
    scb = scale_blocked_for_cublas(sc_B, n, k // sf_vec_size, 0)
    out = torch._scaled_mm(
        a2d,
        b2d,
        scale_a=sca,
        scale_b=scb,
        out_dtype=out_dtype,
    )
    return out if was_2d else out.unsqueeze(0)


def mxfp8_gemm_ref(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Dequantize + plain matmul reference. A=(M,K), B=(K,N)."""
    was_2d = A.dim() == 2
    # (l, m, k)
    A3 = _as_3d(A, A.dim()).float()
    # B is (K, N)/(L, K, N); flip to (l, n, k) for dequant by last-dim
    B3 = _as_3d(B, B.dim()).mT.contiguous().float()
    as3 = _as_3d(A_scale, A_scale.dim()).float()
    bs3 = _as_3d(B_scale, B_scale.dim()).mT.contiguous().float()
    a_dq = A3 * as3.repeat_interleave(_MXFP8_SF_VEC_SIZE, dim=-1)
    b_dq = B3 * bs3.repeat_interleave(_MXFP8_SF_VEC_SIZE, dim=-1)
    out3 = torch.einsum("lmk,lnk->lmn", a_dq, b_dq).to(out_dtype)
    return out3.squeeze(0) if was_2d else out3
