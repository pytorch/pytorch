# Copyright (c) 2025-2026, QuACK team.
# GEMM compilation via TVM-FFI with fake tensors and NamedTuple args.

from typing import NamedTuple, Optional

import torch
from torch import Tensor

import cutlass.cute as cute
from cutlass import Int32, Float32
from cutlass.cute.runtime import make_ptr

from torch._vendor.quack.cache import jit_cache
from torch._vendor.quack.split_k_reduce import split_k_reduce
from torch._vendor.quack.gemm_config import SplitKMode
from torch._vendor.quack.compile_utils import make_fake_tensor as fake_tensor
from torch._vendor.quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters, torch2cute_dtype_map
from torch._vendor.quack.gemm_default_epi import (
    GemmDefaultEpiMixin,
    GemmDefaultSm80,
    GemmDefaultSm90,
    GemmDefaultSm100,
    GemmDefaultSm120,
)
from torch._vendor.quack.rounding import RoundingMode
from torch._vendor.quack.gemm_tvm_ffi_utils import (
    fake_batched,
    get_majors,
    get_dtypes,
    make_scheduler_args,
    make_varlen_args,
    make_fake_scheduler_args,
    make_fake_varlen_args,
    make_fake_gemm_tensors,
    make_fake_sf_tensor,
    compile_gemm_kernel,
    resolve_blockscaled_formats,
    validate_blockscaled_sf,
    tensor_key,
    scalar_mode,
    scalar_arg,
    plan_scheduler_args,
    validate_ag_geometry,
    launch_gemm,
)


@jit_cache
def _compile_gemm(
    a_dtype,
    b_dtype,
    d_dtype,
    c_dtype,
    a_major,
    b_major,
    d_major,
    c_major,
    tile_shape_mn,
    cluster_shape_mnk,
    pingpong,
    persistent,
    is_dynamic_persistent,
    rowvec_dtype,
    colvec_dtype,
    colvec_ndim,
    alpha_mode,
    beta_mode,
    add_to_output,
    concat_layout,
    varlen_m,
    varlen_k,
    gather_A,
    use_tma_gather,
    has_batch_idx_permute,
    device_capacity,
    rounding_mode,
    sr_seed_mode,
    num_warps,
    sf_dtype=None,
    sf_vec_size=None,
    split_k=1,
    split_k_mode=SplitKMode.SERIAL,
    batched=True,
    b_kn=False,
    sf_batched=True,
    sfd_dtype=None,
    sfd_norm_const_mode=0,
    sfd_col_dtype=None,
    a_mma_dtype=None,  # blockscaled: MMA element types when they differ from the
    b_mma_dtype=None,  # storage dtypes (packed fp6 crosses the boundary as bytes)
    has_ag=False,
):
    sm_to_cls = {
        8: GemmDefaultSm80,
        9: GemmDefaultSm90,
        10: GemmDefaultSm100,
        11: GemmDefaultSm100,
        12: GemmDefaultSm120,
    }
    GemmCls = sm_to_cls[device_capacity[0]]
    mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(
        a_dtype,
        b_dtype,
        d_dtype,
        c_dtype,
        a_major,
        b_major,
        d_major,
        c_major,
        varlen_m=varlen_m,
        varlen_k=varlen_k,
        gather_A=gather_A,
        batched=batched,
        b_kn=b_kn,
        a_mma_dtype=a_mma_dtype,
        b_mma_dtype=b_mma_dtype,
    )
    if split_k > 1 and split_k_mode == SplitKMode.SEPARATE:
        # D is the f32 partials workspace; its batch extent is l * split_k, not l,
        # so it needs its own symbolic batch dim.
        mD = fake_batched(
            d_dtype, m, n, cute.sym_int(), 1 if d_major == "n" else 0, 128 // d_dtype.width
        )

    def fake_scalar(mode, dtype=Float32):
        if mode == 0:
            return None
        elif mode == 1:
            return dtype(1.0 if dtype == Float32 else 0)
        else:
            return make_ptr(dtype, 0, cute.AddressSpace.gmem, assumed_align=4)

    mRowVec = fake_tensor(rowvec_dtype, (l, n), leading_dim=1, divisibility=4)
    if colvec_ndim == 2:
        mColVec = fake_tensor(colvec_dtype, (l, m), leading_dim=1, divisibility=4)
    elif colvec_ndim == 1:  # m is total_m in this case
        mColVec = fake_tensor(colvec_dtype, (m,), leading_dim=0, divisibility=4)
    else:
        mColVec = None

    epi_args = GemmCls.EpilogueArguments(
        alpha=fake_scalar(alpha_mode),
        beta=fake_scalar(beta_mode),
        mRowVecBroadcast=mRowVec,
        mColVecBroadcast=mColVec,
        add_to_output=add_to_output,
        rounding_mode=rounding_mode,
        sr_seed=fake_scalar(sr_seed_mode, dtype=Int32),
        split_k_semaphore=(
            # (ntile_m, ntile_n, L) per-tile completion flag; tile counts are
            # runtime-symbolic. SERIAL: turnstile; PARALLEL: arrival counter.
            fake_tensor(Int32, (cute.sym_int(), cute.sym_int(), cute.sym_int()), leading_dim=1)
            if split_k > 1 and split_k_mode != SplitKMode.SEPARATE
            else None
        ),
        split_k_workspace=(
            # (cta_tile_m * cta_tile_n, ntile_m, ntile_n, L) raw-f32-partials regions,
            # one flat fragment stripe per output tile.
            fake_tensor(
                Float32,
                (cute.sym_int(), cute.sym_int(), cute.sym_int(), cute.sym_int()),
                leading_dim=0,
                divisibility=4,
            )
            if split_k > 1 and split_k_mode != SplitKMode.SEPARATE
            else None
        ),
        # varlen_m SFD is one M-padded buffer with a static batch dim of 1
        # (mirroring SFA above); the column-direction SF is dense-batch only.
        mSFD=(
            make_fake_sf_tensor(sfd_dtype, 1 if varlen_m else l) if sfd_dtype is not None else None
        ),
        sfd_norm_const=fake_scalar(sfd_norm_const_mode),
        mSFDCol=(make_fake_sf_tensor(sfd_col_dtype, l) if sfd_col_dtype is not None else None),
    )
    scheduler_args = make_fake_scheduler_args(
        (is_dynamic_persistent and device_capacity[0] <= 9),
        has_batch_idx_permute,
        l,
        has_ag=has_ag,
    )
    aidx_len = m if varlen_m else (k if varlen_k else None)
    varlen_args = make_fake_varlen_args(varlen_m, varlen_k, gather_A, aidx_len)
    if sf_dtype is not None:
        # Padded SF buffers have a static batch dim of exactly 1 (not l): SFA for
        # varlen_m (M-padded) and varlen_k (K-padded); SFB is K-padded too for
        # varlen_k but stays per-batch (l, rn, rk, ...) for varlen_m. Dense
        # unbatched (5-D) SFs get the batch mode prepended at trace time.
        dense_l = l if sf_batched else None
        mSFA = make_fake_sf_tensor(sf_dtype, 1 if (varlen_m or varlen_k) else dense_l)
        mSFB = make_fake_sf_tensor(sf_dtype, 1 if varlen_k else dense_l)
    else:
        mSFA, mSFB = None, None
    return compile_gemm_kernel(
        GemmCls,
        a_dtype,
        tile_shape_mn,
        cluster_shape_mnk,
        pingpong,
        persistent,
        gather_A,
        is_dynamic_persistent,
        device_capacity,
        mA,
        mB,
        mD,
        mC,
        epi_args,
        scheduler_args,
        varlen_args,
        mSFA=mSFA,
        mSFB=mSFB,
        use_tma_gather=use_tma_gather,
        concat_layout=concat_layout or None,
        num_warps=num_warps,
        sf_vec_size=sf_vec_size,
        split_k=split_k,
        split_k_mode=split_k_mode,
        b_transposed=b_kn,
        a_mma_dtype=a_mma_dtype,
        b_mma_dtype=b_mma_dtype,
    )


class _GemmPlan(NamedTuple):
    """Launch plan derived purely from tensor metadata and config flags.

    Cached per metadata key (see ``_gemm_plan_cache``) so warm calls skip
    validation, major/dtype derivation, the compile-cache lookup, and the
    construction of static argument templates. Everything here is immutable,
    so reusing a plan (and its static NamedTuple arg templates) across calls
    is safe.
    """

    compiled_fn: object
    is_sm100_family: bool  # SM100/110 use 2-CTA MMA
    use_d_as_c: bool  # SM8x lowers add_to_output to C = D
    alpha_mode: int
    beta_mode: int
    sr_seed_mode: int
    sfd_norm_const_mode: int
    epi_static: Optional[object]  # EpilogueArguments when it has no per-call values
    scheduler_static: Optional[object]  # TileSchedulerOptions when it has no per-call values
    max_active_clusters: int
    max_swizzle_size: int
    scheduler_uses_semaphore: bool  # SM8x/SM90 dynamic scheduler consumes the semaphore
    # Everything run_gemm_plan needs beyond the per-call tensors/scalars, so an
    # outer layer holding a resolved plan can launch without re-keying here.
    split_k: int
    split_k_mode: object
    add_to_output: bool  # staged split-K reduce routes it there, not into the GEMM
    tile_M: int
    tile_N: int
    cluster_M: int
    cluster_N: int


# (metadata key) -> _GemmPlan. Grows with distinct (shape, stride, dtype, flag)
# combinations, like the autotuner cache; the expensive compile is deduped one
# level down by @jit_cache on _compile_gemm.
_gemm_plan_cache: dict[tuple, _GemmPlan] = {}


# Split-K: the full epilogue (alpha, beta*C, bias) runs exactly once per output tile,
# on the entity that owns the completed f32 sum; non-finalizing splits emit raw f32
# partials only. The buffers below are per-call allocations (the plan is cached and
# immutable); the plan only bakes the redirected compile signature.


def _staged_split_k_workspace(D, split_k):
    """f32 partials workspace for SplitKMode.SEPARATE.

    The GEMM stores raw f32 partials into a workspace whose batch mode is the
    combined (l * split_k + split) index (majorness matches D, contiguous dim
    padded to 16 bytes for TMA); the separate reduction kernel sums the splits
    in fixed order and applies the epilogue — so all epi operands are routed
    there, not into the GEMM.
    """
    num_l, len_m, len_n = D.shape
    if D.stride(-1) == 1:  # n-major
        n_pad = (len_n + 3) // 4 * 4
        return torch.empty((num_l * split_k, len_m, n_pad), dtype=torch.float32, device=D.device)[
            ..., :len_n
        ]
    else:  # m-major
        m_pad = (len_m + 3) // 4 * 4
        return torch.empty((num_l * split_k, len_n, m_pad), dtype=torch.float32, device=D.device)[
            ..., :len_m
        ].mT


def _split_k_buffers(D, split_k_mode, tile_M, tile_N, cluster_M, cluster_N, sm100):
    """Semaphore + partials workspace for SplitKMode.SERIAL / PARALLEL.

    One Int32 completion flag per output tile (SERIAL: turnstile in split order,
    deterministic; PARALLEL: arrival counter, not deterministic) plus a flat f32
    region per tile holding the non-finalizing splits' raw accumulator fragments.
    The scheduler's CTA tile ids are cluster-rounded (a boundary cluster can
    contain CTAs whose tile is fully out of bounds but which still run the
    protocol), so tile counts are rounded up to cluster multiples. The per-CTA
    tile M must match the kernel exactly (the workspace region stride is
    cta_tile_m * tile_N): SM100/110 2-CTA MMA (cluster_M even, tile_M 128/256)
    halves it.
    """
    num_l, len_m, len_n = D.shape
    use_2cta = sm100 and cluster_M % 2 == 0 and tile_M in (128, 256)
    cta_tile_m = tile_M // 2 if use_2cta else tile_M
    ntile_m = (len_m + cta_tile_m - 1) // cta_tile_m
    ntile_n = (len_n + tile_N - 1) // tile_N
    ntile_m = (ntile_m + cluster_M - 1) // cluster_M * cluster_M
    ntile_n = (ntile_n + cluster_N - 1) // cluster_N * cluster_N
    # Kernel-facing layouts are (ntile_m, ntile_n, L) and (E, ntile_m, ntile_n, L):
    # the CuTe layouts own the address computation, so the kernel just slices them
    # at (pid_m, pid_n, batch).
    semaphore = torch.zeros((num_l, ntile_m, ntile_n), dtype=torch.int32, device=D.device)
    tile_elems = cta_tile_m * tile_N
    if split_k_mode == SplitKMode.SERIAL:
        # Split 0's in-order plain store initializes each region.
        workspace = torch.empty(
            (num_l, ntile_m, ntile_n, tile_elems), dtype=torch.float32, device=D.device
        )
    else:
        # PARALLEL: every split red.adds in arrival order; no initializing store.
        workspace = torch.zeros(
            (num_l, ntile_m, ntile_n, tile_elems), dtype=torch.float32, device=D.device
        )
    return semaphore, workspace


def _reduce_staged_split_k(
    split_k_workspace, D, C, split_k, alpha, beta, rowvec_bias, colvec_bias, add_to_output
):
    # The reduction kernel wants the contiguous dim last; transpose the views for
    # m-major outputs (the kernel is elementwise in (m, n), so this is free; the
    # row/col vector roles swap in the transposed space).
    ws_v, out_v, c_v = split_k_workspace, D, C
    rowvec_v, colvec_v = rowvec_bias, colvec_bias
    if D.stride(-1) != 1:
        ws_v, out_v = ws_v.transpose(1, 2), out_v.transpose(1, 2)
        c_v = c_v.transpose(1, 2) if c_v is not None else None
        rowvec_v, colvec_v = colvec_v, rowvec_v
    if c_v is not None and c_v.stride(-1) != 1:
        # The reduce kernel requires C contiguous along out's contiguous dim; a C
        # whose majorness differs from D's (legal in the GEMM epilogue, which
        # tracks c_major independently) is materialized once here.
        c_v = c_v.contiguous()
    split_k_reduce(
        ws_v,
        out_v,
        split_k,
        alpha=alpha,
        beta=beta,
        C=c_v,
        rowvec_bias=rowvec_v,
        colvec_bias=colvec_v,
        add_to_output=add_to_output,
    )


class AllGatherArguments(NamedTuple):
    """AllGather+GEMM scheduler arguments (see quack/distributed/): the typed
    host-side mirror of the ag_* fields in TileSchedulerArguments — the
    kernel-facing structure cute.jit consumes. flags[shard * num_chunks + c]
    >= *epoch gates a tile's first TMA; num_shards/first_shard drive the
    shard-major rotated decode; num_chunks is the sub-shard arrival
    granularity."""

    flags: Tensor  # (num_shards * num_chunks,) int32, symmetric
    epoch: Tensor  # (1,) int32, 4B-aligned, device-resident epoch slot
    num_shards: int  # world_size: shards along M
    # Index of the shard whose tiles are scheduled FIRST; consumption walks
    # the ring from there. Each rank passes its own rank — the local shard
    # is resident immediately, remote shards arrive in ring order.
    first_shard: int
    num_chunks: int = 1  # sub-shard arrival granularity (flags per shard)


def gemm(
    # (l, m, k), or (m, k) unbatched dense (SM90+; B/D/C must be 2D too), or (total_m, k)
    # if varlen_m or (m, total_k) if varlen_k or (whatever, k) if gather_A_varlen_m or
    # (m, whatever) if gather_A_varlen_k
    A: Tensor,
    B: Tensor,  # (l, n, k) or (n, k) or (n, total_k) if varlen_k
    D: Tensor,  # (l, m, n) or (m, n) or (total_m, n) if varlen_m
    C: Optional[Tensor],  # (l, m, n) or (m, n) or (total_m, n) if varlen_m
    tile_count_semaphore: Optional[Tensor],  # (1,)
    tile_M: int,
    tile_N: int,
    cluster_M: int,
    cluster_N: int,
    cluster_K: int = 1,
    tile_K: int | None = None,
    pingpong: bool = False,
    persistent: bool = True,
    is_dynamic_persistent: bool = False,
    max_swizzle_size: int = 8,
    rowvec_bias: Optional[Tensor] = None,  # (l, n)
    colvec_bias: Optional[Tensor] = None,  # (l, m), or (total_m,) if varlen_m
    alpha: float | Tensor = 1.0,
    beta: float | Tensor = 1.0,
    cu_seqlens_m: Optional[Tensor] = None,  # (l+1,) cumulative sum of m values for variable length
    cu_seqlens_k: Optional[Tensor] = None,  # (l+1,) cumulative sum of k values for variable length
    A_idx: Optional[Tensor] = None,  # (total_m,) or (total_k,) indices for gather_A when varlen
    batch_idx_permute: Optional[Tensor] = None,  # (l,) permutation of batch indices for scheduler
    add_to_output: bool = False,
    rounding_mode: int = RoundingMode.RN,
    sr_seed: int | Tensor = 0,
    use_tma_gather: bool = False,
    concat_layout: dict | None = None,
    num_warps: Optional[int] = None,
    # SFA/SFB: (l, rm/rn, rk, 32, 4, 4) blocked scale factors. For varlen_m, SFA is
    # M-padded (1, total_padded_rm, rk, 32, 4, 4) while SFB stays per-batch. For
    # varlen_k, BOTH are K-padded (1, rm/rn, total_padded_rk, 32, 4, 4); pad bytes
    # may be arbitrary (the kernel skips the MMA instructions covering them).
    # See AI/varlen_blockscaled_sf_layout.md.
    SFA: Optional[Tensor] = None,
    SFB: Optional[Tensor] = None,
    # BlockScaledFormat names for A and B (independent; e.g. "mxfp8_e4m3"); both
    # required when SFA/SFB are passed. The descriptors are the single source of
    # scale vec size / packing - formats are never re-derived from tensor dtypes.
    bs_format_a: Optional[str] = None,
    bs_format_b: Optional[str] = None,
    split_k: int = 1,
    split_k_mode: int = SplitKMode.SERIAL,
    # B is passed (k, n) / (l, k, n) and relabeled to kernel order (n, k) at trace
    # time — saves the caller a per-call .mT view. Dense SM90+ only.
    b_kn: bool = False,
    # SFD: (l, rm, rk, 32, 4, 4) blocked output scale factors for quantized D
    # (mxfp8/mxfp4/nvfp4). e8m0 SFD => sf_vec 32 (mx), e4m3 SFD => sf_vec 16
    # (nvfp4). sfd_norm_const is an optional fp32 norm constant (the reciprocal
    # of the nvfp4 per-tensor scale) folded into the stored scale factors.
    SFD: Optional[Tensor] = None,
    sfd_norm_const: Optional[float | Tensor] = None,
    # Column-direction output scale factors: (l, rn, rm_k, 32, 4, 4) blocked
    # over (N, M), for backward consumers contracting over this output's M.
    # Mutually exclusive with SFD.
    SFDCol: Optional[Tensor] = None,
    # AllGather+GEMM (SM90+, see quack/distributed/all_gather_gemm.py): A is the full
    # gathered (M, K) buffer being filled by a copy stream; (flags, seq,
    # num_shards, first_shard) drive the shard-major rotated schedule and the
    # load-warp arrival gate.
    ag_args: Optional[AllGatherArguments] = None,
) -> _GemmPlan:
    alpha_mode = scalar_mode(alpha)
    beta_mode = scalar_mode(beta)
    sr_seed_mode = (
        2 if isinstance(sr_seed, Tensor) else (1 if rounding_mode == RoundingMode.RS else 0)
    )
    concat_key = tuple(sorted(concat_layout)) if concat_layout else ()
    # The key captures every input the plan build below reads (shapes and
    # strides subsume the majors, the validation asserts, and the fp4/SF shape
    # checks), so a cache hit is exactly a replay of a previously validated
    # call with different data pointers.
    if ag_args is not None:
        # Shard/chunk geometry (see validate_ag_geometry): enforced at launch
        # for every frontend via plan_scheduler_args; checked here too so a
        # bad-geometry cold call fails before the (expensive) compile.
        validate_ag_geometry(A, ag_args, tile_M, cluster_M)
    key = (
        tensor_key(A),
        tensor_key(B),
        tensor_key(D),
        tensor_key(C),
        tensor_key(SFA),
        tensor_key(SFB),
        tensor_key(rowvec_bias),
        tensor_key(colvec_bias),
        tensor_key(cu_seqlens_m),
        tensor_key(cu_seqlens_k),
        A_idx is not None,
        batch_idx_permute is not None,
        tile_count_semaphore is not None,
        A.device,
        tile_M,
        tile_N,
        tile_K,
        cluster_M,
        cluster_N,
        cluster_K,
        pingpong,
        persistent,
        is_dynamic_persistent,
        max_swizzle_size,
        add_to_output,
        rounding_mode,
        use_tma_gather,
        num_warps,
        concat_key,
        alpha_mode,
        beta_mode,
        sr_seed_mode,
        split_k,
        split_k_mode,
        b_kn,
        bs_format_a,
        bs_format_b,
        ag_args is not None,
        tensor_key(SFD),
        tensor_key(SFDCol),
        scalar_mode(sfd_norm_const) if sfd_norm_const is not None else 0,
    )
    plan = _gemm_plan_cache.get(key)
    if plan is None:
        plan = _build_gemm_plan(
            A,
            B,
            D,
            C,
            tile_count_semaphore=tile_count_semaphore,
            tile_M=tile_M,
            tile_N=tile_N,
            cluster_M=cluster_M,
            cluster_N=cluster_N,
            cluster_K=cluster_K,
            tile_K=tile_K,
            pingpong=pingpong,
            persistent=persistent,
            is_dynamic_persistent=is_dynamic_persistent,
            max_swizzle_size=max_swizzle_size,
            rowvec_bias=rowvec_bias,
            colvec_bias=colvec_bias,
            cu_seqlens_m=cu_seqlens_m,
            cu_seqlens_k=cu_seqlens_k,
            A_idx=A_idx,
            batch_idx_permute=batch_idx_permute,
            add_to_output=add_to_output,
            rounding_mode=rounding_mode,
            use_tma_gather=use_tma_gather,
            concat_layout=concat_key,
            num_warps=num_warps,
            SFA=SFA,
            SFB=SFB,
            alpha_mode=alpha_mode,
            beta_mode=beta_mode,
            sr_seed_mode=sr_seed_mode,
            split_k=split_k,
            split_k_mode=split_k_mode,
            b_kn=b_kn,
            bs_format_a=bs_format_a,
            bs_format_b=bs_format_b,
            has_ag=ag_args is not None,
            SFD=SFD,
            sfd_norm_const=sfd_norm_const,
            SFDCol=SFDCol,
        )
        _gemm_plan_cache[key] = plan
    run_gemm_plan(
        plan,
        A,
        B,
        D,
        C,
        tile_count_semaphore=tile_count_semaphore,
        rowvec_bias=rowvec_bias,
        colvec_bias=colvec_bias,
        alpha=alpha,
        beta=beta,
        sr_seed=sr_seed,
        cu_seqlens_m=cu_seqlens_m,
        cu_seqlens_k=cu_seqlens_k,
        A_idx=A_idx,
        batch_idx_permute=batch_idx_permute,
        SFA=SFA,
        SFB=SFB,
        ag_args=ag_args,
        SFD=SFD,
        sfd_norm_const=sfd_norm_const,
        SFDCol=SFDCol,
    )
    return plan


def run_gemm_plan(
    plan: _GemmPlan,
    A: Tensor,
    B: Tensor,
    D: Tensor,
    C: Optional[Tensor],
    *,
    tile_count_semaphore: Optional[Tensor] = None,
    rowvec_bias: Optional[Tensor] = None,
    colvec_bias: Optional[Tensor] = None,
    alpha: float | Tensor = 1.0,
    beta: float | Tensor = 1.0,
    sr_seed: int | Tensor = 0,
    cu_seqlens_m: Optional[Tensor] = None,
    cu_seqlens_k: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,
    batch_idx_permute: Optional[Tensor] = None,
    SFA: Optional[Tensor] = None,
    SFB: Optional[Tensor] = None,
    ag_args: Optional[AllGatherArguments] = None,
    # Quantized-output tensors (validated at plan build; see _build_gemm_plan).
    SFD: Optional[Tensor] = None,
    sfd_norm_const: Optional[float | Tensor] = None,
    SFDCol: Optional[Tensor] = None,
) -> None:
    """Launch a resolved plan: only per-call pointers and scalar values here.

    The tensors must match the metadata the plan was built from (``gemm``
    guarantees that via its key; an outer layer holding the returned plan must
    guarantee it with its own key).
    """
    if plan.use_d_as_c:
        C = D
    # Split-K buffers are per-call allocations, never part of the cached plan:
    # SEPARATE redirects the GEMM output to a fresh f32 partials workspace and runs
    # the reduce kernel (which applies the full epilogue) afterwards; SERIAL/PARALLEL
    # thread a per-tile semaphore and partials workspace through the epilogue args.
    D_gemm, C_gemm = D, C
    split_k_semaphore, split_k_workspace = None, None
    staged_split_k = False
    if plan.split_k > 1:
        staged_split_k = plan.split_k_mode == SplitKMode.SEPARATE
        if staged_split_k:
            split_k_workspace = _staged_split_k_workspace(D, plan.split_k)
            D_gemm, C_gemm = split_k_workspace, None
        else:
            split_k_semaphore, split_k_workspace = _split_k_buffers(
                D,
                plan.split_k_mode,
                plan.tile_M,
                plan.tile_N,
                plan.cluster_M,
                plan.cluster_N,
                plan.is_sm100_family,
            )
    # No permutes here: the kernel was compiled batch-first and rotates
    # (l, x, y) -> (x, y, l) at trace time (GemmBase.permute_batch_last).
    epi_args = plan.epi_static
    if epi_args is None:
        epi_args = GemmDefaultEpiMixin.EpilogueArguments(
            alpha=scalar_arg(alpha, plan.alpha_mode),
            beta=scalar_arg(beta, plan.beta_mode),
            mRowVecBroadcast=rowvec_bias,
            mColVecBroadcast=colvec_bias,
            add_to_output=None,
            rounding_mode=None,
            sr_seed=scalar_arg(sr_seed, plan.sr_seed_mode, dtype=Int32),
            split_k_semaphore=(
                split_k_semaphore.permute(1, 2, 0) if split_k_semaphore is not None else None
            ),
            split_k_workspace=(
                split_k_workspace.permute(3, 1, 2, 0) if split_k_semaphore is not None else None
            ),
            mSFD=SFD,
            sfd_norm_const=scalar_arg(sfd_norm_const, plan.sfd_norm_const_mode),
            mSFDCol=SFDCol,
        )
    scheduler_args = plan_scheduler_args(
        plan, tile_count_semaphore, batch_idx_permute, ag_args, A=A
    )
    varlen_args = make_varlen_args(cu_seqlens_m, cu_seqlens_k, A_idx)

    launch_gemm(plan, A, B, D_gemm, C_gemm, epi_args, scheduler_args, varlen_args, SFA, SFB)

    if staged_split_k:
        _reduce_staged_split_k(
            split_k_workspace,
            D,
            C,
            plan.split_k,
            alpha=alpha,
            beta=beta,
            rowvec_bias=rowvec_bias,
            colvec_bias=colvec_bias,
            add_to_output=plan.add_to_output,
        )


def _build_gemm_plan(
    A,
    B,
    D,
    C,
    *,
    tile_count_semaphore,
    tile_M,
    tile_N,
    cluster_M,
    cluster_N,
    cluster_K,
    tile_K,
    pingpong,
    persistent,
    is_dynamic_persistent,
    max_swizzle_size,
    rowvec_bias,
    colvec_bias,
    cu_seqlens_m,
    cu_seqlens_k,
    A_idx,
    batch_idx_permute,
    add_to_output,
    rounding_mode,
    use_tma_gather,
    concat_layout,  # already normalized to a sorted tuple
    num_warps,
    SFA,
    SFB,
    alpha_mode,
    beta_mode,
    sr_seed_mode,
    split_k,
    split_k_mode,
    b_kn=False,
    bs_format_a=None,
    bs_format_b=None,
    has_ag=False,
    SFD=None,
    sfd_norm_const=None,
    SFDCol=None,
) -> _GemmPlan:
    varlen_m = cu_seqlens_m is not None
    varlen_k = cu_seqlens_k is not None
    varlen = varlen_m or varlen_k
    gather_A = A_idx is not None
    blockscaled = SFA is not None
    assert not (varlen_m and varlen_k), "Only one of cu_seqlens_m and cu_seqlens_k"
    if gather_A:
        assert varlen, "gather_A requires varlen"
        assert cluster_N == 1, "gather_A requires cluster_N=1"
    if has_ag:
        assert not varlen and not gather_A, "AllGather+GEMM requires a dense GEMM"
        assert persistent, "AllGather+GEMM requires the persistent scheduler"
        assert split_k == 1, "AllGather+GEMM does not support split_k yet"
    if add_to_output:
        assert not varlen_m, "Add to output not supported with varlen_m"
    assert split_k >= 1, "split_k must be >= 1"
    if split_k > 1:
        if split_k_mode not in tuple(SplitKMode):
            raise ValueError(
                f"split_k_mode must be a SplitKMode (SERIAL, PARALLEL, or SEPARATE), "
                f"got {split_k_mode!r}"
            )
        split_k_mode = SplitKMode(split_k_mode)
        if varlen or gather_A:
            raise ValueError("split_k requires a dense GEMM (no varlen_m/varlen_k/gather_A)")
        if rounding_mode != RoundingMode.RN:
            raise ValueError("split_k does not support stochastic rounding")
        if blockscaled and split_k_mode == SplitKMode.SEPARATE:
            raise NotImplementedError(
                "block-scaled split_k does not support SEPARATE yet; use SERIAL or PARALLEL"
            )
    if varlen_m:
        assert A.stride(-1) == 1, "varlen_m requires A to be k-major"
        assert D.stride(-1) == 1, "varlen_m requires D to be n-major"
    if varlen_k:
        assert A.stride(-2) == 1, "varlen_k requires A to be m-major"
        assert B.stride(-2) == 1, "varlen_k requires B to be n-major"

    device_capacity = get_device_capacity(A.device)
    assert device_capacity[0] in [8, 9, 10, 11, 12], (
        "Only SM8x, SM90, SM100, SM110, and SM120 are supported"
    )
    if has_ag:
        # SM8x is the one arch without the load-warp gate (no specialized
        # load warp in the cp.async kernel); every TMA arch supports AG.
        assert device_capacity[0] >= 9, "AllGather+GEMM requires SM90+"
    sf_dtype, sf_vec_size = None, None
    a_mma_dtype, b_mma_dtype = None, None
    if blockscaled:
        fmt_a, fmt_b = resolve_blockscaled_formats(bs_format_a, bs_format_b)
        # MMA element types come from the descriptors; identical to the storage
        # dtypes except packed fp6, which crosses the FFI boundary as raw bytes
        # (torch has no fp6 dtype) and is reinterpreted in-kernel.
        a_mma_dtype = fmt_a.to_cutlass_dtype()
        b_mma_dtype = fmt_b.to_cutlass_dtype()
        assert not gather_A, "Blockscaled GEMM does not support gather_A yet"
        assert not concat_layout, "Blockscaled GEMM does not support concat_layout"
        assert tile_K is None, "Blockscaled GEMM derives tile_K from the MMA instruction"
        if varlen_m:
            num_batches = cu_seqlens_m.shape[0] - 1
        elif varlen_k:
            num_batches = cu_seqlens_k.shape[0] - 1
        else:
            num_batches = None
        sf_dtype, sf_vec_size = validate_blockscaled_sf(
            A,
            B,
            SFA,
            SFB,
            device_capacity,
            num_batches=num_batches,
            varlen_k=varlen_k,
            b_kn=b_kn,
            fmt_a=fmt_a,
            fmt_b=fmt_b,
        )
    if split_k > 1 and device_capacity[0] not in [9, 10, 11, 12]:
        raise ValueError("split_k > 1 requires SM90, SM100, SM110, or SM120")
    # Quantized output: SFD (row direction, SF vectors along N) / SFDCol
    # (column direction, SF vectors along M). Blocked-atom layout checks here;
    # the epilogue op asserts the kernel-side constraints at trace time.
    assert SFD is None or SFDCol is None, "SFD and SFDCol are mutually exclusive"
    sfd_dtype, sfd_col_dtype = None, None
    if SFD is None and SFDCol is None:
        sfd_norm_const = None
    # Logical output extents for the SF-coverage checks (fp4 D packs N/2 bytes
    # per row). An undersized SF buffer would NOT fault: the kernel clamps its
    # SF stores to the buffer's padded extents, silently dropping scale bytes.
    if SFD is not None or SFDCol is not None:
        d_m = D.shape[-2]
        d_n = D.shape[-1] * (2 if D.dtype == torch.float4_e2m1fn_x2 else 1)
    if SFDCol is not None:
        assert device_capacity[0] in [10, 11, 12], (
            "Column-direction quantized output (SFDCol) requires SM100/SM110/SM120"
        )
        assert not varlen, "Column-direction SFD does not support varlen yet"
        sfd_col_dtype = torch2cute_dtype_map[SFDCol.dtype]
        vec_col = 32 if SFDCol.dtype == torch.float8_e8m0fnu else 16
        assert SFDCol.dim() == 6 and SFDCol.shape[-3:] == (32, 4, 4), (
            f"SFDCol must be (l, rn, rm_k, 32, 4, 4), got {tuple(SFDCol.shape)}"
        )
        assert SFDCol.stride()[-3:] == (16, 4, 1), (
            f"SFDCol inner (32, 4, 4) atom must be contiguous, got {SFDCol.stride()[-3:]}"
        )
        assert SFDCol.shape[2] % (128 // (4 * vec_col)) == 0, (
            "SFDCol rm_k must pad the M extent to whole 128-row stripes"
        )
        assert SFDCol.shape[1] >= -(-d_n // 128) and SFDCol.shape[2] >= -(-d_m // (4 * vec_col)), (
            f"SFDCol (rn={SFDCol.shape[1]}, rm_k={SFDCol.shape[2]}) does not cover the "
            f"({d_m}, {d_n}) output"
        )
    if SFD is not None:
        assert device_capacity[0] in [10, 11, 12], (
            "Quantized output (SFD) requires SM100/SM110/SM120"
        )
        sfd_dtype = torch2cute_dtype_map[SFD.dtype]
        assert SFD.dim() == 6 and SFD.shape[-3:] == (32, 4, 4), (
            f"SFD must be (l, rm, rk, 32, 4, 4), got {tuple(SFD.shape)}"
        )
        assert SFD.stride()[-3:] == (16, 4, 1), (
            f"SFD inner (32, 4, 4) atom must be contiguous, got strides {SFD.stride()[-3:]}"
        )
        vec = 32 if SFD.dtype == torch.float8_e8m0fnu else 16
        assert SFD.shape[2] >= -(-d_n // (4 * vec)), (
            f"SFD rk={SFD.shape[2]} does not cover N={d_n} (vec {vec})"
        )
        if not varlen_m:
            assert SFD.shape[1] >= -(-d_m // 128), f"SFD rm={SFD.shape[1]} does not cover M={d_m}"
        if varlen_m:
            # One M-padded buffer with tile-aligned per-batch padding (same
            # convention as varlen_m input SFA); rows for batch b start at
            # padded tile offset cu_seqlens_m[b] // 128 + b.
            num_batches = cu_seqlens_m.shape[0] - 1
            total_m = A_idx.shape[0] if A_idx is not None else A.shape[0]
            min_rm = -(-total_m // 128) + (num_batches - 1)
            assert SFD.shape[0] == 1 and SFD.shape[1] >= min_rm, (
                f"varlen_m SFD must be (1, total_padded_rm >= {min_rm}, rk, 32, 4, 4), "
                f"got {tuple(SFD.shape)}"
            )
    if SFD is not None or SFDCol is not None:
        assert not add_to_output, "Quantized output (SFD) is incompatible with add_to_output"
        assert split_k == 1, "Quantized output (SFD/SFDCol) does not support split_k yet"
    # SM120 fp8 rides MmaFP8Op (m16n8k32); k-major-only for fp8 operands is
    # enforced by the shared dtype/layout validation below, same as SM90.
    if use_tma_gather:
        assert device_capacity[0] in [10, 11], "TMA gather currently requires SM100/SM110"
    if is_dynamic_persistent and device_capacity[0] <= 9:
        assert tile_count_semaphore is not None, (
            "Dynamic persistent tile scheduler for SM8x and SM90 requires a semaphore in GMEM"
        )
    use_d_as_c = False
    if device_capacity[0] == 8:
        if add_to_output:
            C = D
            add_to_output = False
            use_d_as_c = True

    batched = A.ndim == 3 or varlen
    if not batched:
        # Dense 2D (unbatched) operands: the kernel appends a trivial batch mode
        # at trace time (GemmBase.permute_batch_last), so hosts skip the
        # .unsqueeze() views. Only the rotating SM90+ paths support this.
        assert B.ndim == 2 and D.ndim == 2 and (C is None or C.ndim == 2), (
            "2D (unbatched) A requires 2D B, D, and C"
        )
        assert device_capacity[0] in [9, 10, 11, 12], "2D (unbatched) operands require SM90+"
        assert split_k == 1, "2D (unbatched) operands do not support split_k"
        assert not concat_layout, "2D (unbatched) operands do not support concat_layout"
        assert batch_idx_permute is None, "2D (unbatched) operands do not support batch_idx_permute"
    if b_kn:
        # B crosses the boundary as (k, n) / (l, k, n); the kernel transposes it
        # to (n, k, l) at trace time (GemmBase.rotate_batch_last), saving the
        # caller a per-call .mT view. Same support envelope as the batch append.
        assert not varlen, "b_kn does not support varlen (pass B as (n, total_k))"
        assert device_capacity[0] in [9, 10, 11, 12], "b_kn requires SM90+"
        assert not concat_layout, "b_kn does not support concat_layout"

    a_major, b_major, d_major, c_major = get_majors(A, B, D, C)
    if b_kn:
        # Majors are logical (n, k) labels: with B stored (k, n), a contiguous
        # last dim means n-major.
        b_major = "n" if B.stride(-1) == 1 else "k"
    a_dtype, b_dtype, d_dtype, c_dtype = get_dtypes(A, B, D, C)

    # SEPARATE split-K routes every epi operand (alpha, beta*C, bias, add_to_output)
    # to the reduce kernel; the GEMM itself writes raw f32 partials into a per-call
    # workspace (see _staged_split_k_workspace) with D's majorness, so the compiled
    # signature sees an f32 D and no epilogue extras.
    staged_split_k = split_k > 1 and split_k_mode == SplitKMode.SEPARATE
    rowvec_gemm, colvec_gemm = rowvec_bias, colvec_bias
    gemm_add_to_output = add_to_output
    if staged_split_k:
        d_dtype = Float32
        c_dtype, c_major = None, None
        alpha_mode, beta_mode, sr_seed_mode = 0, 0, 0
        rowvec_gemm, colvec_gemm = None, None
        gemm_add_to_output = False

    colvec_ndim = colvec_gemm.ndim if colvec_gemm is not None else 0
    sfd_norm_const_mode = 0 if sfd_norm_const is None else scalar_mode(sfd_norm_const)
    tile_shape_mnk = (tile_M, tile_N) if tile_K is None else (tile_M, tile_N, tile_K)
    compiled_fn = _compile_gemm(
        a_dtype,
        b_dtype,
        d_dtype,
        c_dtype,
        a_major,
        b_major,
        d_major,
        c_major,
        tile_shape_mnk,
        (cluster_M, cluster_N, cluster_K),
        pingpong,
        persistent,
        is_dynamic_persistent,
        torch2cute_dtype_map[rowvec_gemm.dtype] if rowvec_gemm is not None else None,
        torch2cute_dtype_map[colvec_gemm.dtype] if colvec_gemm is not None else None,
        colvec_ndim,
        alpha_mode,
        beta_mode,
        gemm_add_to_output,
        concat_layout,
        varlen_m,
        varlen_k,
        gather_A,
        use_tma_gather,
        batch_idx_permute is not None,
        device_capacity,
        rounding_mode,
        sr_seed_mode,
        num_warps,
        sf_dtype,
        sf_vec_size,
        split_k,
        split_k_mode,
        batched,
        b_kn,
        SFA.ndim == 6 if blockscaled else True,
        sfd_dtype,
        sfd_norm_const_mode,
        sfd_col_dtype,
        a_mma_dtype,
        b_mma_dtype,
        has_ag,
    )

    cluster_size = cluster_M * cluster_N * cluster_K
    max_active_clusters = (
        get_max_active_clusters(cluster_size, device_capacity=device_capacity) if persistent else 0
    )
    scheduler_uses_semaphore = is_dynamic_persistent and device_capacity[0] <= 9

    # SERIAL/PARALLEL split-K passes per-call semaphore/workspace tensors through the
    # epilogue args, so those can never be static; SEPARATE always is (every epi
    # operand is routed to the reduce kernel, the modes above were zeroed).
    # SFD/SFDCol are per-call tensors, so they disable the static template too.
    epi_static = None
    if (
        alpha_mode == 0
        and beta_mode == 0
        and sr_seed_mode == 0
        and rowvec_gemm is None
        and colvec_gemm is None
        and (split_k == 1 or staged_split_k)
        and SFD is None
        and SFDCol is None
    ):
        epi_static = GemmDefaultEpiMixin.EpilogueArguments(
            alpha=None,
            beta=None,
            mRowVecBroadcast=None,
            mColVecBroadcast=None,
            add_to_output=None,
            rounding_mode=None,
            sr_seed=None,
        )
    scheduler_static = None
    if not scheduler_uses_semaphore and batch_idx_permute is None and not has_ag:
        scheduler_static = make_scheduler_args(max_active_clusters, max_swizzle_size, None, None)

    return _GemmPlan(
        compiled_fn=compiled_fn,
        is_sm100_family=device_capacity[0] in [10, 11],
        use_d_as_c=use_d_as_c,
        alpha_mode=alpha_mode,
        beta_mode=beta_mode,
        sr_seed_mode=sr_seed_mode,
        sfd_norm_const_mode=sfd_norm_const_mode,
        epi_static=epi_static,
        scheduler_static=scheduler_static,
        max_active_clusters=max_active_clusters,
        max_swizzle_size=max_swizzle_size,
        scheduler_uses_semaphore=scheduler_uses_semaphore,
        split_k=split_k,
        split_k_mode=split_k_mode,  # already SplitKMode-normalized above when split_k > 1
        add_to_output=add_to_output,
        tile_M=tile_M,
        tile_N=tile_N,
        cluster_M=cluster_M,
        cluster_N=cluster_N,
    )
