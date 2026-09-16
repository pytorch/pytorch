# Copyright (c) 2025, Tri Dao.
# Shared utilities for TVM-FFI GEMM compilation.

from functools import partial

import torch

import cutlass.cute as cute
from cutlass import Int32, Float32, Float4E2M1FN
from cutlass.cute.runtime import make_ptr

from torch._vendor.quack.compile_utils import make_fake_tensor as fake_tensor
from torch._vendor.quack.compile_utils import div_for_dtype, fake_batched  # noqa: F401  (re-exported)
from torch._vendor.quack.gemm_config import SplitKMode
from torch._vendor.quack.cute_dsl_utils import get_compile_target_capacity, torch2cute_dtype_map
from torch._vendor.quack.tile_scheduler import AgSchedulerArguments, TileSchedulerOptions
from torch._vendor.quack.varlen_utils import VarlenArguments


def resolve_blockscaled_formats(bs_format_a, bs_format_b):
    """Resolve a blockscaled GEMM's per-operand format names into descriptors,
    checking presence and hardware pair legality. Single owner of this step for
    both the gemm_interface path and direct kernel-layer callers
    (quack.gemm.gemm / epilogue-mod .gemm)."""
    if bs_format_a is None or bs_format_b is None:
        raise ValueError(
            "blockscaled GEMM requires bs_format_a and bs_format_b (BlockScaledFormat "
            "names, e.g. 'mxfp8_e4m3'); pass BlockScaledOperand operands to the "
            "torch._vendor.quack.gemm_interface entry points, or set them explicitly here"
        )
    # Lazy import: blockscaled/__init__ pulls blockscaled/utils, which imports
    # this module - a top-level import here would be circular.
    from torch._vendor.quack.blockscaled.operand import BlockScaledFormat, mma_kind_for_pair

    fmt_a = BlockScaledFormat.from_name(bs_format_a)
    fmt_b = BlockScaledFormat.from_name(bs_format_b)
    mma_kind_for_pair(fmt_a, fmt_b)
    return fmt_a, fmt_b


def _validate_tma_unpack_alignment(name, tensor, format_name):
    """Validate the byte-addressing contract of U4/U6 TMA unpack tensor maps."""
    alignment = 32
    if tensor.data_ptr() % alignment != 0:
        raise ValueError(
            f"{name} ({format_name}) TMA-unpack base address must be {alignment}-byte aligned"
        )
    elem_bytes = tensor.element_size()
    for dim, stride in enumerate(tensor.stride()):
        stride_bytes = stride * elem_bytes
        if stride_bytes != 1 and stride_bytes % alignment != 0:
            raise ValueError(
                f"{name} ({format_name}) TMA-unpack non-unit byte strides must be "
                f"{alignment}-byte aligned, got stride[{dim}]={stride_bytes} bytes"
            )


def _validate_tma_unpack_operands(A, B):
    """Per-launch guard for cached blockscaled plans, which do not key on pointers."""
    fp4_dtype = torch.float4_e2m1fn_x2
    subbyte_storage_dtypes = {fp4_dtype, torch.uint8}
    both_fp4 = A.dtype == fp4_dtype and B.dtype == fp4_dtype
    if both_fp4:
        return
    for name, tensor in (("A", A), ("B", B)):
        if tensor.dtype in subbyte_storage_dtypes:
            _validate_tma_unpack_alignment(name, tensor, str(tensor.dtype))


def validate_blockscaled_sf(
    A, B, SFA, SFB, device_capacity, num_batches=None, varlen_k=False, b_kn=False, *, fmt_a, fmt_b
):
    """Validate blockscaled scale factors against kernel-layout operands.

    ``fmt_a`` / ``fmt_b`` are the (independent) A / B
    :class:`quack.blockscaled.operand.BlockScaledFormat` descriptors - the single
    source of scale vec size / scale dtype / element packing per operand. Nothing
    here may re-derive format properties from tensor dtypes. Pair LEGALITY
    (hardware representability) was already checked upstream via
    ``mma_kind_for_pair``; whether a legal (A, B, D) dtype combination is
    IMPLEMENTED is enforced per-architecture by the gemm_smXXX kernel classes
    (e.g. GemmSm100's blockscaled setup assert) - this function validates only
    layout/consistency.

    A is (l, m, k_storage) and B is (l, n, k_storage), where k_storage is the
    format's storage K extent (fp8: k; fp4x2: k/2; packed fp6: 3k/4 bytes);
    SFA/SFB are (l, rm/rn, rk, 32, 4, 4) with the inner (32, 4, 4) block
    contiguous (strides (16, 4, 1) - one 512 B atom per 128 rows x 4 K-blocks).

    When num_batches is not None and varlen_k is False (varlen_m), A is
    (total_m, k) and SFA must be a single M-padded buffer (tile-aligned
    per-batch padding) (1, total_padded_rm, rk, 32, 4, 4) with
    total_padded_rm >= ceil(total_m/128) + (num_batches - 1) — the bound from
    AI/varlen_blockscaled_sf_layout.md that suffices for any per-batch split
    of total_m. SFB stays per-batch: (num_batches, rn, rk, 32, 4, 4).

    When varlen_k, A is (m, total_k) m-major and B is (n, total_k) n-major
    (MXFP8 only — fp4 operands must be K-major), and BOTH SF buffers are
    K-padded with tile-aligned per-batch padding:
    (1, rm/rn, total_padded_rk, 32, 4, 4) with
    total_padded_rk >= ceil(total_k/128) + (num_batches - 1).
    SF pad bytes inside each batch's last atom are loaded by the kernel but
    never consumed: the mma loop skips the MMA instructions for pad k-blocks
    (one instruction per SF block for mxfp8; see GemmSm100.mma), so the pad
    may be arbitrary bytes — torch.empty buffers are fine.
    Returns (sf_dtype, sf_vec_size) as (cutlass dtype, int).
    """
    varlen_m = num_batches is not None and not varlen_k
    assert not varlen_k or num_batches is not None, "varlen_k requires num_batches"
    assert SFB is not None, "SFA and SFB must be provided together"
    assert device_capacity[0] in [10, 11, 12], "Blockscaled GEMM requires SM100/SM110/SM120"
    # SM120 warp-MMA blockscaled: same-dtype MXFP8 (MmaMXF8Op), same-dtype
    # fp4 (packed kind::mxf4 / mxf4nvf4), or any kind::mxf8f6f4 pair with
    # independent a/b dtypes — fp8/fp6/fp4 mixes incl. e4m3 x e5m2
    # (sub-byte sides of mixed pairs ride padded 16U4_ALIGN8B /
    # 16U6_ALIGN16B tensormaps + ldsm.b4x16_p64/b6x16_p32 unpack).
    # MN-major operands (incl. varlen_k's m-major A / n-major B) are fp8-only
    # on SM120: both sides ride the transposing b8 ldmatrix (m16n16.trans.b8);
    # B needs a hand-built TV layout (GemmSm120._nmajor_b_tiled_copy).
    # Per-instruction the scale config is shared: every legal pair has matching
    # scale dtype and vec size (nvfp4 only pairs with itself; all other formats
    # are e8m0 / vec 32).
    assert fmt_a.scale_dtype == fmt_b.scale_dtype and fmt_a.sf_vec_size == fmt_b.sf_vec_size
    assert SFA.dtype == fmt_a.scale_dtype and SFB.dtype == fmt_b.scale_dtype, (
        f"SF dtype mismatch: {SFA.dtype} / {SFB.dtype} vs "
        f"{fmt_a.name}/{fmt_b.name} scales {fmt_a.scale_dtype}/{fmt_b.scale_dtype}"
    )
    assert A.dtype == fmt_a.qdata_dtype, (
        f"A dtype mismatch: {A.dtype} vs {fmt_a.name} qdata {fmt_a.qdata_dtype}"
    )
    assert B.dtype == fmt_b.qdata_dtype, (
        f"B dtype mismatch: {B.dtype} vs {fmt_b.name} qdata {fmt_b.qdata_dtype}"
    )
    sf_vec_size = fmt_a.sf_vec_size
    sf_dtype = torch2cute_dtype_map[fmt_a.scale_dtype]
    # Operand shapes carry the storage K extent (fp4x2: K/2; packed fp6: 3K/4
    # bytes) while dlpack presents the logical extent to the kernel, so validate
    # rk against logical K, and cross-check that A and B agree on it (their
    # packings may differ). Under b_kn, B crosses as (k, n[, l]).
    b_storage_k = B.shape[-2] if b_kn else B.shape[-1]
    k_logical = fmt_a.logical_k(A.shape[-1])
    k_logical_b = fmt_b.logical_k(b_storage_k)
    assert k_logical == k_logical_b, (
        f"logical K mismatch: A storage K {A.shape[-1]} ({fmt_a.name}) => {k_logical} "
        f"vs B storage K {b_storage_k} ({fmt_b.name}) => {k_logical_b}"
    )
    # Packed fp6 is TMA-unpacked through the 16U6_ALIGN16B tensormap, whose
    # granule is 128 logical elements (96 bytes): fail here with a clear message
    # rather than at the compiled arg-spec binding. (Mixed pairs with packed fp4
    # need K % 128 too - already enforced by the arg spec's k divisibility.)
    for fmt in (fmt_a, fmt_b):
        if fmt.elem_bits == 6:
            assert k_logical % 128 == 0, (
                f"{fmt.name} operands require logical K divisible by 128 "
                f"(TMA unpack tensormap granule), got K={k_logical}"
            )
    # Under kind::mxf8f6f4, packed fp4/fp6 operands use U4/U6 TMA unpack into
    # byte-container SMEM. Validate addressing after logical shape/granule checks
    # so malformed K reports the format constraint rather than a derived row pitch.
    # Both-fp4 runs kind::mxf4nvf4 and keeps the ordinary 16-byte contract.
    both_fp4 = fmt_a.elem_bits == 4 and fmt_b.elem_bits == 4
    for name, tensor, fmt in (("A", A, fmt_a), ("B", B, fmt_b)):
        if fmt.elem_bits < 8 and not both_fp4:
            _validate_tma_unpack_alignment(name, tensor, fmt.name)
    rk = (k_logical + 4 * sf_vec_size - 1) // (4 * sf_vec_size)
    if varlen_k:
        assert not fmt_a.is_packed and not fmt_b.is_packed, (
            "varlen_k blockscaled supports 8-bit element formats only: packed "
            "sub-byte operands (fp4/fp6) must be K-major, but varlen_k requires "
            "m-major A / n-major B"
        )
        assert A.ndim == 2 and B.ndim == 2, (
            f"varlen_k expects A (m, total_k) and B (n, total_k), "
            f"got shapes {tuple(A.shape)} / {tuple(B.shape)}"
        )
        # rk here = ceil(total_k/128); K-padded buffers need one extra atom
        # column per additional batch.
        min_rk = rk + (num_batches - 1)
        for name, SF, mn in (("SFA", SFA, A.shape[0]), ("SFB", SFB, B.shape[0])):
            r_mn = (mn + 127) // 128
            assert SF.shape[0] == 1 and SF.shape[1] == r_mn and tuple(SF.shape[3:]) == (32, 4, 4), (
                f"{name} shape {tuple(SF.shape)} != (1, {r_mn}, total_padded_rk, 32, 4, 4)"
            )
            assert SF.shape[2] >= min_rk, (
                f"{name} padded rk {SF.shape[2]} < ceil(total_k/128) + (L-1) = {min_rk}"
            )
        shapes = []
    elif varlen_m:
        assert A.ndim == 2, f"varlen_m expects A as (total_m, k), got shape {tuple(A.shape)}"
        assert B.shape[0] == num_batches, (
            f"B batch dim {B.shape[0]} != len(cu_seqlens_m) - 1 = {num_batches}"
        )
        min_rm = (A.shape[0] + 127) // 128 + (num_batches - 1)
        assert SFA.shape[0] == 1 and tuple(SFA.shape[2:]) == (rk, 32, 4, 4), (
            f"SFA shape {tuple(SFA.shape)} != (1, total_padded_rm, {rk}, 32, 4, 4)"
        )
        assert SFA.shape[1] >= min_rm, (
            f"SFA padded rm {SFA.shape[1]} < ceil(total_m/128) + (L-1) = {min_rm}"
        )
        shapes = [("SFB", SFB, (num_batches, (B.shape[-2] + 127) // 128, rk, 32, 4, 4))]
    else:
        # Dense: 2D operands may carry unbatched 5-D SFs (the kernel prepends
        # the trivial batch mode at trace time) or single-batch 6-D ones.
        l = A.shape[0] if A.ndim == 3 else 1
        n = B.shape[-1] if b_kn else B.shape[-2]
        shapes = []
        for name, SF, mn in (("SFA", SFA, A.shape[-2]), ("SFB", SFB, n)):
            core = ((mn + 127) // 128, rk, 32, 4, 4)
            if SF.ndim == 5:
                assert A.ndim == 2, (
                    f"{name}: unbatched 5-D scale factors require 2D operands, A is {A.ndim}D"
                )
                shapes.append((name, SF, core))
            else:
                shapes.append((name, SF, (l, *core)))
    for name, SF, expected in shapes:
        assert tuple(SF.shape) == expected, f"{name} shape {tuple(SF.shape)} != {expected}"
    for name, SF in (("SFA", SFA), ("SFB", SFB)):
        assert SF.stride()[-3:] == (16, 4, 1), (
            f"{name}: inner (32, 4, 4) block must be contiguous with strides (16, 4, 1), "
            f"got {SF.stride()[-3:]}"
        )
    return sf_dtype, sf_vec_size


def get_major(t, dim0, dim1):
    """Major of the trailing two logical dims: (l, x, y) or (x, y) — batch first.

    Equivalent to the old ``stride(1) == 1`` check on a (x, y, l)-permuted
    view: batched tensors now stay (l, x, y) on the host (the kernel rotates
    them at trace time, see GemmBase.permute_batch_last), and for the 2D
    varlen-flattened case ``stride(-1)`` is the same ``stride(1)``.
    """
    return dim1 if t.stride(-1) == 1 else dim0


def get_majors(A, B, D, C):
    a_major = get_major(A, "m", "k")
    b_major = get_major(B, "n", "k")
    d_major = get_major(D, "m", "n")
    c_major = get_major(C, "m", "n") if C is not None else None
    return a_major, b_major, d_major, c_major


def get_dtypes(A, B, D, C):
    a_dtype = torch2cute_dtype_map[A.dtype]
    b_dtype = torch2cute_dtype_map[B.dtype]
    d_dtype = torch2cute_dtype_map[D.dtype]
    c_dtype = torch2cute_dtype_map[C.dtype] if C is not None else None
    return a_dtype, b_dtype, d_dtype, c_dtype


def make_scheduler_args(
    max_active_clusters,
    max_swizzle_size,
    tile_count_semaphore,
    batch_idx_permute=None,
    ag_args=None,  # quack.gemm.AllGatherArguments or None
):
    return TileSchedulerOptions(
        max_active_clusters=Int32(max_active_clusters),
        raster_order=None,
        max_swizzle_size=max_swizzle_size,
        tile_count_semaphore=(
            tile_count_semaphore.data_ptr() if tile_count_semaphore is not None else None
        ),
        ag=(
            AgSchedulerArguments(
                flags=ag_args.flags,
                epoch=ag_args.epoch,
                num_shards=Int32(ag_args.num_shards),
                first_shard=Int32(ag_args.first_shard),
                num_chunks=Int32(ag_args.num_chunks),
            )
            if ag_args is not None
            else None
        ),
        batch_idx_permute=batch_idx_permute,
    )


def make_fake_scheduler_args(has_semaphore, has_batch_idx_permute, l_sym, has_ag=False):
    return TileSchedulerOptions(
        max_active_clusters=Int32(1),
        max_swizzle_size=Int32(8),
        tile_count_semaphore=(
            make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=4) if has_semaphore else None
        ),
        ag=(
            AgSchedulerArguments(
                flags=fake_tensor(Int32, (cute.sym_int(),), leading_dim=0, divisibility=4),
                # divisibility=1 elem => assumed_align=4 B: the gate reads one
                # scalar int32, and the runtime tensor is a 1-elem VIEW that
                # may sit 4 bytes off its allocation base (parity-1 slot).
                epoch=fake_tensor(Int32, (cute.sym_int(),), leading_dim=0, divisibility=1),
                num_shards=Int32(1),
                first_shard=Int32(0),
                num_chunks=Int32(1),
            )
            if has_ag
            else None
        ),
        batch_idx_permute=(
            fake_tensor(Int32, (l_sym,), leading_dim=0, divisibility=4)
            if has_batch_idx_permute
            else None
        ),
    )


def make_varlen_args(cu_seqlens_m, cu_seqlens_k, A_idx, cu_tiles_m=None):
    if cu_seqlens_m is None and cu_seqlens_k is None:
        return None
    return VarlenArguments(
        mCuSeqlensM=cu_seqlens_m,
        mCuSeqlensK=cu_seqlens_k,
        mAIdx=A_idx,
        mCuTilesM=cu_tiles_m,
    )


def compute_cu_tiles_m(cu_seqlens_m, tile_m_cta):
    """Per-sequence M-tile prefix for varlen M-fold reduce sinks:
    cumsum of ceil(seqlen / tile_m_cta), zero-padded, (num_seqs + 1,) int32.
    Device-only ops (graph-safe); recomputed per launch from the live
    cu_seqlens values."""
    import torch

    seqlens = cu_seqlens_m[1:] - cu_seqlens_m[:-1]
    tiles = torch.div(seqlens + (tile_m_cta - 1), tile_m_cta, rounding_mode="floor")
    return torch.cat([tiles.new_zeros(1), tiles.cumsum(0, dtype=torch.int32)])


def make_fake_varlen_args(varlen_m, varlen_k, gather_A, aidx_len, has_cu_tiles_m=False):
    if not varlen_m and not varlen_k:
        return None
    num_seqlens = cute.sym_int()
    return VarlenArguments(
        mCuSeqlensM=(
            fake_tensor(Int32, (num_seqlens,), leading_dim=0, divisibility=4) if varlen_m else None
        ),
        mCuSeqlensK=(
            fake_tensor(Int32, (num_seqlens,), leading_dim=0, divisibility=4) if varlen_k else None
        ),
        mAIdx=(
            fake_tensor(Int32, (aidx_len,), leading_dim=0, divisibility=4) if gather_A else None
        ),
        # cu_tiles_m has num_seqs + 1 entries, same as cu_seqlens_m — share the sym.
        mCuTilesM=(
            fake_tensor(Int32, (num_seqlens,), leading_dim=0, divisibility=4)
            if has_cu_tiles_m
            else None
        ),
    )


# ---------------------------------------------------------------------------
# Launch-overhead design: three binding tiers
# ---------------------------------------------------------------------------
#
# Every fact about a GEMM call is handled at the EARLIEST tier that can know
# it. That single rule generates the whole host-side architecture; deviations
# below are deliberate and documented so they don't get "fixed" back and forth.
#
# 1. COMPILE TIME (cute.jit trace -> tvm-ffi function, cached by @jit_cache):
#    dtypes, ranks, majors, tile/cluster config, epilogue structure — and
#    every STATIC layout relabel. The FFI signature is the caller's natural
#    tensor form; the trace rearranges it for the kernel:
#      * batch-first (l, x, y) -> (x, y, l) rotation (GemmBase.rotate_batch_last)
#      * dense rank-2 operands get a trivial (1, stride-0) batch mode appended
#      * b_kn: B crosses as (k, n[, l]) and is transposed to (n, k, l)
#      * unbatched 5-D scale factors get the batch mode prepended (SM100)
#    Rule of thumb: NEVER add a torch view (.mT/.unsqueeze/.permute) to a warm
#    path — each costs ~1-1.5us of dispatcher overhead per call. If the relabel
#    is metadata-static, do it at trace time behind a compile flag instead.
#    (Views that survive are semantic or fallbacks: swap_ab's out.mT, the
#    non-2D rank promotion, varlen's flattened forms.)
#
# 2. PLAN TIME (first call per metadata signature; immutable NamedTuple in a
#    per-entry-point dict): validation asserts, major/dtype derivation, config
#    selection (heuristic or autotuner), workspace/output recipes, static arg
#    templates, and WHICH compiled function. The plan key is built from
#    tensor_key() of every tensor argument plus every scalar knob the build
#    reads — shapes/strides subsume the majors and the validation, so a key
#    hit is exactly a replay of a previously validated call with different
#    data pointers. Scalar epilogue operands enter the key as their MODE
#    (scalar_mode: absent / host constant / device pointer) because the modes
#    select different compiled epilogues, while the VALUES stay per-call.
#
#    Plans COMPOSE BY REFERENCE: an outer layer (gemm_interface) holding a
#    resolved plan also captures the dispatch layer's plan and calls the
#    dispatch run_*_plan() directly, so a warm call pays exactly ONE key.
#    THE INVARIANT: an outer key must subsume every input of the captured
#    inner plan's key (that's why the interface key carries alpha/sr modes).
#    Two plan LAYERS are correct, not residual: they cache decisions at
#    different altitudes with different key spaces — during autotuning one
#    interface signature legitimately exercises N dispatch plans (one per
#    candidate config); collapsing the layers would re-resolve kernels on
#    every config switch.
#
# 3. CALL TIME (every call): data pointers, scalar values, stream. The warm
#    path is: one key -> dict hit -> cheap routing -> run_*_plan (epi scalars,
#    static scheduler template, FFI). ~8us of that is the FFI + cudaLaunch
#    floor; everything else is ~1-2us items.
#
# Deliberate deviations / rejected alternatives (do not revisit):
#  * The execute helper re-derives routing flags (b_kn/dense_2d/swap) per call
#    (~0.5us) instead of storing them in the plan: one routing body shared by
#    the cold and warm paths means the two can never drift. Threading flags
#    through return chains was tried on paper and costs more than it saves.
#  * Views cannot be cached (they freeze data pointers) — don't try.
#  * Per-call EpilogueArguments construction is REQUIRED (it carries pointers);
#    only an all-absent epilogue may cache a static instance (epi_static).
#  * Per-call torch.zeros/empty scratch (split-K, SM90 semaphore) is the
#    stream-correct design for free (the caching allocator is stream-aware).
#    Plan-owned scratch would need kernel self-reset protocols + per-stream
#    slabs; that is the (deferred) CUDA-graphs prerequisite, not a cleanup.
#  * An explicit zero-key handle API ("plan = gemm.plan(...); plan(...)") was
#    rejected as user-hostile; implicit metadata keys are the contract.
#
# The helpers below rely only on the common plan field names (compiled_fn,
# is_sm100_family, max_active_clusters, max_swizzle_size,
# scheduler_uses_semaphore, scheduler_static), so each entry point defines its
# own plan NamedTuple with whatever extra fields it needs.
# ---------------------------------------------------------------------------


def tensor_key(t):
    """Metadata key of one tensor for the plan cache: everything a plan build
    reads from it except the data pointer."""
    return (t.dtype, tuple(t.shape), t.stride()) if t is not None else None


def scalar_mode(scalar, neutral=1.0):
    """Compile-time mode of an epilogue scalar: 0 = absent (neutral value, the
    epilogue op is compiled out), 1 = host constant, 2 = device pointer. Part
    of every plan key — the modes select different compiled epilogues."""
    return 2 if isinstance(scalar, torch.Tensor) else (1 if scalar != neutral else 0)


def scalar_arg(scalar, mode, dtype=Float32):
    """Per-call epilogue scalar matching the compiled signature: mode 0 = absent,
    1 = host constant, 2 = device pointer."""
    if mode == 0:
        return None
    elif mode == 1:
        return dtype(scalar)
    else:
        return scalar.data_ptr()


def validate_ag_geometry(A, ag_args, tile_M, cluster_M):
    """The ONE geometry fact the AG transport and kernel share: shard and
    arrival-chunk boundaries must land on whole scheduler clusters along M,
    else a tile spans two shards/chunks and gating it on one flag is unsound.
    Enforced at the launch choke point every frontend goes through
    (plan_scheduler_args — the plan knows the exact tile config); gemm() also
    calls it pre-compile to fail fast on the cold path."""
    m_rows = A.shape[-2]
    assert m_rows % ag_args.num_shards == 0, (
        f"AG+GEMM: M ({m_rows}) must divide into num_shards ({ag_args.num_shards})"
    )
    shard_rows = m_rows // ag_args.num_shards
    assert shard_rows % (tile_M * cluster_M * ag_args.num_chunks) == 0, (
        f"AG+GEMM: shard rows ({shard_rows}) must be a multiple of tile_M * cluster_M * "
        f"arrival_chunks ({tile_M} * {cluster_M} * {ag_args.num_chunks}) so shard/chunk "
        f"boundaries land on whole scheduler clusters"
    )


def plan_scheduler_args(plan, tile_count_semaphore, batch_idx_permute=None, ag_args=None, A=None):
    """Per-call TileSchedulerOptions for a cached plan.

    Must mirror make_fake_scheduler_args in the variant's _compile_* function:
    only the SM8x/SM90 dynamic scheduler consumes the semaphore (SM100 uses CLC
    instead), so when the compiled signature has None there the semaphore the
    caller passed is dropped rather than forwarded.

    AG+GEMM callers must pass A (the gathered kernel-A tensor) so the shard
    geometry is validated against the plan's tile config on every launch —
    frontends can't forget it by construction.
    """
    if ag_args is not None:
        assert A is not None, "plan_scheduler_args: ag_args requires A for geometry validation"
        validate_ag_geometry(A, ag_args, plan.tile_M, plan.cluster_M)
    if plan.scheduler_static is not None:
        return plan.scheduler_static
    return make_scheduler_args(
        plan.max_active_clusters,
        plan.max_swizzle_size,
        tile_count_semaphore if plan.scheduler_uses_semaphore else None,
        batch_idx_permute,
        ag_args=ag_args,
    )


def launch_gemm(plan, A, B, D, C, epi_args, scheduler_args, varlen_args, SFA=None, SFB=None):
    """Invoke the compiled kernel. Kernel signatures uniformly take trailing
    (SFA, SFB) — None unless blockscaled — and the compiled TVM-FFI arg spec
    bakes the full declared arity, so they are always passed. A layout-owning
    transform's A is a TransformAOperand bundle — one slot, the host never
    unpacks it."""
    if SFA is not None:
        _validate_tma_unpack_operands(A, B)
    plan.compiled_fn(A, B, D, C, epi_args, scheduler_args, varlen_args, SFA, SFB)


def make_fake_gemm_tensors(
    a_dtype,
    b_dtype,
    d_dtype,
    c_dtype,
    a_major,
    b_major,
    d_major,
    c_major,
    varlen_m=False,
    varlen_k=False,
    gather_A=False,
    batched=True,
    b_kn=False,
    swap_ab=False,
    packed_cd=None,
    a_mma_dtype=None,
    b_mma_dtype=None,
):
    """Create fake tensors for mA, mB, mD, mC with shared sym_ints.
    Pass dtype=None to get None for that tensor (e.g. optional C).
    Returns (mA, mB, mD, mC, m, n, k, l).
    When varlen_m, m is total_m (flattened M of D/C). When varlen_k, k is total_k.

    ``packed_cd`` ("n" | "m", dgated packed-native): D/C cross in their RAW
    16-bit dtype with the packed extent DOUBLE the f32 view's — it gets its
    OWN sym (no 2n = 2*n compile-time link); the trace derives the f32 view
    by halving (GemmBase._recast_packed_cd).

    3D tensors are built batch-first (l, x, y); see :func:`fake_batched`.
    ``batched=False`` (dense only) builds 2D (x, y) operand fakes: the kernel
    appends a trivial batch mode at trace time (GemmBase.permute_batch_last),
    so hosts pass unbatched torch tensors without .unsqueeze() views.
    ``b_kn`` builds mB as (l, k, n) / (k, n) — B crosses the boundary in the
    caller's (K, N) orientation and the kernel transposes it to (n, k, l) at
    trace time (GemmBase.rotate_batch_last), saving a .mT view. Supported for
    dense and varlen_m (B is batched rank-3 there, so the same trace-time
    select applies); varlen_k keeps the host relabel (its B is the rank-2
    flattened operand, outside rotate_batch_last's transpose path).
    ``a/b_mma_dtype``: blockscaled MMA element types when they differ from the
    storage dtypes - a width-6 MMA dtype marks a packed-fp6 operand whose
    storage tensor is raw bytes.
    """
    assert batched or not (varlen_m or varlen_k), "varlen operands are 2D already"
    assert not (b_kn and varlen_k), "b_kn does not support varlen_k"
    assert not (swap_ab and (varlen_m or varlen_k or b_kn)), "swap_ab: dense, b_kn folded in"
    assert packed_cd is None or (not swap_ab and not varlen_k), "packed_cd: dense/varlen_m only"
    assert packed_cd != "m" or not varlen_m, "varlen packed C/D is n-major"
    a_leading = 1 if a_major == "k" else 0
    b_leading = 1 if b_major == "k" else 0
    d_leading = 1 if d_major == "n" else 0
    c_leading = 1 if c_major == "n" else 0
    m, l = cute.sym_int(), cute.sym_int()
    # Sub-byte (fp4) D is n-major with a statically packed contiguous extent.
    n_div = div_for_dtype(d_dtype) if d_dtype is not None and d_dtype.width < 8 else 1
    n = cute.sym_int(divisibility=n_div)
    a_packed_f6 = a_mma_dtype is not None and a_mma_dtype.width == 6
    b_packed_f6 = b_mma_dtype is not None and b_mma_dtype.width == 6
    # Sub-byte tensors need their contiguous extent statically divisible; sub-byte
    # operands are k-major, so mark k. Both-packed-fp4 runs kind::mxf4nvf4 with
    # packed smem (16-byte rule: 32 elements). A sub-byte operand in any other
    # pair is TMA-unpacked under kind::mxf8f6f4: logical K must be a multiple of
    # 128, and the FFI signature must enforce the unpack tensor map's 32-byte
    # base/non-unit-stride contract.
    both_fp4 = a_dtype is Float4E2M1FN and b_dtype is Float4E2M1FN
    a_unpack = (a_dtype.width < 8 or a_packed_f6) and not both_fp4
    b_unpack = (b_dtype.width < 8 or b_packed_f6) and not both_fp4
    if (
        any(dt.width < 8 for dt in (a_dtype, b_dtype)) or a_packed_f6 or b_packed_f6
    ) and not both_fp4:
        k_div = 128
    else:
        k_div = max((div_for_dtype(dt) for dt in (a_dtype, b_dtype) if dt.width < 8), default=1)
    k = cute.sym_int(divisibility=k_div)
    # Packed-fp6 operands cross the FFI boundary as raw bytes (torch has no fp6
    # dtype), so their K extent is 3k/4 bytes: an independent sym - the 4/3
    # relation between shared syms is not expressible in the arg spec; logical-K
    # consistency is validated host-side. 96-byte divisibility == 128 fp6
    # elements (the unpack-tensormap granule), and rows of whole 96-byte groups
    # also satisfy the tensormap's 32-byte stride rule.
    a_k_sym = cute.sym_int(divisibility=96) if a_packed_f6 else k
    b_k_sym = cute.sym_int(divisibility=96) if b_packed_f6 else k
    div_a = 256 // a_dtype.width if a_unpack else div_for_dtype(a_dtype)
    div_b = 256 // b_dtype.width if b_unpack else div_for_dtype(b_dtype)
    div_d = div_for_dtype(d_dtype) if d_dtype is not None else 1
    div_c = div_for_dtype(c_dtype) if c_dtype is not None else 1
    # Doubled packed extent for raw 16-bit D/C — its own independent sym.
    pd = cute.sym_int() if packed_cd is not None else None
    if varlen_m:
        # m is total_m in this case: the flattened M dimension of D/C
        m = cute.sym_int()
        a_m = cute.sym_int() if gather_A else m
        mA = fake_batched(a_dtype, a_m, a_k_sym, None, a_leading, div_a)
        if b_kn:
            mB = fake_batched(b_dtype, b_k_sym, n, l, 1 - b_leading, div_b)
        else:
            mB = fake_batched(b_dtype, n, b_k_sym, l, b_leading, div_b)
        dc_n = pd if packed_cd is not None else n  # "n" form only under varlen
        mD = fake_batched(d_dtype, m, dc_n, None, d_leading, div_d)
        mC = fake_batched(c_dtype, m, dc_n, None, c_leading, div_c)
    elif varlen_k:
        # k is total_k in this case: the flattened K dimension of A/B
        k = cute.sym_int()
        a_k = cute.sym_int() if gather_A else k
        mA = fake_batched(a_dtype, m, a_k, None, a_leading, div_a)
        mB = fake_batched(b_dtype, n, k, None, b_leading, div_b)
        mD = fake_batched(d_dtype, m, n, l, d_leading, div_d)
        mC = fake_batched(c_dtype, m, n, l, c_leading, div_c)
    else:
        bl = l if batched else None
        if swap_ab:
            # Swap-at-trace (dims are KERNEL m/n; m = caller n). Slot-A is the
            # caller's (k, n) B crossing natively -> a_transposed relabels to
            # (m, k, l); slot-B is the caller's (m, k) A, already kernel-
            # ordered (n_k, k); D/C cross caller-oriented (n_k, m_k) ->
            # cd_transposed. The caller-stride major labels flip with the
            # shape order, so the standard leading formulas hold (build_
            # gemm_epi_plan flips d/c majors, the only non-cancelling pair).
            mA = fake_batched(a_dtype, a_k_sym, m, bl, 1 - a_leading, div_a)
            mB = fake_batched(b_dtype, n, b_k_sym, bl, b_leading, div_b)
            mD = fake_batched(d_dtype, n, m, bl, 1 - d_leading, div_d)
            mC = fake_batched(c_dtype, n, m, bl, 1 - c_leading, div_c)
            return mA, mB, mD, mC, m, n, k, l
        mA = fake_batched(a_dtype, m, a_k_sym, bl, a_leading, div_a)
        if b_kn:
            # (k, n) orientation: b_major is still the logical (n, k) label, so
            # "k"-major means dim 0 of (k, n) is contiguous.
            mB = fake_batched(b_dtype, b_k_sym, n, bl, 1 - b_leading, div_b)
        else:
            mB = fake_batched(b_dtype, n, b_k_sym, bl, b_leading, div_b)
        dc_x, dc_y = (m, pd) if packed_cd == "n" else (pd, n) if packed_cd == "m" else (m, n)
        mD = fake_batched(d_dtype, dc_x, dc_y, bl, d_leading, div_d)
        mC = fake_batched(c_dtype, dc_x, dc_y, bl, c_leading, div_c)
    return mA, mB, mD, mC, m, n, k, l


def make_fake_sf_tensor(sf_dtype, l):
    """Fake (l, rm, rk, 32, 4, 4) blockscaled scale-factor tensor.

    The inner (32, 4, 4) block has static strides (16, 4, 1) — one contiguous
    512 B atom per 128 rows x 4 K-blocks, so TMA loads it as one box. The
    kernel only consumes the base pointer and the outer (l, rm, rk) strides
    (the atom layout is hardware-fixed); outer strides are dynamic but
    atom-granular, so slices of larger scale buffers are accepted without a
    copy.

    Pass ``l=None`` for an unbatched 5-D (rm, rk, 32, 4, 4) fake (dense 2D
    operands): the kernel prepends the trivial batch mode at trace time.
    """
    rm, rk = cute.sym_int(), cute.sym_int()
    shape = (rm, rk, 32, 4, 4) if l is None else (l, rm, rk, 32, 4, 4)
    outer_strides = tuple(cute.sym_int64(divisibility=512) for _ in range(2 if l is None else 3))
    return cute.runtime.make_fake_tensor(
        sf_dtype,
        shape,
        stride=(*outer_strides, 16, 4, 1),
        assumed_align=16,
    )


def compile_gemm_kernel(
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
    post_init=None,
    transform_a=None,  # A-operand transform factory (SM90 RS mainloop);
    # layout-owning transforms take mA as a TransformAOperand bundle
    mSFA=None,
    mSFB=None,
    use_tma_gather=False,
    concat_layout=None,
    num_warps=None,
    sf_vec_size=None,
    split_k=1,
    split_k_mode=SplitKMode.SERIAL,
    b_transposed=False,
    a_transposed=False,
    cd_transposed=False,
    cd_packed=None,
    a_mma_dtype=None,
    b_mma_dtype=None,
):
    """Build GemmCls instance, apply SM90 partial, and cute.compile with TVM-FFI."""
    split_k_kwargs = {}
    if split_k != 1:
        assert device_capacity[0] in [9, 10, 11, 12], "split_k requires SM90/SM100/SM120"
        split_k_kwargs = {"split_k": split_k, "split_k_mode": split_k_mode}
    if transform_a is not None:
        assert device_capacity[0] in (9, 12), "A-operand transforms are SM90/SM120 only"
        split_k_kwargs["transform_a"] = transform_a
    if device_capacity[0] == 8:
        sm8x_kwargs = {"is_persistent": persistent, "num_warps": num_warps}
        sm8x_kwargs["arch"] = device_capacity[0] * 10 + device_capacity[1]
        GemmCls = partial(GemmCls, **sm8x_kwargs)
    elif device_capacity[0] in [9, 12]:
        if device_capacity[0] == 12:
            if sf_vec_size is not None:
                # SM120 blockscaled (real SFA/SFB); SM90 has no blockscaled
                # path. MMA element types ride along when they differ from
                # storage dtypes (packed fp6 crosses the FFI boundary as raw
                # bytes).
                split_k_kwargs["sf_vec_size"] = sf_vec_size
                split_k_kwargs["a_mma_dtype"] = a_mma_dtype
                split_k_kwargs["b_mma_dtype"] = b_mma_dtype
            # SM120's dynamic persistence is CLC (cluster launch control, same
            # Blackwell feature as sm100's) — no tile-count semaphore needed.
            # Pingpong included: it consumes CLC responses one-at-a-time
            # (128x128 pingpong + CLC is the best-measured mxfp8 config, see
            # AI/sm120_blockscaled_gemm_worklog.md).
            # Gate on the COMPILE TARGET, not the dispatch arch: CLC
            # instructions are sm_100+, so the H100 CI proxy legs
            # (QUACK_ARCH=120 compiled for sm_90a) must fall back to the
            # static persistent scheduler or NVVM rejects the kernel.
            split_k_kwargs["use_clc_persistence"] = (
                is_dynamic_persistent and persistent and get_compile_target_capacity()[0] >= 10
            )
        GemmCls = partial(GemmCls, pingpong=pingpong, is_persistent=persistent, **split_k_kwargs)
    elif device_capacity[0] in [10, 11]:
        GemmCls = partial(
            GemmCls,
            use_clc_persistence=is_dynamic_persistent,
            use_tma_gather=use_tma_gather,
            sf_vec_size=sf_vec_size,
            a_mma_dtype=a_mma_dtype,
            b_mma_dtype=b_mma_dtype,
            **split_k_kwargs,
        )
    gemm_obj = GemmCls(
        Float32,
        a_dtype,
        tile_shape_mn,
        cluster_shape_mnk,
        gather_A=gather_A,
        concat_layout=concat_layout,
    )
    # mB crosses the boundary as (l, k, n); rotate_batch_last transposes it to
    # kernel order (n, k, l) at trace time. a/cd_transposed are the
    # swap-at-trace analogues (see GemmBase).
    gemm_obj.b_transposed = b_transposed
    gemm_obj.a_transposed = a_transposed
    gemm_obj.cd_transposed = cd_transposed
    gemm_obj.cd_packed = cd_packed
    if post_init:
        post_init(gemm_obj)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    # Unified signature across archs: every kernel class takes trailing
    # (SFA, SFB) — None unless blockscaled (SM120), and SM100 consumes them
    # natively. The compiled TVM-FFI arg spec bakes the full declared arity
    # (defaults included), so they are always passed here and at launch.
    sf_args = (mSFA, mSFB)
    return cute.compile(
        gemm_obj,
        mA,
        mB,
        mD,
        mC,
        epi_args,
        scheduler_args,
        varlen_args,
        stream,
        *sf_args,
        options="--enable-tvm-ffi",
    )
