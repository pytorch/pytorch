# Copyright (c) 2026, Han Guo, Tri Dao.
"""Generic host-side plan/compile/launch layer for epilogue GEMM variants.

The per-variant host plumbing (fake-tensor construction, jit_cache'd compile
wrapper, plan NamedTuple + cache key + build/run pair) used to be ~400
near-identical lines per variant file. This module replaces it with one
generic implementation driven by the variant's EpiOp schema:

* each EpiOp describes its own host argument via ``host_arg_key`` (torch value
  -> picklable descriptor), ``host_fake_arg`` (descriptor -> fake trace-time
  tensor/scalar), and ``host_call_arg`` (torch value -> runtime argument);
* a reconstructable ``GemmClassRef`` is the jit_cache key component — static
  classes resolve by module+qualname, while dynamic epilogue classes resolve
  through a module-global EpiMod and mint locally in async workers.

A variant file keeps only: its mixin(s), the per-SM class stampings, its
validation asserts, and thin ``gemm_X`` / ``run_gemm_X_plan`` wrappers that
map the public signature onto ``epi_values`` dicts.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import cutlass.cute as cute
from cutlass import Float32, Int32

from torch._vendor.quack.cache import jit_cache
from torch._vendor.quack.compile_utils import make_fake_tensor
from torch._vendor.quack.cute_dsl_utils import get_max_active_clusters, torch2cute_dtype_map
from torch._vendor.quack.gemm_config import SplitKMode, cta_tile_shape_m
from torch._vendor.quack.epilogue.ops import EpiOp
from torch._vendor.quack.gemm_runtime.identity import (
    resolve_gemm_class,
    resolve_transform_a,
    static_gemm_class_ref,
)
from torch._vendor.quack.gemm_tvm_ffi_utils import (
    compile_gemm_kernel,
    compute_cu_tiles_m,
    launch_gemm,
    make_fake_gemm_tensors,
    make_fake_scheduler_args,
    make_fake_sf_tensor,
    make_fake_varlen_args,
    make_scheduler_args,
    make_varlen_args,
    plan_scheduler_args,
)


class FakeArgCtx(NamedTuple):
    """Shared symbolic dims + flags handed to EpiOp.host_fake_arg.

    ``swapped`` (swap-at-trace): m/n are KERNEL dims (m = caller n); tile-
    shaped args cross the boundary caller-oriented, i.e. (l, n, m) in kernel
    labels, and are transposed at trace time (GemmBase.cd_transposed)."""

    m: object
    n: object
    k: object
    l: object  # noqa: E741
    batched: bool
    varlen_m: bool
    swapped: bool = False


def _ops_by_name(GemmCls):
    return {op.name: op for op in GemmCls._epi_ops}


def _has_m_fold_sink(ops, epi_keys):
    """True when an ACTIVE (present in epi_keys) op is an M-fold (dim==1)
    reduce sink — the trigger for varlen zero-fill ragging and the cu_tiles_m
    per-sequence tile prefix."""
    return any(
        getattr(ops[name], "fn_port", None) == "sink" and getattr(ops[name], "dim", None) == 1
        for name, _ in epi_keys
        if name in ops
    )


@jit_cache
def _compile_gemm_epi(
    gemm_cls_ref,
    device_capacity,
    a_dtype,  # the gemm ctor a_dtype: the MMA dtype (= A's dtype except for
    # layout-owning transforms, where A crosses as a storage blob)
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
    varlen_m,
    gather_A,
    batched,
    b_kn,
    epi_keys,  # ((op_name, op.host_arg_key(value)), ...) — name-sorted
    swap_ab=False,
    use_tma_gather=False,
    concat_layout=(),
    sf_dtype=None,
    sf_vec_size=None,
    sf_batched=True,
    # Blockscaled MMA element types when they differ from the boundary dtypes
    # (packed fp6 crosses TVM-FFI as raw uint8); None: same as the tensor dtypes.
    a_mma_dtype=None,
    b_mma_dtype=None,
    post_init_attrs=(),  # ((attr, value), ...) setattr'd on the gemm object pre-trace
    packed_cd=None,  # "n" | "m": raw 16-bit D/C, f32-recast at trace (dgated)
    has_ag=False,  # AllGather+GEMM: ag scheduler fields in the compiled signature
    split_k=1,  # K-dim split factor, constexpr kernel specialization
    split_k_mode=SplitKMode.SERIAL,  # SERIAL/PARALLEL only (SEPARATE rejected upstream)
    # A-operand transform (SM90 RS / SM120 warp-MMA mainloop,
    # quack.operand_transform): a
    # picklable TransformARef. Layout-owning (packed W4) transforms replace
    # the standard fake operands: A is the repacked blob (static geometry
    # from transform_dims = (n_full, k)), the SF strip rides a trailing AuxA
    # arg, kernel-N (the activation M) is the only symbolic dim.
    transform_a_ref=None,
    transform_dims=None,
):
    """Compile one epilogue-GEMM variant against fake symbolic tensors.

    Every argument is a picklable primitive (jit_cache pickles the tuple for
    the disk key and to ship cold misses to async-compile workers).
    """
    GemmCls = resolve_gemm_class(gemm_cls_ref)
    transform_mod = None
    owned_fmt = None
    if transform_a_ref is not None:
        transform_mod = resolve_transform_a(transform_a_ref)
        owned_fmt = transform_mod.owned_fmt
    if owned_fmt is not None:
        assert not (varlen_m or gather_A or batched or b_kn or swap_ab or concat_layout), (
            "layout-owning transforms support the plain dense path only"
        )
        n_full, k_static = transform_dims
        # mA is the TransformAOperand bundle (blob + optional strip): ONE
        # argument slot, the same shape as the runtime views
        mA = transform_mod.fake_operands(n_full, k_static, tile_shape_mn[0])
        n_sym = cute.sym_int()
        # activations: (m_act, k) k-major with a STATIC compact stride
        # (mark_compact_shape_dynamic(mode=0)). D crosses CALLER-oriented
        # (m_act, n_full) row-major and is relabeled at trace via
        # cd_transposed (kernel D = (n_full, m_act) m-major) — the swap-at-
        # trace convention, so tile-shaped epi operands stay caller-oriented
        # too (fctx.swapped below).
        mB = cute.runtime.make_fake_tensor(
            b_dtype, (n_sym, k_static), stride=(k_static, 1), assumed_align=16
        )
        mD = cute.runtime.make_fake_tensor(
            d_dtype, (n_sym, n_full), stride=(n_full, 1), assumed_align=16
        )
        mC = None
        m, n, k, l = n_full, n_sym, k_static, 1
    else:
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
            gather_A=gather_A,
            batched=batched,
            b_kn=b_kn,
            swap_ab=swap_ab,
            packed_cd=packed_cd,
            a_mma_dtype=a_mma_dtype,
            b_mma_dtype=b_mma_dtype,
        )
        if transform_mod is not None and transform_mod.needs_operands:
            # value transform with runtime operands: A crosses as a (plain A,
            # operand view) bundle in the one mA slot (same-arity trick as W4)
            assert not (varlen_m or gather_A or batched or swap_ab or concat_layout), (
                "transform runtime operands support the plain dense path only"
            )
            mA = transform_mod.fake_bundle(
                mA, a_dtype, tile_shape_mn[0], tile_shape_mn[2] if len(tile_shape_mn) == 3 else None
            )
    fctx = FakeArgCtx(m, n, k, l, batched, varlen_m, swap_ab or owned_fmt is not None)
    ops = _ops_by_name(GemmCls)
    fields = {}
    for name, key in epi_keys:
        fake = ops[name].host_fake_arg(key, fctx)
        if fake is not None:
            fields[name] = fake
    if split_k > 1 and split_k_mode != SplitKMode.SEPARATE:
        # Mirrors quack.gemm: (ntile_m, ntile_n, L) Int32 per-tile flag and
        # (cta_tile_m * cta_tile_n, ntile_m, ntile_n, L) f32 partials stripes.
        fields["split_k_semaphore"] = make_fake_tensor(
            Int32, (cute.sym_int(), cute.sym_int(), cute.sym_int()), leading_dim=1
        )
        fields["split_k_workspace"] = make_fake_tensor(
            Float32,
            (cute.sym_int(), cute.sym_int(), cute.sym_int(), cute.sym_int()),
            leading_dim=0,
            divisibility=4,
        )
    epi_args = GemmCls.EpilogueArguments(**fields)

    scheduler_args = make_fake_scheduler_args(
        (is_dynamic_persistent and device_capacity[0] == 9), False, l, has_ag=has_ag
    )
    varlen_args = make_fake_varlen_args(
        varlen_m,
        False,
        gather_A,
        m if varlen_m else None,
        has_cu_tiles_m=varlen_m and _has_m_fold_sink(ops, epi_keys),
    )
    mSFA = make_fake_sf_tensor(sf_dtype, l if sf_batched else None) if sf_dtype else None
    mSFB = make_fake_sf_tensor(sf_dtype, l if sf_batched else None) if sf_dtype else None
    post_init = None
    if post_init_attrs:

        def post_init(gemm_obj):
            for attr, value in post_init_attrs:
                setattr(gemm_obj, attr, value)

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
        post_init=post_init,
        transform_a=transform_mod,
        mSFA=mSFA,
        mSFB=mSFB,
        use_tma_gather=use_tma_gather,
        concat_layout=concat_layout or None,
        sf_vec_size=sf_vec_size,
        a_mma_dtype=a_mma_dtype,
        b_mma_dtype=b_mma_dtype,
        b_transposed=b_kn,
        a_transposed=swap_ab,
        cd_transposed=swap_ab or owned_fmt is not None,
        cd_packed=packed_cd,
        split_k=split_k,
        split_k_mode=split_k_mode,
    )


class GemmEpiPlan(NamedTuple):
    """Launch plan derived purely from tensor metadata and config flags.

    Cached by the variant wrapper per metadata key, so warm calls skip
    validation, major/dtype derivation, and the compile-cache lookup.
    ``epi_arg_keys`` replays each op's compile-time descriptor at launch
    (host_call_arg needs e.g. the scalar mode); ``gemm_cls`` carries the op
    schema and EpilogueArguments type into run_gemm_epi_plan.
    """

    compiled_fn: object
    gemm_cls: type
    is_sm100_family: bool  # SM100/110 use 2-CTA MMA
    max_active_clusters: int
    max_swizzle_size: int
    scheduler_uses_semaphore: bool  # only the SM90 dynamic scheduler consumes the semaphore
    scheduler_static: Optional[object]  # TileSchedulerOptions when it has no per-call values
    epi_arg_keys: tuple  # ((op_name, key), ...) as compiled
    tile_M: int  # scheduler-cluster geometry, for launch-time AG validation
    cluster_M: int  # (see validate_ag_geometry in plan_scheduler_args)
    # Launch-overhead precomputation (host hot path): (name, converter, key)
    # triples — converter is the op's bound ``host_call_arg``, or None when the
    # op inherits the identity default (the value passes straight through) —
    # plus an all-None EpilogueArguments field template in field order, so warm
    # calls do a dict .copy() + positional ``_make`` instead of rebuilding both
    # dicts and parsing kwargs.
    call_ops: tuple = ()
    arg_template: dict = {}
    # Split-K (SERIAL/PARALLEL): run allocates the per-call flag/workspace
    # buffers, sized from D and the tile/cluster geometry below.
    split_k: int = 1
    split_k_mode: object = SplitKMode.SERIAL
    tile_N: int = 0
    cluster_N: int = 1
    # D crosses caller-oriented and is transposed at trace (layout-owning
    # transforms): split-k buffer sizing must use kernel orientation (D.mT).
    d_transposed: bool = False
    # varlen_m + active M-fold sink: per-CTA M-tile extent used to compute the
    # cu_tiles_m prefix at launch (None = no prefix needed). Also consumed by
    # the interface layer's per-sequence finalize.
    cu_tiles_tile_m: Optional[int] = None


def _get_major(t, m_label, n_label):
    return n_label if t.stride(-1) == 1 else m_label


def build_gemm_epi_plan(
    GemmCls,
    device_capacity,
    A,
    B,
    D,
    C,
    *,
    epi_values,  # {op_name: torch value or scalar}; missing/None = op inactive
    epi_key_overrides=None,  # {op_name: key} when the wrapper owns the key rule (scalar modes)
    tile_M,
    tile_N,
    cluster_M,
    cluster_N,
    tile_K=None,
    pingpong=False,
    persistent=True,
    is_dynamic_persistent=False,
    max_swizzle_size=8,
    varlen_m=False,
    gather_A=False,
    b_kn=False,
    swap_ab=False,  # swap-at-trace: slot tensors in, caller-oriented D/C
    use_tma_gather=False,
    concat_layout=(),
    sf_dtype=None,
    sf_vec_size=None,
    sf_batched=True,
    a_mma_dtype=None,
    b_mma_dtype=None,
    post_init_attrs=(),
    gemm_cls_ref=None,
    packed_cd=None,  # "n" | "m": D/C passed RAW 16-bit, f32-recast at trace (dgated)
    has_ag=False,  # AllGather+GEMM (see quack/distributed/): dense persistent only
    split_k=1,
    split_k_mode=SplitKMode.SERIAL,
    # A-operand transform handle (format name / DecodeFormat / a_transform
    # mod). Layout-owning transforms: A is the TransformAOperand bundle from
    # operand_transform.host.w4_operand_views (blob view + optional strip);
    # the ctor a_dtype comes from the format's mma_dtype.
    transform_a=None,
) -> GemmEpiPlan:
    """Derive majors/dtypes/epi keys from tensor metadata and compile (or hit
    the jit cache). Variant wrappers call this after their validation asserts."""
    transform_ref, owned_fmt, transform_dims = None, None, None
    if transform_a is not None:
        from torch._vendor.quack.operand_transform.host import as_transform_mod

        transform_mod = as_transform_mod(transform_a)
        transform_ref = transform_mod.compile_ref()
        owned_fmt = transform_mod.owned_fmt
        if owned_fmt is not None:
            # A is the TransformAOperand bundle; recover the static problem
            # geometry for the fake construction (the handle owns the blob
            # anatomy). Metadata derivation below reads the blob; the bundle
            # itself never crosses into the picklable compile args.
            transform_dims = transform_mod.compile_dims(A)
            A = A.blob
        elif transform_mod.needs_operands:
            # value transform with runtime operands: A arrives bundled with
            # the operand view (transform_a_operand); metadata derivation
            # reads the plain operand, the bundle itself crosses at launch.
            from torch._vendor.quack.operand_transform.transform import TransformAOperand

            assert isinstance(A, TransformAOperand), (
                "a transform with runtime operands takes A as "
                "transform_a_operand(mod, A, values, tile_M)"
            )
            A = A.blob
    batched = A.ndim == 3 or varlen_m
    a_major = _get_major(A, "m", "k")
    b_major = _get_major(B, "n", "k")
    if b_kn:
        # Majors are logical (n, k) labels: with B stored (k, n), a contiguous
        # last dim means n-major.
        b_major = "n" if B.stride(-1) == 1 else "k"
    d_major = _get_major(D, "m", "n") if D is not None else None
    c_major = _get_major(C, "m", "n") if C is not None else None
    if swap_ab:
        # Slot tensors: A-slot = caller B (k, n) native (a_transposed relabels
        # at trace) — kernel-A is (m_k, k) with m_k the caller n, so the label
        # flips vs the standard derivation. B-slot = caller A (m, k) is
        # already kernel-ordered (n_k, k): the standard formula holds. D/C
        # cross caller-oriented, so their kernel labels flip like A's.
        a_major = "m" if A.stride(-1) == 1 else "k"
        d_major = ("m" if D.stride(-1) == 1 else "n") if D is not None else None
        c_major = ("m" if C.stride(-1) == 1 else "n") if C is not None else None
    if owned_fmt is not None:
        # the ctor a_dtype is the MMA compute dtype the format decodes to,
        # decoupled from the blob's storage dtype
        a_dtype = owned_fmt.mma_dtype
        batched = False
    else:
        a_dtype = torch2cute_dtype_map[A.dtype]
    b_dtype = torch2cute_dtype_map[B.dtype]
    d_dtype = torch2cute_dtype_map[D.dtype] if D is not None else None
    c_dtype = torch2cute_dtype_map[C.dtype] if C is not None else None

    ops = _ops_by_name(GemmCls)
    overrides = epi_key_overrides or {}
    epi_keys = []
    for name, op in ops.items():
        key = overrides[name] if name in overrides else op.host_arg_key(epi_values.get(name))
        if key is not None:
            epi_keys.append((name, key))
    epi_keys = tuple(sorted(epi_keys, key=lambda nk: nk[0]))

    if gemm_cls_ref is None:
        gemm_cls_ref = static_gemm_class_ref(GemmCls)
    compiled_fn = _compile_gemm_epi(
        gemm_cls_ref,
        device_capacity,
        a_dtype,
        b_dtype,
        d_dtype,
        c_dtype,
        a_major,
        b_major,
        d_major,
        c_major,
        (tile_M, tile_N, tile_K) if tile_K is not None else (tile_M, tile_N),
        (cluster_M, cluster_N, 1),
        pingpong,
        persistent,
        is_dynamic_persistent,
        varlen_m,
        gather_A,
        batched,
        b_kn,
        epi_keys,
        swap_ab=swap_ab,
        use_tma_gather=use_tma_gather,
        concat_layout=concat_layout,
        sf_dtype=sf_dtype,
        sf_vec_size=sf_vec_size,
        sf_batched=sf_batched,
        a_mma_dtype=a_mma_dtype,
        b_mma_dtype=b_mma_dtype,
        post_init_attrs=post_init_attrs,
        packed_cd=packed_cd,
        has_ag=has_ag,
        split_k=split_k,
        split_k_mode=split_k_mode,
        transform_a_ref=transform_ref,
        transform_dims=transform_dims,
    )

    max_active_clusters = get_max_active_clusters(cluster_M * cluster_N) if persistent else 0
    # Must mirror make_fake_scheduler_args above: only the SM90 dynamic
    # scheduler consumes the semaphore, so it's the only non-static case.
    scheduler_uses_semaphore = is_dynamic_persistent and device_capacity[0] == 9
    # AG plans get PER-CALL scheduler args (the flags seq advances every
    # iteration), never the prebuilt static tuple.
    scheduler_static = (
        make_scheduler_args(max_active_clusters, max_swizzle_size, None)
        if not scheduler_uses_semaphore and not has_ag
        else None
    )
    plan_ops = _ops_by_name(GemmCls)
    call_ops = tuple(
        (
            name,
            None
            if type(plan_ops[name]).host_call_arg is EpiOp.host_call_arg
            else plan_ops[name].host_call_arg,
            key,
        )
        for name, key in epi_keys
    )
    return GemmEpiPlan(
        compiled_fn=compiled_fn,
        gemm_cls=GemmCls,
        call_ops=call_ops,
        arg_template={name: None for name in GemmCls.EpilogueArguments._fields},
        is_sm100_family=device_capacity[0] in [10, 11],
        max_active_clusters=max_active_clusters,
        max_swizzle_size=max_swizzle_size,
        scheduler_uses_semaphore=scheduler_uses_semaphore,
        scheduler_static=scheduler_static,
        epi_arg_keys=epi_keys,
        tile_M=tile_M,
        cluster_M=cluster_M,
        split_k=split_k,
        split_k_mode=split_k_mode,
        tile_N=tile_N,
        cluster_N=cluster_N,
        d_transposed=owned_fmt is not None,
        cu_tiles_tile_m=(
            cta_tile_shape_m(tile_M, cluster_M, device_capacity[0], sf_dtype is not None)
            if varlen_m and _has_m_fold_sink(plan_ops, epi_keys)
            else None
        ),
    )


def run_gemm_epi_plan(
    plan: GemmEpiPlan,
    A,
    B,
    D,
    C,
    epi_values,
    *,
    ag_args=None,  # forwarded to the scheduler (AllGather+GEMM flags contract)
    tile_count_semaphore=None,
    cu_seqlens_m=None,
    cu_seqlens_k=None,
    A_idx=None,
    SFA=None,
    SFB=None,
    split_k_buffers=None,  # (sem, ws) raw from _split_k_buffers: reuse across
    # calls when the kernel leaves them reusable (serial self-resets)
) -> None:
    """Launch a resolved plan: only per-call pointers and scalar values here.

    The tensors must match the metadata the plan was built from (the variant
    wrapper guarantees that via its plan-cache key). Constexpr fields are
    passed None — they are baked into the compiled kernel.
    """
    # arg_template preserves EpilogueArguments field order, so the values()
    # view feeds _make positionally (no kwargs parsing).
    fields = plan.arg_template.copy()
    for name, convert, key in plan.call_ops:
        value = epi_values.get(name)
        if convert is not None:
            value = convert(value, key)
        if value is not None:
            fields[name] = value
    if plan.split_k > 1:
        if split_k_buffers is None:
            # Fresh per-call buffers (mirrors quack.gemm.run_gemm_plan); lazy import —
            # quack.gemm sits above this module in the import graph.
            from torch._vendor.quack.gemm import _split_k_buffers

            Dk = D.mT if plan.d_transposed else D  # size in KERNEL orientation
            split_k_buffers = _split_k_buffers(
                Dk if Dk.ndim == 3 else Dk[None],
                plan.split_k_mode,
                plan.tile_M,
                plan.tile_N,
                plan.cluster_M,
                plan.cluster_N,
                plan.is_sm100_family,
            )
        sem, ws = split_k_buffers
        fields["split_k_semaphore"] = sem.permute(1, 2, 0)
        fields["split_k_workspace"] = ws.permute(3, 1, 2, 0)
    epi_args = plan.gemm_cls.EpilogueArguments._make(fields.values())
    scheduler_args = plan_scheduler_args(plan, tile_count_semaphore, ag_args=ag_args, A=A)
    cu_tiles_m = (
        compute_cu_tiles_m(cu_seqlens_m, plan.cu_tiles_tile_m)
        if cu_seqlens_m is not None and plan.cu_tiles_tile_m is not None
        else None
    )
    varlen_args = make_varlen_args(cu_seqlens_m, cu_seqlens_k, A_idx, cu_tiles_m=cu_tiles_m)
    launch_gemm(plan, A, B, D, C, epi_args, scheduler_args, varlen_args, SFA, SFB)
