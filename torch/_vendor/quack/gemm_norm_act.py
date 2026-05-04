# Copyright (c) 2025-2026, Tri Dao.
# GEMM + normalize (multiply by colvec and rowvec) + activation:
# PostAct = act((A @ B + C) * colvec * rowvec)
# colvec is typically rstd (M,), rowvec is typically norm_weight (N,).

from typing import Optional, Tuple

from torch import Tensor

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr
from cutlass.cute.runtime import make_ptr

from .compile_utils import make_fake_tensor as fake_tensor
from .cute_dsl_utils import (
    torch2cute_dtype_map,
    get_device_capacity,
    get_max_active_clusters,
)
from .gemm_sm80 import GemmSm80
from .gemm_sm90 import GemmSm90
from .gemm_sm100 import GemmSm100
from .gemm_sm120 import GemmSm120
from .gemm_act import GemmActMixin, GemmGatedMixin, GemmGatedSm120Mixin
from .epi_ops import vec_multiply
from .activation import act_fn_map, gate_fn_map
from .cache_utils import jit_cache
from .rounding import RoundingMode
from .gemm_tvm_ffi_utils import (
    get_major,
    perm3d_single,
    make_scheduler_args,
    make_varlen_args,
    make_fake_scheduler_args,
    make_fake_varlen_args,
    div_for_dtype,
    make_fake_gemm_tensors,
    compile_gemm_kernel,
)
from . import utils


class GemmNormActMixin(GemmActMixin):
    """GEMM + normalize + activation: PostAct = act((A @ B + C) * colvec * rowvec).

    colvec is typically rstd (M,), rowvec is typically norm_weight (N,).
    D stores the normalized (pre-activation) value, PostAct stores act(D).
    """

    @cute.jit
    def epi_visit_subtile(
        self,
        params: GemmActMixin.EpilogueParams,
        epi_loop_tensors: Tuple[cute.Tensor, ...],
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor] = None,
    ) -> Optional[cute.Tensor]:
        tDrRowVec = epi_loop_tensors["mRowVecBroadcast"]
        tDrColVec = epi_loop_tensors["mColVecBroadcast"]
        # Load accumulator and apply alpha/beta/C
        rD = tRS_rD.load()
        if const_expr(hasattr(params, "alpha") and params.alpha is not None):
            alpha = utils.load_scalar_or_pointer(params.alpha)
            rD *= alpha
        if const_expr(tRS_rC is not None):
            if const_expr(not hasattr(params, "beta") or params.beta is None):
                rD += tRS_rC.load().to(tRS_rD.element_type)
            else:
                beta = utils.load_scalar_or_pointer(params.beta)
                rD += beta * tRS_rC.load().to(tRS_rD.element_type)
        tRS_rD.store(rD)
        # Multiply by colvec (rstd) and rowvec (norm_weight)
        vec_multiply(self, tRS_rD, tDrColVec, tDrRowVec)
        # Apply activation
        if const_expr(params.act_fn is not None):
            tRS_rAuxOut = cute.make_rmem_tensor(tRS_rD.layout.shape, self.acc_dtype)
            if const_expr(self.arch != 100):
                for i in cutlass.range(cute.size(tRS_rAuxOut), unroll_full=True):
                    tRS_rAuxOut[i] = params.act_fn(tRS_rD[i])
            else:
                for i in cutlass.range(cute.size(tRS_rAuxOut) // 2, unroll_full=True):
                    tRS_rAuxOut[2 * i], tRS_rAuxOut[2 * i + 1] = params.act_fn(
                        (tRS_rD[2 * i], tRS_rD[2 * i + 1])
                    )
        else:
            tRS_rAuxOut = tRS_rD
        return tRS_rAuxOut


class GemmNormActSm90(GemmNormActMixin, GemmSm90):
    pass


class GemmNormActSm80(GemmNormActMixin, GemmSm80):
    pass


class GemmNormActSm100(GemmNormActMixin, GemmSm100):
    pass


class GemmNormActSm120(GemmNormActMixin, GemmSm120):
    pass


class GemmNormGatedMixin(GemmGatedMixin):
    """GEMM + normalize + gated activation: PostAct = gated_act((A @ B + C) * colvec * rowvec)."""

    @cute.jit
    def epi_visit_subtile(
        self,
        params: GemmActMixin.EpilogueParams,
        epi_loop_tensors: Tuple[cute.Tensor, ...],
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor] = None,
    ) -> Optional[cute.Tensor]:
        tDrRowVec = epi_loop_tensors["mRowVecBroadcast"]
        tDrColVec = epi_loop_tensors["mColVecBroadcast"]
        # Load accumulator and apply alpha/beta/C
        rD = tRS_rD.load()
        if const_expr(hasattr(params, "alpha") and params.alpha is not None):
            alpha = utils.load_scalar_or_pointer(params.alpha)
            rD *= alpha
        if const_expr(tRS_rC is not None):
            if const_expr(not hasattr(params, "beta") or params.beta is None):
                rD += tRS_rC.load().to(tRS_rD.element_type)
            else:
                beta = utils.load_scalar_or_pointer(params.beta)
                rD += beta * tRS_rC.load().to(tRS_rD.element_type)
        tRS_rD.store(rD)
        # Multiply by colvec (rstd) and rowvec (norm_weight)
        vec_multiply(self, tRS_rD, tDrColVec, tDrRowVec)
        # Gated activation on normalized D
        tRS_rAuxOut_layout = cute.recast_layout(2, 1, tRS_rD.layout)
        tRS_rAuxOut = cute.make_rmem_tensor(tRS_rAuxOut_layout.shape, self.acc_dtype)
        if const_expr(self.arch != 100):
            for i in cutlass.range(cute.size(tRS_rAuxOut), unroll_full=True):
                tRS_rAuxOut[i] = params.act_fn(tRS_rD[2 * i], tRS_rD[2 * i + 1])
        else:
            for i in cutlass.range(cute.size(tRS_rAuxOut) // 2, unroll_full=True):
                tRS_rAuxOut[2 * i], tRS_rAuxOut[2 * i + 1] = params.act_fn(
                    (tRS_rD[4 * i], tRS_rD[4 * i + 2]),
                    (tRS_rD[4 * i + 1], tRS_rD[4 * i + 3]),
                )
        return tRS_rAuxOut


class GemmNormGatedSm90(GemmNormGatedMixin, GemmSm90):
    pass


class GemmNormGatedSm80(GemmNormGatedMixin, GemmSm80):
    pass


class GemmNormGatedSm100(GemmNormGatedMixin, GemmSm100):
    pass


class GemmNormGatedSm120(GemmGatedSm120Mixin, GemmNormGatedMixin, GemmSm120):
    pass


@jit_cache
def _compile_gemm_norm_act(
    a_dtype,
    b_dtype,
    d_dtype,
    c_dtype,
    postact_dtype,
    a_major,
    b_major,
    d_major,
    c_major,
    postact_major,
    tile_shape_mn,
    cluster_shape_mnk,
    pingpong,
    persistent,
    is_dynamic_persistent,
    activation,
    rowvec_dtype,
    colvec_dtype,
    colvec_ndim,
    varlen_m,
    gather_A,
    device_capacity,
    gemm_cls_name,
    rounding_mode=RoundingMode.RN,
    sr_seed_mode=0,
):
    sm_to_cls = {
        "norm_act": {
            8: GemmNormActSm80,
            9: GemmNormActSm90,
            10: GemmNormActSm100,
            11: GemmNormActSm100,
            12: GemmNormActSm120,
        },
        "norm_gated": {
            8: GemmNormGatedSm80,
            9: GemmNormGatedSm90,
            10: GemmNormGatedSm100,
            11: GemmNormGatedSm100,
            12: GemmNormGatedSm120,
        },
    }
    GemmCls = sm_to_cls[gemm_cls_name][device_capacity[0]]
    pa_leading = 1 if postact_major == "n" else 0
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
    )
    div_pa = div_for_dtype(postact_dtype)
    pa_n = cute.sym_int() if gemm_cls_name == "norm_gated" else n
    pa_leading_dim = 1 if gemm_cls_name == "norm_gated" else pa_leading
    pa_shape = (m, pa_n) if varlen_m else (m, pa_n, l)
    mAuxOut = fake_tensor(postact_dtype, pa_shape, leading_dim=pa_leading_dim, divisibility=div_pa)

    mRowVec = fake_tensor(rowvec_dtype, (l, n), leading_dim=1, divisibility=4)
    if colvec_ndim == 2:
        mColVec = fake_tensor(colvec_dtype, (l, m), leading_dim=1, divisibility=4)
    elif colvec_ndim == 1:
        mColVec = fake_tensor(colvec_dtype, (m,), leading_dim=0, divisibility=4)
    else:
        mColVec = None

    act_fn = act_fn_map[activation] if gemm_cls_name == "norm_act" else gate_fn_map[activation]

    def fake_scalar(mode, dtype=Int32):
        if mode == 0:
            return None
        elif mode == 1:
            return dtype(0)
        else:
            return make_ptr(dtype, 0, cute.AddressSpace.gmem, assumed_align=4)

    epi_args = GemmCls.EpilogueArguments(
        mAuxOut,
        act_fn,
        mRowVecBroadcast=mRowVec,
        mColVecBroadcast=mColVec,
        rounding_mode=rounding_mode,
        sr_seed=fake_scalar(sr_seed_mode),
    )
    scheduler_args = make_fake_scheduler_args(
        (is_dynamic_persistent and device_capacity[0] == 9), False, l
    )
    varlen_args = make_fake_varlen_args(varlen_m, False, gather_A, m if varlen_m else None)
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
    )


def gemm_norm_act_fn(
    A: Tensor,  # (l, m, k) or (total_m, k) if varlen_m
    B: Tensor,  # (l, n, k)
    D: Optional[Tensor],  # (l, m, n) or (total_m, n) if varlen_m
    C: Optional[Tensor],  # (l, m, n) or (total_m, n) if varlen_m
    PostAct: Tensor,  # (l, m, n) or (total_m, n//2) if gated
    tile_count_semaphore: Optional[Tensor],
    activation: Optional[str],
    tile_M: int,
    tile_N: int,
    cluster_M: int,
    cluster_N: int,
    tile_K: int | None = None,
    pingpong: bool = False,
    persistent: bool = True,
    is_dynamic_persistent: bool = False,
    max_swizzle_size: int = 8,
    rowvec: Optional[Tensor] = None,  # (l, n) — norm_weight
    colvec: Optional[Tensor] = None,  # (l, m) or (total_m,) — rstd
    cu_seqlens_m: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,
    rounding_mode: int = RoundingMode.RN,
    sr_seed: int | Tensor = 0,
) -> None:
    if activation in gate_fn_map:
        gemm_cls_name = "norm_gated"
    else:
        assert activation in act_fn_map, f"Unsupported activation {activation}"
        gemm_cls_name = "norm_act"

    varlen_m = cu_seqlens_m is not None
    gather_A = A_idx is not None
    if varlen_m:
        assert persistent, "varlen_m requires persistent=True"
        assert A.stride(-1) == 1, "varlen_m requires A to be k-major"
        if D is not None:
            assert D.stride(-1) == 1, "varlen_m requires D to be n-major"
        assert PostAct.stride(-1) == 1, "varlen_m requires PostAct to be n-major"
    if gather_A:
        assert cu_seqlens_m is not None, "gather_A requires varlen"
        assert cluster_N == 1, "gather_A requires cluster_N=1"

    A_p = perm3d_single(A, varlen_m)
    B_p = perm3d_single(B)
    D_p = perm3d_single(D, varlen_m)
    C_p = perm3d_single(C, varlen_m)
    PostAct_p = perm3d_single(PostAct, varlen_m)

    a_major = get_major(A_p, "m", "k")
    b_major = get_major(B_p, "n", "k")
    d_major = get_major(D_p, "m", "n") if D_p is not None else None
    c_major = get_major(C_p, "m", "n") if C_p is not None else None
    postact_major = get_major(PostAct_p, "m", "n")

    a_dtype = torch2cute_dtype_map[A.dtype]
    b_dtype = torch2cute_dtype_map[B.dtype]
    d_dtype = torch2cute_dtype_map[D.dtype] if D is not None else None
    c_dtype = torch2cute_dtype_map[C.dtype] if C is not None else None
    postact_dtype = torch2cute_dtype_map[PostAct.dtype]
    colvec_ndim = colvec.ndim if colvec is not None else 0

    device_capacity = get_device_capacity(A.device)
    assert device_capacity[0] in [8, 9, 10, 11, 12], (
        "Only SM8x, SM90, SM100, SM110, and SM120 are supported"
    )
    if rounding_mode == RoundingMode.RS:
        assert device_capacity[0] == 10, "Stochastic rounding requires SM100"

    if is_dynamic_persistent and device_capacity[0] == 9:
        assert tile_count_semaphore is not None, (
            "Dynamic persistent tile scheduler in SM90 requires a semaphore in GMEM"
        )

    sr_seed_mode = (
        2 if isinstance(sr_seed, Tensor) else (1 if rounding_mode == RoundingMode.RS else 0)
    )
    compiled_fn = _compile_gemm_norm_act(
        a_dtype,
        b_dtype,
        d_dtype,
        c_dtype,
        postact_dtype,
        a_major,
        b_major,
        d_major,
        c_major,
        postact_major,
        (tile_M, tile_N, tile_K) if tile_K is not None else (tile_M, tile_N),
        (cluster_M, cluster_N, 1),
        pingpong,
        persistent,
        is_dynamic_persistent,
        activation,
        torch2cute_dtype_map[rowvec.dtype] if rowvec is not None else None,
        torch2cute_dtype_map[colvec.dtype] if colvec is not None else None,
        colvec_ndim,
        varlen_m,
        gather_A,
        device_capacity,
        gemm_cls_name,
        rounding_mode=rounding_mode,
        sr_seed_mode=sr_seed_mode,
    )

    from .cache_utils import COMPILE_ONLY

    if COMPILE_ONLY:
        return

    max_active_clusters = get_max_active_clusters(cluster_M * cluster_N) if persistent else 0

    def scalar_arg(scalar, mode, dtype=Int32):
        if mode == 0:
            return None
        elif mode == 1:
            return dtype(scalar)
        else:
            return scalar.data_ptr()

    epi_args = GemmActMixin.EpilogueArguments(
        PostAct_p,
        None,  # act_fn is Constexpr, pass None at call time
        mRowVecBroadcast=rowvec,
        mColVecBroadcast=colvec,
        rounding_mode=None,
        sr_seed=scalar_arg(sr_seed, sr_seed_mode),
    )
    scheduler_args = make_scheduler_args(
        max_active_clusters, max_swizzle_size, tile_count_semaphore
    )
    varlen_args = make_varlen_args(cu_seqlens_m, None, A_idx)

    if device_capacity[0] in [10, 11]:
        compiled_fn(A_p, B_p, D_p, C_p, epi_args, scheduler_args, varlen_args, None, None, None)
    else:
        compiled_fn(A_p, B_p, D_p, C_p, epi_args, scheduler_args, varlen_args, None)
