# Copyright (c) 2025-2026, Tri Dao.
from __future__ import annotations
from typing import NamedTuple, Optional, Tuple, Callable

import torch
from torch import Tensor

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32, const_expr
from .gemm_sm80 import GemmSm80
from .gemm_sm90 import GemmSm90
from .gemm_sm100 import GemmSm100
from .gemm_sm120 import GemmSm120
from .gemm_default_epi import GemmDefaultEpiMixin
from .gemm_act import GemmActMixin
from .epi_ops import ColVecLoad, ColVecReduce, Scalar, TileStore, colvec_reduce_accumulate
from .compile_utils import make_fake_tensor as fake_tensor
from .cute_dsl_utils import (
    mlir_namedtuple,
    torch2cute_dtype_map,
    get_device_capacity,
    get_max_active_clusters,
)
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
from .cache_utils import jit_cache
from .rounding import RoundingMode
from . import layout_utils
from .activation import dact_fn_map, dgate_fn_map


class GemmDActMixin(GemmActMixin):
    # Different from GemmActSm90, here act_bwd_fn must take in 2 arguments (x, dout)
    # and return 2 arguments (dx, out)
    EpilogueArguments = GemmActMixin.EpilogueArguments

    @cute.jit
    def epi_visit_subtile(
        self,
        params,
        epi_loop_tensors: Tuple[cute.Tensor, ...],
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor] = None,
    ) -> Optional[cute.Tensor]:
        assert tRS_rC is not None
        # We don't add C to the accumulator
        GemmDefaultEpiMixin.epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None)
        tRS_rC_acc = cute.make_rmem_tensor_like(tRS_rC, self.acc_dtype)
        tRS_rC_acc.store(tRS_rC.load().to(self.acc_dtype))
        # If we don't have .shape here, the compiler generates local stores and loads
        if const_expr(params.act_fn is not None):
            tRS_rAuxOut = cute.make_rmem_tensor(tRS_rD.layout.shape, self.acc_dtype)
            if const_expr(self.arch != 100):
                for i in cutlass.range(cute.size(tRS_rAuxOut), unroll_full=True):
                    tRS_rD[i], tRS_rAuxOut[i] = params.act_fn(tRS_rC_acc[i], tRS_rD[i])
            else:
                for i in cutlass.range(cute.size(tRS_rAuxOut) // 2, unroll_full=True):
                    (
                        (tRS_rD[2 * i], tRS_rD[2 * i + 1]),
                        (tRS_rAuxOut[2 * i], tRS_rAuxOut[2 * i + 1]),
                    ) = params.act_fn(
                        (tRS_rC_acc[2 * i], tRS_rC_acc[2 * i + 1]),
                        (tRS_rD[2 * i], tRS_rD[2 * i + 1]),
                    )
        else:
            tRS_rAuxOut = tRS_rC_acc
        return tRS_rAuxOut


class GemmDActSm90(GemmDActMixin, GemmSm90):
    pass


class GemmDActSm80(GemmDActMixin, GemmSm80):
    pass


class GemmDActSm100(GemmDActMixin, GemmSm100):
    pass


class GemmDActSm120(GemmDActMixin, GemmSm120):
    pass


class GemmDGatedMixin(GemmActMixin):
    # Different from GemmActMixin, here act_bwd_fn must take in 3 arguments (x, y, dout)
    # and return 3 arguments (dx, dy, out)
    _epi_ops = (
        ColVecLoad("mColVecBroadcast"),
        Scalar("sr_seed", dtype=Int32),
        TileStore("mAuxOut"),
        ColVecReduce("mColVecReduce"),
    )
    _extra_param_fields = (("act_bwd_fn", cutlass.Constexpr, None),)

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mAuxOut: cute.Tensor
        act_bwd_fn: cutlass.Constexpr[Callable] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        mColVecReduce: Optional[cute.Tensor] = None
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN
        sr_seed: Optional[Int32 | cute.Tensor] = None

    # EpilogueParams auto-generated from _epi_ops + _extra_param_fields

    def epi_to_underlying_arguments(self, args: EpilogueArguments, *, loc=None, ip=None):
        # C and D are implicitly 2 16-bit elements packed into 32 bits, simply for the purpose
        # for reusing the existing load/store code.
        assert self.implicit_dtype.width == 16, "GemmDGated only supports 16bit for now"
        assert self.d_dtype.width == 32, "D storage type must be 32 bit"
        assert self.c_dtype.width == 32, "C storage type must be 32 bit"
        self.rounding_mode = args.rounding_mode
        self.aux_out_dtype = args.mAuxOut.element_type
        self.aux_out_layout = cutlass.utils.LayoutEnum.from_tensor(args.mAuxOut)
        self.cta_tile_shape_aux_out_mn = self.cta_tile_shape_mnk[:2]
        d = self._epi_ops_to_params_dict(args)
        d["act_bwd_fn"] = args.act_bwd_fn
        return self.EpilogueParams(**d)

    # epi_begin, epi_begin_loop, epi_end are inherited from ComposableEpiMixin via _epi_ops.

    @cute.jit
    def epi_visit_subtile(
        self,
        params,
        epi_loop_tensors: Tuple[cute.Tensor, ...],
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor] = None,
    ) -> Optional[cute.Tensor]:
        tDrColVec = epi_loop_tensors["mColVecBroadcast"]
        tDrColVecReduce = epi_loop_tensors["mColVecReduce"]
        assert tRS_rC is not None
        implicit_dtype = self.implicit_dtype
        assert implicit_dtype.width == 16, "GemmDGatedMixin only supports 16bit for now"
        tRS_rXY_f16x2 = cute.recast_tensor(tRS_rC, implicit_dtype)
        tRS_rXY_f32x2 = cute.make_rmem_tensor(tRS_rXY_f16x2.layout, Float32)
        tRS_rXY_f32x2.store(tRS_rXY_f16x2.load().to(Float32))
        tRS_rdXY_f32x2 = cute.make_rmem_tensor_like(tRS_rXY_f32x2, Float32)
        tRS_rOut = cute.make_rmem_tensor_like(tRS_rD, Float32)
        tRS_rD_scaled = cute.make_rmem_tensor_like(tRS_rD)
        if const_expr(tDrColVec is not None):  # Scale D by colvec
            if const_expr(self.arch != 100):
                tRS_rD_scaled.store(tRS_rD.load() * tDrColVec.load().to(tRS_rD.element_type))
            else:
                tDrColVec_mn = layout_utils.convert_layout_zero_stride(tDrColVec, tDrColVec.layout)
                tRS_rD_mn = layout_utils.convert_layout_zero_stride(tRS_rD, tDrColVec.layout)
                tRS_rD_scaled_mn = layout_utils.convert_layout_zero_stride(
                    tRS_rD_scaled, tDrColVec.layout
                )
                for m in cutlass.range(cute.size(tDrColVec_mn, mode=[0]), unroll_full=True):
                    for n in cutlass.range(
                        cute.size(tDrColVec_mn, mode=[1]) // 2, unroll_full=True
                    ):
                        (
                            tRS_rD_scaled_mn[m, 2 * n],
                            tRS_rD_scaled_mn[m, 2 * n + 1],
                        ) = cute.arch.mul_packed_f32x2(
                            (tRS_rD_mn[m, 2 * n], tRS_rD_mn[m, 2 * n + 1]),
                            (tDrColVec_mn[m, 0], tDrColVec_mn[m, 0]),
                        )
        else:
            tRS_rD_scaled.store(tRS_rD.load())
        if const_expr(self.arch != 100):
            for i in cutlass.range(cute.size(tRS_rD)):
                (
                    tRS_rdXY_f32x2[2 * i],
                    tRS_rdXY_f32x2[2 * i + 1],
                    tRS_rOut[i],
                ) = params.act_bwd_fn(
                    tRS_rXY_f32x2[2 * i], tRS_rXY_f32x2[2 * i + 1], tRS_rD_scaled[i]
                )
        else:
            for i in cutlass.range(cute.size(tRS_rD) // 2):
                (
                    (tRS_rdXY_f32x2[4 * i], tRS_rdXY_f32x2[4 * i + 2]),
                    (tRS_rdXY_f32x2[4 * i + 1], tRS_rdXY_f32x2[4 * i + 3]),
                    (tRS_rOut[2 * i], tRS_rOut[2 * i + 1]),
                ) = params.act_bwd_fn(
                    (tRS_rXY_f32x2[4 * i], tRS_rXY_f32x2[4 * i + 2]),
                    (tRS_rXY_f32x2[4 * i + 1], tRS_rXY_f32x2[4 * i + 3]),
                    (tRS_rD_scaled[2 * i], tRS_rD_scaled[2 * i + 1]),
                )
        if const_expr(tDrColVecReduce is not None):
            # Accumulate postact * dout before D is scaled by colvec_scale
            colvec_reduce_accumulate(self, tDrColVecReduce, tRS_rOut, rScale=tRS_rD)

        if const_expr(tDrColVec is not None):  # Scale Out by colvec
            if const_expr(self.arch != 100):
                tRS_rOut.store(tRS_rOut.load() * tDrColVec.load().to(tRS_rD.element_type))
            else:
                tDrColVec_mn = layout_utils.convert_layout_zero_stride(tDrColVec, tDrColVec.layout)
                tRS_rOut_mn = layout_utils.convert_layout_zero_stride(tRS_rOut, tDrColVec.layout)
                for m in cutlass.range(cute.size(tDrColVec_mn, mode=[0]), unroll_full=True):
                    for n in cutlass.range(
                        cute.size(tDrColVec_mn, mode=[1]) // 2, unroll_full=True
                    ):
                        tRS_rOut_mn[m, 2 * n], tRS_rOut_mn[m, 2 * n + 1] = (
                            cute.arch.mul_packed_f32x2(
                                (tRS_rOut_mn[m, 2 * n], tRS_rOut_mn[m, 2 * n + 1]),
                                (tDrColVec_mn[m, 0], tDrColVec_mn[m, 0]),
                            )
                        )
        # Type conversion
        tRS_rdXY_f16x2 = cute.make_rmem_tensor(tRS_rdXY_f32x2.layout, implicit_dtype)
        tRS_rdXY_f16x2.store(tRS_rdXY_f32x2.load().to(implicit_dtype))
        tRS_rD.store(cute.recast_tensor(tRS_rdXY_f16x2, Float32).load())
        return tRS_rOut

    # epi_end is inherited from ComposableEpiMixin → delegates to ColVecReduce.end()


class GemmDGatedSm90(GemmDGatedMixin, GemmSm90):
    pass


class GemmDGatedSm80(GemmDGatedMixin, GemmSm80):
    pass


class GemmDGatedSm100(GemmDGatedMixin, GemmSm100):
    pass


class GemmDGatedSm120(GemmDGatedMixin, GemmSm120):
    pass


@jit_cache
def _compile_gemm_dact(
    a_dtype,
    b_dtype,
    d_dtype,
    c_dtype,
    postact_dtype,
    implicit_dtype,
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
    colvec_scale_dtype,
    colvec_scale_ndim,
    colvec_reduce_dtype,
    colvec_reduce_ndim,
    varlen_m,
    gather_A,
    device_capacity,
    gemm_cls_name,
    use_tma_gather=False,
):
    is_dgated = gemm_cls_name == "dgated"
    sm_to_cls = {
        "dact": {
            8: GemmDActSm80,
            9: GemmDActSm90,
            10: GemmDActSm100,
            11: GemmDActSm100,
            12: GemmDActSm120,
        },
        "dgated": {
            8: GemmDGatedSm80,
            9: GemmDGatedSm90,
            10: GemmDGatedSm100,
            11: GemmDGatedSm100,
            12: GemmDGatedSm120,
        },
    }
    GemmCls = sm_to_cls[gemm_cls_name][device_capacity[0]]
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
    pa_leading = 1 if postact_major == "n" else 0
    pa_shape = (m, n) if varlen_m else (m, n, l)
    mAuxOut = fake_tensor(postact_dtype, pa_shape, leading_dim=pa_leading, divisibility=div_pa)

    if is_dgated:
        act_fn = dgate_fn_map[activation]

        mColVec = None
        if colvec_scale_ndim == 2:
            mColVec = fake_tensor(colvec_scale_dtype, (l, m), leading_dim=1, divisibility=4)
        elif colvec_scale_ndim == 1:
            mColVec = fake_tensor(colvec_scale_dtype, (m,), leading_dim=0, divisibility=4)
        mColVecReduce = None
        n_tiles = cute.sym_int()
        if colvec_reduce_ndim == 3:
            mColVecReduce = fake_tensor(
                colvec_reduce_dtype,
                (l, m, n_tiles),
                leading_dim=2,
                divisibility=1,
            )
        elif colvec_reduce_ndim == 2:
            mColVecReduce = fake_tensor(
                colvec_reduce_dtype,
                (m, n_tiles),
                leading_dim=1,
                divisibility=1,
            )
        epi_args = GemmCls.EpilogueArguments(
            mAuxOut,
            act_fn,
            mColVecBroadcast=mColVec,
            mColVecReduce=mColVecReduce,
        )

        def _set_implicit_dtype(gemm_obj):
            gemm_obj.implicit_dtype = implicit_dtype

        post_init = _set_implicit_dtype
    else:
        act_fn = dact_fn_map[activation]
        epi_args = GemmCls.EpilogueArguments(mAuxOut, act_fn)
        post_init = None

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
        post_init=post_init,
        use_tma_gather=use_tma_gather,
    )


def gemm_dact(
    A: Tensor,  # (l, m, k) or (total_m, k) if varlen_m or (whatever, k) if gather_A with varlen_m
    B: Tensor,  # (l, n, k)
    Out: Tensor,  # (l, m, n) or (total_m, n) if varlen_m; or (l, m, 2*n)/(total_m, 2*n) if dgated
    PreAct: Tensor,  # same shape as Out
    PostAct: Tensor,  # (l, m, n) or (total_m, n) if varlen_m
    tile_count_semaphore: Optional[Tensor],  # (1,)
    activation: Optional[str],
    tile_M: int,
    tile_N: int,
    cluster_M: int,
    cluster_N: int,
    tile_K: int | None = None,
    pingpong: bool = True,
    persistent: bool = True,
    is_dynamic_persistent: bool = False,
    max_swizzle_size: int = 8,
    colvec_scale: Optional[Tensor] = None,  # (l, m), or (total_m,) if varlen_m (dgated only)
    # (l, m, ceildiv(n, tile_n)), or (total_m, ceildiv(n, tile_n)) if varlen_m (dgated only)
    colvec_reduce: Optional[Tensor] = None,
    cu_seqlens_m: Optional[Tensor] = None,  # (l+1,) cumulative sum of m values for variable length
    A_idx: Optional[Tensor] = None,  # (total_m,) if gather_A with varlen_m
    use_tma_gather: bool = False,
) -> None:
    is_dgated = activation in dgate_fn_map
    if not is_dgated:
        assert activation in dact_fn_map, f"Unsupported activation {activation}"
        assert colvec_scale is None, "colvec_scale is only supported for gated activations"
        assert colvec_reduce is None, "colvec_reduce is only supported for gated activations"
    gemm_cls_name = "dgated" if is_dgated else "dact"

    varlen_m = cu_seqlens_m is not None
    gather_A = A_idx is not None
    if varlen_m:
        assert persistent, "varlen_m requires persistent=True"
        assert A.stride(-1) == 1, "varlen_m requires A to be k-major"
        assert Out.stride(-1) == 1, "varlen_m requires Out to be n-major"
        assert PreAct.stride(-1) == 1, "varlen_m requires PreAct to be n-major"
        assert PostAct.stride(-1) == 1, "varlen_m requires PostAct to be n-major"
    if gather_A:
        assert cu_seqlens_m is not None, "gather_A requires varlen"
        assert cluster_N == 1, "gather_A requires cluster_N=1"

    # For dgated, capture implicit_dtype before viewing Out/PreAct as f32
    implicit_dtype = None
    if is_dgated:
        AB_swapped = Out.stride(-1) != 1
        implicit_dtype = torch2cute_dtype_map[Out.dtype]
        assert Out.element_size() == 2, "Out dtype must be fp16 or bf16"
        assert PreAct.element_size() == 2, "Preact dtype must be fp16 or bf16"
        if varlen_m or not AB_swapped:
            Out = Out.view(torch.float32)
            PreAct = PreAct.view(torch.float32)
        else:
            Out = Out.mT.view(torch.float32).mT
            PreAct = PreAct.mT.view(torch.float32).mT

    A_p = perm3d_single(A, varlen_m)
    B_p = perm3d_single(B)
    Out_p = perm3d_single(Out, varlen_m)
    PreAct_p = perm3d_single(PreAct, varlen_m)
    PostAct_p = perm3d_single(PostAct, varlen_m)

    a_major = get_major(A_p, "m", "k")
    b_major = get_major(B_p, "n", "k")
    d_major = get_major(Out_p, "m", "n")
    c_major = get_major(PreAct_p, "m", "n")
    postact_major = get_major(PostAct_p, "m", "n")

    a_dtype = torch2cute_dtype_map[A.dtype]
    b_dtype = torch2cute_dtype_map[B.dtype]
    d_dtype = torch2cute_dtype_map[Out.dtype]
    c_dtype = torch2cute_dtype_map[PreAct.dtype]
    postact_dtype = torch2cute_dtype_map[PostAct.dtype]

    device_capacity = get_device_capacity(A.device)
    assert device_capacity[0] in [8, 9, 10, 11, 12], (
        "Only SM8x, SM90, SM100, SM110, and SM120 are supported"
    )

    if is_dynamic_persistent and device_capacity[0] == 9:
        assert tile_count_semaphore is not None, (
            "Dynamic persistent tile scheduler in SM90 requires a semaphore in GMEM"
        )

    compiled_fn = _compile_gemm_dact(
        a_dtype,
        b_dtype,
        d_dtype,
        c_dtype,
        postact_dtype,
        implicit_dtype,
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
        torch2cute_dtype_map[colvec_scale.dtype] if colvec_scale is not None else None,
        colvec_scale.ndim if colvec_scale is not None else 0,
        torch2cute_dtype_map[colvec_reduce.dtype] if colvec_reduce is not None else None,
        colvec_reduce.ndim if colvec_reduce is not None else 0,
        varlen_m,
        gather_A,
        device_capacity,
        gemm_cls_name,
        use_tma_gather=use_tma_gather,
    )

    from .cache_utils import COMPILE_ONLY

    if COMPILE_ONLY:
        return

    max_active_clusters = get_max_active_clusters(cluster_M * cluster_N) if persistent else 0
    if is_dgated:
        epi_args = GemmDGatedMixin.EpilogueArguments(
            PostAct_p,
            None,  # act_bwd_fn is Constexpr
            mColVecBroadcast=colvec_scale,
            mColVecReduce=colvec_reduce,
            rounding_mode=None,
            sr_seed=None,
        )
    else:
        epi_args = GemmDActMixin.EpilogueArguments(
            PostAct_p,
            None,
            rounding_mode=None,
            sr_seed=None,
        )
    scheduler_args = make_scheduler_args(
        max_active_clusters,
        max_swizzle_size,
        tile_count_semaphore,
    )
    varlen_args = make_varlen_args(cu_seqlens_m, None, A_idx)

    if device_capacity[0] in [10, 11]:
        compiled_fn(
            A_p, B_p, Out_p, PreAct_p, epi_args, scheduler_args, varlen_args, None, None, None
        )
    else:
        compiled_fn(A_p, B_p, Out_p, PreAct_p, epi_args, scheduler_args, varlen_args, None)


gemm_dgated = gemm_dact
