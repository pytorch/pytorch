# Copyright (c) 2025, Siyu Wang, Shengbin Di, Yuxi Chi, Johnsonms, Linfeng Zheng, Haoyan Huang, Lanbo Li, Yun Zhong, Man Yuan, Minmin Sun, Yong Li, Wei Lin.

"""Fused multi-head attention (FMHA) backward for the SM100 architecture using CUTE DSL.

Constraints:
* Supported head dimensions: 256 only
* cta_tiler_mn must be 64,128
* Batch size must be the same for Q, K, and V tensors
"""

import enum
import math

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.typing import Int32
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from cutlass.utils import ClcDynamicPersistentTileScheduler
from .tile_scheduler import (
    ClcState,
    SM100_TMEM_CAPACITY_COLUMNS,
    make_sm100_thread_cooperative_group as make_thread_cooperative_group,
    Sm100FmhaClcDynamicTileSchedulerParams as FmhaClcDynamicTileSchedulerParams,
    Sm100FmhaClcDynamicTileScheduler as FmhaClcDynamicTileScheduler,
    Sm100FmhaStaticTileSchedulerParams as FmhaStaticTileSchedulerParams,
)

from . import copy_utils as fa_copy_utils

LAYOUT_RANK_CONSTANT = 3


@cute.jit
def split_wg(
    t: cute.Tensor,
    num_warp_groups: Int32,
    wg_idx: Int32,
) -> cute.Tensor:
    """Split warp group."""
    # dishengbin, TODO：need to double check if more efficient to split in other dimensions
    ret = None
    if cutlass.const_expr(cute.rank(t.layout) == LAYOUT_RANK_CONSTANT):
        p = cute.composition(
            t,
            cute.make_layout(
                (
                    t.shape[0],
                    t.shape[1],
                    (num_warp_groups, cute.size(t, mode=[2]) // num_warp_groups),
                )
            ),
        )
        ret = p[None, None, (wg_idx, None)]
    else:
        p = cute.composition(
            t,
            cute.make_layout(
                (
                    t.shape[0],
                    t.shape[1],
                    t.shape[2],
                    (num_warp_groups, cute.size(t, mode=[3]) // num_warp_groups),
                )
            ),
        )
        ret = p[None, None, None, (wg_idx, None)]
    return ret


class MaskType(enum.Enum):
    """Mask type used in FMHA backward."""

    NO_MASK = enum.auto()
    RESIDUAL_MASK_FOR_BACKWARD = enum.auto()
    CAUSAL_MASK_FOR_BACKWARD = enum.auto()


def Tmemory_offset(lane, col):
    """Tensor memory offset."""
    return (lane << 16) + col


permute_order = (0, 1, 2, 3, 4)


class BlackwellFusedMultiHeadAttentionBackwardDKDVKernel:
    """FMHA backward class for executing CuTeDSL kernel."""

    def __init__(
        self,
        acc_dtype: type[cutlass.Numeric],
        cta_tiler: tuple[int, int, int],
        is_causal: bool,
        window_size_left: int | None,
        window_size_right: int | None,
        use_clc_scheduler: bool = False,
    ):
        """Initialization."""
        self.acc_dtype = acc_dtype
        self.cta_tiler = cta_tiler
        self.use_clc_scheduler = use_clc_scheduler
        self.sched_warp_id = 10 if use_clc_scheduler else None
        # TODO: need check, not sure whether need to *2 if 2cta
        self.tile_shape_Q = cta_tiler[0]
        self.tile_shape_K = cta_tiler[1]
        self.tile_shape_dQ_K = cta_tiler[2]
        self.tile_shape_dV_dO = cta_tiler[2]
        # For S
        self.KQ_mma_tiler = (
            cta_tiler[1] * 2,
            cta_tiler[0],
            cta_tiler[2],
        )
        # For dP
        self.VdO_mma_tiler = (
            cta_tiler[1] * 2,
            cta_tiler[0],
            cta_tiler[2],
        )
        # For dV
        self.PdO_mma_tiler = (
            cta_tiler[1] * 2,
            cta_tiler[2],
            cta_tiler[0],
        )
        # For dK
        self.dSQ_mma_tiler = (
            cta_tiler[1] * 2,
            cta_tiler[2],
            cta_tiler[0],
        )
        # For dQ, dishengbin, need to remove
        self.dSK_mma_tiler = (
            cta_tiler[0] * 2,
            cta_tiler[2],
            cta_tiler[1],
        )
        self.cluster_shape_mn = (2, 1)
        self.is_causal = is_causal
        self.window_size_left: int = -1 if window_size_left is None else window_size_left
        self.window_size_right: int = -1 if window_size_right is None else window_size_right
        self.has_sliding_window = False
        if self.window_size_left > 0 or self.window_size_right > 0:
            self.has_sliding_window = True
        if self.is_causal:
            self.window_size_right = 0

        self.compute_warp_id = (0, 1, 2, 3, 4, 5, 6, 7)
        self.mma_warp_id = 8
        self.load_warp_id = 9
        self.empty_warp_id = 10

        self.num_compute_warps = 8

        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * (self.num_compute_warps + 4)

        self.cta_sync_bar_id = 0
        self.tmem_alloc_sync_bar_id = 1
        self.compute_sync_bar_id = 2
        self.epilogue_sync_bar_id = 3
        self.reduce_sync_bar_id = 4

        self.tmem_dK_offset = 0
        self.tmem_dV_offset = Tmemory_offset(0, cta_tiler[2] // 2)
        self.tmem_dP_offset = Tmemory_offset(0, cta_tiler[2] + cta_tiler[0] // 2)
        self.tmem_S_offset = Tmemory_offset(0, cta_tiler[2])

        self.num_regs_reduce = 152
        self.num_regs_compute = 128
        self.num_regs_mma = 128
        self.num_regs_empty = 96
        self.num_regs_load = 96

        self.buffer_align_bytes = 128

    def _setup_attributes(self):
        """Settings for pipeline stage."""
        self.load_mma_Q_stage = 1
        self.load_mma_K_stage = 1
        self.load_mma_V_stage = 1
        self.load_mma_QT_stage = 1
        self.load_mma_dO_stage = 1
        self.load_compute_LSE_stage = 1
        self.load_compute_sum_OdO_stage = 1
        self.mma_compute_S_stage = 1
        self.mma_compute_dP_stage = 1
        self.compute_mma_P_stage = 1
        self.compute_mma_dS_stage = 1
        self.mma_compute_dKdV_stage = 2

        if cutlass.const_expr(self.use_clc_scheduler):
            self.num_clc_stage = 1
            self.num_clc_response_bytes = 16

    @cute.jit
    def __call__(
        self,
        Q: cute.Tensor,
        K: cute.Tensor,
        V: cute.Tensor,
        dK: cute.Tensor,
        dV: cute.Tensor,
        dO: cute.Tensor,
        scaled_LSE: cute.Tensor,
        sum_OdO: cute.Tensor,
        cumulative_s_q: cute.Tensor | None,
        cumulative_s_k: cute.Tensor | None,
        scale_softmax: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        """Host function to launch CuTeDSL kernel."""
        varlen = cumulative_s_q is not None or cumulative_s_k is not None
        # Infer shape metadata from normalized 5D tensors (B, S, H_k, H_r, D).
        h_r = Q.shape[3]
        h_k = Q.shape[2]
        if cutlass.const_expr(cumulative_s_q is not None):
            b = cumulative_s_q.shape[0] - 1
        elif cutlass.const_expr(cumulative_s_k is not None):
            b = cumulative_s_k.shape[0] - 1
        else:
            b = Q.shape[0]
        problem_shape = (
            Q.shape[1],
            K.shape[1],
            Q.shape[4],
            ((h_r, h_k), b),
        )
        hb = ((h_r, h_k), b)
        # (b, s, h_k, h_r, d) -> (s, d, ((h_r, h_k), b))
        Q = cute.make_tensor(
            Q.iterator,
            cute.make_layout(
                (Q.shape[1], Q.shape[4], hb),
                stride=(
                    cute.assume(Q.stride[1], divby=64),
                    Q.stride[4],
                    (
                        (Q.stride[3], Q.stride[2]),
                        0 if cumulative_s_q is not None else cute.assume(Q.stride[0], divby=64),
                    ),
                ),
            ),
        )
        # (b, s, h_k, 1, d) -> (s, d, ((1, h_k), b))
        K = cute.make_tensor(
            K.iterator,
            cute.make_layout(
                (K.shape[1], K.shape[4], hb),
                stride=(
                    cute.assume(K.stride[1], divby=64),
                    K.stride[4],
                    (
                        (0, K.stride[2]),
                        0 if cumulative_s_k is not None else cute.assume(K.stride[0], divby=64),
                    ),
                ),
            ),
        )
        # (b, s, h_k, 1, d) -> (s, d, ((1, h_k), b))
        V = cute.make_tensor(
            V.iterator,
            cute.make_layout(
                (V.shape[1], V.shape[4], hb),
                stride=(
                    cute.assume(V.stride[1], divby=64),
                    V.stride[4],
                    (
                        (0, V.stride[2]),
                        0 if cumulative_s_k is not None else cute.assume(V.stride[0], divby=64),
                    ),
                ),
            ),
        )
        # (s, d, ((h_r, h_k), b)) -> (d, s, ((h_r, h_k), b))
        QT = cute.make_tensor(
            Q.iterator,
            cute.make_layout(
                (Q.shape[1], Q.shape[0], Q.shape[2]),
                stride=(
                    Q.stride[1],
                    Q.stride[0],
                    Q.stride[2],
                ),
            ),
        )
        dK = cute.make_tensor(
            dK.iterator,
            cute.make_layout(
                (dK.shape[1], dK.shape[4], hb),
                stride=(
                    cute.assume(dK.stride[1], divby=64),
                    dK.stride[4],
                    (
                        (0, dK.stride[2]),
                        0 if cumulative_s_k is not None else cute.assume(dK.stride[0], divby=64),
                    ),
                ),
            ),
        )
        dV = cute.make_tensor(
            dV.iterator,
            cute.make_layout(
                (dV.shape[1], dV.shape[4], hb),
                stride=(
                    cute.assume(dV.stride[1], divby=64),
                    dV.stride[4],
                    (
                        (0, dV.stride[2]),
                        0 if cumulative_s_k is not None else cute.assume(dV.stride[0], divby=64),
                    ),
                ),
            ),
        )
        # (s, d, ((h_r, h_k), b))
        dO = cute.make_tensor(
            dO.iterator,
            cute.make_layout(
                (dO.shape[1], dO.shape[4], hb),
                stride=(
                    cute.assume(dO.stride[1], divby=64),
                    dO.stride[4],
                    (
                        (dO.stride[3], dO.stride[2]),
                        0 if cumulative_s_q is not None else cute.assume(dO.stride[0], divby=64),
                    ),
                ),
            ),
        )

        # (s, d, ((h_r, h_k), b)) -> (d, s, ((h_r, h_k), b))
        dOT = cute.make_tensor(
            dO.iterator,
            cute.make_layout(
                (dO.shape[1], dO.shape[0], dO.shape[2]),
                stride=(
                    dO.stride[1],
                    dO.stride[0],
                    dO.stride[2],
                ),
            ),
        )

        self.Q_major_mode = utils.LayoutEnum.from_tensor(Q).mma_major_mode()
        self.K_major_mode = utils.LayoutEnum.from_tensor(K).mma_major_mode()
        self.dK_major_mode = utils.LayoutEnum.from_tensor(dK).mma_major_mode()
        self.V_major_mode = utils.LayoutEnum.from_tensor(V).mma_major_mode()
        self.dV_major_mode = utils.LayoutEnum.from_tensor(dV).mma_major_mode()
        self.dO_major_mode = utils.LayoutEnum.from_tensor(dO).mma_major_mode()

        if cutlass.const_expr(self.Q_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError(f"The layout of q is not supported: {self.Q_major_mode}")
        if cutlass.const_expr(self.K_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of k is not supported")
        if cutlass.const_expr(self.dK_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of dk is not supported")
        if cutlass.const_expr(self.V_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of v is not supported")
        if cutlass.const_expr(self.dV_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of dv is not supported")

        self._setup_attributes()

        cta_group = tcgen05.CtaGroup.TWO
        PT_source = tcgen05.OperandSource.SMEM

        # compute S
        KQ_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            K.element_type,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            cta_group,
            self.KQ_mma_tiler[:2],
        )
        # compute dP
        VdO_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            V.element_type,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            cta_group,
            self.VdO_mma_tiler[:2],
        )
        # compute dV
        PdO_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            dO.element_type,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            cta_group,
            self.PdO_mma_tiler[:2],
            PT_source,
        )
        # compute dK
        dSQ_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            Q.element_type,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            cta_group,
            self.dSQ_mma_tiler[:2],
        )
        # compute dQ
        # dishengbin, need to remove, but used in dS_mem_layout_staged
        dSK_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            K.element_type,
            tcgen05.OperandMajorMode.MN,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            cta_group,
            self.dSK_mma_tiler[:2],
        )

        atom_thr_size = cute.size(KQ_tiled_mma.thr_id.shape)
        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)  # type: ignore[assignment]
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (atom_thr_size,),
        )

        K_smem_layout_staged = sm100_utils.make_smem_layout_a(
            KQ_tiled_mma,
            self.KQ_mma_tiler,
            K.element_type,
            1,
        )
        Q_smem_layout_staged = sm100_utils.make_smem_layout_b(
            KQ_tiled_mma,
            self.KQ_mma_tiler,
            Q.element_type,
            self.load_mma_Q_stage,
        )
        V_smem_layout_staged = sm100_utils.make_smem_layout_a(
            VdO_tiled_mma,
            self.VdO_mma_tiler,
            V.element_type,
            1,
        )
        dO_smem_layout_staged = sm100_utils.make_smem_layout_b(
            VdO_tiled_mma,
            self.VdO_mma_tiler,
            dO.element_type,
            self.load_mma_dO_stage,
        )
        # dishengbin, need to remove, but used for sPT, need to double check
        dS_smem_layout_staged = sm100_utils.make_smem_layout_a(
            dSK_tiled_mma,
            self.dSK_mma_tiler,
            Q.element_type,
            self.compute_mma_dS_stage,
        )
        dST_smem_layout_staged = sm100_utils.make_smem_layout_a(
            dSQ_tiled_mma,
            self.dSQ_mma_tiler,
            Q.element_type,
            self.compute_mma_dS_stage,
        )
        tiled_mma = dSQ_tiled_mma
        is_k_major = tiled_mma.op.a_major_mode == tcgen05.OperandMajorMode.K
        a_major_mode = tcgen05.OperandMajorMode.K if is_k_major else tcgen05.OperandMajorMode.MN
        tmp = cute.dice(self.dSQ_mma_tiler, (1, None, 1))
        a_smem_shape = tiled_mma.partition_shape_A(
            cute.dice(self.dSQ_mma_tiler, (1, None, 1)),
        )
        a_smem_shape_mn_k = (
            cute.size(a_smem_shape[0][0]) * a_smem_shape[1],
            cute.size(a_smem_shape[0][1]) * a_smem_shape[2],
        )
        smem_layout_atom_kind = sm100_utils.get_smem_layout_atom_ab(
            a_major_mode,
            K.element_type,
            a_smem_shape_mn_k,
        )
        a_smem_layout_atom = sm100_utils.make_smem_layout_atom(
            smem_layout_atom_kind,
            K.element_type,
        )

        a_smem_shape = cute.append(
            a_smem_shape,
            self.compute_mma_dS_stage,
        )
        order = (2, 1, 3) if not is_k_major else (1, 2, 3)
        dST_smem_layout_staged_tmp = sm100_utils.tile_to_mma_shape(
            a_smem_layout_atom,
            a_smem_shape,
            order=order,
        )
        QT_smem_layout_staged = sm100_utils.make_smem_layout_b(
            dSQ_tiled_mma,
            self.dSQ_mma_tiler,
            Q.element_type,
            self.load_mma_QT_stage,
        )
        P_smem_layout_staged = sm100_utils.make_smem_layout_a(
            PdO_tiled_mma,
            self.PdO_mma_tiler,
            Q.element_type,
            self.compute_mma_P_stage,
        )
        dOT_smem_layout_staged = sm100_utils.make_smem_layout_b(
            PdO_tiled_mma,
            self.PdO_mma_tiler,
            dO.element_type,
            self.load_mma_dO_stage,
        )
        LSE_smem_layout = cute.make_layout((self.cta_tiler[0], self.load_compute_LSE_stage))
        sum_OdO_smem_layout = cute.make_layout((self.cta_tiler[0], self.load_compute_sum_OdO_stage))

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_reduce_op = cpasync.CopyReduceBulkTensorTileS2GOp()

        K_smem_layout = cute.select(K_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_K, tma_tensor_K = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            K,
            K_smem_layout,
            self.KQ_mma_tiler,
            KQ_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        V_smem_layout = cute.select(V_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_V, tma_tensor_V = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            V,
            V_smem_layout,
            self.VdO_mma_tiler,
            VdO_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        Q_smem_layout = cute.select(Q_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            Q,
            Q_smem_layout,
            self.KQ_mma_tiler,
            KQ_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        QT_smem_layout = cute.select(QT_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_QT, tma_tensor_QT = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            QT,
            QT_smem_layout,
            self.dSQ_mma_tiler,
            dSQ_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        dO_smem_layout = cute.select(dO_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_dO, tma_tensor_dO = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            dO,
            dO_smem_layout,
            self.VdO_mma_tiler,
            VdO_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        dOT_smem_layout = cute.select(dOT_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_dOT, tma_tensor_dOT = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            dOT,
            dOT_smem_layout,
            self.PdO_mma_tiler,
            PdO_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # for 2cta, tma_copy_QT_bytes is same as the tma_copy_Q_bytes
        self.tma_copy_Q_bytes = cute.size_in_bytes(Q.element_type, Q_smem_layout) * atom_thr_size
        self.tma_copy_K_bytes = cute.size_in_bytes(K.element_type, K_smem_layout) * atom_thr_size
        self.tma_copy_V_bytes = cute.size_in_bytes(V.element_type, V_smem_layout) * atom_thr_size
        self.tma_copy_dO_bytes = cute.size_in_bytes(dO.element_type, dO_smem_layout) * atom_thr_size

        # Variant 3a epilogue: TMA store atoms (S2G) for dK / dV.
        # Each compute warp group owns half the hd_v output via split_wg, so
        # the per-WG epi tile is (cta_tiler[1], cta_tiler[2] / num_compute_wgs).
        # SMEM staging will alias onto sP+sdST in subsequent commits; for now
        # the atoms and layouts are built and threaded through but unused.
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
        num_compute_wgs = self.num_compute_warps // 4
        # Variant 3a Path 2: CTA-shared epilogue SMEM. epi_tile is (M, gcd(128B, ...))
        # = (M, 64) for bf16 hd=256. Total stages = num_compute_wgs * num_epi_stages
        # = 4 stages of (64, 64), virtually a per-CTA (64, 256) buffer aliased onto
        # sP+sdST. Both warp-groups cooperatively populate this buffer; TMA fires
        # one (64, 64) box per stage to the corresponding (64, 64) GMEM slice.
        epi_cols_dKV = math.gcd(
            128 // (dK.element_type.width // 8), self.cta_tiler[2] // num_compute_wgs
        )
        num_epi_stages_dKV = (self.cta_tiler[2] // num_compute_wgs) // epi_cols_dKV
        epi_tile_dKV = (self.cta_tiler[1], epi_cols_dKV)
        total_epi_stages = num_compute_wgs * num_epi_stages_dKV
        dK_layout_enum = utils.LayoutEnum.from_tensor(dK)
        dV_layout_enum = utils.LayoutEnum.from_tensor(dV)
        sdK_epi_layout = sm100_utils.make_smem_layout_epi(
            dK.element_type,
            dK_layout_enum,
            epi_tile_dKV,
            total_epi_stages,
        )
        sdV_epi_layout = sm100_utils.make_smem_layout_epi(
            dV.element_type,
            dV_layout_enum,
            epi_tile_dKV,
            total_epi_stages,
        )
        tma_atom_dK, tma_tensor_dK = cpasync.make_tiled_tma_atom(
            tma_store_op,
            dK,
            cute.select(sdK_epi_layout, mode=[0, 1]),
            epi_tile_dKV,
        )
        tma_atom_dV, tma_tensor_dV = cpasync.make_tiled_tma_atom(
            tma_store_op,
            dV,
            cute.select(sdV_epi_layout, mode=[0, 1]),
            epi_tile_dKV,
        )

        @cute.struct
        class SharedStorage:
            # Pipeline barriers
            load_mma_Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_mma_Q_stage * 2]
            load_mma_K_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_mma_K_stage * 2]
            load_mma_V_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_mma_V_stage * 2]
            load_mma_QT_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_mma_QT_stage * 2]
            load_mma_dO_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_mma_dO_stage * 2]
            load_mma_dOT_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_mma_dO_stage * 2]
            load_compute_lse_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.load_compute_LSE_stage * 2
            ]
            load_compute_sum_OdO_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.load_compute_sum_OdO_stage * 2
            ]
            mma_compute_S_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.mma_compute_S_stage * 2
            ]
            mma_compute_dP_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.mma_compute_dP_stage * 2
            ]
            compute_mma_P_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.compute_mma_P_stage * 2
            ]
            compute_mma_dS_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.compute_mma_dS_stage * 2
            ]
            mma_compute_dKdV_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.mma_compute_dKdV_stage * 2
            ]
            tmem_holding_buf: cutlass.Int32
            tmem_dealloc_mbar: cutlass.Int64
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            clc_response: cute.struct.MemRange[Int32, 4]
            # Smem tensors
            sK: cute.struct.Align[
                cute.struct.MemRange[K.element_type, cute.cosize(K_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            # only used in 2cta
            sV: cute.struct.Align[
                cute.struct.MemRange[V.element_type, cute.cosize(V_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[Q.element_type, cute.cosize(Q_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sQT: cute.struct.Align[
                cute.struct.MemRange[Q.element_type, cute.cosize(QT_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sdO: cute.struct.Align[
                cute.struct.MemRange[dO.element_type, cute.cosize(dO_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sdOT: cute.struct.Align[
                cute.struct.MemRange[dO.element_type, cute.cosize(dOT_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            # only used in 2cta
            # dishengbin checked whether we need sP
            sP: cute.struct.Align[
                cute.struct.MemRange[Q.element_type, cute.cosize(P_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sdST: cute.struct.Align[
                cute.struct.MemRange[Q.element_type, cute.cosize(dST_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

            sLSE: cute.struct.Align[
                cute.struct.MemRange[self.acc_dtype, cute.cosize(LSE_smem_layout)],
                self.buffer_align_bytes,
            ]
            sSum_OdO: cute.struct.Align[
                cute.struct.MemRange[self.acc_dtype, cute.cosize(sum_OdO_smem_layout)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # =============================== bwd ===============================
        K_val = problem_shape[1]
        _, H_K = problem_shape[3][0]
        B = problem_shape[3][1]
        problem_shape_mbh = (
            cute.ceil_div(K_val, self.cta_tiler[1]),
            cute.size(B),
            cute.size(H_K),
        )
        if cutlass.const_expr(self.use_clc_scheduler):
            self.tile_sched_params = FmhaClcDynamicTileSchedulerParams(
                problem_shape_mbh,
                (*self.cluster_shape_mn, 1),
            )
            bwd_grid = FmhaClcDynamicTileScheduler.get_grid_shape(self.tile_sched_params)
        else:
            self.tile_sched_params = FmhaStaticTileSchedulerParams(
                is_persistent=False,
                problem_shape_mbh=problem_shape_mbh,
            )
            bwd_grid = self._compute_bwd_grid(problem_shape, self.cta_tiler[1])
            bwd_grid = cute.round_up(bwd_grid, self.cluster_shape_mnk)

        self.dkdv_bwd(
            KQ_tiled_mma,
            VdO_tiled_mma,
            PdO_tiled_mma,
            dSQ_tiled_mma,
            tma_atom_K,
            tma_tensor_K,
            K,
            tma_atom_V,
            tma_tensor_V,
            tma_atom_Q,
            tma_tensor_Q,
            Q,
            tma_atom_QT,
            tma_tensor_QT,
            tma_atom_dO,
            tma_tensor_dO,
            tma_atom_dOT,
            tma_tensor_dOT,
            dK,
            dV,
            tma_atom_dK,
            tma_tensor_dK,
            tma_atom_dV,
            tma_tensor_dV,
            scaled_LSE,
            scale_softmax,
            sum_OdO,
            problem_shape,
            cumulative_s_q,
            cumulative_s_k,
            self.cluster_layout_vmnk,
            K_smem_layout_staged,
            Q_smem_layout_staged,
            V_smem_layout_staged,
            dO_smem_layout_staged,
            dS_smem_layout_staged,
            dST_smem_layout_staged,
            QT_smem_layout_staged,
            dOT_smem_layout_staged,
            P_smem_layout_staged,
            LSE_smem_layout,
            sum_OdO_smem_layout,
            sdK_epi_layout,
            sdV_epi_layout,
            self.tile_sched_params,
        ).launch(
            grid=bwd_grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            smem=self.shared_storage.size_in_bytes(),  # type: ignore[attr-defined]
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def dkdv_bwd(
        self,
        KQ_tiled_mma: cute.TiledMma,
        VdO_tiled_mma: cute.TiledMma,
        PdO_tiled_mma: cute.TiledMma,
        dSQ_tiled_mma: cute.TiledMma,
        tma_atom_K: cute.CopyAtom,
        K_in: cute.Tensor,
        K_ref: cute.Tensor,
        tma_atom_V: cute.CopyAtom,
        V_in: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        Q_in: cute.Tensor,
        Q_ref: cute.Tensor,
        tma_atom_QT: cute.CopyAtom,
        QT_in: cute.Tensor,
        tma_atom_dO: cute.CopyAtom,
        dO_in: cute.Tensor,
        tma_atom_dOT: cute.CopyAtom,
        dOT_in: cute.Tensor,
        dK: cute.Tensor,
        dV: cute.Tensor,
        tma_atom_dK: cute.CopyAtom,
        dK_tma: cute.Tensor,
        tma_atom_dV: cute.CopyAtom,
        dV_tma: cute.Tensor,
        LSE: cute.Tensor,
        scale_softmax: cutlass.Float32,
        sum_OdO: cute.Tensor,
        problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32], Int32]],
        cumulative_s_q: cute.Tensor | None,
        cumulative_s_k: cute.Tensor | None,
        cluster_layout_vmnk: cute.Layout,
        K_smem_layout_staged: cute.ComposedLayout,
        Q_smem_layout_staged: cute.ComposedLayout,
        V_smem_layout_staged: cute.ComposedLayout,
        dO_smem_layout_staged: cute.ComposedLayout,
        dS_smem_layout_staged: cute.ComposedLayout,
        dST_smem_layout_staged: cute.ComposedLayout,
        QT_smem_layout_staged: cute.ComposedLayout,
        dOT_smem_layout_staged: cute.ComposedLayout,
        P_smem_layout_staged: cute.ComposedLayout,
        LSE_smem_layout: cute.Layout,
        sum_OdO_smem_layout: cute.Layout,
        sdK_epi_layout: cute.ComposedLayout,
        sdV_epi_layout: cute.ComposedLayout,
        tile_sched_params: FmhaStaticTileSchedulerParams | FmhaClcDynamicTileSchedulerParams,
    ):
        """Core CuTeDSL backward kernel."""
        bidx, bidy, bidz = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        varlen = cumulative_s_q is not None or cumulative_s_k is not None

        mma_tile_coord_v = bidx % cute.size(KQ_tiled_mma.thr_id.shape)

        if warp_idx == self.load_warp_id:
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_QT)
            cpasync.prefetch_descriptor(tma_atom_V)
            cpasync.prefetch_descriptor(tma_atom_dO)
            cpasync.prefetch_descriptor(tma_atom_dOT)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_mma_Q_producer, load_mma_Q_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.load_mma_Q_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_Q_bytes,
            barrier_storage=storage.load_mma_Q_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_mma_K_producer, load_mma_K_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.load_mma_K_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_K_bytes,
            barrier_storage=storage.load_mma_K_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_mma_V_producer, load_mma_V_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.load_mma_V_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_V_bytes,
            barrier_storage=storage.load_mma_V_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_mma_QT_producer, load_mma_QT_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.load_mma_QT_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_Q_bytes,
            barrier_storage=storage.load_mma_QT_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_mma_dO_producer, load_mma_dO_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.load_mma_dO_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_dO_bytes,
            barrier_storage=storage.load_mma_dO_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_mma_dOT_producer, load_mma_dOT_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.load_mma_dO_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_dO_bytes,
            barrier_storage=storage.load_mma_dOT_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_compute_LSE_producer, load_compute_LSE_consumer = pipeline.PipelineCpAsync.create(
            num_stages=self.load_compute_LSE_stage,
            producer_group=make_thread_cooperative_group(self.threads_per_warp),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * self.num_compute_warps
            ),
            barrier_storage=storage.load_compute_lse_mbar_ptr.data_ptr(),
        ).make_participants()
        load_compute_sum_OdO_producer, load_compute_sum_OdO_consumer = (
            pipeline.PipelineCpAsync.create(
                num_stages=self.load_compute_sum_OdO_stage,
                producer_group=make_thread_cooperative_group(self.threads_per_warp),
                consumer_group=make_thread_cooperative_group(
                    self.threads_per_warp * self.num_compute_warps
                ),
                barrier_storage=storage.load_compute_sum_OdO_mbar_ptr.data_ptr(),
            ).make_participants()
        )
        mma_compute_S_producer, mma_compute_S_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_compute_S_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                self.num_compute_warps * self.threads_per_warp * cluster_layout_vmnk.shape[0][0]
            ),
            barrier_storage=storage.mma_compute_S_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        mma_compute_dP_producer, mma_compute_dP_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_compute_dP_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                self.num_compute_warps * self.threads_per_warp * cluster_layout_vmnk.shape[0][0]
            ),
            barrier_storage=storage.mma_compute_dP_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        compute_mma_P_producer, compute_mma_P_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.compute_mma_P_stage,
            producer_group=make_thread_cooperative_group(
                self.num_compute_warps * self.threads_per_warp * cluster_layout_vmnk.shape[0][0]
            ),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.compute_mma_P_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        compute_mma_dS_producer, compute_mma_dS_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.compute_mma_dS_stage,
            producer_group=make_thread_cooperative_group(
                self.num_compute_warps * self.threads_per_warp * cluster_layout_vmnk.shape[0][0]
            ),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.compute_mma_dS_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        mma_compute_dKdV_producer, mma_compute_dKdV_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_compute_dKdV_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                self.num_compute_warps * self.threads_per_warp * cluster_layout_vmnk.shape[0][0]
            ),
            barrier_storage=storage.mma_compute_dKdV_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        cute.arch.barrier(barrier_id=self.cta_sync_bar_id, number_of_threads=self.threads_per_cta)

        # setup mma
        sQ = storage.sQ.get_tensor(Q_smem_layout_staged.outer, swizzle=Q_smem_layout_staged.inner)
        sK = storage.sK.get_tensor(K_smem_layout_staged.outer, swizzle=K_smem_layout_staged.inner)
        sV = storage.sV.get_tensor(V_smem_layout_staged.outer, swizzle=V_smem_layout_staged.inner)
        sdO = storage.sdO.get_tensor(
            dO_smem_layout_staged.outer, swizzle=dO_smem_layout_staged.inner
        )
        sLSE = storage.sLSE.get_tensor(LSE_smem_layout)
        sSum_OdO = storage.sSum_OdO.get_tensor(sum_OdO_smem_layout)
        tmem_holding_buf = storage.tmem_holding_buf
        # for 2cta, QT use different mem from Q

        sQT = storage.sQT.get_tensor(
            QT_smem_layout_staged.outer, swizzle=QT_smem_layout_staged.inner
        )
        sdST = storage.sdST.get_tensor(
            dST_smem_layout_staged.outer, swizzle=dST_smem_layout_staged.inner
        )
        tP_fake_ptr = cute.make_ptr(sQ.element_type, 0, cute.AddressSpace.tmem)
        tP = cute.make_tensor(tP_fake_ptr, P_smem_layout_staged.outer)

        sP = storage.sP.get_tensor(P_smem_layout_staged.outer, swizzle=P_smem_layout_staged.inner)

        sdOT = storage.sdOT.get_tensor(
            dOT_smem_layout_staged.outer, swizzle=dOT_smem_layout_staged.inner
        )

        # tSTrK shape : (MMA, MMA_M, MMA_K, STAGE)
        tSTrK = KQ_tiled_mma.make_fragment_A(sK)
        # tSTrQ shape : (MMA, MMA_N, MMA_K, STAGE)
        tSTrQ = KQ_tiled_mma.make_fragment_B(sQ)

        # tdPTrV shape : (MMA, MMA_M, MMA_K, STAGE)
        tdPTrV = VdO_tiled_mma.make_fragment_A(sV)
        # tdPTrdO shape : (MMA, MMA_N, MMA_K, STAGE)
        tdPTrdO = VdO_tiled_mma.make_fragment_B(sdO)

        # tdKrdST shape: (MMA, MMA_M, MMA_K, STAGE)
        tdKrdST = dSQ_tiled_mma.make_fragment_A(sdST)
        # tdKrQT shape : (MMA, MMA_N, MMA_K, STAGE)
        tdKrQT = dSQ_tiled_mma.make_fragment_B(sQT)

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=self.threads_per_cta,
        )

        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.load_warp_id,
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        tmem.allocate(self.tmem_alloc_cols)

        # wait for tmem allocation and retrieve the pointer
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

        # Cluster arrive after barrier init
        # is_relaxed=False has memory consistency guarantee
        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=False)

        if cutlass.const_expr(self.use_clc_scheduler):
            clc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
            cluster_size = cute.size(self.cluster_shape_mnk)
            num_clc_consumer_threads = self.threads_per_warp * (
                1  # sched_warp (CTA 0 only)
                + cluster_size
                * (
                    len(self.compute_warp_id)
                    + 1  # mma_warp
                    + 1  # load_warp
                )
            )
            clc_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_clc_consumer_threads
            )
            clc_response_ptr = storage.clc_response.data_ptr()
            clc = ClcState.create(
                hw_scheduler=ClcDynamicPersistentTileScheduler.create(
                    self.tile_sched_params.clc_hw_params(),
                    cute.arch.block_idx(),
                    cute.arch.grid_dim(),
                    clc_response_ptr,
                ),
                pipeline=pipeline.PipelineClcFetchAsync.create(
                    barrier_storage=storage.clc_mbar_ptr.data_ptr(),
                    num_stages=self.num_clc_stage,
                    producer_group=clc_pipeline_producer_group,
                    consumer_group=clc_pipeline_consumer_group,
                    tx_count=self.num_clc_response_bytes,
                    cta_layout_vmnk=cluster_layout_vmnk,
                    defer_sync=True,
                ),
                consumer_state=pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_clc_stage
                ),
                producer_state=pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_clc_stage
                ),
            )
            tile_sched = FmhaClcDynamicTileScheduler.create(
                tile_sched_params,
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
                clc_response_ptr,
                clc,
            )
            work_tile = tile_sched.initial_work_tile_info()
        else:
            clc = None
            clc_response_ptr = None

        tSTtST_shape = KQ_tiled_mma.partition_shape_C(cute.select(self.KQ_mma_tiler, mode=[0, 1]))
        tSTtST = KQ_tiled_mma.make_fragment_C(tSTtST_shape)
        # tSTtST shape : (MMA, MMA_M, MMA_N)
        tSTtST = cute.make_tensor(tmem_ptr + self.tmem_S_offset, tSTtST.layout)

        # tdVrP shape : (MMA, MMA_M, MMA_K, STAGE)
        tdVrP = PdO_tiled_mma.make_fragment_A(sP)
        # tdVrdOT shape : (MMA, MMA_N, MMA_K, STAGE)
        tdVrdOT = PdO_tiled_mma.make_fragment_B(sdOT)

        tdPTtdPT_shape = VdO_tiled_mma.partition_shape_C(
            cute.select(self.VdO_mma_tiler, mode=[0, 1])
        )
        tdPTtdPT = VdO_tiled_mma.make_fragment_C(tdPTtdPT_shape)
        # tdPTtdPT shape : (MMA, MMA_M, MMA_N)
        tdPTtdPT = cute.make_tensor(tmem_ptr + self.tmem_dP_offset, tdPTtdPT.layout)

        tdKtdK_shape = dSQ_tiled_mma.partition_shape_C(cute.select(self.dSQ_mma_tiler, mode=[0, 1]))
        tdKtdK = dSQ_tiled_mma.make_fragment_C(tdKtdK_shape)
        # tdKtdK shape : (MMA, MMA_M, MMA_N)
        tdKtdK = cute.make_tensor(tmem_ptr + self.tmem_dK_offset, tdKtdK.layout)

        tdVtdV_shape = PdO_tiled_mma.partition_shape_C(cute.select(self.PdO_mma_tiler, mode=[0, 1]))
        tdVtdV = PdO_tiled_mma.make_fragment_C(tdVtdV_shape)
        # tdVtdV shape : (MMA, MMA_M, MMA_N)
        tdVtdV = cute.make_tensor(tmem_ptr + self.tmem_dV_offset, tdVtdV.layout)

        # get the current batch problem shape

        if cutlass.const_expr(self.use_clc_scheduler):
            # ===== CLC PERSISTENT PATH: per-warp while loops =====
            cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            is_first_cta_in_cluster = cta_rank_in_cluster == 0
            is_sched_warp = warp_idx == self.sched_warp_id and is_first_cta_in_cluster

            # Register allocation ONCE (before any loop)
            if warp_idx == self.load_warp_id:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_load)
            elif warp_idx == self.mma_warp_id:
                cute.arch.warpgroup_reg_alloc(self.num_regs_mma)
            elif warp_idx >= self.compute_warp_id[0] and warp_idx <= self.compute_warp_id[-1]:
                cute.arch.warpgroup_reg_alloc(self.num_regs_compute)
            else:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)

            # Cluster wait
            pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

            # ===== SCHEDULER WARP =====
            if is_sched_warp:
                while work_tile.is_valid_tile:
                    tile_sched.prefetch_next_work()
                    work_tile = tile_sched.advance_to_next_work()
                tile_sched.producer_tail()

            # ===== LOAD WARP =====
            elif warp_idx == self.load_warp_id:
                while work_tile.is_valid_tile:
                    # Decode coordinates from CLC work tile
                    blk_coord_k = work_tile.tile_idx[0]
                    blk_coord_b = work_tile.tile_idx[2][0]
                    blk_coord_h_k = work_tile.tile_idx[2][1]
                    blk_coord = (
                        Int32(0),
                        blk_coord_k,
                        Int32(0),
                        ((Int32(0), blk_coord_h_k), blk_coord_b),
                    )
                    seqlen_q_cur_batch = Q_ref.shape[0]
                    seqlen_k_cur_batch = K_ref.shape[0]
                    blk_offset = (Int32(0), Int32(0), Int32(0), ((Int32(0), Int32(0)), Int32(0)))
                    if cutlass.const_expr(varlen):
                        assert isinstance(cumulative_s_q, cute.Tensor)
                        assert isinstance(cumulative_s_k, cute.Tensor)
                        seqlen_q_cur_batch = (
                            cumulative_s_q[blk_coord_b + 1] - cumulative_s_q[blk_coord_b]
                        )
                        seqlen_k_cur_batch = (
                            cumulative_s_k[blk_coord_b + 1] - cumulative_s_k[blk_coord_b]
                        )
                        blk_offset = (
                            cumulative_s_q[blk_coord_b],
                            cumulative_s_k[blk_coord_b],
                            Int32(0),
                            ((Int32(0), Int32(0)), Int32(0)),
                        )
                    iter_start, iter_end = self.get_Q_block_min_max(
                        seqlen_q_cur_batch,
                        seqlen_k_cur_batch,
                        blk_coord_k,
                        is_2cta=True,
                    )
                    iter_count = (iter_end - iter_start) * problem_shape[3][0][0]
                    if iter_count <= 0:
                        if blk_coord_k * self.tile_shape_K < seqlen_k_cur_batch:
                            problem_shape_cur_batch = (
                                seqlen_q_cur_batch,
                                seqlen_k_cur_batch,
                                problem_shape[2],
                                problem_shape[3],
                            )
                            self.epilogue_clear(
                                blk_coord, blk_offset, problem_shape_cur_batch, dK, dV
                            )
                    else:
                        problem_shape_cur_batch = (
                            seqlen_q_cur_batch,
                            seqlen_k_cur_batch,
                            problem_shape[2],
                            problem_shape[3],
                        )
                        (
                            load_mma_Q_producer,
                            load_mma_K_producer,
                            load_mma_V_producer,
                            load_compute_LSE_producer,
                            load_mma_dO_producer,
                            load_mma_dOT_producer,
                            load_compute_sum_OdO_producer,
                            load_mma_QT_producer,
                        ) = self.load(
                            K_in,
                            V_in,
                            Q_in,
                            QT_in,
                            dO_in,
                            dOT_in,
                            LSE,
                            sum_OdO,
                            sK,
                            sQ,
                            sQT,
                            sV,
                            sdO,
                            sdOT,
                            sLSE,
                            sSum_OdO,
                            KQ_tiled_mma,
                            VdO_tiled_mma,
                            PdO_tiled_mma,
                            dSQ_tiled_mma,
                            tma_atom_K,
                            tma_atom_Q,
                            tma_atom_QT,
                            tma_atom_V,
                            tma_atom_dO,
                            tma_atom_dOT,
                            blk_offset,
                            problem_shape_cur_batch,
                            varlen,
                            iter_count,
                            iter_start,
                            iter_end,
                            load_mma_Q_producer,
                            load_mma_Q_consumer,
                            load_mma_K_producer,
                            load_mma_K_consumer,
                            load_mma_V_producer,
                            load_mma_V_consumer,
                            load_compute_LSE_producer,
                            load_compute_LSE_consumer,
                            load_mma_dO_producer,
                            load_mma_dO_consumer,
                            load_mma_dOT_producer,
                            load_mma_dOT_consumer,
                            load_compute_sum_OdO_producer,
                            load_compute_sum_OdO_consumer,
                            load_mma_QT_producer,
                            load_mma_QT_consumer,
                            blk_coord_k,
                            blk_coord_h_k,
                            blk_coord_b,
                        )
                    # CLC advance
                    work_tile = tile_sched.advance_to_next_work()
                # producer_tail after loop
                load_mma_K_producer.tail()
                load_mma_V_producer.tail()
                load_mma_Q_producer.tail()
                load_compute_LSE_producer.tail()
                load_mma_dO_producer.tail()
                load_mma_dOT_producer.tail()
                load_compute_sum_OdO_producer.tail()
                load_mma_QT_producer.tail()

            # ===== MMA WARP =====
            elif warp_idx == self.mma_warp_id:
                while work_tile.is_valid_tile:
                    blk_coord_k = work_tile.tile_idx[0]
                    blk_coord_b = work_tile.tile_idx[2][0]
                    blk_coord_h_k = work_tile.tile_idx[2][1]
                    blk_coord = (
                        Int32(0),
                        blk_coord_k,
                        Int32(0),
                        ((Int32(0), blk_coord_h_k), blk_coord_b),
                    )
                    seqlen_q_cur_batch = Q_ref.shape[0]
                    seqlen_k_cur_batch = K_ref.shape[0]
                    blk_offset = (Int32(0), Int32(0), Int32(0), ((Int32(0), Int32(0)), Int32(0)))
                    if cutlass.const_expr(varlen):
                        assert isinstance(cumulative_s_q, cute.Tensor)
                        assert isinstance(cumulative_s_k, cute.Tensor)
                        seqlen_q_cur_batch = (
                            cumulative_s_q[blk_coord_b + 1] - cumulative_s_q[blk_coord_b]
                        )
                        seqlen_k_cur_batch = (
                            cumulative_s_k[blk_coord_b + 1] - cumulative_s_k[blk_coord_b]
                        )
                        blk_offset = (
                            cumulative_s_q[blk_coord_b],
                            cumulative_s_k[blk_coord_b],
                            Int32(0),
                            ((Int32(0), Int32(0)), Int32(0)),
                        )
                    iter_start, iter_end = self.get_Q_block_min_max(
                        seqlen_q_cur_batch,
                        seqlen_k_cur_batch,
                        blk_coord_k,
                        is_2cta=True,
                    )
                    iter_count = (iter_end - iter_start) * problem_shape[3][0][0]
                    if iter_count <= 0:
                        if blk_coord_k * self.tile_shape_K < seqlen_k_cur_batch:
                            problem_shape_cur_batch = (
                                seqlen_q_cur_batch,
                                seqlen_k_cur_batch,
                                problem_shape[2],
                                problem_shape[3],
                            )
                            self.epilogue_clear(
                                blk_coord, blk_offset, problem_shape_cur_batch, dK, dV
                            )
                    else:
                        (
                            mma_compute_S_producer,
                            mma_compute_dP_producer,
                            mma_compute_dKdV_producer,
                            load_mma_Q_consumer,
                            load_mma_K_consumer,
                            load_mma_V_consumer,
                            load_mma_dO_consumer,
                            load_mma_dOT_consumer,
                            compute_mma_P_consumer,
                            compute_mma_dS_consumer,
                            load_mma_QT_consumer,
                        ) = self.mma_2cta(
                            KQ_tiled_mma,
                            VdO_tiled_mma,
                            PdO_tiled_mma,
                            dSQ_tiled_mma,
                            tSTtST,
                            tSTrQ,
                            tSTrK,
                            tdPTtdPT,
                            tdPTrV,
                            tdPTrdO,
                            tdVtdV,
                            tdVrP,
                            tdVrdOT,
                            tdKrdST,
                            tdKtdK,
                            tdKrQT,
                            iter_count,
                            load_mma_Q_consumer,
                            load_mma_K_consumer,
                            load_mma_V_consumer,
                            mma_compute_S_producer,
                            load_mma_dO_consumer,
                            mma_compute_dP_producer,
                            load_mma_dOT_consumer,
                            compute_mma_P_consumer,
                            compute_mma_dS_consumer,
                            load_mma_QT_consumer,
                            mma_compute_dKdV_producer,
                        )
                    # CLC advance
                    work_tile = tile_sched.advance_to_next_work()
                # producer_tail after loop
                mma_compute_S_producer.tail()
                mma_compute_dP_producer.tail()
                mma_compute_dKdV_producer.tail()

            # ===== COMPUTE WARPS =====
            elif warp_idx >= self.compute_warp_id[0] and warp_idx <= self.compute_warp_id[-1]:
                while work_tile.is_valid_tile:
                    blk_coord_k = work_tile.tile_idx[0]
                    blk_coord_b = work_tile.tile_idx[2][0]
                    blk_coord_h_k = work_tile.tile_idx[2][1]
                    blk_coord = (
                        Int32(0),
                        blk_coord_k,
                        Int32(0),
                        ((Int32(0), blk_coord_h_k), blk_coord_b),
                    )
                    seqlen_q_cur_batch = Q_ref.shape[0]
                    seqlen_k_cur_batch = K_ref.shape[0]
                    blk_offset = (Int32(0), Int32(0), Int32(0), ((Int32(0), Int32(0)), Int32(0)))
                    if cutlass.const_expr(varlen):
                        assert isinstance(cumulative_s_q, cute.Tensor)
                        assert isinstance(cumulative_s_k, cute.Tensor)
                        seqlen_q_cur_batch = (
                            cumulative_s_q[blk_coord_b + 1] - cumulative_s_q[blk_coord_b]
                        )
                        seqlen_k_cur_batch = (
                            cumulative_s_k[blk_coord_b + 1] - cumulative_s_k[blk_coord_b]
                        )
                        blk_offset = (
                            cumulative_s_q[blk_coord_b],
                            cumulative_s_k[blk_coord_b],
                            Int32(0),
                            ((Int32(0), Int32(0)), Int32(0)),
                        )
                    iter_start, iter_end = self.get_Q_block_min_max(
                        seqlen_q_cur_batch,
                        seqlen_k_cur_batch,
                        blk_coord_k,
                        is_2cta=True,
                    )
                    iter_count = (iter_end - iter_start) * problem_shape[3][0][0]
                    if iter_count <= 0:
                        if blk_coord_k * self.tile_shape_K < seqlen_k_cur_batch:
                            problem_shape_cur_batch = (
                                seqlen_q_cur_batch,
                                seqlen_k_cur_batch,
                                problem_shape[2],
                                problem_shape[3],
                            )
                            self.epilogue_clear(
                                blk_coord, blk_offset, problem_shape_cur_batch, dK, dV
                            )
                    else:
                        problem_shape_cur_batch = (
                            seqlen_q_cur_batch,
                            seqlen_k_cur_batch,
                            problem_shape[2],
                            problem_shape[3],
                        )
                        (
                            compute_mma_P_producer,
                            compute_mma_dS_producer,
                            mma_compute_S_consumer,
                            compute_mma_P_consumer,
                            load_compute_LSE_consumer,
                            load_compute_sum_OdO_consumer,
                            mma_compute_dP_consumer,
                            compute_mma_dS_consumer,
                            mma_compute_dKdV_consumer,
                        ) = self.compute(
                            tSTtST,
                            tdPTtdPT,
                            tdVrP,
                            sP,
                            sLSE,
                            sdST,
                            sdOT,
                            sSum_OdO,
                            dK,
                            dV,
                            tdKtdK,
                            tdVtdV,
                            PdO_tiled_mma,
                            dSQ_tiled_mma,
                            blk_coord,
                            blk_offset,
                            problem_shape_cur_batch,
                            iter_count,
                            iter_start,
                            iter_end,
                            scale_softmax,
                            mma_compute_S_producer,
                            mma_compute_S_consumer,
                            compute_mma_P_producer,
                            compute_mma_P_consumer,
                            load_compute_LSE_producer,
                            load_compute_LSE_consumer,
                            load_compute_sum_OdO_producer,
                            load_compute_sum_OdO_consumer,
                            mma_compute_dP_producer,
                            mma_compute_dP_consumer,
                            compute_mma_dS_producer,
                            compute_mma_dS_consumer,
                            mma_compute_dKdV_producer,
                            mma_compute_dKdV_consumer,
                            varlen,
                            sK,
                            seqlen_k_cur_batch,
                            tma_atom_dK,
                            dK_tma,
                            tma_atom_dV,
                            dV_tma,
                            sdK_epi_layout,
                            sdV_epi_layout,
                        )
                        cute.arch.barrier(
                            barrier_id=self.epilogue_sync_bar_id,
                            number_of_threads=self.num_compute_warps * self.threads_per_warp,
                        )
                    # CLC advance
                    work_tile = tile_sched.advance_to_next_work()
                # producer_tail after loop
                compute_mma_P_producer.tail()
                compute_mma_dS_producer.tail()

        else:
            # ===== STATIC PATH: original non-persistent code =====
            blk_coord = (Int32(0), bidx, Int32(0), ((Int32(0), bidy), bidz))
            seqlen_q_cur_batch = Q_ref.shape[0]
            seqlen_k_cur_batch = K_ref.shape[0]
            blk_offset = (Int32(0), Int32(0), Int32(0), ((Int32(0), Int32(0)), Int32(0)))
            if cutlass.const_expr(varlen):
                assert isinstance(cumulative_s_q, cute.Tensor)
                assert isinstance(cumulative_s_k, cute.Tensor)
                seqlen_q_cur_batch = cumulative_s_q[bidz + 1] - cumulative_s_q[bidz]
                seqlen_k_cur_batch = cumulative_s_k[bidz + 1] - cumulative_s_k[bidz]
                blk_offset = (
                    cumulative_s_q[bidz],
                    cumulative_s_k[bidz],
                    Int32(0),
                    ((Int32(0), Int32(0)), Int32(0)),
                )

            iter_start, iter_end = self.get_Q_block_min_max(
                seqlen_q_cur_batch,
                seqlen_k_cur_batch,
                blk_coord[1],
                is_2cta=True,
            )

            # Cluster wait
            pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

            iter_count = (iter_end - iter_start) * problem_shape[3][0][0]
            problem_shape_cur_batch = (
                seqlen_q_cur_batch,
                seqlen_k_cur_batch,
                problem_shape[2],
                problem_shape[3],
            )
            if iter_count <= 0:
                if bidx * self.tile_shape_K < seqlen_k_cur_batch:
                    self.epilogue_clear(
                        blk_coord,
                        blk_offset,
                        problem_shape_cur_batch,
                        dK,
                        dV,
                    )
            # ///////////////////////////////////////////////////////////////////////////////
            #  LOAD
            # ///////////////////////////////////////////////////////////////////////////////
            elif warp_idx == self.load_warp_id:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_load)

                self.load(
                    K_in,
                    V_in,
                    Q_in,
                    QT_in,
                    dO_in,
                    dOT_in,
                    LSE,
                    sum_OdO,
                    sK,
                    sQ,
                    sQT,
                    sV,
                    sdO,
                    sdOT,
                    sLSE,
                    sSum_OdO,
                    KQ_tiled_mma,
                    VdO_tiled_mma,
                    PdO_tiled_mma,
                    dSQ_tiled_mma,
                    tma_atom_K,
                    tma_atom_Q,
                    tma_atom_QT,
                    tma_atom_V,
                    tma_atom_dO,
                    tma_atom_dOT,
                    blk_offset,
                    problem_shape_cur_batch,
                    varlen,
                    iter_count,
                    iter_start,
                    iter_end,
                    load_mma_Q_producer,
                    load_mma_Q_consumer,
                    load_mma_K_producer,
                    load_mma_K_consumer,
                    load_mma_V_producer,
                    load_mma_V_consumer,
                    load_compute_LSE_producer,
                    load_compute_LSE_consumer,
                    load_mma_dO_producer,
                    load_mma_dO_consumer,
                    load_mma_dOT_producer,
                    load_mma_dOT_consumer,
                    load_compute_sum_OdO_producer,
                    load_compute_sum_OdO_consumer,
                    load_mma_QT_producer,
                    load_mma_QT_consumer,
                )

            # ///////////////////////////////////////////////////////////////////////////////
            #  MMA
            # ///////////////////////////////////////////////////////////////////////////////
            elif warp_idx == self.mma_warp_id:
                cute.arch.warpgroup_reg_alloc(self.num_regs_mma)

                self.mma_2cta(
                    KQ_tiled_mma,
                    VdO_tiled_mma,
                    PdO_tiled_mma,
                    dSQ_tiled_mma,
                    tSTtST,
                    tSTrQ,
                    tSTrK,
                    tdPTtdPT,
                    tdPTrV,
                    tdPTrdO,
                    tdVtdV,
                    tdVrP,
                    tdVrdOT,
                    tdKrdST,
                    tdKtdK,
                    tdKrQT,
                    iter_count,
                    load_mma_Q_consumer,
                    load_mma_K_consumer,
                    load_mma_V_consumer,
                    mma_compute_S_producer,
                    load_mma_dO_consumer,
                    mma_compute_dP_producer,
                    load_mma_dOT_consumer,
                    compute_mma_P_consumer,
                    compute_mma_dS_consumer,
                    load_mma_QT_consumer,
                    mma_compute_dKdV_producer,
                )

            # ///////////////////////////////////////////////////////////////////////////////
            #  Compute
            # ///////////////////////////////////////////////////////////////////////////////
            elif warp_idx >= self.compute_warp_id[0] and warp_idx <= self.compute_warp_id[-1]:
                cute.arch.warpgroup_reg_alloc(self.num_regs_compute)

                self.compute(
                    tSTtST,
                    tdPTtdPT,
                    tdVrP,
                    sP,
                    sLSE,
                    sdST,
                    sdOT,
                    sSum_OdO,
                    dK,
                    dV,
                    tdKtdK,
                    tdVtdV,
                    PdO_tiled_mma,
                    dSQ_tiled_mma,
                    blk_coord,
                    blk_offset,
                    problem_shape_cur_batch,
                    iter_count,
                    iter_start,
                    iter_end,
                    scale_softmax,
                    mma_compute_S_producer,
                    mma_compute_S_consumer,
                    compute_mma_P_producer,
                    compute_mma_P_consumer,
                    load_compute_LSE_producer,
                    load_compute_LSE_consumer,
                    load_compute_sum_OdO_producer,
                    load_compute_sum_OdO_consumer,
                    mma_compute_dP_producer,
                    mma_compute_dP_consumer,
                    compute_mma_dS_producer,
                    compute_mma_dS_consumer,
                    mma_compute_dKdV_producer,
                    mma_compute_dKdV_consumer,
                    varlen,
                    sK,
                    seqlen_k_cur_batch,
                    tma_atom_dK,
                    dK_tma,
                    tma_atom_dV,
                    dV_tma,
                    sdK_epi_layout,
                    sdV_epi_layout,
                )

                cute.arch.barrier(
                    barrier_id=self.epilogue_sync_bar_id,
                    number_of_threads=self.num_compute_warps * self.threads_per_warp,
                )

            else:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)

        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        # dishengbin Deallocate tmem for early exit
        # Dealloc the tensor memory
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)

    @cute.jit
    def get_Q_block_min_max(
        self,
        seq_Q: Int32,
        seq_K: Int32,
        blk_coord_k: Int32,
        is_2cta: bool,
    ):
        """Get Q tiles range."""
        Q_block_max = cute.ceil_div(seq_Q, self.tile_shape_Q)
        Q_block_min = cutlass.Int32(0)
        if cutlass.const_expr(self.has_sliding_window):
            # For 2cta, use the last K block in the cluster so both CTAs get the same Q_block_max
            blk_coord_k_for_max = (blk_coord_k // 2) * 2 + 1
            Q_block_max_tmp = cute.ceil_div(
                (blk_coord_k_for_max + 1) * self.tile_shape_K
                + seq_Q
                - seq_K
                + self.window_size_left,
                self.tile_shape_Q,
            )
            Q_block_max = min(Q_block_max, Q_block_max_tmp)
        if cutlass.const_expr(self.is_causal or self.has_sliding_window):
            # For 2cta, use the first K block in the cluster so both CTAs get the same Q_block_min.
            # This ensures both CTAs in a cluster run the same number of pipeline iterations,
            # avoiding hang from mismatched producer_commit / consumer_wait counts.
            blk_coord_k_for_min = (blk_coord_k // 2) * 2
            Q_block_min_tmp = (
                blk_coord_k_for_min * self.tile_shape_K + seq_Q - seq_K - self.window_size_right
            ) // self.tile_shape_Q
            # Consider the case of 2cta, we need to ensure the K block is aligned to 2
            Q_block_min_tmp = Q_block_min_tmp - Q_block_min_tmp % 2
            Q_block_min = max(Q_block_min_tmp, Q_block_min)
        return Q_block_min, Q_block_max

    @cute.jit
    def load(
        self,
        K_in: cute.Tensor,
        V_in: cute.Tensor,
        Q_in: cute.Tensor,
        QT_in: cute.Tensor,
        dO_in: cute.Tensor,
        dOT_in: cute.Tensor,
        LSE_in: cute.Tensor,
        sum_OdO_in: cute.Tensor,
        sK: cute.Tensor,
        sQ: cute.Tensor,
        sQT: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sdOT: cute.Tensor,
        sLSE: cute.Tensor,
        sSum_OdO: cute.Tensor,
        KQ_tiled_mma: cute.TiledMma,
        VdO_tiled_mma: cute.TiledMma,
        PdO_tiled_mma: cute.TiledMma,
        dSQ_tiled_mma: cute.TiledMma,
        tma_atom_K: cute.CopyAtom,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_QT: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dOT: cute.CopyAtom,
        blk_offset: cute.Shape,
        problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32], Int32]],
        varlen: bool,
        iter_count: Int32,
        iter_start: Int32,
        iter_end: Int32,
        load_mma_Q_producer,
        load_mma_Q_consumer,
        load_mma_K_producer,
        load_mma_K_consumer,
        load_mma_V_producer,
        load_mma_V_consumer,
        load_compute_LSE_producer,
        load_compute_LSE_consumer,
        load_mma_dO_producer,
        load_mma_dO_consumer,
        load_mma_dOT_producer,
        load_mma_dOT_consumer,
        load_compute_sum_OdO_producer,
        load_compute_sum_OdO_consumer,
        load_mma_QT_producer,
        load_mma_QT_consumer,
        blk_coord_k_override: Int32 = Int32(-1),
        blk_coord_h_k_override: Int32 = Int32(-1),
        blk_coord_b_override: Int32 = Int32(-1),
    ):
        """TMA load."""
        tidx, _, _ = cute.arch.thread_idx()
        if cutlass.const_expr(self.use_clc_scheduler):
            blk_coord_k = blk_coord_k_override
            blk_coord_h_k = blk_coord_h_k_override
            blk_coord_b = blk_coord_b_override
        else:
            blk_coord_k, blk_coord_h_k, blk_coord_b = cute.arch.block_idx()
        blk_coord_h_r = Int32(0)
        blk_coord_h = (blk_coord_h_r, blk_coord_h_k)
        iter_index = iter_start
        mma_tile_coord_v = blk_coord_k % cute.size(KQ_tiled_mma.thr_id.shape)
        mma_tile_coord_m = blk_coord_k // cute.size(KQ_tiled_mma.thr_id.shape)

        K = cute.domain_offset(cute.select(blk_offset, mode=[1, 2, 3]), K_in)
        V = cute.domain_offset(cute.select(blk_offset, mode=[1, 2, 3]), V_in)
        Q = cute.domain_offset(cute.select(blk_offset, mode=[0, 2, 3]), Q_in)
        QT = cute.domain_offset(cute.select(blk_offset, mode=[2, 0, 3]), QT_in)
        dO = cute.domain_offset(cute.select(blk_offset, mode=[0, 2, 3]), dO_in)
        dOT = cute.domain_offset(cute.select(blk_offset, mode=[2, 0, 3]), dOT_in)
        blk_offset_stats = blk_offset
        if cutlass.const_expr(varlen):
            cuseqlen_q_stats = cute.assume(
                (blk_offset[0] + blk_coord_b * self.tile_shape_Q)
                // self.tile_shape_Q
                * self.tile_shape_Q,
                divby=self.tile_shape_Q,
            )
            blk_offset_stats = (
                cuseqlen_q_stats,
                blk_offset[1],
                blk_offset[2],
                blk_offset[3],
            )
        LSE = cute.domain_offset(cute.select(blk_offset_stats, mode=[0, 3]), LSE_in)
        sum_OdO = cute.domain_offset(cute.select(blk_offset_stats, mode=[0, 3]), sum_OdO_in)

        gK = cute.local_tile(K, cute.select(self.KQ_mma_tiler, mode=[0, 2]), (None, None, None))
        gQ = cute.local_tile(Q, cute.select(self.KQ_mma_tiler, mode=[1, 2]), (None, None, None))
        gQT = cute.local_tile(QT, cute.select(self.dSQ_mma_tiler, mode=[1, 2]), (None, None, None))
        gV = cute.local_tile(V, cute.select(self.VdO_mma_tiler, mode=[0, 2]), (None, None, None))
        gdO = cute.local_tile(dO, cute.select(self.VdO_mma_tiler, mode=[1, 2]), (None, None, None))
        gdOT = cute.local_tile(
            dOT, cute.select(self.PdO_mma_tiler, mode=[1, 2]), (None, None, None)
        )

        KQ_thr_mma = KQ_tiled_mma.get_slice(mma_tile_coord_v)
        VdO_thr_mma = VdO_tiled_mma.get_slice(mma_tile_coord_v)
        PdO_thr_mma = PdO_tiled_mma.get_slice(mma_tile_coord_v)
        dSQ_thr_mma = dSQ_tiled_mma.get_slice(mma_tile_coord_v)

        tSTgK = KQ_thr_mma.partition_A(gK)
        tSTgQ = KQ_thr_mma.partition_B(gQ)
        tdKgQT = dSQ_thr_mma.partition_B(gQT)
        tdPTgV = VdO_thr_mma.partition_A(gV)
        tdPTgdO = VdO_thr_mma.partition_B(gdO)
        tdVgdOT = PdO_thr_mma.partition_B(gdOT)

        cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)
        cta_layout_vmnk = cute.tiled_divide(cta_layout_mnk, (KQ_tiled_mma.thr_id,))
        cta_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(cute.arch.block_idx_in_cluster())

        tKsK, tKgK_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_K,
            cta_in_cluster_coord_vmnk[2],
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[2])),
            cute.group_modes(sK, 0, 3),
            cute.group_modes(tSTgK, 0, 3),
        )
        tQsQ, tQgQ_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_Q,
            cta_in_cluster_coord_vmnk[1],
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(tSTgQ, 0, 3),
        )
        tQTsQT, tQTgQT_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_QT,
            cta_in_cluster_coord_vmnk[1],
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
            cute.group_modes(sQT, 0, 3),
            cute.group_modes(tdKgQT, 0, 3),
        )
        tVsV, tVgV_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_V,
            cta_in_cluster_coord_vmnk[2],
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[2])),
            cute.group_modes(sV, 0, 3),
            cute.group_modes(tdPTgV, 0, 3),
        )
        tdOsdO, tdOgdO_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_dO,
            cta_in_cluster_coord_vmnk[1],
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
            cute.group_modes(sdO, 0, 3),
            cute.group_modes(tdPTgdO, 0, 3),
        )
        tdOTsdOT, tdOTgdOT_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_dOT,
            cta_in_cluster_coord_vmnk[1],
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
            cute.group_modes(sdOT, 0, 3),
            cute.group_modes(tdVgdOT, 0, 3),
        )

        k_handle = load_mma_K_producer.acquire_and_advance()
        cute.copy(
            tma_atom_K,
            tKgK_mkl[(None, mma_tile_coord_m, 0, (blk_coord_h, blk_coord_b))],
            tKsK[None, 0],
            tma_bar_ptr=k_handle.barrier,
        )

        q_handle = load_mma_Q_producer.acquire_and_advance()
        cute.copy(
            tma_atom_Q,
            tQgQ_mkl[(None, iter_index, 0, (blk_coord_h, blk_coord_b))],
            tQsQ[None, q_handle.index],
            tma_bar_ptr=q_handle.barrier,
        )

        lse_handle = load_compute_LSE_producer.acquire_and_advance()
        thread_idx = tidx % self.threads_per_warp
        async_copy_num_elts = sLSE.shape[0] // self.threads_per_warp
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
            self.acc_dtype,
            num_bits_per_copy=self.acc_dtype.width,
        )
        sLSE_for_copy = cute.flat_divide(sLSE, (1,))
        LSE_for_copy = cute.flat_divide(LSE, (1,))
        # Warp-coalesced: at each i, lane T accesses index `T + i*W` (stride-1
        # across the warp) instead of `T*N + i` (stride-N across the warp).
        for i in cutlass.range_constexpr(async_copy_num_elts):
            LSE_idx = self.tile_shape_Q * iter_index + thread_idx + i * self.threads_per_warp
            sLSE_idx = thread_idx + i * self.threads_per_warp
            if cute.elem_less(LSE_idx, problem_shape[0]):
                cute.copy(
                    atom_async_copy,
                    LSE_for_copy[None, LSE_idx, (blk_coord_h, blk_coord_b)],
                    sLSE_for_copy[None, sLSE_idx, lse_handle.index],
                )
            else:
                sLSE_for_copy[None, sLSE_idx, lse_handle.index].fill(0.0)
        lse_handle.commit()

        v_handle = load_mma_V_producer.acquire_and_advance()
        cute.copy(
            tma_atom_V,
            tVgV_mkl[(None, mma_tile_coord_m, 0, (blk_coord_h, blk_coord_b))],
            tVsV[(None, 0)],
            tma_bar_ptr=v_handle.barrier,
        )

        do_handle = load_mma_dO_producer.acquire_and_advance()
        cute.copy(
            tma_atom_dO,
            tdOgdO_mkl[(None, iter_index, 0, (blk_coord_h, blk_coord_b))],
            tdOsdO[(None, do_handle.index)],
            tma_bar_ptr=do_handle.barrier,
        )

        sum_odo_handle = load_compute_sum_OdO_producer.acquire_and_advance()
        sSum_OdO_for_copy = cute.flat_divide(sSum_OdO, (1,))
        sum_OdO_for_copy = cute.flat_divide(sum_OdO, (1,))
        for i in cutlass.range_constexpr(async_copy_num_elts):
            sum_OdO_idx = self.tile_shape_Q * iter_index + thread_idx + i * self.threads_per_warp
            sSum_OdO_idx = thread_idx + i * self.threads_per_warp
            if cute.elem_less(sum_OdO_idx, problem_shape[0]):
                cute.copy(
                    atom_async_copy,
                    sum_OdO_for_copy[None, sum_OdO_idx, (blk_coord_h, blk_coord_b)],
                    sSum_OdO_for_copy[None, sSum_OdO_idx, sum_odo_handle.index],
                )
            else:
                sSum_OdO_for_copy[None, sSum_OdO_idx, sum_odo_handle.index].fill(0.0)
        sum_odo_handle.commit()

        dot_handle = load_mma_dOT_producer.acquire_and_advance()
        cute.copy(
            tma_atom_dOT,
            tdOTgdOT_mkl[(None, 0, iter_index, (blk_coord_h, blk_coord_b))],
            tdOTsdOT[None, dot_handle.index],
            tma_bar_ptr=dot_handle.barrier,
        )

        qt_handle = load_mma_QT_producer.acquire_and_advance()
        cute.copy(
            tma_atom_QT,
            tQTgQT_mkl[(None, 0, iter_index, (blk_coord_h, blk_coord_b))],
            tQTsQT[None, qt_handle.index],
            tma_bar_ptr=qt_handle.barrier,
        )

        iter_count -= 1
        iter_index += 1

        while iter_count > 0:
            if iter_index == iter_end:
                iter_index = iter_start
                blk_coord_h_r += 1
                blk_coord_h = (blk_coord_h_r, blk_coord_h_k)

            q_handle = load_mma_Q_producer.acquire_and_advance()
            cute.copy(
                tma_atom_Q,
                tQgQ_mkl[(None, iter_index, 0, (blk_coord_h, blk_coord_b))],
                tQsQ[None, q_handle.index],
                tma_bar_ptr=q_handle.barrier,
            )

            lse_handle = load_compute_LSE_producer.acquire_and_advance()
            sLSE_for_copy = cute.flat_divide(sLSE, (1,))
            LSE_for_copy = cute.flat_divide(LSE, (1,))
            for i in cutlass.range_constexpr(async_copy_num_elts):
                LSE_idx = self.tile_shape_Q * iter_index + thread_idx + i * self.threads_per_warp
                sLSE_idx = thread_idx + i * self.threads_per_warp
                if cute.elem_less(LSE_idx, problem_shape[0]):
                    cute.copy(
                        atom_async_copy,
                        LSE_for_copy[None, LSE_idx, (blk_coord_h, blk_coord_b)],
                        sLSE_for_copy[None, sLSE_idx, lse_handle.index],
                    )
                else:
                    sLSE_for_copy[None, sLSE_idx, lse_handle.index].fill(0.0)
            lse_handle.commit()

            do_handle = load_mma_dO_producer.acquire_and_advance()
            cute.copy(
                tma_atom_dO,
                tdOgdO_mkl[(None, iter_index, 0, (blk_coord_h, blk_coord_b))],
                tdOsdO[None, do_handle.index],
                tma_bar_ptr=do_handle.barrier,
            )

            sum_odo_handle = load_compute_sum_OdO_producer.acquire_and_advance()
            sSum_OdO_for_copy = cute.flat_divide(sSum_OdO, (1,))
            sum_OdO_for_copy = cute.flat_divide(sum_OdO, (1,))
            for i in cutlass.range_constexpr(async_copy_num_elts):
                sum_OdO_idx = (
                    self.tile_shape_Q * iter_index + thread_idx + i * self.threads_per_warp
                )
                sSum_OdO_idx = thread_idx + i * self.threads_per_warp
                if cute.elem_less(sum_OdO_idx, problem_shape[0]):
                    cute.copy(
                        atom_async_copy,
                        sum_OdO_for_copy[None, sum_OdO_idx, (blk_coord_h, blk_coord_b)],
                        sSum_OdO_for_copy[None, sSum_OdO_idx, sum_odo_handle.index],
                    )
                else:
                    sSum_OdO_for_copy[None, sSum_OdO_idx, sum_odo_handle.index].fill(0.0)
            sum_odo_handle.commit()

            dot_handle = load_mma_dOT_producer.acquire_and_advance()
            cute.copy(
                tma_atom_dOT,
                tdOTgdOT_mkl[(None, 0, iter_index, (blk_coord_h, blk_coord_b))],
                tdOTsdOT[None, dot_handle.index],
                tma_bar_ptr=dot_handle.barrier,
            )

            qt_handle = load_mma_QT_producer.acquire_and_advance()
            cute.copy(
                tma_atom_QT,
                tQTgQT_mkl[(None, 0, iter_index, (blk_coord_h, blk_coord_b))],
                tQTsQT[None, qt_handle.index],
                tma_bar_ptr=qt_handle.barrier,
            )

            iter_count -= 1
            iter_index += 1

        if not cutlass.const_expr(self.use_clc_scheduler):
            load_mma_K_producer.tail()
            load_mma_V_producer.tail()
            load_mma_Q_producer.tail()
            load_compute_LSE_producer.tail()
            load_mma_dO_producer.tail()
            load_mma_dOT_producer.tail()
            load_compute_sum_OdO_producer.tail()
            load_mma_QT_producer.tail()

        return (
            load_mma_Q_producer,
            load_mma_K_producer,
            load_mma_V_producer,
            load_compute_LSE_producer,
            load_mma_dO_producer,
            load_mma_dOT_producer,
            load_compute_sum_OdO_producer,
            load_mma_QT_producer,
        )

    @cute.jit
    def mma_2cta(
        self,
        KQ_tiled_mma: cute.TiledMma,
        VdO_tiled_mma: cute.TiledMma,
        PdO_tiled_mma: cute.TiledMma,
        dSQ_tiled_mma: cute.TiledMma,
        tSTtST: cute.Tensor,
        tSTrQ: cute.Tensor,
        tSTrK: cute.Tensor,
        tdPTtdPT: cute.Tensor,
        tdPTrV: cute.Tensor,
        tdPTrdO: cute.Tensor,
        tdVtdV: cute.Tensor,
        tdVrP: cute.Tensor,
        tdVrdOT: cute.Tensor,
        tdKrdST: cute.Tensor,
        tdKtdK: cute.Tensor,
        tdKrQT: cute.Tensor,
        iter_count: Int32,
        load_mma_Q_consumer,
        load_mma_K_consumer,
        load_mma_V_consumer,
        mma_compute_S_producer,
        load_mma_dO_consumer,
        mma_compute_dP_producer,
        load_mma_dOT_consumer,
        compute_mma_P_consumer,
        compute_mma_dS_consumer,
        load_mma_QT_consumer,
        mma_compute_dKdV_producer,
    ):
        """CuTeDSL kernel for mma pipeline."""
        load_mma_Q_releaser = load_mma_Q_consumer.clone()
        load_mma_K_releaser = load_mma_K_consumer.clone()
        load_mma_V_releaser = load_mma_V_consumer.clone()

        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        is_leader_cta = cta_rank_in_cluster % 2 == 0

        if is_leader_cta:
            s_handle = mma_compute_S_producer.acquire_and_advance()
            k_handle = load_mma_K_consumer.wait_and_advance()
            q_handle = load_mma_Q_consumer.wait_and_advance()

            # Compute S = K * Q
            for k_block in cutlass.range(0, cute.size(tSTrQ, mode=[2]), unroll_full=True):
                KQ_tiled_mma.set(tcgen05.Field.ACCUMULATE, k_block != 0)
                cute.gemm(
                    KQ_tiled_mma,
                    tSTtST,
                    tSTrK[None, None, k_block, 0],
                    tSTrQ[None, None, k_block, q_handle.index],
                    tSTtST,
                )
            q_handle.release()

            cute.arch.fence_view_async_tmem_store()
            s_handle.commit()

            do_handle = load_mma_dO_consumer.wait_and_advance()
            v_handle = load_mma_V_consumer.wait_and_advance()

            dp_handle = mma_compute_dP_producer.acquire_and_advance()

            # Compute dP = V * dO
            VdO_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(0, cute.size(tdPTrV, mode=[2]), unroll_full=True):
                cute.gemm(
                    VdO_tiled_mma,
                    tdPTtdPT,
                    tdPTrV[None, None, k_block, 0],
                    tdPTrdO[None, None, k_block, do_handle.index],
                    tdPTtdPT,
                )
                VdO_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            dp_handle.commit()
            do_handle.release()
            # V only produced once by load(); hold v_handle until end, release there via releaser

            p_handle = compute_mma_P_consumer.wait_and_advance()
            dot_handle = load_mma_dOT_consumer.wait_and_advance()

            # Compute dV = P * dO (First iteration)
            PdO_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(0, cute.size(tdVrP, mode=[2]), unroll_full=True):
                cute.gemm(
                    PdO_tiled_mma,
                    tdVtdV,
                    tdVrP[None, None, k_block, 0],
                    tdVrdOT[None, None, k_block, dot_handle.index],
                    tdVtdV,
                )
                PdO_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            dot_handle.release()
            p_handle.release()

        iter_count -= 1

        dSQ_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        while iter_count > 0:
            if is_leader_cta:
                q_handle = load_mma_Q_consumer.wait_and_advance()
                s_handle = mma_compute_S_producer.acquire_and_advance()

                # Compute S = K * Q
                for k_block in cutlass.range(0, cute.size(tSTrQ, mode=[2]), unroll_full=True):
                    KQ_tiled_mma.set(tcgen05.Field.ACCUMULATE, k_block != 0)
                    cute.gemm(
                        KQ_tiled_mma,
                        tSTtST,
                        tSTrK[None, None, k_block, 0],
                        tSTrQ[None, None, k_block, q_handle.index],
                        tSTtST,
                    )
                q_handle.release()
                s_handle.commit()

            if is_leader_cta:
                qt_handle = load_mma_QT_consumer.wait_and_advance()
                ds_handle = compute_mma_dS_consumer.wait_and_advance()

                # Compute dK = dS * QT
                for k_block in cutlass.range(0, cute.size(tdKrdST, mode=[2]), unroll_full=True):
                    cute.gemm(
                        dSQ_tiled_mma,
                        tdKtdK,
                        tdKrdST[
                            None,
                            None,
                            k_block,
                            ds_handle.index,
                        ],
                        tdKrQT[None, None, k_block, qt_handle.index],
                        tdKtdK,
                    )
                    dSQ_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                qt_handle.release()
                ds_handle.release()

            if is_leader_cta:
                dp_handle = mma_compute_dP_producer.acquire_and_advance()
                do_handle = load_mma_dO_consumer.wait_and_advance()
                # V only produced once by load(); reuse same V (index 0) for all loop iterations
                VdO_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k_block in cutlass.range(0, cute.size(tdPTrV, mode=[2]), unroll_full=True):
                    cute.gemm(
                        VdO_tiled_mma,
                        tdPTtdPT,
                        tdPTrV[None, None, k_block, 0],
                        tdPTrdO[None, None, k_block, do_handle.index],
                        tdPTtdPT,
                    )
                    VdO_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                dp_handle.commit()
                do_handle.release()

            if is_leader_cta:
                p_handle = compute_mma_P_consumer.wait_and_advance()
                dot_handle = load_mma_dOT_consumer.wait_and_advance()

                # Compute dV = P * dO (Loop iterations)
                for k_block in cutlass.range(0, cute.size(tdVrP, mode=[2]), unroll_full=True):
                    cute.gemm(
                        PdO_tiled_mma,
                        tdVtdV,
                        tdVrP[None, None, k_block, 0],
                        tdVrdOT[None, None, k_block, dot_handle.index],
                        tdVtdV,
                    )
                    PdO_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                p_handle.release()
                dot_handle.release()

            iter_count -= 1

        if is_leader_cta:
            dkdv_handle = mma_compute_dKdV_producer.acquire_and_advance()
            dkdv_handle.commit()

            load_mma_K_releaser.release()
            load_mma_K_releaser.advance()
            load_mma_V_releaser.release()
            load_mma_V_releaser.advance()

        if is_leader_cta:
            dkdv_handle = mma_compute_dKdV_producer.acquire_and_advance()

            ds_handle = compute_mma_dS_consumer.wait_and_advance()
            qt_handle = load_mma_QT_consumer.wait_and_advance()

            # Compute dK = dS * Q
            for k_block in cutlass.range(0, cute.size(tdKrdST, mode=[2]), unroll_full=True):
                cute.gemm(
                    dSQ_tiled_mma,
                    tdKtdK,
                    tdKrdST[None, None, k_block, ds_handle.index],
                    tdKrQT[None, None, k_block, qt_handle.index],
                    tdKtdK,
                )
                dSQ_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            dkdv_handle.commit()
            qt_handle.release()
            ds_handle.release()

        if not cutlass.const_expr(self.use_clc_scheduler):
            mma_compute_S_producer.tail()
            mma_compute_dP_producer.tail()
            mma_compute_dKdV_producer.tail()

        return (
            mma_compute_S_producer,
            mma_compute_dP_producer,
            mma_compute_dKdV_producer,
            load_mma_Q_consumer,
            load_mma_K_consumer,
            load_mma_V_consumer,
            load_mma_dO_consumer,
            load_mma_dOT_consumer,
            compute_mma_P_consumer,
            compute_mma_dS_consumer,
            load_mma_QT_consumer,
        )

    @cute.jit
    def reg_to_smem_mma64x64(
        self,
        regs: cute.Tensor,
        smem: cute.Tensor,
        index: Int32,
        tiler_mn: tuple[Int32, Int32],
        dp_idx: Int32,
        wg_idx: Int32,
    ):
        smem_slice = smem[None, None, None, index]
        # TODO: double check the layout of the data in reg.
        # TODO: this may introduce additional smem transpose.
        thread_layout = cute.make_ordered_layout(
            tiler_mn,
            (0, 1),  # TODO: (0,1) or (1,0) ???
        )
        smem_slice_tmp = cute.composition(smem_slice, thread_layout)

        # TODO: temporary code for tile 64 x 64.
        tmp_shape = ((8, 2, 4), (2, 4, 2, 2, 2))
        tmp_stride = ((64, 512, 1024), (1, 2, 8, 16, 32))
        smem_copy = cute.composition(smem_slice_tmp, cute.make_layout(tmp_shape, stride=tmp_stride))

        # TODO: the following code is only for tile 64 x 64.
        # TODO: need to modify the code for other tile sizes.
        lane_idx = dp_idx % 32
        reg_shape = regs.shape
        atom_loops = reg_shape[0][0][2]
        block_loops = reg_shape[2]
        # | 00 ~ 07 | 08 ~ 15 | 16 ~ 23 | 24 ~ 31 | 32 ~ 39 | 40 ~ 47 | 48 ~ 55 | 56 ~ 63 |
        # |---- atom size ----|---- atom size ----|---- atom size ----|---- atom size ----|
        # |----     wg0   ----|----     wg1   ----|----     wg0   ----|----     wg1   ----|
        for ia in cutlass.range(atom_loops):
            for ib in cutlass.range(block_loops):
                # the lower 8 lines
                regs_copy = regs[((None, 0, ia), 0), 0, ib]  # two elements
                smem_copy_slice = smem_copy[
                    (lane_idx // 4, 0, dp_idx // 32),
                    (None, lane_idx % 4, ia, wg_idx, ib),
                ]
                cute.autovec_copy(regs_copy, smem_copy_slice)
                # the upper 8 lines
                regs_copy = regs[((None, 1, ia), 0), 0, ib]
                smem_copy_slice = smem_copy[
                    (lane_idx // 4, 1, dp_idx // 32),
                    (None, lane_idx % 4, ia, wg_idx, ib),
                ]
                cute.autovec_copy(regs_copy, smem_copy_slice)

    @cute.jit
    def reg_to_smem_mma128x128_2cta(
        self,
        regs: cute.Tensor,
        smem: cute.Tensor,
        index: Int32,
        tiler_mn: tuple[Int32, Int32],
        dp_idx: Int32,
        wg_idx: Int32,
        smem_RowMajor: bool = True,
    ):
        smem_slice = smem[None, None, None, index]
        # K>> smem_slice:  tensor<ptr<f16, smem, align<1024>, S<3,4,3>> o ((64,16),1,(4,2)):((64,1),0,(16,4096))>
        thread_layout = cute.make_ordered_layout(
            # (tileN, tileM)
            tiler_mn,
            (0, 1),
        )
        # K>> thread_layout:  (64,128):(128,1)
        smem_slice_tmp = cute.composition(smem_slice, thread_layout)

        # NOTE: hardcode for tcgen05.ld.32x32b.x8 & mma128x64+2cta
        # tmp_shape = ((32, 2), (8, 2, 2, 2)) # for 64x64 tile
        # tmp_stride = ((64, 32*64), (1, 8, 16, 32))
        # NOTE: hardcode for tcgen05.ld.32x32b.x16 & mma128x64+2cta
        tmp_shape = ((32, 2), (16, 2, 2, 2))  # for 128x64 tile
        tmp_stride = ((64, 32 * 64), (1, 16, 32, 64 * 64))
        # smem_copy = cute.composition(smem_slice_tmp, cute.make_layout(tmp_shape, stride=tmp_stride))
        smem_copy = cute.make_tensor(
            smem_slice_tmp.iterator, cute.make_layout(tmp_shape, stride=tmp_stride)
        )

        warp_idx = dp_idx // 32
        warp_row_idx = warp_idx % 2
        warp_col_idx = warp_idx // 2  # corresponding to the second 64 cols in smem
        lane_idx = dp_idx % 32
        reg_shape = (
            regs.shape
        )  # ((8,1),1,2):((1,0),0,8) for 64x64, ((16,1),1,2):((1,0),0,16) for 128x64
        block_loops = reg_shape[2]

        # TODO: maybe can use cp.async for optimization
        for ib in cutlass.range(block_loops):
            regs_copy = regs[(None, 0), 0, ib]
            smem_copy_slice = smem_copy[(lane_idx, warp_row_idx), (None, wg_idx, ib, warp_col_idx)]
            cute.autovec_copy(regs_copy, smem_copy_slice)

    @cute.jit
    def reg_to_smem_mma128x64_2cta(
        self,
        regs: cute.Tensor,
        smem: cute.Tensor,
        index: Int32,
        tiler_mn: tuple[Int32, Int32],
        dp_idx: Int32,
        wg_idx: Int32,
        smem_RowMajor: bool = True,
    ):
        smem_slice = smem[None, None, None, index]
        thread_layout = cute.make_ordered_layout(
            # (tileN, tileM)
            tiler_mn,
            (1, 0) if smem_RowMajor else (0, 1),
        )
        smem_slice_tmp = cute.composition(smem_slice, thread_layout)
        # NOTE: hardcode for tcgen05.ld.32x32b.x8 & mma128x64+2cta
        tmp_shape = ((32, 2), (8, 2, 2, 2))
        tmp_stride = ((64, 32 * 64), (1, 8, 16, 32))
        smem_copy = cute.composition(smem_slice_tmp, cute.make_layout(tmp_shape, stride=tmp_stride))

        warp_idx = dp_idx // 32
        warp_row_idx = warp_idx % 2
        warp_col_idx = warp_idx // 2
        lane_idx = dp_idx % 32
        reg_shape = regs.shape  # ((8,1),1,2):((1,0),0,8)
        block_loops = reg_shape[2]

        # TODO: maybe can use cp.async for optimization
        for ib in cutlass.range(block_loops):
            regs_copy = regs[(None, 0), 0, ib]
            smem_copy_slice = smem_copy[(lane_idx, warp_row_idx), (None, wg_idx, ib, warp_col_idx)]
            cute.autovec_copy(regs_copy, smem_copy_slice)

    @cute.jit
    def compute(
        self,
        tSTtST: cute.Tensor,
        tdPTtdPT: cute.Tensor,
        tdVrP: cute.Tensor,
        sP: cute.Tensor,
        sLSE: cute.Tensor,
        # sdS: cute.Tensor,
        sdST: cute.Tensor,
        sdOT: cute.Tensor,
        sSum_OdO: cute.Tensor,
        dK: cute.Tensor,
        dV: cute.Tensor,
        tdKtdK: cute.Tensor,
        tdVtdV: cute.Tensor,
        PdO_tiled_mma: cute.TiledMma,
        dSQ_tiled_mma: cute.TiledMma,
        blk_coord: cute.Coord,
        blk_offset: cute.Shape,
        problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32], Int32]],
        iter_count: Int32,
        iter_start: Int32,
        iter_end: Int32,
        scale_softmax: cutlass.Float32,
        mma_compute_S_producer,
        mma_compute_S_consumer,
        compute_mma_P_producer,
        compute_mma_P_consumer,
        load_compute_LSE_producer,
        load_compute_LSE_consumer,
        load_compute_sum_OdO_producer,
        load_compute_sum_OdO_consumer,
        mma_compute_dP_producer,
        mma_compute_dP_consumer,
        compute_mma_dS_producer,
        compute_mma_dS_consumer,
        mma_compute_dKdV_producer,
        mma_compute_dKdV_consumer,
        varlen: bool,
        sK: cute.Tensor,
        problem_shape_k_cur_batch: Int32,
        tma_atom_dK: cute.CopyAtom,
        dK_tma: cute.Tensor,
        tma_atom_dV: cute.CopyAtom,
        dV_tma: cute.Tensor,
        sdK_epi_layout: cute.ComposedLayout,
        sdV_epi_layout: cute.ComposedLayout,
    ):
        """CuTeDSL kernel for recomputing softmax and producing dk and dv."""
        tidx, _, _ = cute.arch.thread_idx()
        Q, K, _, _ = problem_shape
        _, blk_coord_k, _, _ = blk_coord

        iter_index = iter_start

        # adi: TMEM_ST, TMEM_DPT
        tmem_load_op = tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16))
        tmem_load_atom = cute.make_copy_atom(
            tmem_load_op,
            self.acc_dtype,
        )

        tSTtST = tSTtST[(None, None), 0, 0]
        tdPTtdPT = tdPTtdPT[(None, None), 0, 0]

        cST = cute.make_identity_tensor(cute.select(self.cta_tiler, mode=[1, 0]))
        cdPT = cute.make_identity_tensor(cute.select(self.cta_tiler, mode=[1, 0]))

        num_warp_groups = self.num_compute_warps // 4
        dp_idx = tidx % 128
        wg_idx = (tidx % (self.num_compute_warps * self.threads_per_warp)) // 128
        tiled_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tSTtST)
        thr_t2r = tiled_t2r.get_slice(dp_idx)

        tTR_cST = thr_t2r.partition_D(cST)
        tTR_cST = split_wg(tTR_cST, num_warp_groups, wg_idx)
        tTR_rST = cute.make_rmem_tensor(tTR_cST.shape, self.acc_dtype)

        tTR_tST = thr_t2r.partition_S(tSTtST)
        tTR_tST = split_wg(tTR_tST, num_warp_groups, wg_idx)

        tTR_cdPT_p = thr_t2r.partition_D(cdPT)
        tTR_cdPT = split_wg(tTR_cdPT_p, num_warp_groups, wg_idx)
        tTR_rdPT = cute.make_rmem_tensor(tTR_cdPT.shape, self.acc_dtype)

        tTR_tdPT = thr_t2r.partition_S(tdPTtdPT)
        tTR_tdPT = split_wg(tTR_tdPT, num_warp_groups, wg_idx)

        tdVcST = PdO_tiled_mma.get_slice(0).partition_A(cST)

        is_residual_k = blk_coord_k * self.tile_shape_K + self.tile_shape_K > K
        last_iter = iter_end - 1

        while iter_count > 0:
            s_handle = mma_compute_S_consumer.wait_and_advance()
            p_handle = compute_mma_P_producer.acquire_and_advance()
            lse_handle = load_compute_LSE_consumer.wait_and_advance()

            leading_causal_masking = cutlass.Boolean(False)
            if cutlass.const_expr(self.is_causal):
                # TODO: could be optimized by specify an exact iter_index
                #
                # NOTE (causal + 2CTA correctness):
                # `iter_start` can be rounded down to an even index for 2CTA. When the true
                # causal boundary Q tile is odd, it becomes (iter_start + 1). In that case,
                # we must treat both (iter_start, iter_start + 1) as "masked tiles" so the
                # per-element causal mask is applied on the boundary tile too.
                leading_causal_masking = iter_index == iter_start
                q_block_min_unaligned = (
                    blk_coord_k * self.tile_shape_K + Q - K - self.window_size_right
                ) // self.tile_shape_Q
                boundary_in_second = (q_block_min_unaligned % 2) == 1
                leading_causal_masking = leading_causal_masking or (
                    boundary_in_second and (iter_index == iter_start + 1)
                )
                offset_partial_tile = (K - Q) % self.tile_shape_K
                need_additional_mask = offset_partial_tile and iter_index == iter_start + 1
                leading_causal_masking = leading_causal_masking or need_additional_mask
                leading_causal_masking = cute.arch.shuffle_sync(leading_causal_masking, 0)

            trailing_residual_masking = cutlass.Boolean(False)
            trailing_residual_masking = iter_index == last_iter or is_residual_k
            trailing_residual_masking = cute.arch.shuffle_sync(trailing_residual_masking, 0)

            # For causal, every tile may contain (q,k) with k > q; we must apply per-element mask for all Q tiles.
            is_masked_tile = (
                leading_causal_masking
                or trailing_residual_masking
                or self.has_sliding_window
                or cutlass.const_expr(self.is_causal)
            )

            # Compute P = softmax(S, LSE)
            cute.copy(tiled_t2r, tTR_tST, tTR_rST)

            if is_masked_tile:
                for i in cutlass.range(cute.size(tTR_rST), unroll_full=True):
                    c_transpose = tTR_cST[i]
                    pos = (
                        cute.get(c_transpose, mode=[1]) + iter_index * self.tile_shape_Q,
                        cute.get(c_transpose, mode=[0]) + blk_coord_k * self.tile_shape_K,
                    )
                    if cutlass.const_expr(self.has_sliding_window):
                        if cutlass.const_expr(self.window_size_left < 0):
                            tTR_rST[i] = (
                                -cutlass.Float32.inf
                                if pos[1] > pos[0] + K - Q + self.window_size_right
                                else tTR_rST[i]
                            )
                        else:
                            max_K_index = min(pos[0] + K - Q + self.window_size_right, K)
                            min_K_index = max(0, pos[0] + K - Q - self.window_size_left)
                            tTR_rST[i] = (
                                -cutlass.Float32.inf
                                if pos[1] > max_K_index or pos[1] < min_K_index
                                else tTR_rST[i]
                            )
                    if cutlass.const_expr(self.is_causal) and (
                        pos[0] + K - Q < pos[1] or not cute.elem_less(pos, (Q, K))
                    ):
                        tTR_rST[i] = -cutlass.Float32.inf
                    if not cute.elem_less(pos, (Q, K)):
                        tTR_rST[i] = -cutlass.Float32.inf

            log2_e = cutlass.Float32(math.log2(math.e))
            softmax_scale_log2_e = scale_softmax * log2_e

            for i in cutlass.range(0, cute.size(tTR_rST), 2, unroll_full=True):
                lse = (
                    -sLSE[
                        cute.get(tTR_cST[i], mode=[1]),
                        lse_handle.index,
                    ],
                    -sLSE[
                        cute.get(tTR_cST[i + 1], mode=[1]),
                        lse_handle.index,
                    ],
                )
                tTR_rST[i], tTR_rST[i + 1] = cute.arch.fma_packed_f32x2(
                    (tTR_rST[i], tTR_rST[i + 1]),
                    (softmax_scale_log2_e, softmax_scale_log2_e),
                    lse,
                )
                tTR_rST[i] = cute.math.exp2(tTR_rST[i], fastmath=True)
                tTR_rST[i + 1] = cute.math.exp2(tTR_rST[i + 1], fastmath=True)

            # convert fp32 P to fp16 P which will be used in the PdO
            tTR_rPT = self.quantize(tTR_rST, dV.element_type)  # tTR_rST is ST in fp32 in RF.
            self.reg_to_smem_mma128x128_2cta(
                tTR_rPT,
                sP,
                p_handle.index,
                (self.tile_shape_K, self.tile_shape_Q),
                dp_idx,
                wg_idx,
            )
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(
                barrier_id=self.compute_sync_bar_id,
                number_of_threads=self.num_compute_warps * self.threads_per_warp,
            )

            p_handle.commit()

            s_handle.release()
            lse_handle.release()

            sum_odo_handle = load_compute_sum_OdO_consumer.wait_and_advance()
            dp_handle = mma_compute_dP_consumer.wait_and_advance()
            ds_handle = compute_mma_dS_producer.acquire_and_advance()

            # Compute dS = dsoftmax(P, dP, sum_OdO)
            cute.copy(tiled_t2r, tTR_tdPT, tTR_rdPT)

            for i in cutlass.range(0, cute.size(tTR_rdPT), 2, unroll_full=True):
                dpsum_0 = -sSum_OdO[
                    cute.get(tTR_cdPT[i], mode=[1]),
                    sum_odo_handle.index,
                ]
                dpsum_1 = -sSum_OdO[
                    cute.get(tTR_cdPT[i + 1], mode=[1]),
                    sum_odo_handle.index,
                ]
                if cutlass.const_expr(varlen):
                    if not cute.elem_less(cute.get(tTR_cdPT[i], mode=[1]), Q):
                        dpsum_0 = 0.0
                    if not cute.elem_less(cute.get(tTR_cdPT[i + 1], mode=[1]), Q):
                        dpsum_1 = 0.0
                tTR_rdPT[i], tTR_rdPT[i + 1] = cute.arch.add_packed_f32x2(
                    (tTR_rdPT[i], tTR_rdPT[i + 1]),
                    (dpsum_0, dpsum_1),
                )
                tTR_rdPT[i], tTR_rdPT[i + 1] = cute.arch.mul_packed_f32x2(
                    (tTR_rdPT[i], tTR_rdPT[i + 1]), (tTR_rST[i], tTR_rST[i + 1])
                )
            # For causal, force dS to zero at masked (q,k) so dK/dV accumulation is correct
            if cutlass.const_expr(self.is_causal):
                for i in cutlass.range(cute.size(tTR_rdPT), unroll_full=True):
                    c_transpose = tTR_cdPT[i]
                    pos = (
                        cute.get(c_transpose, mode=[1]) + iter_index * self.tile_shape_Q,
                        cute.get(c_transpose, mode=[0]) + blk_coord_k * self.tile_shape_K,
                    )
                    if pos[0] + K - Q < pos[1] or not cute.elem_less(pos, (Q, K)):
                        tTR_rdPT[i] = cutlass.Float32(0.0)
            # convert fp32 dS to fp16 dS which will be used in the computation of dK and DQ
            tTR_rdST = self.quantize(tTR_rdPT, dV.element_type)

            cute.arch.fence_view_async_tmem_load()
            dp_handle.release()

            self.reg_to_smem_mma128x128_2cta(
                tTR_rdST,
                sdST,
                ds_handle.index,
                (self.tile_shape_K, self.tile_shape_Q),
                dp_idx,
                wg_idx,
            )
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(
                barrier_id=self.compute_sync_bar_id,
                number_of_threads=self.num_compute_warps * self.threads_per_warp,
            )

            ds_handle.commit()
            sum_odo_handle.release()

            iter_count -= 1
            iter_index += 1
            if iter_index == iter_end:
                iter_index = iter_start

        # Epilogue
        mma_compute_dKdV_consumer = self.epilogue(
            blk_coord,
            blk_offset,
            problem_shape,
            dK,
            dV,
            tdKtdK,
            tdVtdV,
            scale_softmax,
            mma_compute_dKdV_producer,
            mma_compute_dKdV_consumer,
            problem_shape_k_cur_batch,
            tma_atom_dK,
            dK_tma,
            tma_atom_dV,
            dV_tma,
            sdK_epi_layout,
            sdV_epi_layout,
            varlen,
            sdOT,
            sP,
        )

        if not cutlass.const_expr(self.use_clc_scheduler):
            compute_mma_P_producer.tail()
            compute_mma_dS_producer.tail()

        return (
            compute_mma_P_producer,
            compute_mma_dS_producer,
            mma_compute_S_consumer,
            compute_mma_P_consumer,
            load_compute_LSE_consumer,
            load_compute_sum_OdO_consumer,
            mma_compute_dP_consumer,
            compute_mma_dS_consumer,
            mma_compute_dKdV_consumer,
        )

    @cute.jit
    def quantize(
        self,
        input_t: cute.Tensor,
        element_dtype: type[cutlass.Numeric],
    ) -> cute.Tensor:
        """Convert Float32 to element dtype."""
        output = cute.make_rmem_tensor(input_t.shape, element_dtype)
        output.store(input_t.load().to(element_dtype))
        return output

    @cute.jit
    def store(
        self,
        gmem: cute.Tensor,
        regs: cute.Tensor,
        coord: cute.Tensor,
        tensor_shape: cute.Shape,
    ):
        for i in cutlass.range(cute.size(coord, mode=[2]), unroll_full=True):
            coord_i = coord[None, 0, i]
            gmem_i = gmem[None, 0, i]
            regs_i = regs[None, 0, i]
            if cute.elem_less(coord_i[0], tensor_shape):
                gmem_i.store(regs_i.load().to(gmem.element_type))

    @cute.jit
    def epilogue_clear(
        self,
        blk_coord: cute.Coord,
        blk_offset: cute.Shape,
        problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32], Int32]],
        dK: cute.Tensor,
        dV: cute.Tensor,
    ):
        """Early stopping needs to clear dK and dV."""
        tidx, _, _ = cute.arch.thread_idx()
        block_dim_x, _, _ = cute.arch.block_dim()
        _, K, _, HB = problem_shape
        _, blk_coord_k, _, blk_coord_batch = blk_coord

        mdK_offset = cute.assume(blk_offset[1] * dK.stride[0], divby=64)
        mdK = cute.make_tensor(
            dK.iterator + mdK_offset,
            cute.make_layout((K, self.tile_shape_dQ_K, HB), stride=dK.stride),
        )
        gdK = cute.local_tile(mdK, (self.cta_tiler[1], self.cta_tiler[2]), (None, None, None))
        gdK = gdK[None, None, blk_coord_k, 0, blk_coord_batch]
        cdK = cute.domain_offset(
            (blk_coord_k * self.tile_shape_K, 0),
            cute.make_identity_tensor((self.cta_tiler[1], self.cta_tiler[2])),
        )

        mdV_offset = cute.assume(blk_offset[1] * dV.stride[0], divby=64)
        mdV = cute.make_tensor(
            dV.iterator + mdV_offset,
            cute.make_layout((K, self.tile_shape_dV_dO, HB), stride=dV.stride),
        )
        gdV = cute.local_tile(mdV, (self.cta_tiler[1], self.cta_tiler[2]), (None, None, None))
        gdV = gdV[None, None, blk_coord_k, 0, blk_coord_batch]
        cdV = cute.domain_offset(
            (blk_coord_k * self.tile_shape_K, 0),
            cute.make_identity_tensor((self.cta_tiler[1], self.cta_tiler[2])),
        )

        num_zero_epi_threads = 256

        tiled_copy_r2g = fa_copy_utils.tiled_copy_2d(
            dK.element_type, self.cta_tiler[2], num_zero_epi_threads
        )

        thr_copy_r2g = tiled_copy_r2g.get_slice(tidx)

        tRG_gdK = thr_copy_r2g.partition_D(gdK)
        tRG_cdK = thr_copy_r2g.partition_D(cdK)
        tRG_gdV = thr_copy_r2g.partition_D(gdV)
        tRG_cdV = thr_copy_r2g.partition_D(cdV)

        zero_frg = cute.make_rmem_tensor_like(tRG_gdK[None, 0, None])
        zero_frg.fill(dK.element_type(0.0))

        # check we don't need zero fragment duplication
        V_frg_size = cute.size(tRG_gdV[None, 0, None])
        assert cute.size(zero_frg) == V_frg_size

        if tidx < num_zero_epi_threads:
            for n in cutlass.range(cute.size(tRG_gdK.shape[1]), unroll_full=True):
                if cute.elem_less(tRG_cdK[0, n, 0][0], problem_shape[1]):
                    cute.copy(tiled_copy_r2g, zero_frg, tRG_gdK[None, n, None])

            for n in cutlass.range(cute.size(tRG_gdV.shape[1]), unroll_full=True):
                if cute.elem_less(tRG_cdV[0, n, 0][0], problem_shape[1]):
                    cute.copy(tiled_copy_r2g, zero_frg, tRG_gdV[None, n, None])

    @cute.jit
    def epilogue(
        self,
        blk_coord: cute.Coord,
        blk_offset: cute.Shape,
        problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32], Int32]],
        dK: cute.Tensor,
        dV: cute.Tensor,
        tdKtdK: cute.Tensor,
        tdVtdV: cute.Tensor,
        scale_softmax: cutlass.Float32,
        mma_compute_dKdV_producer,
        mma_compute_dKdV_consumer,
        problem_shape_k_cur_batch: Int32,
        tma_atom_dK: cute.CopyAtom,
        dK_tma: cute.Tensor,
        tma_atom_dV: cute.CopyAtom,
        dV_tma: cute.Tensor,
        sdK_epi_layout: cute.ComposedLayout,
        sdV_epi_layout: cute.ComposedLayout,
        varlen: bool,
        sdOT: cute.Tensor,
        sP: cute.Tensor,
    ):
        """Variant 3a (5/5) Path 2: CTA-shared SMEM with cooperative WG writes + TMA bulk store.

        Both warp-groups cooperatively populate a per-CTA (64, 256) virtual SMEM
        buffer (4 stages of (64, 64) aliased onto sP+sdST). Per-thread t2r N
        coverage is interleaved across the full hd=256, so per-WG TMA is not
        viable — instead we treat SMEM as one shared per-CTA buffer and let
        each thread's `self.store`-equivalent write into it via a (64, 256)
        virtual tensor whose N axis maps (n%64, n//64) → (N_within, stage).
        After an inter-WG barrier (256 threads), the leader warp fires 4 TMA
        bulk stores, one per stage, to the corresponding (64, 64) GMEM slice.
        Varlen falls back to per-thread self.store as in flash_bwd_sm100.py.
        """
        tidx, _, _ = cute.arch.thread_idx()
        _, K, D, HB = problem_shape
        _, blk_coord_k, _, blk_coord_batch = blk_coord

        # adi: TMEM_DK, TMEM_DV
        tmem_copy_op = tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32))
        load_op = cute.make_copy_atom(
            tmem_copy_op,
            self.acc_dtype,
        )

        tdKtdK = tdKtdK[(None, None), 0, 0]
        mdK_offset = cute.assume(blk_offset[1] * dK.stride[0], divby=64)
        mdK = cute.make_tensor(
            dK.iterator + mdK_offset,
            cute.make_layout((K, self.tile_shape_dQ_K, HB), stride=dK.stride),
        )
        gdK = cute.local_tile(mdK, (self.cta_tiler[1], self.cta_tiler[2]), (None, None, None))
        gdK = gdK[None, None, blk_coord_k, 0, blk_coord_batch]
        cdK = cute.domain_offset(
            (blk_coord_k * self.tile_shape_K, 0),
            cute.make_identity_tensor((self.cta_tiler[1], self.cta_tiler[2])),
        )

        num_warp_groups = self.num_compute_warps // 4
        dp_idx = tidx % 128
        wg_idx = (tidx % (self.num_compute_warps * self.threads_per_warp)) // 128
        leader_warp = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) == 0

        # Path 2 SMEM staging. dV stages through sdOT (already-consumed by the
        # dV MMA before the dV epilogue begins). dK stages through sP+sdST
        # (dead after dK MMA completes, before dK epilogue runs).
        s_epi_dK = cute.make_tensor(
            cute.recast_ptr(sP.iterator, sdK_epi_layout.inner, dK.element_type),
            sdK_epi_layout.outer,
        )
        s_epi_dV = cute.make_tensor(
            cute.recast_ptr(sdOT.iterator, sdV_epi_layout.inner, dV.element_type),
            sdV_epi_layout.outer,
        )

        # Compile-time: stage tile shape and number of stages.
        epi_cols_dKV = math.gcd(
            128 // (dV.element_type.width // 8), self.cta_tiler[2] // num_warp_groups
        )
        num_epi_stages_dKV = (self.cta_tiler[2] // num_warp_groups) // epi_cols_dKV
        total_epi_stages = num_warp_groups * num_epi_stages_dKV
        epi_tile_dKV = (self.cta_tiler[1], epi_cols_dKV)

        # Local (M, N) coord tensor for SMEM indexing (no global domain offset
        # — cdK/cdV are domain-offset by blk_coord_k * tile_shape_K to match
        # the GMEM destination, but the SMEM indexing must be per-CTA-local).
        cdV_local = cute.make_identity_tensor((self.cta_tiler[1], self.cta_tiler[2]))
        cdK_local = cdV_local

        tiled_t2r_dK = tcgen05.make_tmem_copy(load_op, tdKtdK)
        thread_t2r_dK = tiled_t2r_dK.get_slice(dp_idx)

        tTR_cdK = thread_t2r_dK.partition_D(cdK)
        tTR_cdK = split_wg(tTR_cdK, num_warp_groups, wg_idx)
        tTR_cdK_local = thread_t2r_dK.partition_D(cdK_local)
        tTR_cdK_local = split_wg(tTR_cdK_local, num_warp_groups, wg_idx)
        tTR_gdK = thread_t2r_dK.partition_D(gdK)
        tTR_gdK = split_wg(tTR_gdK, num_warp_groups, wg_idx)
        tTR_rdK = cute.make_rmem_tensor(tTR_cdK.shape, self.acc_dtype)
        tTR_tdK = thread_t2r_dK.partition_S(tdKtdK)
        tTR_tdK = split_wg(tTR_tdK, num_warp_groups, wg_idx)

        mdV_in = cute.make_tensor(
            dV.iterator, cute.make_layout((K, self.cta_tiler[2], HB), stride=dV.stride)
        )
        offset_mdV = cute.assume(blk_offset[1] * mdV_in.stride[0], divby=64)
        mdV = cute.make_tensor(mdV_in.iterator + offset_mdV, mdV_in.layout)
        gdV = cute.local_tile(mdV, (self.cta_tiler[1], self.cta_tiler[2]), (None, None, None))
        gdV = gdV[None, None, blk_coord_k, 0, blk_coord_batch]

        cdV = cute.domain_offset(
            (blk_coord_k * self.cta_tiler[1], 0),
            cute.make_identity_tensor((self.cta_tiler[1], self.cta_tiler[2])),
        )

        tdVtdV = tdVtdV[(None, None), 0, 0]

        tiled_t2r_dV = tcgen05.make_tmem_copy(load_op, tdVtdV)
        thread_t2r_dV = tiled_t2r_dV.get_slice(dp_idx)

        tTR_cdV = thread_t2r_dV.partition_D(cdV)
        tTR_cdV = split_wg(tTR_cdV, num_warp_groups, wg_idx)
        tTR_cdV_local = thread_t2r_dV.partition_D(cdV_local)
        tTR_cdV_local = split_wg(tTR_cdV_local, num_warp_groups, wg_idx)
        tTR_gdV = thread_t2r_dV.partition_D(gdV)
        tTR_gdV = split_wg(tTR_gdV, num_warp_groups, wg_idx)
        tTR_rdV = cute.make_rmem_tensor(tTR_cdV.shape, self.acc_dtype)
        tTR_tdV = thread_t2r_dV.partition_S(tdVtdV)
        tTR_tdV = split_wg(tTR_tdV, num_warp_groups, wg_idx)

        # GMEM destinations for the multi-stage TMA path (gated on not-varlen).
        if cutlass.const_expr(not varlen):
            mdV_tma_3d = cute.make_tensor(
                dV_tma.iterator,
                cute.make_layout((K, self.cta_tiler[2], HB), stride=dV_tma.stride),
            )
            mdV_tma_cur = mdV_tma_3d[None, None, blk_coord_batch]
            gdV_tma = cute.local_tile(
                mdV_tma_cur, (self.cta_tiler[1], self.cta_tiler[2]), (blk_coord_k, 0)
            )
            gdV_tma_epi = cute.local_tile(gdV_tma, epi_tile_dKV, (0, None))

            mdK_tma_3d = cute.make_tensor(
                dK_tma.iterator,
                cute.make_layout((K, self.cta_tiler[2], HB), stride=dK_tma.stride),
            )
            mdK_tma_cur = mdK_tma_3d[None, None, blk_coord_batch]
            gdK_tma = cute.local_tile(
                mdK_tma_cur, (self.cta_tiler[1], self.cta_tiler[2]), (blk_coord_k, 0)
            )
            gdK_tma_epi = cute.local_tile(gdK_tma, epi_tile_dKV, (0, None))

        cta_threads = self.num_compute_warps * self.threads_per_warp

        dkdv_handle = mma_compute_dKdV_consumer.wait_and_advance()

        if blk_coord_k * self.tile_shape_K < problem_shape_k_cur_batch:
            cute.copy(tiled_t2r_dV, tTR_tdV, tTR_rdV)
            tTR_rdV_cast = cute.make_rmem_tensor(tTR_rdV.shape, dV.element_type)
            tTR_rdV_cast.store(tTR_rdV.load().to(dV.element_type))

            if cutlass.const_expr(not varlen):
                # reg -> SMEM via per-element indexed stores using tTR_cdV's
                # per-thread (M, N) coords. (M, N) is per-CTA cdV space (M=0..63,
                # N=0..255). We map N=(n%epi_cols, n//epi_cols) → (N_within, stage)
                # of the 3D s_epi_dV tensor.
                for _i in cutlass.range_constexpr(cute.size(tTR_cdV_local, mode=[2])):
                    for _j in cutlass.range_constexpr(cute.size(tTR_cdV_local[None, 0, _i])):
                        c = tTR_cdV_local[None, 0, _i][_j]
                        m_pos = c[0]
                        n_pos = c[1]
                        stage_pos = n_pos // epi_cols_dKV
                        n_within_pos = n_pos % epi_cols_dKV
                        v = tTR_rdV_cast[None, 0, _i][_j]
                        s_epi_dV[m_pos, n_within_pos, stage_pos] = v
                cute.arch.fence_view_async_shared()
                # Inter-WG barrier — both warp-groups must finish their writes
                # before the leader warp reads SMEM via TMA.
                cute.arch.barrier(barrier_id=5, number_of_threads=cta_threads)
                # TMA bulk store, one (64, 64) box per stage.
                if leader_warp and wg_idx == 0:
                    for _stage in cutlass.range_constexpr(total_epi_stages):
                        sdV_stage = s_epi_dV[None, None, _stage]
                        gdV_stage = gdV_tma_epi[None, None, _stage]
                        td_sdV, td_gdV = cpasync.tma_partition(
                            tma_atom_dV,
                            0,
                            cute.make_layout(1),
                            cute.group_modes(sdV_stage, 0, 2),
                            cute.group_modes(gdV_stage, 0, 2),
                        )
                        cute.copy(tma_atom_dV, td_sdV, td_gdV)
                        cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            else:
                self.store(tTR_gdV, tTR_rdV, tTR_cdV, (K, D))

        cute.arch.fence_view_async_tmem_load()
        dkdv_handle.release()

        dkdv_handle = mma_compute_dKdV_consumer.wait_and_advance()

        if blk_coord_k * self.tile_shape_K < problem_shape_k_cur_batch:
            cute.copy(tiled_t2r_dK, tTR_tdK, tTR_rdK)

            for i in cutlass.range(cute.size(tTR_rdK), unroll_full=True):
                tTR_rdK[i] = scale_softmax * tTR_rdK[i]

            tTR_rdK_cast = cute.make_rmem_tensor(tTR_rdK.shape, dK.element_type)
            tTR_rdK_cast.store(tTR_rdK.load().to(dK.element_type))

            if cutlass.const_expr(not varlen):
                for _i in cutlass.range_constexpr(cute.size(tTR_cdK_local, mode=[2])):
                    for _j in cutlass.range_constexpr(cute.size(tTR_cdK_local[None, 0, _i])):
                        c = tTR_cdK_local[None, 0, _i][_j]
                        m_pos = c[0]
                        n_pos = c[1]
                        stage_pos = n_pos // epi_cols_dKV
                        n_within_pos = n_pos % epi_cols_dKV
                        v = tTR_rdK_cast[None, 0, _i][_j]
                        s_epi_dK[m_pos, n_within_pos, stage_pos] = v
                cute.arch.fence_view_async_shared()
                cute.arch.barrier(barrier_id=6, number_of_threads=cta_threads)
                if leader_warp and wg_idx == 0:
                    for _stage in cutlass.range_constexpr(total_epi_stages):
                        sdK_stage = s_epi_dK[None, None, _stage]
                        gdK_stage = gdK_tma_epi[None, None, _stage]
                        td_sdK, td_gdK = cpasync.tma_partition(
                            tma_atom_dK,
                            0,
                            cute.make_layout(1),
                            cute.group_modes(sdK_stage, 0, 2),
                            cute.group_modes(gdK_stage, 0, 2),
                        )
                        cute.copy(tma_atom_dK, td_sdK, td_gdK)
                        cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            else:
                self.store(tTR_gdK, tTR_rdK, tTR_cdK, (K, D))

        cute.arch.fence_view_async_tmem_load()
        dkdv_handle.release()

        return mma_compute_dKdV_consumer

    def get_workspace_tensor(
        self,
        problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32], Int32]],
        workspace: cute.Tensor,
        acc_dtype: type[cutlass.Numeric],
        varlen: bool,
    ) -> tuple[cute.Tensor, cute.Tensor, cute.Tensor]:
        """Get workspace tensor."""
        D = problem_shape[2]
        H, B = cute.size(problem_shape[3][0]), cute.size(problem_shape[3][1])
        H_r, H_k = problem_shape[3][0]
        D = cute.round_up(D, 8)

        # b = 1 for varlen, else batch_size
        b = workspace.shape[0]
        # s_q_sum for varlen, else s_q_max, already rounded to 8
        S_Q = workspace.shape[1]

        acc_bytes = acc_dtype.width // 8
        sum_OdO_bytes = cute.assume(b * H * S_Q * acc_bytes, divby=acc_bytes)
        scaled_lse_bytes = cute.assume(b * H * S_Q * acc_bytes, divby=acc_bytes)

        sum_OdO_iter = workspace.iterator
        scaled_lse_iter = sum_OdO_iter + sum_OdO_bytes

        sum_OdO_iter = cute.recast_ptr(sum_OdO_iter, dtype=self.acc_dtype)
        scaled_lse_iter = cute.recast_ptr(scaled_lse_iter, dtype=self.acc_dtype)

        sum_OdO = cute.make_tensor(
            sum_OdO_iter,
            cute.make_layout(
                (S_Q, ((H_r, H_k), B)),
                stride=(1, ((S_Q, S_Q * H_r), 0 if varlen else S_Q * H)),
            ),
        )
        scaled_lse = cute.make_tensor(
            scaled_lse_iter,
            cute.make_layout(
                (S_Q, ((H_r, H_k), B)),
                stride=(1, ((S_Q, S_Q * H_r), 0 if varlen else S_Q * H)),
            ),
        )

        return sum_OdO, scaled_lse

    @staticmethod
    def _compute_sum_OdO_grid(
        problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32], Int32]],
        block_q: int,
    ) -> tuple[Int32, Int32, Int32]:
        """Compute grid shape for sum_OdO kernel."""
        return (
            cute.ceil_div(cute.size(problem_shape[0]), block_q),
            cute.size(problem_shape[3][0]),  # H
            cute.size(problem_shape[3][1]),  # B
        )

    @staticmethod
    def _compute_bwd_grid(
        problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32], Int32]],
        block_k: int,
    ) -> tuple[Int32, Int32, Int32]:
        """Compute grid shape for bwd kernel."""
        K = problem_shape[1]
        _, H_K = problem_shape[3][0]
        B = problem_shape[3][1]
        return (cute.ceil_div(K, block_k), cute.size(H_K), cute.size(B))

    @staticmethod
    def get_workspace_size(s_q: int, d: int, h: int, b: int, acc_dtype: type[cutlass.Numeric]):
        """Get workspace size."""
        d = (d + 7) // 8 * 8  # round up to 8
        s_q = (s_q + 7) // 8 * 8  # round up to 8
        workspace_bytes = 0
        # OdO vector
        workspace_bytes += acc_dtype.width // 8
        # scaled LSE vector
        workspace_bytes += acc_dtype.width // 8
        # FP32 versions of outputs that are churned (start off with Q only)
        workspace_bytes += d * acc_dtype.width // 8
        return (b, s_q, h, workspace_bytes)

    def make_and_init_load_mma_K_pipeline(self, load_mma_K_mbar_ptr, cluster_layout_vmnk):
        """Create and initialize barrier for load mma Q."""
        load_mma_K_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.load_warp_id])
        )
        load_mma_K_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        return pipeline.PipelineTmaUmma.create(
            barrier_storage=load_mma_K_mbar_ptr,
            num_stages=self.load_mma_K_stage,
            producer_group=load_mma_K_producer_group,
            consumer_group=load_mma_K_consumer_group,
            tx_count=self.tma_copy_K_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

    def make_and_init_load_mma_V_pipeline(self, load_mma_V_mbar_ptr, cluster_layout_vmnk):
        """Create and initialize barrier for load mma Q."""
        load_mma_V_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.load_warp_id])
        )
        load_mma_V_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        return pipeline.PipelineTmaUmma.create(
            barrier_storage=load_mma_V_mbar_ptr,
            num_stages=self.load_mma_V_stage,
            producer_group=load_mma_V_producer_group,
            consumer_group=load_mma_V_consumer_group,
            tx_count=self.tma_copy_V_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

    def make_and_init_load_mma_Q_pipeline(self, load_mma_Q_mbar_ptr, cluster_layout_vmnk):
        """Create and initialize barrier for load mma Q."""
        load_mma_Q_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.load_warp_id])
        )
        load_mma_Q_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        return pipeline.PipelineTmaUmma.create(
            barrier_storage=load_mma_Q_mbar_ptr,
            num_stages=self.load_mma_Q_stage,
            producer_group=load_mma_Q_producer_group,
            consumer_group=load_mma_Q_consumer_group,
            tx_count=self.tma_copy_Q_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

    def make_and_init_load_mma_QT_pipeline(self, load_mma_QT_mbar_ptr, cluster_layout_vmnk):
        """Create and initialize barrier for load mma QT."""
        load_mma_QT_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.load_warp_id])
        )
        load_mma_QT_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        return pipeline.PipelineTmaUmma.create(
            barrier_storage=load_mma_QT_mbar_ptr,
            num_stages=self.load_mma_QT_stage,
            producer_group=load_mma_QT_producer_group,
            consumer_group=load_mma_QT_consumer_group,
            tx_count=self.tma_copy_Q_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

    def make_and_init_load_mma_dO_pipeline(self, load_mma_dO_mbar_ptr, cluster_layout_vmnk):
        """Create and initialize barrier for load mma dO."""
        load_mma_dO_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.load_warp_id])
        )
        load_mma_dO_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        return pipeline.PipelineTmaUmma.create(
            barrier_storage=load_mma_dO_mbar_ptr,
            num_stages=self.load_mma_dO_stage,
            producer_group=load_mma_dO_producer_group,
            consumer_group=load_mma_dO_consumer_group,
            tx_count=self.tma_copy_dO_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

    def make_and_init_load_compute_LSE_pipeline(self, load_compute_lse_mbar_ptr):
        """Create and initialize barrier for load compute lse."""
        load_compute_lse_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp,
            # self.threads_per_warp,
        )
        load_compute_lse_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * self.num_compute_warps,
            # self.threads_per_warp * self.num_compute_warps,
        )
        return pipeline.PipelineCpAsync.create(
            barrier_storage=load_compute_lse_mbar_ptr,
            num_stages=self.load_compute_LSE_stage,
            producer_group=load_compute_lse_producer_group,
            consumer_group=load_compute_lse_consumer_group,
        )

    def make_and_init_load_compute_sum_OdO_pipeline(self, load_compute_sum_OdO_mbar_ptr):
        """Create and initialize barrier for load sum OdO."""
        load_compute_sum_OdO_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp,
            # self.threads_per_warp,
        )
        load_compute_sum_OdO_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * self.num_compute_warps,
            # self.threads_per_warp * self.num_compute_warps,
        )
        return pipeline.PipelineCpAsync.create(
            barrier_storage=load_compute_sum_OdO_mbar_ptr,
            num_stages=self.load_compute_sum_OdO_stage,
            producer_group=load_compute_sum_OdO_producer_group,
            consumer_group=load_compute_sum_OdO_consumer_group,
        )

    def make_and_init_mma_compute_S_pipeline(self, mma_compute_S_mbar_ptr, cluster_layout_vmnk):
        """Create and initialize barrier for mma S."""
        mma_compute_S_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )
        mma_compute_S_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_compute_warps * self.threads_per_warp * cluster_layout_vmnk.shape[0][0],
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_compute_S_mbar_ptr,
            num_stages=self.mma_compute_S_stage,
            producer_group=mma_compute_S_producer_group,
            consumer_group=mma_compute_S_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

    #  Barrier to between dP = v * dO and consume of dP in compute()
    def make_and_init_mma_compute_dP_pipeline(self, mma_compute_dP_mbar_ptr, cluster_layout_vmnk):
        """Create and initialize barrier for mma Q."""
        mma_compute_dP_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )
        mma_compute_dP_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_compute_warps * self.threads_per_warp * cluster_layout_vmnk.shape[0][0],
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_compute_dP_mbar_ptr,
            num_stages=self.mma_compute_dP_stage,
            producer_group=mma_compute_dP_producer_group,
            consumer_group=mma_compute_dP_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

    def make_and_init_compute_mma_P_pipeline(self, compute_mma_P_mbar_ptr, cluster_layout_vmnk):
        """Create and initialize barrier for mma P."""
        compute_mma_P_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_compute_warps * self.threads_per_warp * cluster_layout_vmnk.shape[0][0],
            # self.num_compute_warps * self.threads_per_warp,
        )
        compute_mma_P_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )
        return pipeline.PipelineAsyncUmma.create(
            barrier_storage=compute_mma_P_mbar_ptr,
            num_stages=self.compute_mma_P_stage,
            producer_group=compute_mma_P_producer_group,
            consumer_group=compute_mma_P_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

    def make_and_init_compute_mma_dS_pipeline(self, compute_mma_dS_mbar_ptr, cluster_layout_vmnk):
        """Create and initialize barrier for mma dS."""

        compute_mma_dS_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_compute_warps * self.threads_per_warp * cluster_layout_vmnk.shape[0][0],
            # self.num_compute_warps * self.threads_per_warp,
        )
        compute_mma_dS_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )

        return pipeline.PipelineAsyncUmma.create(
            barrier_storage=compute_mma_dS_mbar_ptr,
            num_stages=self.compute_mma_dS_stage,
            producer_group=compute_mma_dS_producer_group,
            consumer_group=compute_mma_dS_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

    def make_and_init_mma_compute_dKdV_pipeline(
        self, mma_compute_dKdV_mbar_ptr, cluster_layout_vmnk
    ):
        """Create and initialize barrier for mma dKdV."""
        mma_compute_dKdV_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )
        mma_compute_dKdV_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_compute_warps * self.threads_per_warp * cluster_layout_vmnk.shape[0][0],
            # self.num_compute_warps * self.threads_per_warp,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_compute_dKdV_mbar_ptr,
            num_stages=self.mma_compute_dKdV_stage,
            producer_group=mma_compute_dKdV_producer_group,
            consumer_group=mma_compute_dKdV_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )
