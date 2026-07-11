# Supported features:
# - BF16 & FP16 dtype
# - noncausal & causal attention
# - MHA, GQA, MQA
# - hdim 64, 96, 128, (192, 128).
# - varlen
# - sliding window
# - split-kv
# Unsupported features that will be added later:
# - page size != 128
# - more hdim (192, 256)
# Based on the cutlass example and cute-dsl example:
# https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha
# https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/fmha.py

import math
from typing import Tuple, Callable, Optional, Literal, NamedTuple
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Boolean, const_expr
from cutlass.cute.nvgpu import cpasync
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from cutlass import pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import ClcDynamicPersistentTileScheduler
from cutlass.base_dsl.arch import Arch
from cutlass.cutlass_dsl import BaseDSL

from ..quack import copy_utils, layout_utils

from .paged_kv import PagedKVManager
from .cute_dsl_utils import assume_tensor_aligned
from . import utils
from . import pipeline as pipeline_custom
import cutlass.pipeline as cutlass_pipeline
from .mask import AttentionMask
from .softmax import SoftmaxSm100, apply_score_mod_inner
from .seqlen_info import SeqlenInfoQK
from .block_info import BlockInfo
from .block_sparsity import BlockSparseTensors
from .block_sparse_utils import (
    get_total_block_count,
    produce_block_sparse_loads_sm100,
    softmax_block_sparse_sm100,
    handle_block_sparse_empty_tile_correction_sm100,
)
from .pack_gqa import PackGQA, pack_gqa_layout
from . import mma_sm100_desc as sm100_desc
from . import blackwell_helpers as sm100_utils
from .named_barrier import NamedBarrierFwdSm100
from cutlass.cute import FastDivmodDivisor
from ..quack.cute_dsl_utils import ParamsBase
from .tile_scheduler import (
    ClcState,
    SchedulingMode,
    TileSchedulerArguments,
    TileSchedulerProtocol,
    SingleTileScheduler,
    StaticPersistentTileScheduler,
    SingleTileLPTScheduler,
    SingleTileVarlenScheduler,
)
from .fa_logging import fa_log, fa_printf
from .utils import smid
from .utils import AuxData

# === TUNING KNOBS (agent-editable) ===
# Keys: (use_2cta_instrs: bool, is_causal: bool, head_dim_padded: int, is_sm103: bool)
# Values:
#   ex2_emu_freq: int — how often to use emulated exp2 (0=all hardware exp2, higher=more emulation).
#                        SM103 has fast native exp2, so set freq=0 there.
#   ex2_emu_res: int — (hd256 only) number of fragment-pairs per freq period to emulate.
#   ex2_emu_start_frg: int — fragment index to start emulation from
#   num_regs_softmax: int — register count for softmax warps (multiple of 8)
#   num_regs_correction: int — register count for correction warps (multiple of 8)
#   num_regs_other is derived: 512 - num_regs_softmax * 2 - num_regs_correction
#                  (hd256 exception: num_regs_other is fixed at 32, not derived)
_TUNING_CONFIG = {
    (True, False, 128, False): {"ex2_emu_freq": 10, "ex2_emu_start_frg": 1, "num_regs_softmax": 176, "num_regs_correction": 88},
    (False, True, 128, False): {"ex2_emu_freq": 16, "ex2_emu_start_frg": 1, "num_regs_softmax": 192, "num_regs_correction": 72},
    (True, False, 192, False): {"ex2_emu_freq": 16, "ex2_emu_start_frg": 0, "num_regs_softmax": 184, "num_regs_correction": 80},
    (False, True, 192, False): {"ex2_emu_freq": 32, "ex2_emu_start_frg": 1, "num_regs_softmax": 192, "num_regs_correction": 72},
    (True, False, 128, True): {"ex2_emu_freq": 0, "ex2_emu_start_frg": 0, "num_regs_softmax": 176, "num_regs_correction": 80},
    (False, True, 128, True): {"ex2_emu_freq": 0, "ex2_emu_start_frg": 0, "num_regs_softmax": 176, "num_regs_correction": 64},
    (True, False, 192, True): {"ex2_emu_freq": 0, "ex2_emu_start_frg": 0, "num_regs_softmax": 176, "num_regs_correction": 64},
    (False, True, 192, True): {"ex2_emu_freq": 0, "ex2_emu_start_frg": 0, "num_regs_softmax": 176, "num_regs_correction": 72},
    (True, False, 256, False): {"ex2_emu_freq": 14, "ex2_emu_res": 6, "ex2_emu_start_frg": 0, "num_regs_softmax": 256, "num_regs_correction": 160},
    (True, True, 256, False): {"ex2_emu_freq": 14, "ex2_emu_res": 6, "ex2_emu_start_frg": 0, "num_regs_softmax": 256, "num_regs_correction": 160},
}
_FP8_TUNING_CONFIG = {
    (True, False, 128, False): {'ex2_emu_freq': 10, 'ex2_emu_start_frg': 1, 'num_regs_softmax': 160, 'num_regs_correction': 72},
    # Causal hd128 FP8 previously inherited bf16's freq=16. FP8 fwd is MUFU/ex2-bound, so a more
    # aggressive emulation freq=8 offloads more exp from MUFU: +3.4%(4k)..+5.5%(16k) on B200,
    # MHA & GQA, accuracy-neutral (3-run validated, locked clocks). freq=8 would regress non-causal
    # (0.94x), hence keyed on is_causal=True only.
    (False, True, 128, False): {'ex2_emu_freq': 8, 'ex2_emu_start_frg': 1},
}
_FP8_SMALL_HDIM_REGS = {
    False: {"num_regs_softmax": 168, "num_regs_correction": 96, "num_regs_other": 80},
    True: {"num_regs_softmax": 152, "num_regs_correction": 96, "num_regs_other": 112},
}
# === END TUNING KNOBS ===


class DescaleTensors(NamedTuple):
    q_descale: Optional[cute.Tensor] = None
    k_descale: Optional[cute.Tensor] = None
    v_descale: Optional[cute.Tensor] = None

    def __new_from_mlir_values__(self, values):
        return DescaleTensors(*((*values, None, None, None)[:3]))


class FlashAttentionForwardSm100:

    def __init__(
        self,
        # dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
        is_causal: bool = False,
        is_local: bool = False,
        is_split_kv: bool = False,
        pack_gqa: bool = False,
        q_subtile_factor: int = 1,
        m_block_size: int = 128,
        n_block_size: int = 128,
        q_stage: cutlass.Constexpr[int] = 2,
        is_persistent: bool = True,
        score_mod: cutlass.Constexpr | None = None,
        mask_mod: cutlass.Constexpr | None = None,
        has_aux_tensors: cutlass.Constexpr = False,
        paged_kv_non_tma: bool = False,
        is_varlen_q: bool = False,
        use_2cta_instrs: bool = False,
        use_clc_scheduler: bool = False,
    ):
        self.use_tma_KV = not paged_kv_non_tma
        # self.dtype = dtype
        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 16
        self.head_dim_padded = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.head_dim_v_padded = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        self.same_hdim_kv_padded = self.head_dim_padded == self.head_dim_v_padded
        self.check_hdim_oob = head_dim != self.head_dim_padded
        self.check_hdim_v_oob = head_dim_v != self.head_dim_v_padded
        self.m_block_size = m_block_size
        self.n_block_size = n_block_size
        self.q_stage = q_stage
        assert self.q_stage in [1, 2]
        self.use_2cta_instrs = use_2cta_instrs
        # If split_P_arrive, the softmax warps write some columns of P first, signal to the MMA warp
        # to being the P @ V MMA, then write the rest of P and signal again. This allows some overlap
        # between compute the last couple columns of P and the P @ V MMA.
        self.split_P_arrive = n_block_size // 4 * 3
        self.split_P_arrive = int(self.split_P_arrive / 32) * 32  # multiple of 32
        assert self.split_P_arrive % 32 == 0
        assert self.split_P_arrive < self.n_block_size
        self.arch = BaseDSL._get_dsl().get_arch_enum()
        assert self.arch.is_family_of(Arch.sm_100f) or self.arch.is_family_of(Arch.sm_110f), \
            "Only SM 10.x and 11.x are supported"

        self.cta_group_size = 2 if self.use_2cta_instrs else 1
        # cta_tiler M includes only 1 CTA, the scheduler will take into account the cluster shape
        self.cta_tiler = (self.q_stage * m_block_size, n_block_size, self.head_dim_padded)
        # With 2CTA, the MMA tiler M covers both CTAs, so it's cta_group_size * m_block_size.
        # Each CTA owns m_block_size rows; the 2CTA MMA instruction spans both.
        self.mma_tiler_qk = (self.cta_group_size * m_block_size, n_block_size, self.head_dim_padded)
        self.mma_tiler_pv = (self.cta_group_size * m_block_size, self.head_dim_v_padded, n_block_size)
        self.qk_acc_dtype = Float32
        self.pv_acc_dtype = Float32
        self.cluster_shape_mn = (2, 1) if self.use_2cta_instrs else (1, 1)
        self.is_persistent = is_persistent
        self.is_causal = is_causal
        self.is_local = is_local
        self.is_varlen_q = is_varlen_q
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_split_kv = is_split_kv
        self.pack_gqa = pack_gqa
        self.use_tma_O = (
            not (self.pack_gqa and self.m_block_size % self.qhead_per_kvhead != 0)
            and not (self.pack_gqa and self.is_split_kv)
            and not is_varlen_q
        )
        self.use_correction_warps_for_epi = not self.use_tma_O
        self.q_subtile_factor = q_subtile_factor
        assert not (self.is_split_kv and self.head_dim_v_padded >= 192), (
            "SplitKV is not supported for hdim >= 192"
        )
        self.score_mod = score_mod
        self.mask_mod = mask_mod
        self.score_vec_size: cutlass.Constexpr = getattr(
            score_mod, "__vec_size__", 1 if cutlass.const_expr(has_aux_tensors) else 2
        )
        self.mask_vec_size: cutlass.Constexpr = getattr(mask_mod, "__vec_size__", 1)
        # Does S1 need to wait for S0 to finish
        # self.s0_s1_barrier = self.head_dim_padded in [64, 96] and (not self.is_causal and not self.is_local)
        # NOTE: is_family_of also matches any future sm_10x with x > 3 — intentional.
        # The flag gates ex2 emulation; sm_103 (B300) has fast hardware ex2 and later
        # Blackwell variants are assumed to inherit this, so forward-inclusion is correct
        # despite the literal `is_sm103` name.
        is_sm103 = self.arch.is_family_of(Arch.sm_103f)
        self.is_sm103 = is_sm103
        # enable_ex2_emu is derived: True if tuning config has freq > 0, else fallback to default logic
        _default_enable_ex2_emu = (self.head_dim_padded <= 128 or (self.head_dim_padded == 192 and self.use_2cta_instrs and not self.is_causal and not self.is_local)) and not is_sm103
        self.enable_ex2_emu = _default_enable_ex2_emu
        self.s0_s1_barrier = False
        self.overlap_sO_sQ = (
            (self.head_dim_padded == 192 and self.head_dim_v_padded >= 64) or
            (self.head_dim_v_padded >= 128 and self.is_split_kv)
        )
        if self.overlap_sO_sQ:
            self.is_persistent = False

        assert self.use_tma_KV or not (self.check_hdim_oob or self.check_hdim_v_oob), (
            "Paged KV does not support irregular head dim"
        )

        # ClC does not compose with these other features, so disable even if requested
        self.use_clc_scheduler = (
            use_clc_scheduler
            and self.use_tma_KV
            and not self.overlap_sO_sQ
        )
        self.sched_stages = 1
        if self.use_clc_scheduler:
            assert self.cluster_shape_mn[1] == 1, f"CLC requires cluster N == 1: {self.cluster_shape_mn}"
            assert self.cluster_shape_mn[0] in (1, 2), f"bad CLC cluster M: {self.cluster_shape_mn}"
            assert self.cluster_shape_mn[0] == self.cta_group_size, (
                f"CLC cluster M != cta_group_size: {self.cluster_shape_mn}, {self.cta_group_size}"
            )

        self.scheduling_mode = SchedulingMode.CLC if self.use_clc_scheduler else SchedulingMode.STATIC

        if is_varlen_q:
            self.TileScheduler = SingleTileVarlenScheduler
        elif self.is_causal or self.is_local or self.use_clc_scheduler:
            self.TileScheduler = SingleTileLPTScheduler
        elif self.is_persistent:
            self.TileScheduler = StaticPersistentTileScheduler
        else:
            self.TileScheduler = SingleTileScheduler

        fa_log(1, f"TileScheduler={self.TileScheduler.__name__}, scheduling_mode={self.scheduling_mode.name}, USE_2CTA={self.use_2cta_instrs}")

        self.softmax0_warp_ids = (0, 1, 2, 3)
        self.softmax1_warp_ids = (4, 5, 6, 7)
        self.correction_warp_ids = (8, 9, 10, 11)
        self.mma_warp_id = 12
        self.epilogue_warp_ids = (13,)
        self.load_warp_ids = (14,)
        self.empty_warp_ids = (15,)
        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")

        self.threads_per_cta = cute.arch.WARP_SIZE * len(
            (
                *self.softmax0_warp_ids,
                *self.softmax1_warp_ids,
                *self.correction_warp_ids,
                self.mma_warp_id,
                *self.load_warp_ids,
                *self.epilogue_warp_ids,
                *self.empty_warp_ids,
            )
        )

        self.use_tma_Q = not (self.pack_gqa and self.m_block_size % self.qhead_per_kvhead != 0)

        if self.q_stage == 1:
            if not self.use_tma_KV or not self.use_tma_Q:
                self.empty_warp_ids = self.empty_warp_ids + self.load_warp_ids
                self.load_warp_ids = self.softmax1_warp_ids
            else:
                self.empty_warp_ids = self.empty_warp_ids + self.softmax1_warp_ids
            self.softmax1_warp_ids = ()
        elif not self.use_tma_KV:
            self.load_warp_ids = (14, 15)
            self.empty_warp_ids = ()

        if self.use_correction_warps_for_epi:
            self.empty_warp_ids = self.empty_warp_ids + self.epilogue_warp_ids
            self.epilogue_warp_ids = self.correction_warp_ids

        self.clc_scheduler_warp_id = self.empty_warp_ids[0] if self.use_clc_scheduler else None

        self.tmem_s_offset = [0, self.n_block_size]  # e.g., 0, 128
        self.tmem_o_offset = [
            self.tmem_s_offset[-1] + self.n_block_size + i * self.head_dim_v_padded
            for i in range(self.q_stage)
        ]  # e.g., 256, 384
        self.tmem_total = self.tmem_o_offset[-1] + self.head_dim_v_padded
        assert self.tmem_total <= self.tmem_alloc_cols
        self.tmem_s_to_p_offset = self.n_block_size // 2
        self.tmem_p_offset = [
            self.tmem_s_offset[i] + self.tmem_s_to_p_offset for i in range(2)
        ]  # 0, 128

        # vec buffer for row_max & row_sum
        self.tmem_vec_offset = self.tmem_s_offset

        # Look up tuning config for register counts and ex2_emu params
        _tune_key = (self.use_2cta_instrs, self.is_causal, self.head_dim_padded, self.is_sm103)
        self._tune = _TUNING_CONFIG.get(_tune_key, {})
        if "ex2_emu_freq" in self._tune:
            self.enable_ex2_emu = self._tune["ex2_emu_freq"] > 0
        if self.head_dim_padded < 96:
            self.num_regs_softmax = 200 if not paged_kv_non_tma else 184
            self.num_regs_correction = 64
            self.num_regs_other = 48 if not paged_kv_non_tma else 80
        else:
            if not paged_kv_non_tma and "num_regs_softmax" in self._tune:
                self.num_regs_softmax = self._tune["num_regs_softmax"]
                self.num_regs_correction = self._tune["num_regs_correction"]
            elif not paged_kv_non_tma:
                self.num_regs_softmax = 192
                self.num_regs_correction = 80
            else:
                self.num_regs_softmax = 184
                self.num_regs_correction = 64
            self.num_regs_other = 512 - self.num_regs_softmax * 2 - self.num_regs_correction

        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        """Set up configurations and parameters for the FMHA kernel operation.

        This method initializes and configures various attributes required for the
        execution of the fused multi-head attention kernel, mainly about the pipeline stages:

        - Sets up staging parameters for Q, K, V inputs and accumulator data
        - Configures pipeline stages for softmax, correction, and epilogue operations
        """

        smem_size_q = self.q_stage * self.m_block_size * self.head_dim_padded * self.q_dtype.width // 8
        smem_size_o = self.q_stage * self.m_block_size * self.head_dim_v_padded * self.o_dtype.width // 8
        smem_size_q_o = smem_size_q + smem_size_o if not self.overlap_sO_sQ else max(smem_size_q, smem_size_o)
        smem_size_k_per_stage = self.n_block_size * self.head_dim_padded * self.k_dtype.width // 8
        smem_size_v_per_stage = self.n_block_size * self.head_dim_v_padded * self.v_dtype.width // 8
        smem_size_kv_per_stage = max(smem_size_k_per_stage, smem_size_v_per_stage) // self.cta_group_size
        # Cap small head_dim from over-staging: the 224*1024 budget undercounts
        # per-stage state, so at hd_padded=16 the unbounded formula picks 52 stages
        # and overflows the 227 KB SMEM cap. No-op for hd_padded >= 32 (max 26).
        kv_stage = min((224 * 1024 - smem_size_q_o) // smem_size_kv_per_stage, 32)
        if self.head_dim_padded == 192 and self.head_dim_v_padded == 128 and kv_stage == 2:
            # For hdim 192,128, we can fit 3 stages if we use uneven_kv_smem
             kv_stage = 3
        self.kv_stage = kv_stage
        # print("kv_stage", self.kv_stage)
        self.s_stage = 2
        assert self.s_stage >= self.q_stage
        # For hdim 192,128 1CTA, we don't have enough smem to store all 3 stages of KV:
        # 128 x 192 x 2 bytes x 3 stages = 144KB, and we need 96KB for Q.
        # Instead we store smem as [smem_large, smem_small, smem_large], where smem_large is
        # 128 x 192 and smem_small is 128 x 128. We set the stride between the stages to be
        # 128 * 160, so that indexing the 0th and 2nd stages will get the right address,
        # but for the 1st stage we need to add or subtract (depending on phase) 128 x 64.
        self.uneven_kv_smem = (
            self.head_dim_padded == 192 and self.head_dim_v_padded == 128 and self.kv_stage == 3
        )
        self.uneven_kv_smem_offset = (
            self.n_block_size * (self.head_dim_padded - self.head_dim_v_padded) // 2
            if self.uneven_kv_smem
            else 0
        )
        assert self.uneven_kv_smem_offset % 1024 == 0

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
        mK: cute.Tensor,  # (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size, h_k, d) if there is page_table
        mV: cute.Tensor,  # (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages, page_size, h_k, dv) if there is page_table
        mO: cute.Tensor,  # (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,  # (b_k, max_num_pages_per_seq)
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        learnable_sink: Optional[cute.Tensor] = None,
        descale_tensors: Optional[DescaleTensors] = None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        aux_data: AuxData = AuxData(),
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        """Execute the Fused Multi-Head Attention operation on the provided tensors.

        This method prepares the input tensors for processing, validates their shapes and types,
        configures the computation parameters, and launches the CUDA kernel.

        The method handles:
        1. Tensor layout transformations for specific memory access patterns
        2. Validation of tensor shapes and data types
        3. Initialization of hardware-specific parameters and memory layouts
        4. Configuration of TMA (Tensor Memory Access) operations
        5. Grid and work scheduling computation
        6. Kernel launch with appropriate parameters
        """
        # setup static attributes before smem/grid/tma computation
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.o_dtype = mO.element_type
        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        Q_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=Q_layout_transpose))
        # (s_k, d, h_k, b_k) or (total_k, d, h_k) if there's cu_seqlens_k or (page_size, d, h_k, num_pages) if there's page_table
        KV_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        mK, mV = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=KV_layout_transpose))
            for t in (mK, mV)
        ]
        if const_expr(self.is_split_kv):
            O_layout_transpose = [2, 4, 3, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 3, 2, 0]
            LSE_layout_transpose = [3, 2, 1, 0] if const_expr(mCuSeqlensQ is None) else [2, 1, 0]
            num_splits = mO.shape[0]
        else:
            O_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
            LSE_layout_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
            num_splits = Int32(1)
        mO = cute.make_tensor(mO.iterator, cute.select(mO.layout, mode=O_layout_transpose))
        mLSE = (
            cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose))
            if const_expr(mLSE is not None)
            else None
        )
        # (s, d, h, b) -> (d, s, h, b)
        V_layout_transpose = [1, 0, 2, 3] if const_expr(mCuSeqlensK is None) else [1, 0, 2]
        mV = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=V_layout_transpose))

        # check type consistency
        if const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if const_expr(self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")
        if const_expr(self.q_dtype.width == 8):
            paged_kv_non_tma = not self.use_tma_KV
            if const_expr(self.head_dim_padded < 96):
                fp8_regs = _FP8_SMALL_HDIM_REGS[paged_kv_non_tma]
                self.num_regs_softmax = fp8_regs["num_regs_softmax"]
                self.num_regs_correction = fp8_regs["num_regs_correction"]
                self.num_regs_other = fp8_regs["num_regs_other"]
            else:
                fp8_tune = _FP8_TUNING_CONFIG.get(
                    (self.use_2cta_instrs, self.is_causal, self.head_dim_padded, self.is_sm103), {}
                )
                if const_expr("ex2_emu_freq" in fp8_tune):
                    self._tune = {**self._tune, **fp8_tune}
                    self.enable_ex2_emu = self._tune["ex2_emu_freq"] > 0
                if const_expr(not paged_kv_non_tma and "num_regs_softmax" in fp8_tune):
                    self.num_regs_softmax = fp8_tune["num_regs_softmax"]
                    self.num_regs_correction = fp8_tune["num_regs_correction"]
                    self.num_regs_other = 512 - self.num_regs_softmax * 2 - self.num_regs_correction
        self._setup_attributes()
        self.ex2_emu_freq = 0
        self.ex2_emu_start_frg = self._tune.get("ex2_emu_start_frg", 1)
        if const_expr(self.enable_ex2_emu):
            self.ex2_emu_freq = self._tune.get("ex2_emu_freq", 16)
            if const_expr(
                self.pack_gqa and self.head_dim_padded > 64 and not self.is_causal and not self.is_local
            ):
                self.ex2_emu_freq = 32 if mCuSeqlensQ is not None or mSeqUsedQ is not None else self._tune.get("ex2_emu_freq", 10)

        cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        q_major_mode = tcgen05.OperandMajorMode.K
        k_major_mode = tcgen05.OperandMajorMode.K
        v_major_mode = tcgen05.OperandMajorMode.MN
        self.o_layout = cutlass.utils.LayoutEnum.from_tensor(mO)
        # the intermediate tensor p is from tmem & mK-major
        p_source = tcgen05.OperandSource.TMEM
        p_major_mode = tcgen05.OperandMajorMode.K
        tiled_mma_qk = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            q_major_mode,
            k_major_mode,
            self.qk_acc_dtype,
            cta_group,
            self.mma_tiler_qk[:2],
        )
        tiled_mma_pv = sm100_utils_basic.make_trivial_tiled_mma(
            self.v_dtype,
            p_major_mode,
            v_major_mode,
            self.pv_acc_dtype,
            cta_group,
            self.mma_tiler_pv[:2],
            p_source,
        )

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,)
        )

        # epi_tile is per-CTA (not full 2CTA) since each CTA writes its own O portion
        self.epi_tile = (self.m_block_size, self.head_dim_v_padded)

        sQ_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_qk, self.mma_tiler_qk, self.q_dtype, self.q_stage
        )
        sK_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_qk, self.mma_tiler_qk, self.k_dtype, self.kv_stage
        )
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_pv, self.mma_tiler_pv, self.q_dtype, self.s_stage
        )
        sV_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_pv, self.mma_tiler_pv, self.v_dtype, self.kv_stage
        )
        sO_layout = sm100_utils_basic.make_smem_layout_epi(
            self.o_dtype, self.o_layout, self.epi_tile, self.q_stage
        )
        if const_expr(not self.same_hdim_kv_padded):
            # sK and sV are using the same physical smem so we need to adjust the stride so that they line up
            stride_sK = const_expr(
                max(sK_layout.outer.stride[-1], 0)
            )  # take max to turn tuple to Int32
            stride_sV = const_expr(max(sV_layout.outer.stride[-1], 0))
            stage_stride = const_expr(
                max(stride_sK, stride_sV)
                if not self.uneven_kv_smem
                else (stride_sK + stride_sV) // 2
            )
            sK_layout = cute.make_composed_layout(
                sK_layout.inner,
                0,
                cute.make_layout(
                    (*sK_layout.outer.shape[:-1], self.kv_stage),
                    stride=(*sK_layout.outer.stride[:-1], stage_stride),
                ),
            )
            sV_layout = cute.make_composed_layout(
                sV_layout.inner,
                0,
                cute.make_layout(
                    (*sV_layout.outer.shape[:-1], self.kv_stage),
                    stride=(*sV_layout.outer.stride[:-1], stage_stride),
                ),
            )

        if const_expr(self.pack_gqa):
            nheads_kv = mK.shape[2]
            mQ = pack_gqa_layout(mQ, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            mO = pack_gqa_layout(mO, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            if const_expr(mLSE is not None):
                mLSE = pack_gqa_layout(mLSE, self.qhead_per_kvhead, nheads_kv, head_idx=1)

        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1, 2]))
            for name, mX, layout in [
                ("Q", mQ, sQ_layout),
                ("K", mK, sK_layout),
                ("V", mV, sV_layout),
            ]
        }
        for name in ("Q", "K", "V"):
            self.tma_copy_bytes[name] *= self.cta_group_size

        # TMA load for Q
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()

        if const_expr(self.use_tma_Q):
            tma_atom_Q, mQ = cute.nvgpu.make_tiled_tma_atom_A(
                tma_load_op,
                mQ,
                cute.select(sQ_layout, mode=[0, 1, 2]),
                self.mma_tiler_qk,
                tiled_mma_qk,
                cta_layout_vmnk.shape,
            )
            gmem_tiled_copy_Q = None
        else:
            tma_atom_Q = None
            async_copy_elems = 128 // self.q_dtype.width
            num_load_threads = cute.arch.WARP_SIZE * len(self.load_warp_ids)
            threads_per_row = math.gcd(self.head_dim_padded // async_copy_elems, num_load_threads)
            gmem_tiled_copy_Q = copy_utils.tiled_copy_2d(
                self.q_dtype, threads_per_row, num_load_threads, async_copy_elems, is_async=True
            )

        tma_atom_K = None
        tma_atom_V = None
        if const_expr(self.use_tma_KV):
            # TMA load for K
            tma_atom_K, mK = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mK,
                cute.select(sK_layout, mode=[0, 1, 2]),
                self.mma_tiler_qk,
                tiled_mma_qk,
                cta_layout_vmnk.shape,
            )
            # TMA load for V
            tma_atom_V, mV = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mV,
                cute.select(sV_layout, mode=[0, 1, 2]),
                self.mma_tiler_pv,
                tiled_mma_pv,
                cta_layout_vmnk.shape,
            )

        self.num_epilogue_threads = cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
        if const_expr(self.use_tma_O):
            tma_atom_O, mO = cpasync.make_tiled_tma_atom(
                tma_store_op, mO, cute.select(sO_layout, mode=[0, 1]), self.epi_tile
            )
            gmem_tiled_copy_O = None
        else:
            tma_atom_O = None
            universal_copy_bits = 128
            async_copy_elems = universal_copy_bits // self.o_dtype.width
            atom_universal_copy = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.o_dtype,
                num_bits_per_copy=universal_copy_bits,
            )
            tO_shape_dim_1 = sO_layout.outer.shape[1][0] // async_copy_elems
            tO_layout = cute.make_ordered_layout(
                (self.num_epilogue_threads // tO_shape_dim_1, tO_shape_dim_1),
                order=(1, 0),
            )
            # So that we don't have to check if we overshoot kBlockM when we store O
            assert self.m_block_size % tO_layout.shape[0] == 0
            vO_layout = cute.make_layout((1, async_copy_elems))
            gmem_tiled_copy_O = cute.make_tiled_copy_tv(atom_universal_copy, tO_layout, vO_layout)

        TileScheduler = self.TileScheduler
        _num_block_divisor = self.cta_tiler[0] * (self.cta_group_size if not self.is_persistent and self.cta_group_size > 1 else 1)
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), _num_block_divisor),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3])
            if const_expr(mCuSeqlensQ is None)
            else cute.size(mCuSeqlensQ.shape[0] - 1),
            num_splits,
            cute.size(mK.shape[0])
            if const_expr(mPageTable is None)
            else mK.shape[0] * mPageTable.shape[1],
            mQ.shape[1],
            mV.shape[0],  # Note that this is different from Sm90 since we transpose mV in Sm100
            total_q=cute.size(mQ.shape[0])
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=self.cta_tiler[:2],
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            element_size=self.k_dtype.width // 8,
            is_persistent=self.is_persistent,
            lpt=self.is_causal or self.is_local,
            is_split_kv=self.is_split_kv,
            cluster_shape_mn=self.cluster_shape_mn,
            use_cluster_idx=not self.is_persistent and self.cta_group_size > 1,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(
            tile_sched_args, scheduling_mode=self.scheduling_mode
        )
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        sO_size = cute.cosize(sO_layout) if const_expr(not self.overlap_sO_sQ) else 0
        sQ_size = (
            cute.cosize(sQ_layout) if const_expr(not self.overlap_sO_sQ) else
            cutlass.max(cute.cosize(sQ_layout), cute.cosize(sO_layout) * self.o_dtype.width // self.q_dtype.width)
        )

        clc_response_size = self.sched_stages * 4 if self.use_clc_scheduler else 0
        clc_mbar_size = self.sched_stages * 2 if self.use_clc_scheduler else 0

        @cute.struct
        class SharedStorage:
            # m_barriers for pipelines
            mbar_load_Q: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_load_KV: cute.struct.MemRange[Int64, self.kv_stage * 2]
            mbar_S_full_P_full_O_rescaled: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_P_full_lastsplit: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_O_full: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_softmax_stats: cute.struct.MemRange[Int64, self.q_stage * 2]
            # mbar_softmax_stats: cute.struct.MemRange[Int64, self.q_stage * 4 * 2]
            mbar_O_epi: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_s0_s1_sequence: cute.struct.MemRange[Int64, 2 * 2]
            # Tmem dealloc cluster barrier
            tmem_dealloc_mbar: Int64
            # Tmem holding buffer
            tmem_holding_buf: Int32
            # Smem tensors
            # store row max and row sum
            sScale: cute.struct.MemRange[Float32, self.q_stage * self.m_block_size * 2]
            # CLC buffers placed here to utilize padding before sO's 1024-byte alignment.
            # This avoids adding bytes at the end when we're at the smem limit.
            # PipelineClcFetchAsync expects 2 * sched_stages mbarriers (full + empty).
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, clc_mbar_size]
            # CLC response storage (16 bytes per stage, stored as 4 Int32s).
            clc_response: cute.struct.MemRange[Int32, clc_response_size]
            # Large TMA buffers with 1024-byte alignment
            sO: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, sO_size], self.buffer_align_bytes
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, sQ_size], self.buffer_align_bytes
            ]
            sK: cute.struct.Align[
                # cute.cosize(sK_layout) is correct even in the case of self.uneven_kv_smem
                cute.struct.MemRange[self.k_dtype, cute.cosize(sK_layout)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(softmax_scale, self.score_mod)
        window_size_left = Int32(window_size_left) if window_size_left is not None else None
        window_size_right = Int32(window_size_right) if window_size_right is not None else None
        fastdiv_mods = utils.compute_fastdiv_mods(mQ, mK, self.qhead_per_kvhead, self.pack_gqa, aux_data.tensors, mPageTable)

        head_divmod = None
        if cutlass.const_expr(self.pack_gqa):
            head_divmod = FastDivmodDivisor(self.qhead_per_kvhead)

        self.use_block_sparsity = cutlass.const_expr(blocksparse_tensors is not None)
        if cutlass.const_expr(self.use_block_sparsity and mPageTable is not None):
            raise NotImplementedError("Block sparsity + paged KV not supported on SM100")
        if cutlass.const_expr(self.use_block_sparsity and self.is_varlen_q):
            assert const_expr(blocksparse_tensors.cu_total_m_blocks is not None), (
                "blocksparse_tensors.cu_total_m_blocks must be provided for varlen blocksparsity"
            )

        # Launch the kernel synchronously
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mPageTable,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            softmax_scale_log2,
            softmax_scale,
            window_size_left,
            window_size_right,
            learnable_sink,
            descale_tensors,
            blocksparse_tensors,
            sQ_layout,
            sK_layout,
            tP_layout,
            sV_layout,
            sO_layout,
            gmem_tiled_copy_Q,
            gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
            num_splits,
            aux_data,
            fastdiv_mods,
            head_divmod,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk if cute.size(self.cluster_shape_mnk) > 1 else None,
            stream=stream,
            min_blocks_per_mp=1,
        )

    def _generate_attention_mask_cls(self, window_size_left, window_size_right):
        return partial(
            AttentionMask,
            self.m_block_size,
            self.n_block_size,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            qhead_per_kvhead_packgqa=(
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
            ),
        )

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,  # (s_q, d, h, b) or (total_q, d, h) if there is cu_seqlens_q
        mK: cute.Tensor,  # (s_k, d, h_k, b_k) or (total_k, d, h_k) if there is cu_seqlens_k or (page_size, d, h_k, num_pages) if there is page_table
        mV: cute.Tensor,  # (d, s_k, h_k, b_k) or (d, total_k, h_k) if there is cu_seqlens_k or (d, page_size, h_k, num_pages) if there is page_table
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        tma_atom_O: Optional[cute.CopyAtom],
        softmax_scale_log2: Float32,
        softmax_scale: Float32 | None,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        learnable_sink: Optional[cute.Tensor],
        descale_tensors: Optional[DescaleTensors],
        blocksparse_tensors: Optional[BlockSparseTensors],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_Q: Optional[cute.TiledCopy],
        gmem_tiled_copy_O: Optional[cute.TiledCopy],
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        num_splits: Int32,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        head_divmod=None,
    ):
        """The device kernel implementation of the Fused Multi-Head Attention.

        This kernel coordinates multiple specialized warps to perform different phases of the FMHA computation:
        1. Load warp: Loads Q, K, V data from global memory to shared memory using TMA
        2. MMA warp: Performs matrix multiplications (Q*K^T and P*V)
        3. Softmax warps: Compute softmax normalization on attention scores
        4. Correction warps: Apply adjustments to intermediate results
        5. Epilogue warp: Handles final output transformation and storage

        The kernel implements a complex pipeline with overlapping computation and memory operations,
        using tensor memory access (TMA) for efficient data loading, warp specialization for different
        computation phases, and optional attention masking.
        """

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch tma descriptor
        if warp_idx == 0:
            for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,)
        )
        # Setup cta/thread coordinates
        bidx, _, _ = cute.arch.block_idx()
        if const_expr(cute.size(tiled_mma_qk.thr_id.shape) == 1):
            mma_tile_coord_v = 0
        else:
            mma_tile_coord_v = bidx % cute.size(tiled_mma_qk.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0

        # Alloc
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.TmemPtr),
            num_threads=cute.arch.WARP_SIZE * len(
                (self.mma_warp_id,
                 *self.softmax0_warp_ids,
                 *self.softmax1_warp_ids,
                 *self.correction_warp_ids)
            ),
        )
        # Tensor memory dealloc barrier init
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        ThreadCooperativeGroup = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
        mma_warp = ThreadCooperativeGroup(len([self.mma_warp_id]))
        tma_warp = ThreadCooperativeGroup(1)
        load_threads = ThreadCooperativeGroup(len(self.load_warp_ids) * cute.arch.WARP_SIZE)
        softmax_warps = ThreadCooperativeGroup(len(self.softmax0_warp_ids))
        softmax_threads = ThreadCooperativeGroup(cute.arch.WARP_SIZE * len(self.softmax0_warp_ids))
        # softmax_threads = ThreadCooperativeGroup(cute.arch.WARP_SIZE)
        correction_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.correction_warp_ids)
        )
        # correction_threads = ThreadCooperativeGroup(cute.arch.WARP_SIZE)
        softmax_correction_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.softmax0_warp_ids + self.correction_warp_ids)
        )
        epilogue_threads = ThreadCooperativeGroup(cute.arch.WARP_SIZE * len(self.epilogue_warp_ids))
        # For UMMA-bridging pipelines: the non-MMA side spans both CTAs in the cluster,
        # so the thread count must include warps from both CTAs.
        softmax_warps_cluster = ThreadCooperativeGroup(
            len(self.softmax0_warp_ids) * self.cta_group_size
        )
        correction_threads_cluster = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.correction_warp_ids) * self.cta_group_size
        )
        softmax_correction_threads_cluster = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.softmax0_warp_ids + self.correction_warp_ids) * self.cta_group_size
        )
        if const_expr(self.use_tma_Q):
            pipeline_q = pipeline_custom.PipelineTmaUmma.create(
                barrier_storage=storage.mbar_load_Q.data_ptr(),
                num_stages=self.q_stage,
                producer_group=tma_warp,
                consumer_group=mma_warp,
                tx_count=self.tma_copy_bytes["Q"],
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,
            )
        else:
            pipeline_q = pipeline_custom.PipelineAsyncUmma.create(
                barrier_storage=storage.mbar_load_Q.data_ptr(),
                num_stages=self.q_stage,
                producer_group=load_threads,
                consumer_group=mma_warp,
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,
            )
        if const_expr(self.use_tma_KV):
            pipeline_kv = pipeline_custom.PipelineTmaUmma.create(
                barrier_storage=storage.mbar_load_KV.data_ptr(),
                num_stages=self.kv_stage,
                producer_group=tma_warp,
                consumer_group=mma_warp,
                tx_count=self.tma_copy_bytes["K"],
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,
            )
        else:
            pipeline_kv = pipeline.PipelineAsyncUmma.create(
                barrier_storage=storage.mbar_load_KV.data_ptr(),
                num_stages=self.kv_stage,
                producer_group=load_threads,
                consumer_group=mma_warp,
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,
            )
        # This pipeline is not the typical producer-consumer pipeline. The "producer" mma warp
        # uses it to signal that S is ready, and the softmax threads wait for S to be ready.
        # When softmax threads write P to tmem and the correction threads have rescaled O, they
        # signal as "consumer". The mma warp then waits for that signal to do the P @ V gemm.
        pipeline_s_p_o = pipeline_custom.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_S_full_P_full_O_rescaled.data_ptr(),
            num_stages=self.q_stage,
            producer_group=mma_warp,
            consumer_group=softmax_correction_threads_cluster,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_p_lastsplit = pipeline_custom.PipelineAsyncUmma.create(
            barrier_storage=storage.mbar_P_full_lastsplit.data_ptr(),
            num_stages=self.q_stage,
            producer_group=softmax_warps_cluster,
            consumer_group=mma_warp,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        # MMA warp uses this to signal to the correction warps that O is ready.
        pipeline_o_acc = pipeline_custom.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_O_full.data_ptr(),
            num_stages=self.q_stage,
            producer_group=mma_warp,
            consumer_group=correction_threads_cluster,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_s0_s1_sequence = None
        if const_expr(self.s0_s1_barrier and self.q_stage > 1):
            # This is not a typical producer-consumer pipeline. We will directly use
            # pipeline_s0_s1_sequence.sync_object_full and will not use
            # pipeline_s0_s1_sequence.sync_object_empty.
            pipeline_s0_s1_sequence = pipeline_custom.PipelineAsync.create(
                barrier_storage=storage.mbar_s0_s1_sequence.data_ptr(),
                num_stages=2,
                producer_group=softmax_threads,
                consumer_group=softmax_threads,
                defer_sync=True,
            )
        pipeline_sm_stats = pipeline_custom.PipelineAsync.create(
            barrier_storage=storage.mbar_softmax_stats.data_ptr(),
            num_stages=self.q_stage,
            producer_group=softmax_threads,
            consumer_group=correction_threads,
            defer_sync=True,
        )
        # Should put the NamedBarrier inside the pipeline class so we'll just have pipeline_sm_stats
        sm_stats_barrier = pipeline_custom.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.SoftmaxStatsW0), num_threads=cute.arch.WARP_SIZE * 2
        )
        pipeline_o_epi = None
        if const_expr(not self.use_correction_warps_for_epi):
            pipeline_o_epi = pipeline_custom.PipelineAsync.create(
                barrier_storage=storage.mbar_O_epi.data_ptr(),
                num_stages=self.q_stage,
                producer_group=correction_threads,
                consumer_group=epilogue_threads,
                defer_sync=True,
            )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=cta_layout_vmnk, is_relaxed=True)

        #  Generate smem tensor Q/K/V/O
        # (MMA, MMA_Q, MMA_D, PIPE)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        # Strip swizzle info to reuse smem
        sV = cute.make_tensor(cute.recast_ptr(sK.iterator, sV_layout.inner), sV_layout.outer)
        if const_expr(not self.overlap_sO_sQ):
            sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)
        else:
            sO = cute.make_tensor(cute.recast_ptr(sQ.iterator, sO_layout.inner, self.o_dtype), sO_layout.outer)

        sScale = storage.sScale.get_tensor(cute.make_layout(self.q_stage * self.m_block_size * 2))

        thr_mma_qk = tiled_mma_qk.get_slice(mma_tile_coord_v)
        thr_mma_pv = tiled_mma_pv.get_slice(mma_tile_coord_v)

        qk_acc_shape = thr_mma_qk.partition_shape_C(self.mma_tiler_qk[:2])
        # This is a fake tensor, by right we need to retrieve tmem_ptr. But we know that we always
        # request 512 columns of tmem, so we know that it starts at 0.
        tStS = thr_mma_qk.make_fragment_C(cute.append(qk_acc_shape, self.s_stage))
        pv_acc_shape = thr_mma_pv.partition_shape_C(self.mma_tiler_pv[:2])
        tOtO = thr_mma_pv.make_fragment_C(cute.append(pv_acc_shape, self.q_stage))
        tOtO = cute.make_tensor(tOtO.iterator + self.tmem_o_offset[0], tOtO.layout)
        tP = cute.make_tensor(tStS.iterator, tP_layout.outer)
        tOrP = thr_mma_pv.make_fragment_A(tP)[None, None, None, 0]
        # Need to multiply by width ratio bc tP is in v_dtype but tmem offsets are in FP32
        tP_width_ratio = Float32.width // self.v_dtype.width
        # Need to adjust the stage stride manually since the two stages aren't contiguous in tmem
        tP_stage_stride = (self.tmem_p_offset[1] - self.tmem_p_offset[0]) * tP_width_ratio
        tOrP = cute.make_tensor(
            tOrP.iterator + self.tmem_p_offset[0] * tP_width_ratio,
            cute.append(tOrP.layout, cute.make_layout((self.s_stage,), stride=(tP_stage_stride,)))
        )

        block_info = BlockInfo(
            # This is cta_tiler, not mma_tiler_qk, since we move by block by (2 * mma_tiler[0], mma_tiler[1])
            self.cta_tiler[0],
            self.cta_tiler[1],
            self.is_causal,
            self.is_local,
            self.is_split_kv,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1],
            seqlen_k_static=mK.shape[0]
            if const_expr(mPageTable is None)
            else mK.shape[0] * mPageTable.shape[1],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            mCuTotalMBlocks=(
                blocksparse_tensors.cu_total_m_blocks if blocksparse_tensors is not None else None
            ),
            mCuBlockIdxOffsets=(
                blocksparse_tensors.cu_block_idx_offsets if blocksparse_tensors is not None else None
            ),
        )
        AttentionMaskCls = self._generate_attention_mask_cls(
            window_size_left, window_size_right
        )
        # Cluster wait before tensor memory alloc
        pipeline_init_wait(cluster_shape_mn=cta_layout_vmnk)

        if const_expr(self.use_clc_scheduler):
            clc_response_ptr = storage.clc_response.data_ptr()
            clc_mbar_ptr = storage.clc_mbar_ptr.data_ptr()

            clc_pipeline_producer_group = cutlass_pipeline.CooperativeGroup(
                cutlass_pipeline.Agent.Thread
            )
            num_clc_consumer_warps_per_cta = self.threads_per_cta // cute.arch.WARP_SIZE
            # NB on CTA0 warp15 == scheduler on CTA1 == empty but still both consume
            num_clc_consumer_warps = num_clc_consumer_warps_per_cta * self.cta_group_size
            clc_pipeline_consumer_group = cutlass_pipeline.CooperativeGroup(
                cutlass_pipeline.Agent.Thread, cute.arch.WARP_SIZE * num_clc_consumer_warps
            )

            block_idx = cute.arch.block_idx()
            clc = ClcState.create(
                hw_scheduler=ClcDynamicPersistentTileScheduler.create(
                    self.tile_scheduler_cls.clc_problem_shape(tile_sched_params),
                    block_idx,
                    cute.arch.grid_dim(),
                    clc_response_ptr,
                ),
                pipeline=cutlass_pipeline.PipelineClcFetchAsync.create(
                    barrier_storage=clc_mbar_ptr,
                    num_stages=self.sched_stages,
                    producer_group=clc_pipeline_producer_group,
                    consumer_group=clc_pipeline_consumer_group,
                    tx_count=16,
                    cta_layout_vmnk=cta_layout_vmnk,
                ),
                consumer_state=cutlass_pipeline.make_pipeline_state(
                    cutlass_pipeline.PipelineUserType.Consumer, self.sched_stages
                ),
                producer_state=cutlass_pipeline.make_pipeline_state(
                    cutlass_pipeline.PipelineUserType.Producer, self.sched_stages
                ),
            )
            tile_scheduler = self.tile_scheduler_cls.create(tile_sched_params, clc=clc)
        else:
            tile_scheduler = self.tile_scheduler_cls.create(tile_sched_params)
        assert isinstance(tile_scheduler, TileSchedulerProtocol), f"tile_scheduler is not a TileSchedulerProtocol: {type(tile_scheduler)}"

        # ///////////////////////////////////////////////////////////////////////////////
        #  EMPTY / CLC SCHEDULER WARP
        # ///////////////////////////////////////////////////////////////////////////////
        if const_expr(self.use_clc_scheduler):
            if warp_idx == self.clc_scheduler_warp_id:
                cute.arch.setmaxregister_decrease(self.num_regs_other)
                if is_leader_cta:
                    self.clc_scheduler_warp(tile_scheduler)
                else:
                    self.empty_warp(tile_scheduler)
            for i in cutlass.range_constexpr(len(self.empty_warp_ids)):
                if warp_idx == self.empty_warp_ids[i] and warp_idx != self.clc_scheduler_warp_id:
                    cute.arch.setmaxregister_decrease(self.num_regs_other)
                    self.empty_warp(tile_scheduler)
        else:
            for i in cutlass.range_constexpr(len(self.empty_warp_ids)):
                if warp_idx == self.empty_warp_ids[i]:
                    cute.arch.setmaxregister_decrease(self.num_regs_other)

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.load_warp_ids[0] and warp_idx <= self.load_warp_ids[-1]:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self.load(
                thr_mma_qk,
                thr_mma_pv,
                mQ,
                mK,
                mV,
                sQ,
                sK,
                sV,
                mPageTable,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                gmem_tiled_copy_Q,
                pipeline_q,
                pipeline_kv,
                block_info,
                num_splits,
                SeqlenInfoCls,
                blocksparse_tensors,
                tile_scheduler=tile_scheduler,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            # Alloc tensor memory buffer
            tmem.allocate(cute.arch.get_max_tmem_alloc_cols("sm_100"))
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                sQ,
                sK,
                sV,
                tStS,
                tOtO,
                tOrP,
                pipeline_q,
                pipeline_kv,
                pipeline_s_p_o,
                pipeline_p_lastsplit,
                pipeline_o_acc,
                is_leader_cta,
                block_info,
                num_splits,
                SeqlenInfoCls,
                blocksparse_tensors,
                tile_scheduler=tile_scheduler,
            )
            # Dealloc the tensor memory buffer
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Epilogue
        # ///////////////////////////////////////////////////////////////////////////////
        if const_expr(not self.use_correction_warps_for_epi):
            if warp_idx >= self.epilogue_warp_ids[0] and warp_idx <= self.epilogue_warp_ids[-1]:
                cute.arch.setmaxregister_decrease(self.num_regs_other)
                self.epilogue_s2g(
                    mO,
                    sO,
                    gmem_tiled_copy_O,
                    tma_atom_O,
                    pipeline_o_epi,
                    block_info,
                    num_splits,
                    SeqlenInfoCls,
                    mma_tile_coord_v,
                    blocksparse_tensors=blocksparse_tensors,
                    tile_scheduler=tile_scheduler,
                )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax
        # ///////////////////////////////////////////////////////////////////////////////
        if (
            (const_expr(self.q_stage == 2) and warp_idx <= self.softmax1_warp_ids[-1]) or
            (const_expr(self.q_stage == 1) and warp_idx <= self.softmax0_warp_ids[-1])
        ):
            # increase register after decreasing
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            # sync with mma warp before retrieving tmem ptr
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            softmax_loop = partial(
                self.softmax_loop,
                softmax_scale_log2=softmax_scale_log2,
                softmax_scale=softmax_scale,
                descale_tensors=descale_tensors,
                thr_mma_qk=thr_mma_qk,
                sScale=sScale,
                mLSE=mLSE,
                pipeline_s_p_o=pipeline_s_p_o,
                pipeline_p_lastsplit=pipeline_p_lastsplit,
                pipeline_sm_stats=pipeline_sm_stats,
                sm_stats_barrier=sm_stats_barrier,
                pipeline_s0_s1_sequence=pipeline_s0_s1_sequence,
                learnable_sink=learnable_sink,
                block_info=block_info,
                num_splits=num_splits,
                SeqlenInfoCls=SeqlenInfoCls,
                AttentionMaskCls=AttentionMaskCls,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods,
                head_divmod=head_divmod,
                blocksparse_tensors=blocksparse_tensors,
                tile_scheduler=tile_scheduler,
            )

            if const_expr(not self.s0_s1_barrier):
                stage = Int32(0 if const_expr(self.q_stage == 1) or warp_idx < self.softmax1_warp_ids[0] else 1)
                softmax_loop(stage=stage, tStS=tStS)
            else:
                # If there's s0_s1_barrier, it's faster to have 2 WGs having different code
                if warp_idx < self.softmax1_warp_ids[0]:
                    softmax_loop(stage=0, tStS=tStS)
                if warp_idx < self.correction_warp_ids[0] and warp_idx >= self.softmax1_warp_ids[0]:
                    softmax_loop(stage=1, tStS=tStS)

            tmem_alloc_barrier.arrive()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Correction
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_correction)
            # sync with mma warp before retrieving tmem ptr
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            self.correction_loop(
                thr_mma_qk,
                thr_mma_pv,
                tStS,
                tOtO,
                sScale,
                mO,
                mLSE,
                sO,
                pipeline_s_p_o,
                pipeline_o_acc,
                pipeline_sm_stats,
                sm_stats_barrier,
                pipeline_o_epi,
                learnable_sink,
                descale_tensors,
                gmem_tiled_copy_O,
                tma_atom_O,
                softmax_scale_log2,
                block_info,
                num_splits,
                SeqlenInfoCls,
                blocksparse_tensors,
                tile_scheduler=tile_scheduler,
            )
            tmem_alloc_barrier.arrive()

        return

    @cute.jit
    def load(
        self,
        thr_mma_qk: cute.ThrMma,
        thr_mma_pv: cute.ThrMma,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        gmem_tiled_copy_Q: Optional[cute.TiledCopy],
        pipeline_q: pipeline.PipelineAsync,
        pipeline_kv: pipeline.PipelineAsync,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors],
        tile_scheduler: TileSchedulerProtocol,
    ):
        num_load_threads = len(self.load_warp_ids) * cute.arch.WARP_SIZE
        tidx = cute.arch.thread_idx()[0] % num_load_threads
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        issue_kv_for_this_warp = (
            const_expr(not self.use_tma_KV or len(self.load_warp_ids) == 1) or
            warp_idx == self.load_warp_ids[0]
        )
        issue_q_for_this_warp = (
            const_expr(not self.use_tma_Q or len(self.load_warp_ids) == 1) or
            warp_idx == self.load_warp_ids[0]
        )
        q_producer_phase = Int32(1)
        kv_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.kv_stage
        )
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]

            head_idx_kv = (
                head_idx // self.qhead_per_kvhead if const_expr(not self.pack_gqa) else head_idx
            )
            if const_expr(mPageTable is None):
                if const_expr(not seqlen.has_cu_seqlens_k):
                    mK_cur, mV_cur = [t[None, None, head_idx_kv, batch_idx] for t in (mK, mV)]
                else:
                    mK_cur = cute.domain_offset((seqlen.offset_k, 0), mK[None, None, head_idx_kv])
                    mV_cur = cute.domain_offset((0, seqlen.offset_k), mV[None, None, head_idx_kv])
                gK = cute.local_tile(mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0))
                gV = cute.local_tile(mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None))
            else:
                # Need to keep batch coord None since we'll index into it with page idx
                mK_cur, mV_cur = [t[None, None, head_idx_kv, None] for t in (mK, mV)]
                gK = cute.local_tile(
                    mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0, None)
                )
                gV = cute.local_tile(
                    mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None, None)
                )
            tSgK = thr_mma_qk.partition_B(gK)
            tOgV = thr_mma_pv.partition_B(gV)
            if const_expr(self.use_tma_Q):
                tiler_gQ = ((self.mma_tiler_qk[0] * self.q_stage), self.head_dim_padded)
                gQ = cute.local_tile(mQ_cur, tiler_gQ, (m_block, 0))  # (128 * 2, 128)
                gQ = layout_utils.select(
                    cute.flat_divide(gQ, (self.mma_tiler_qk[0],)), mode=[0, 2, 1]
                )  # (128, 128, 2)
                tSgQ = thr_mma_qk.partition_A(gQ)
                load_Q_fn, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_Q, 0, cute.make_layout(1), tSgQ, sQ
                )
                load_Q = partial(self.load_Q, load_Q_fn, pipeline_q=pipeline_q, phase=q_producer_phase)
            else:
                assert gmem_tiled_copy_Q is not None
                load_Q = partial(
                    self.load_Q_non_tma,
                    mQ_cur,
                    sQ,
                    gmem_tiled_copy_Q,
                    pipeline_q,
                    tidx,
                    seqlen.seqlen_q,
                    m_block,
                    phase=q_producer_phase,
                )

            if const_expr(self.use_tma_KV):
                tKsK, tKgK = cpasync.tma_partition(
                    tma_atom_K,
                    0,  # no multicast
                    cute.make_layout(1),
                    cute.group_modes(sK, 0, 3),
                    cute.group_modes(tSgK, 0, 3),
                )
                tVsV, tVgV = cpasync.tma_partition(
                    tma_atom_V,
                    0,  # no multicast
                    cute.make_layout(1),
                    cute.group_modes(sV, 0, 3),
                    cute.group_modes(tOgV, 0, 3),
                )
                paged_kv_manager = None
            else:
                page_size = mK.shape[0]
                paged_kv_manager = PagedKVManager.create(
                    mPageTable,
                    mK,
                    mV,
                    FastDivmodDivisor(page_size),
                    batch_idx,
                    head_idx_kv,
                    tidx,
                    seqlen.seqlen_k,
                    0,  # leftpad_k
                    self.n_block_size,
                    self.head_dim_padded,
                    self.head_dim_v_padded,
                    num_load_threads,
                    mK.element_type,
                )
                tKsK, tKgK = None, None
                tVsV, tVgV = None, None

            load_K = partial(
                self.load_KV,
                tma_atom_K,
                tKgK,
                tKsK,
                paged_kv_manager,
                sK,
                pipeline_kv=pipeline_kv,
                K_or_V="K",
            )
            load_V = partial(
                self.load_KV,
                tma_atom_V,
                tVgV,
                tVsV,
                paged_kv_manager,
                sV,
                pipeline_kv=pipeline_kv,
                K_or_V="V",
            )

            if const_expr(not self.use_block_sparsity):
                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen, m_block, split_idx, num_splits
                )
                if const_expr(not self.is_split_kv) or n_block_min < n_block_max:
                    n_block_first = n_block_max - 1 if n_block_max > 0 else 0
                    page_idx = (
                        mPageTable[batch_idx, n_block_first]
                        if const_expr(mPageTable is not None and self.use_tma_KV)
                        else None
                    )
                    if const_expr(not self.use_tma_KV):
                        paged_kv_manager.load_page_table(n_block_first)
                    if issue_kv_for_this_warp:
                        load_K(block=n_block_max - 1, producer_state=kv_producer_state, page_idx=page_idx)  # K0
                    # load_K(block=n_block_max - 1, producer_state=kv_producer_state, page_idx=page_idx, extra_tx_count=self.tma_copy_bytes["Q"])  # K0
                    if issue_q_for_this_warp:
                        load_Q(block=0, stage=0)
                    if issue_kv_for_this_warp:
                        kv_producer_state.advance()
                    if const_expr(self.q_stage == 2) and issue_q_for_this_warp:
                        load_Q(block=1, stage=1)
                    q_producer_phase ^= 1
                    if issue_kv_for_this_warp:
                        load_V(block=n_block_max - 1, producer_state=kv_producer_state, page_idx=page_idx)  # V0
                        kv_producer_state.advance()
                    for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                        n_block = n_block_max - 2 - i
                        page_idx = (
                            mPageTable[batch_idx, n_block]
                            if const_expr(mPageTable is not None and self.use_tma_KV)
                            else None
                        )
                        if const_expr(not self.use_tma_KV):
                            paged_kv_manager.load_page_table(n_block)
                    # if cute.arch.thread_idx()[0] % 32 == 0: cute.printf("n_block = {}, page_idx = {}", n_block, page_idx)
                        if issue_kv_for_this_warp:
                            load_K(block=n_block, producer_state=kv_producer_state, page_idx=page_idx)  # Ki
                            kv_producer_state.advance()
                            load_V(block=n_block, producer_state=kv_producer_state, page_idx=page_idx)  # Vi
                            kv_producer_state.advance()

            else:
                kv_producer_state, q_producer_phase = produce_block_sparse_loads_sm100(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    seqlen,
                    split_idx,
                    num_splits,
                    kv_producer_state,
                    load_Q,
                    load_K,
                    load_V,
                    pipeline_kv,
                    self.q_stage,
                    q_producer_phase,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                    self.q_subtile_factor,
                )


            work_tile = tile_scheduler.advance_to_next_work()
            # End of persistent scheduler loop

        if issue_kv_for_this_warp:
            pipeline_kv.producer_tail(kv_producer_state)
        # This is equivalent to pipeline_q.producer_tail for the TMA-Q producer warp.
        if issue_q_for_this_warp:
            pipeline_q.producer_acquire_w_index_phase(self.q_stage - 1, q_producer_phase)

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.ThrMma,
        tiled_mma_pv: cute.ThrMma,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tStS: cute.Tensor,
        tOtO: cute.Tensor,
        tOrP: cute.Tensor,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_kv: pipeline.PipelineAsync,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        is_leader_cta: Boolean,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors],
        tile_scheduler=None,
    ):
        tSrQ = tiled_mma_qk.make_fragment_A(sQ)
        tSrK = tiled_mma_qk.make_fragment_B(sK)
        tOrV = tiled_mma_pv.make_fragment_B(sV)
        if const_expr(self.q_stage == 2):
            tSrQs = (tSrQ[None, None, None, 0], tSrQ[None, None, None, 1])
        else:
            tSrQs = (tSrQ[None, None, None, 0],)

        qk_mma_op, pv_mma_op = tiled_mma_qk.op, tiled_mma_pv.op
        qk_mma_idesc, pv_mma_idesc = sm100_desc.mma_op_to_idesc(qk_mma_op), sm100_desc.mma_op_to_idesc(pv_mma_op)
        qk_mma_kind = sm100_utils._tcgen05_mma_kind(qk_mma_op)
        q_smem_base = sm100_desc.smem_desc_base_from_tensor(sQ, sm100_desc.Major.K)
        k_smem_base = sm100_desc.smem_desc_base_from_tensor(sK, sm100_desc.Major.K)
        v_smem_base = sm100_desc.smem_desc_base_from_tensor(sV, sm100_desc.Major.MN)
        q_smem_start = [sm100_desc.make_smem_desc_start_addr(sQ[None, None, None, stage].iterator) for stage in range(self.q_stage)]

        sm100_utils.declare_ptx_smem_desc(q_smem_start[self.q_stage - 1], q_smem_base, tSrQ[None, None, None, 0].layout, var_name_prefix="fa_fwd_q_smem_desc")
        sm100_utils.declare_ptx_idesc(qk_mma_op, var_name="fa_fwd_qk_mma_idesc")
        sm100_utils.declare_ptx_idesc(pv_mma_op, var_name="fa_fwd_pv_mma_idesc")

        sQ_stage_stride = (sQ.layout.stride[-1] * sQ.element_type.width // 8) >> 4
        if const_expr(self.q_stage == 1):
            sQ_stage_stride = 0
        gemm_Si = [
            partial(
                # sm100_utils.gemm_ptx_precomputed,
                # self.tmem_s_offset[stage],
                # smem_desc_start_a=q_smem_start[stage],
                # idesc=qk_mma_idesc,
                # smem_desc_base_a=q_smem_base,
                # smem_desc_base_b=k_smem_base,
                # tCrA_layout=tSrQ[None, None, None, 0].layout,
                sm100_utils.gemm_ptx_precomputed_varname,
                self.tmem_s_offset[stage],
                # idesc=qk_mma_idesc,
                smem_desc_base_b=k_smem_base,
                tCrB_layout=tSrK[None, None, None, 0].layout,
                smem_var_name_prefix="fa_fwd_q_smem_desc",
                idesc_var_name="fa_fwd_qk_mma_idesc",
                kind=qk_mma_kind,
                smem_offset=-sQ_stage_stride if stage == 0 else sQ_stage_stride,
                zero_init=True,
                cta_group=self.cta_group_size,
            )
            for stage in range(self.q_stage)
        ]
        # gemm_Si = [
        #     partial(
        #         sm100_utils.gemm,
        #         tiled_mma_qk,
        #         tStS[None, None, None, stage],
        #         tCrA=tSrQ[None, None, None, stage],
        #         zero_init=True,
        #     )
        #     for stage in range(self.q_stage)
        # ]
        gemm_Pi = [
            partial(
                # sm100_utils.gemm_ptx_precomputed,
                sm100_utils.gemm_ptx_partial,
                pv_mma_op,
                self.tmem_o_offset[stage],
                tOrP[None, None, None, stage],
                sA=None,
                split_arrive=self.split_P_arrive if self.split_P_arrive > 0 else None,
                # smem_desc_start_a=tOrP[None, None, None, stage].iterator.toint(),
                # smem_desc_start_a=self.tmem_p_offset[stage],
                # idesc=pv_mma_idesc,
                # smem_desc_base_a=None,
                # smem_desc_base_b=v_smem_base,
                # tCrA_layout=tOrP[None, None, None, 0].layout,
                # tCrB_layout=tOrV[None, None, None, 0].layout
                cta_group=self.cta_group_size,
            )
            for stage in range(self.q_stage)
        ]
        # gemm_Pi = [
        #     partial(
        #         sm100_utils.gemm, tOtO[None, None, None, stage], tCrA=tOrP[None, None, None, stage]
        #     )
        #     for stage in range(self.q_stage)
        # ]

        mma_q_consumer_phase = Int32(0)
        mma_kv_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.kv_stage
        )
        P_full_O_rescaled_phase = Int32(0)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)

            block_iter_count = Int32(0)
            process_tile = False

            if const_expr(self.use_block_sparsity):
                block_iter_count = get_total_block_count(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    split_idx,
                    num_splits,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                    self.q_subtile_factor,
                    seqlen_info=seqlen,
                )
                process_tile = block_iter_count > Int32(0)
            else:
                n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block, split_idx, num_splits)
                block_iter_count = n_block_max - n_block_min
                if const_expr(not self.is_split_kv):
                    process_tile = True
                else:
                    process_tile = n_block_min < n_block_max

            if process_tile and is_leader_cta:
                for stage in cutlass.range_constexpr(self.q_stage):
                    # GEMM_QK00 (Q0 * K0 -> S0) or GEMM_QK01 (Q1 * K0 -> S1)
                    # 1. wait for Q0 / Q1
                    pipeline_q.consumer_wait_w_index_phase(stage, mma_q_consumer_phase)
                    # 2. wait for K0
                    if const_expr(stage == 0):
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    Ki_index, Ki_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                    tSrKi = tSrK[None, None, None, Ki_index]
                    # We don't need to acquire empty S0 / S1.
                    # For the first iteration, we don't need to wait as we're guaranteed S0 / S1
                    # are empty. For subsequent iterations, the wait happened at the end
                    # of the while loop.
                    # 3. gemm
                    # sm100_utils.gemm(tiled_mma_qk, tStS[None, None, None, stage], tSrQ[None, None, None, stage], tSrKi, zero_init=True)
                    sK_cur = sK[None, None, None, Ki_index]
                    if const_expr(self.uneven_kv_smem):
                        sK_cur = self.offset_kv_smem(sK_cur, Ki_index, Ki_phase)
                    # gemm_Si[stage](tCrB=tSrKi, sB=sK_cur)
                    gemm_Si[stage](
                        smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sK_cur.iterator)
                    )
                    # gemm_Si[stage](tCrB=tSrKi)
                    # 4. release S0 / S1
                    pipeline_s_p_o.producer_commit_w_index(stage)
                mma_q_consumer_phase ^= 1
                # 5. release K0
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()
                # End of GEMM (Q1 * K0 -> S1)
                # Note: Q0 & Q1 are still needed in the seqlen_kv loop
                # so we need to release them after the seqlen_kv loop

                # O hasn't been accumulated yet, its first MMA calculation doesn't need to accumulate
                block_loop_count = block_iter_count - 1
                O_should_accumulate = False
                for i in cutlass.range(block_loop_count, unroll=1):
                    # GEMM_PV00 (P0 * V0 -> O0_partial), O0 needs to be accumulated in the seqlen_kv loop
                    # 1. wait for V0
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    mma_kv_release_state = mma_kv_consumer_state.clone()
                    Vi_index, Vi_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                    tOrVi = tOrV[None, None, None, Vi_index]
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # 2. acquire corrected O0/O1_partial and P0 / P1
                        # For the first iteration in this work tile, waiting for O0/O1_partial
                        # means that the correction warps has finished reading tO during
                        # the last iteration of the previous work tile.
                        pipeline_s_p_o.producer_acquire_w_index_phase(stage, P_full_O_rescaled_phase)
                        # 3. gemm
                        # sm100_utils.gemm(tiled_mma_pv, tOtO0, tOrP0, tOrVi, zero_init=True)
                        # gemm_Pi[stage](tCrB=tOrVi, sB=sV[None, None, None, Vi_index], zero_init=not O_should_accumulate)
                        sV_cur = sV[None, None, None, Vi_index]
                        if const_expr(self.uneven_kv_smem):
                            sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)
                        gemm_Pi[stage](
                            tCrB=tOrVi,
                            sB=sV_cur,
                            # smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sV_cur.iterator),
                            zero_init=not O_should_accumulate,
                            mbar_ptr=pipeline_p_lastsplit.sync_object_full.get_barrier(stage) if self.split_P_arrive > 0 else None,
                            mbar_phase=P_full_O_rescaled_phase,
                        )
                        # Don't need to signal O_full to the correction warps since the
                        # correction warps wait for the softmax warps anyway. By the time the softmax
                        # warps finished, S_i for the next iteration must have been done, so O_i-1
                        # must have been done as well.
                        # pipeline_o_acc.producer_commit_w_index(stage)
                        # 4. release V(i-1)
                        if const_expr(stage == self.q_stage - 1):
                            pipeline_kv.consumer_release(mma_kv_release_state)
                            mma_kv_release_state.advance()
                        # End of GEMM_PV00 (P0 * V0 -> O0_partial)

                        # GEMM_QK0i (Q0 * Ki -> S0)
                        # 1. wait for Ki
                        if const_expr(stage == 0):
                            mma_kv_consumer_state.advance()
                            pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        Ki_index, Ki_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                        # 2. gemm
                        # Don't need to wait for the softmax warp to have finished reading the previous
                        # Si, since this gemm is scheduled after the PV gemm, which guaranteed that Si
                        # has been read and Pi has been written.
                        # sm100_utils.gemm(tiled_mma_qk, tStS[None, None, None, stage], tSrQ[None, None, None, stage], tSrK[None, None, None, Ki_index], zero_init=True)
                        sK_cur = sK[None, None, None, Ki_index]
                        if const_expr(self.uneven_kv_smem):
                            sK_cur = self.offset_kv_smem(sK_cur, Ki_index, Ki_phase)
                        # gemm_Si[stage](tCrB=tSrK[None, None, None, Ki_index], sB=sK_cur)
                        gemm_Si[stage](
                            smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sK_cur.iterator)
                        )
                        # gemm_Si[stage](tCrB=tSrK[None, None, None, Ki_index])
                        # 3. release S0 / S1
                        pipeline_s_p_o.producer_commit_w_index(stage)
                        # End of GEMM_QK0i (Q0 * Ki -> S0)
                    # 4. release Ki
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()
                    P_full_O_rescaled_phase ^= 1
                    O_should_accumulate = True
                # End of seqlen_kv loop

                # release Q0 & Q1
                for stage in cutlass.range(self.q_stage):
                    pipeline_q.consumer_release_w_index(stage)

                # GEMM_PV00 (P0 * V0 -> O0_partial), O0 needs to be accumulated in the seqlen_kv loop
                # 1. wait for V0
                pipeline_kv.consumer_wait(mma_kv_consumer_state)
                Vi_index, Vi_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                tOrVi = tOrV[None, None, None, Vi_index]
                for stage in cutlass.range_constexpr(self.q_stage):
                    # 2. acquire corrected Oi_partial and Pi
                    pipeline_s_p_o.producer_acquire_w_index_phase(stage, P_full_O_rescaled_phase)
                    # 3. gemm
                    # sm100_utils.gemm(tiled_mma_pv, tOtO0, tOrP0, tOrVi, zero_init=True)
                    # gemm_Pi[stage](tCrB=tOrVi, sB=sV[None, None, None, Vi_index], zero_init=not O_should_accumulate)
                    sV_cur = sV[None, None, None, Vi_index]
                    if const_expr(self.uneven_kv_smem):
                        sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)
                    gemm_Pi[stage](
                        tCrB=tOrVi,
                        sB=sV_cur,
                        # smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sV_cur.iterator),
                        zero_init=not O_should_accumulate,
                        mbar_ptr=pipeline_p_lastsplit.sync_object_full.get_barrier(stage) if self.split_P_arrive > 0 else None,
                        mbar_phase=P_full_O_rescaled_phase,
                    )
                    # 4. release accumulated O0_partial
                    # We do need O_full here since for the last tile, by the time the softmax warp
                    # has signaled to the correction warps, the softmax warp has just finished
                    # computing the row sum of the current tile. It does not guarantee that the 1st
                    # tile of the next work tile has been computed yet.
                    pipeline_o_acc.producer_commit_w_index(stage)
                    # End of GEMM_PV00 (P0 * V0 -> O0_partial)
                P_full_O_rescaled_phase ^= 1
                # 5. release Vi_end
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()
                # End of GEMM_PV1(i_end) (P1 * Vi_end -> O1)

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()
        # End of persistent scheduler loop

        # We don't need pipeline_s_p_o.producer_tail() since there's no dangling mbarrier at the end
        # pipeline_s_p_o.producer_acquire_w_index_phase(self.q_stage - 1, P_full_O_rescaled_phase)
        # We don't need pipeline_o_acc.producer_tail() since we don't call
        # pipeline_o_acc.producer_acquire() inside the loop.

    # for both softmax0 and softmax1 warp group
    @cute.jit
    def _kv_head_idx(self, head_idx: Int32) -> Int32:
        """Map query-head tile index -> KV-head index (FA3 descale semantics)."""
        if cutlass.const_expr(self.pack_gqa):
            return head_idx
        return head_idx // self.qhead_per_kvhead

    @cute.jit
    def _load_effective_descales(
        self,
        descale_tensors: Optional[DescaleTensors],
        batch_idx: Int32,
        kv_head_idx: Int32,
    ) -> Tuple[Float32, Float32]:
        """Load effective QK and V descales, defaulting unspecified tensors to identity."""
        qk_descale = Float32(1.0)
        v_descale = Float32(1.0)
        if cutlass.const_expr(descale_tensors is not None):
            if cutlass.const_expr(descale_tensors.q_descale is not None):
                qk_descale = qk_descale * Float32(descale_tensors.q_descale[batch_idx, kv_head_idx])
            if cutlass.const_expr(descale_tensors.k_descale is not None):
                qk_descale = qk_descale * Float32(descale_tensors.k_descale[batch_idx, kv_head_idx])
            if cutlass.const_expr(descale_tensors.v_descale is not None):
                v_descale = Float32(descale_tensors.v_descale[batch_idx, kv_head_idx])
        return qk_descale, v_descale

    @cute.jit
    def softmax_loop(
        self,
        stage: int | Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Float32 | None,
        descale_tensors: Optional[DescaleTensors],
        thr_mma_qk: cute.ThrMma,
        tStS: cute.Tensor,  # ((TILE_M, TILE_N), 1, 1, q_stage)
        sScale: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        pipeline_s0_s1_sequence: Optional[pipeline.PipelineAsync],
        learnable_sink: Optional[cute.Tensor],
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        head_divmod=None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        tile_scheduler=None,
    ):
        """Compute softmax on attention scores from QK matrix multiplication.

        This method handles the softmax computation for either the first or second half of the
        attention matrix, depending on the 'stage' parameter. It calculates row-wise maximum
        and sum values needed for stable softmax computation, applies optional masking, and
        transforms raw attention scores into probability distributions.

        The implementation uses specialized memory access patterns and efficient math operations
        for computing exp(x) using exp2 functions. It also coordinates pipeline
        synchronization between MMA, correction, and sequence processing stages.
        """
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE
            # * (len(self.softmax0_warp_ids) if stage == 0 else len(self.softmax1_warp_ids)
            * (len(self.softmax0_warp_ids))
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        aux_tensors = aux_data.tensors

        cta_qk_tiler = (self.mma_tiler_qk[0] // thr_mma_qk.thr_id.shape, self.mma_tiler_qk[1])
        tSAcc = tStS[(None, None), 0, 0, stage]  # (128, 128)
        tStScale = cute.composition(tSAcc, cute.make_layout((self.m_block_size, 1)))
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScS = tScS[(None, None), 0, 0]  # (128, 128)
        tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))

        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        tStP_layout = cute.composition(
            tSAcc.layout, cute.make_layout((self.m_block_size, tilePlikeFP32))
        )
        tStP = cute.make_tensor(tSAcc.iterator + self.tmem_s_to_p_offset, tStP_layout)

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.qk_acc_dtype
        )
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tSAcc).get_slice(tidx)
        tStS_t2r = thr_tmem_load.partition_S(tSAcc)  # (((32,32),1),1,4)

        tmem_store_scale_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(1)), Float32
        )
        thr_tmem_store_scale = tcgen05.make_tmem_copy(tmem_store_scale_atom, tStScale).get_slice(
            tidx
        )
        tStScale_r2t = thr_tmem_store_scale.partition_D(tStScale)
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(
                tcgen05.copy.Repetition(8 if const_expr(self.q_dtype.width == 8) else 16)
            ),
            Float32,
        )
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStP).get_slice(tidx)
        tStP_r2t = thr_tmem_store.partition_D(tStP)  # (((16,32),1),1,4)

        mma_si_consumer_phase = Int32(0)
        sm_stats_producer_phase = Int32(1)
        s0_s1_sequence_phase = Int32(1 if stage == 0 else 0)

        # self.warp_scheduler_barrier_init()

        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            kv_head_idx = self._kv_head_idx(head_idx)
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block, split_idx, num_splits)

            mask = AttentionMaskCls(seqlen)
            shared_mask_kwargs = dict(
                m_block=(self.q_stage * m_block + stage) * self.cta_group_size,
                thr_mma=thr_mma_qk,
                thr_tmem_load=thr_tmem_load,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
                batch_idx=batch_idx,
                head_idx=head_idx,
                aux_data=aux_data,
                vec_size=self.mask_vec_size,
            )

            # Recompute fastdiv_mods if necessary
            recompute_fastdiv_mods_q = cutlass.const_expr(
                aux_tensors is not None and (seqlen.has_cu_seqlens_q or seqlen.has_seqused_q)
            )
            recompute_fastdiv_mods_k = cutlass.const_expr(
                aux_tensors is not None and (seqlen.has_cu_seqlens_k or seqlen.has_seqused_k)
            )

            if cutlass.const_expr(fastdiv_mods is not None):
                seqlen_q_divmod, seqlen_k_divmod = fastdiv_mods
                fastdiv_mods = (
                    seqlen_q_divmod
                    if not recompute_fastdiv_mods_q
                    else FastDivmodDivisor(seqlen.seqlen_q),
                    seqlen_k_divmod
                    if not recompute_fastdiv_mods_k
                    else FastDivmodDivisor(seqlen.seqlen_k),
                )

            mask_mod = self.mask_mod if const_expr(self.mask_mod is not None) else None
            mask_fn = partial(
                mask.apply_mask_sm100,
                mask_mod=mask_mod,
                fastdiv_mods=fastdiv_mods,
                head_divmod=head_divmod,
                **shared_mask_kwargs,
            )
            if const_expr(self.use_block_sparsity):
                #  Full blocks dont need mask_mod
                mask_fn_none = partial(
                    mask.apply_mask_sm100,
                    mask_mod=None,
                    fastdiv_mods=fastdiv_mods,
                    head_divmod=head_divmod,
                    **shared_mask_kwargs,
                )
            else:
                mask_fn_none = None

            qk_descale, _ = self._load_effective_descales(descale_tensors, batch_idx, kv_head_idx)

            max_offset = 8 if cutlass.const_expr(self.q_dtype.width == 8) else 0
            if const_expr(self.score_mod is None):
                softmax_scale_log2_eff = softmax_scale_log2 * qk_descale
                softmax_scale_eff = None
            else:
                softmax_scale_log2_eff = softmax_scale_log2
                softmax_scale_eff = softmax_scale * qk_descale

            rescale_threshold = (
                8.0 if const_expr(self.q_dtype.width == 16) else
                4.0 if const_expr(self.q_dtype.width == 8) else
                0.0
            )
            softmax = SoftmaxSm100.create(
                softmax_scale_log2_eff,
                rescale_threshold=rescale_threshold,
                softmax_scale=softmax_scale_eff,
                max_offset=max_offset,
            )
            softmax.reset()

            if const_expr(self.use_block_sparsity):
                tile_block_count = get_total_block_count(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    split_idx,
                    num_splits,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                    self.q_subtile_factor,
                    seqlen_info=seqlen,
                )
                has_work = tile_block_count > Int32(0)
            else:
                tile_block_count = n_block_max - n_block_min
                has_work = const_expr(not self.is_split_kv) or tile_block_count > Int32(0)

            softmax_step = partial(
                self.softmax_step,
                softmax=softmax,
                thr_mma_qk=thr_mma_qk,
                pipeline_s_p_o=pipeline_s_p_o,
                pipeline_p_lastsplit=pipeline_p_lastsplit,
                pipeline_sm_stats=pipeline_sm_stats,
                sm_stats_barrier=sm_stats_barrier,
                pipeline_s0_s1_sequence=pipeline_s0_s1_sequence,
                thr_tmem_load=thr_tmem_load,
                thr_tmem_store=thr_tmem_store,
                thr_tmem_store_scale=thr_tmem_store_scale,
                tStS_t2r=tStS_t2r,
                tStScale_r2t=tStScale_r2t,
                tStP_r2t=tStP_r2t,
                sScale=sScale,
                stage=stage,
                batch_idx=batch_idx,
                head_idx=head_idx,
                m_block=(self.q_stage * m_block + stage) * self.cta_group_size,
                seqlen=seqlen,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods,
                head_divmod=head_divmod,
            )

            if const_expr(self.use_block_sparsity) or has_work:
                pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)
                sm_stats_producer_phase ^= 1

            # Block sparse or dense iteration
            if const_expr(self.use_block_sparsity):
                # When aux_tensors exist, Q indices beyond seqlen_q must be wrapped to avoid
                # OOB aux_tensor access. Only edge tiles (where m_tile_end > seqlen_q) need this.
                if const_expr(aux_tensors is not None):
                    m_tile_end = ((self.q_stage * m_block + stage + 1) * self.cta_group_size) * self.m_block_size
                    check_m_boundary = m_tile_end > seqlen.seqlen_q
                else:
                    check_m_boundary = False
                (
                    mma_si_consumer_phase,
                    sm_stats_producer_phase,
                    s0_s1_sequence_phase,
                    empty_tile,
                ) = softmax_block_sparse_sm100(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    seqlen,
                    split_idx,
                    num_splits,
                    softmax_step,
                    mask_fn,
                    mask_fn_none,
                    mma_si_consumer_phase,
                    sm_stats_producer_phase,
                    s0_s1_sequence_phase,
                    pipeline_sm_stats,
                    sm_stats_barrier,
                    self.q_stage,
                    Int32(stage),
                    check_m_boundary,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                    self.q_subtile_factor,
                )
                if not empty_tile:
                    sScale[tidx + stage * self.m_block_size] = softmax.row_sum[0]
                    if const_expr(mLSE is not None or learnable_sink is not None):
                        sScale[
                            tidx + stage * self.m_block_size + self.q_stage * self.m_block_size
                        ] = softmax.row_max[0]
                    # if tidx == 0:
                    #     cute.printf("softmax row sum stage %d: %f, row_max = %f\n", stage, softmax.row_sum[0], softmax.row_max[0])
                    # See block_sparse_utils.py NOTE [SM100 block-sparse empty tiles: mbarrier contract].
                    # pipeline_sm_stats.producer_commit_w_index(stage)
                    sm_stats_barrier.arrive_w_index(index=stage * 4 + warp_idx)
                    # if tidx == 0: cute.printf("softmax row sum stage %d: %f\n", stage, softmax.row_sum[0])
            else:
                if const_expr(not self.is_split_kv) or tile_block_count > Int32(0):
                    mma_si_consumer_phase, sm_stats_producer_phase, s0_s1_sequence_phase = softmax_step(
                        mma_si_consumer_phase,
                        sm_stats_producer_phase,
                        s0_s1_sequence_phase,
                        n_block_max - 1,
                        is_first=True,
                        mask_fn=partial(mask_fn, mask_seqlen=True),
                    )
                    n_block_max -= 1
                    # Next couple of iterations with causal masking
                    if const_expr(self.is_causal or self.is_local):
                        n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
                            seqlen, m_block, n_block_min
                        )
                        for n_tile in cutlass.range(n_block_max - n_block_min_causal_local_mask, unroll=1):
                            n_block = n_block_max - 1 - n_tile
                            mma_si_consumer_phase, sm_stats_producer_phase, s0_s1_sequence_phase = (
                                softmax_step(
                                    mma_si_consumer_phase,
                                    sm_stats_producer_phase,
                                    s0_s1_sequence_phase,
                                    n_block,
                                    mask_fn=partial(mask_fn, mask_seqlen=False),
                                )
                            )
                        n_block_max = cutlass.min(n_block_max, n_block_min_causal_local_mask)
                    # The remaining iterations have no masking (but may still need mask_mod)
                    n_block_min_before_local_mask = block_info.get_n_block_min_before_local_mask(
                        seqlen, m_block, n_block_min
                    )
                    for n_tile in cutlass.range(n_block_max - n_block_min_before_local_mask, unroll=1):
                        n_block = n_block_max - n_tile - 1
                        if const_expr(self.mask_mod is not None):
                            mma_si_consumer_phase, sm_stats_producer_phase, s0_s1_sequence_phase = softmax_step(
                                mma_si_consumer_phase, sm_stats_producer_phase, s0_s1_sequence_phase, n_block,
                                mask_fn=partial(mask_fn, mask_seqlen=False),
                            )
                        else:
                            mma_si_consumer_phase, sm_stats_producer_phase, s0_s1_sequence_phase = softmax_step(
                                mma_si_consumer_phase, sm_stats_producer_phase, s0_s1_sequence_phase, n_block,
                            )
                    # Separate iterations with local masking on the left
                    if const_expr(self.is_local and block_info.window_size_left is not None):
                        n_block_max = cutlass.min(n_block_max, n_block_min_before_local_mask)
                        for n_tile in cutlass.range(0, n_block_max - n_block_min, unroll=1):
                            n_block = n_block_max - 1 - n_tile
                            mma_si_consumer_phase, sm_stats_producer_phase, s0_s1_sequence_phase = (
                                softmax_step(
                                    mma_si_consumer_phase,
                                    sm_stats_producer_phase,
                                    s0_s1_sequence_phase,
                                    n_block,
                                    mask_fn=partial(mask_fn, mask_seqlen=False),
                                )
                            )
                            # Now that we no longer already have the 1st iteration, need mask_seqlen=True here

                    # Dense path always writes scale / signals
                    sScale[tidx + stage * self.m_block_size] = softmax.row_sum[0]
                    if const_expr(mLSE is not None or learnable_sink is not None):
                        sScale[
                            tidx + stage * self.m_block_size + self.q_stage * self.m_block_size
                        ] = softmax.row_max[0]
                    # pipeline_sm_stats.producer_commit_w_index(stage)
                    sm_stats_barrier.arrive_w_index(index=stage * 4 + warp_idx)

            # # Write LSE to gmem
            # if const_expr(mLSE is not None):
            #     acc_O_mn_row_is_zero_or_nan = softmax.row_sum[0] == 0.0 or softmax.row_sum[0] != softmax.row_sum[0]
            #     scale = (
            #         cute.arch.rcp_approx(softmax.row_sum[0] if not acc_O_mn_row_is_zero_or_nan else 1.0)
            #     )
            #     LN2 = math.log(2.0)
            #     lse = (
            #         (softmax.row_max[0] * softmax.scale_log2 + cute.math.log2(softmax.row_sum[0], fastmath=True)) * LN2
            #         if not acc_O_mn_row_is_zero_or_nan else -Float32.inf
            #     )
            #     if const_expr(not seqlen.has_cu_seqlens_q):
            #         mLSE_cur = mLSE[None, head_idx, batch_idx]
            #     else:
            #         mLSE_cur = cute.domain_offset((seqlen.offset_q,), mLSE[None, head_idx])
            #     gLSE = cute.local_tile(mLSE_cur, (self.m_block_size,), (m_block * 2 + stage,))
            #     if tidx < seqlen.seqlen_q - (m_block * 2 + stage) * self.m_block_size:
            #         gLSE[tidx] = lse

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()
        # End of persistent scheduler loop

        # This is equivalent to pipeline_sm_stats.producer_tail
        pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)
        # This is equivalent to pipeline_s0_s1.producer_tail
        if const_expr(self.s0_s1_barrier):
            if stage == 0:
                pipeline_s0_s1_sequence.sync_object_full.wait(stage, s0_s1_sequence_phase)

    @cute.jit
    def softmax_step(
        self,
        mma_si_consumer_phase: Int32,
        sm_stats_producer_phase: Int32,
        s0_s1_sequence_phase: Int32,
        n_block: Int32,
        softmax: SoftmaxSm100,
        thr_mma_qk: cute.ThrMma,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        pipeline_s0_s1_sequence: Optional[pipeline.PipelineAsync],
        thr_tmem_load: cute.CopyAtom,
        thr_tmem_store: cute.CopyAtom,
        thr_tmem_store_scale: cute.CopyAtom,
        tStS_t2r: cute.Tensor,
        tStScale_r2t: cute.Tensor,
        tStP_r2t: cute.Tensor,
        sScale: cute.Tensor,
        stage: int | Int32,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        head_divmod=None,
        mask_fn: Optional[Callable] = None,
        is_first: bool = False,
    ) -> Tuple[cute.Int32, cute.Int32, cute.Int32]:
        """Perform a single step of the softmax computation on a block of attention scores.

        This method processes one block of the attention matrix, computing numerically stable
        softmax by first finding the row maximum, subtracting it from all elements, applying
        exponential function, and then normalizing by the sum of exponentials. It also handles
        optional masking of attention scores.

        The method involves several key operations:
        1. Loading attention scores from tensor memory
        2. Applying optional masking based on position
        3. Computing row-wise maximum values for numerical stability
        4. Transforming scores using exp2(x*scale - max*scale)
        5. Computing row sums for normalization
        6. Coordinating pipeline synchronization between different processing stages
        """
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScS = tScS[(None, None), 0, 0]  # (128, 128)
        # tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))
        cta_qk_tiler = (self.mma_tiler_qk[0] // thr_mma_qk.thr_id.shape, self.mma_tiler_qk[1])
        tScS_shape = cta_qk_tiler  # (128, 128)
        tScP_shape = (tScS_shape[0], tilePlikeFP32)  # (128, 64)

        # Wait for Si
        pipeline_s_p_o.consumer_wait_w_index_phase(stage, mma_si_consumer_phase)
        tSrS_t2r = cute.make_rmem_tensor(thr_tmem_load.partition_D(tScS).shape, self.qk_acc_dtype)
        cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)
        # tSrS_t2r = copy_utils.load_t2r(thr_tmem_load, tScS_shape, tStS_t2r)
        if cutlass.const_expr(self.score_mod is not None):
            self.apply_score_mod(
                tSrS_t2r,
                thr_tmem_load,
                thr_mma_qk,
                batch_idx,
                head_idx,
                m_block,
                n_block,
                softmax,
                seqlen,
                aux_data,
                fastdiv_mods,
                head_divmod,
            )

        if const_expr(mask_fn is not None):
            mask_fn(tSrS_t2r, n_block=n_block)
        row_max, acc_scale = softmax.update_row_max(tSrS_t2r.load(), is_first)

        if const_expr(not is_first):
            # tSrScale_r2t = cute.make_rmem_tensor(thr_tmem_store_scale.partition_S(tScScale).shape, Float32)
            # tSrScale_r2t[0] = acc_scale
            # cute.copy(thr_tmem_store_scale, tSrScale_r2t, tStScale_r2t)
            # cute.arch.fence_view_async_tmem_store()
            thread_idx = thr_tmem_load.thr_idx
            sScale[thread_idx + stage * self.m_block_size] = acc_scale
            # if thread_idx == 0: cute.printf("softmax acc_scale stage %d: %f, row_max = %f\n", stage, acc_scale, row_max)
        # Notify correction wg that row_max is ready
        # pipeline_sm_stats.producer_commit_w_index(stage)
        sm_stats_barrier.arrive_w_index(index=stage * 4 + warp_idx)

        # if thread_idx == 0 and stage == 0: cute.print_tensor(tSrS_t2r)
        softmax.scale_subtract_rowmax(tSrS_t2r, row_max)
        # Sequence barrier wait
        if const_expr(self.s0_s1_barrier):
            pipeline_s0_s1_sequence.sync_object_full.wait(stage, s0_s1_sequence_phase)
        tSrP_r2t_f32 = cute.make_rmem_tensor(
            thr_tmem_store.partition_S(cute.make_identity_tensor(tScP_shape)).shape, Float32
        )
        tSrP_r2t = cute.make_tensor(
            cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.q_dtype), tSrS_t2r.layout
        )
        # softmax.scale_apply_exp2_convert(tSrS_t2r, row_max, tSrP_r2t)
        softmax.apply_exp2_convert(
            tSrS_t2r,
            tSrP_r2t,
            ex2_emu_freq=self.ex2_emu_freq,
            ex2_emu_start_frg=self.ex2_emu_start_frg,
        )
        # Sequence barrier arrive
        if const_expr(self.s0_s1_barrier):
            pipeline_s0_s1_sequence.sync_object_full.arrive(1 - stage, dst=None)
        # print(tSrP_r2t_f32, tStP_r2t)
        # cute.copy(thr_tmem_store, tSrP_r2t_f32, tStP_r2t)
        for i in cutlass.range_constexpr(cute.size(tStP_r2t.shape[2])):
            cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])
            if const_expr(self.split_P_arrive > 0):
                split_P_arrive_idx = cute.size(tStP_r2t.shape[2]) * self.split_P_arrive // self.n_block_size
                if const_expr(i + 1 == split_P_arrive_idx):
                    # Notify mma warp that the 1st half of P is ready
                    cute.arch.fence_view_async_tmem_store()
                    pipeline_s_p_o.consumer_release_w_index(stage)
        # Notify mma warp that the 2nd half of P is ready
        cute.arch.fence_view_async_tmem_store()
        if const_expr(self.split_P_arrive > 0):
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                pipeline_p_lastsplit.producer_commit_w_index(stage)
        else:
            pipeline_s_p_o.consumer_release_w_index(stage)
        pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)
        softmax.update_row_sum(tSrS_t2r.load(), acc_scale, is_first)
        # acc_scale = cute.math.exp2(acc_scale_, fastmath=True)
        return mma_si_consumer_phase ^ 1, sm_stats_producer_phase ^ 1, s0_s1_sequence_phase ^ 1

    @cute.jit
    def correction_loop(
        self,
        thr_mma_qk: cute.ThrMma,
        thr_mma_pv: cute.ThrMma,
        tStS: cute.Tensor,
        tOtO: cute.Tensor,
        sScale: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        sO: cute.Tensor,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        pipeline_o_epi: pipeline.PipelineAsync,
        learnable_sink: Optional[cute.Tensor],
        descale_tensors: Optional[DescaleTensors],
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: cute.CopyAtom,
        softmax_scale_log2: Float32,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        tile_scheduler=None,
    ):
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.correction_warp_ids))
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        mma_tile_coord_v = thr_mma_qk.thr_idx

        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tStScale_layout = cute.composition(tStS.layout, cute.make_layout((self.m_block_size, 1)))
        tStScales = tuple(
            cute.make_tensor(tStS.iterator + self.tmem_vec_offset[stage], tStScale_layout)
            for stage in range(self.q_stage)
        )
        tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))
        tmem_load_v_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(1)), self.qk_acc_dtype
        )
        thr_tmem_load_vec = tcgen05.make_tmem_copy(tmem_load_v_atom, tStScales[0]).get_slice(tidx)

        tStScales_t2r = [thr_tmem_load_vec.partition_S(tStScales[stage]) for stage in range(self.q_stage)]
        tSrScale_t2r_shape = thr_tmem_load_vec.partition_D(tScScale).shape

        # First iter: no correction is required
        # Notify mma warp that O has been rescaled
        for stage in cutlass.range(self.q_stage):
            pipeline_s_p_o.consumer_release_w_index(stage)

        sm_stats_consumer_phase = Int32(0)
        o_corr_consumer_phase = Int32(0)
        corr_epi_producer_phase = Int32(1)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            kv_head_idx = self._kv_head_idx(head_idx)
            qk_descale, v_descale = self._load_effective_descales(descale_tensors, batch_idx, kv_head_idx)
            if const_expr(self.score_mod is None):
                softmax_scale_log2_eff = softmax_scale_log2 * qk_descale
            else:
                softmax_scale_log2_eff = softmax_scale_log2

            max_offset = Float32(8.0) if cutlass.const_expr(self.q_dtype.width == 8) else Float32(0.0)
            max_offset_scale = (
                Float32(256.0) if cutlass.const_expr(self.q_dtype.width == 8) else Float32(1.0)
            )
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block, split_idx, num_splits)

            if const_expr(self.is_split_kv):
                mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx, split_idx]
            else:
                mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx]
            gO = None
            if const_expr(self.use_tma_O or not self.pack_gqa):
                tiler_gO = ((self.mma_tiler_pv[0] * self.q_stage), self.head_dim_v_padded)
                gO = cute.local_tile(mO_cur, tiler_gO, (m_block, 0))  # (128 * 2, 128)
                gO = layout_utils.select(
                    cute.flat_divide(gO, (self.mma_tiler_pv[0],)), mode=[0, 2, 1]
                )  # (128, 128, 2)
                gO = cute.flat_divide(gO, (self.mma_tiler_pv[0] // self.cta_group_size,))[None, mma_tile_coord_v, None, None]

            # Default LSE to -inf for invalid split_idx tiles
            stats = [(0.0, -Float32.inf if const_expr(mLSE is not None or learnable_sink is not None) else None, True)] * self.q_stage

            if const_expr(self.use_block_sparsity):
                total_block_count = get_total_block_count(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    split_idx,
                    num_splits,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                    self.q_subtile_factor,
                    seqlen_info=seqlen,
                )
                has_work = total_block_count > Int32(0)
            else:
                total_block_count = n_block_max - n_block_min
                has_work = const_expr(not self.is_split_kv) or total_block_count > Int32(0)

            if has_work:
                # Ignore first signal from softmax as no correction is required
                # pipeline_sm_stats.consumer_wait_w_index_phase(0, sm_stats_consumer_phase)
                sm_stats_barrier.arrive_and_wait_w_index(index=0 * 4 + warp_idx)
                pipeline_sm_stats.consumer_release_w_index(0)
                if const_expr(self.q_stage == 2):
                    # pipeline_sm_stats.consumer_wait_w_index_phase(1, sm_stats_consumer_phase)
                    sm_stats_barrier.arrive_and_wait_w_index(index=1 * 4 + warp_idx)
                sm_stats_consumer_phase ^= 1

                tSrScale_t2r = cute.make_rmem_tensor(tSrScale_t2r_shape, Float32)
                for i in cutlass.range(total_block_count - 1, unroll=1):
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # wait for S0 / S1
                        # pipeline_sm_stats.consumer_wait_w_index_phase(stage, sm_stats_consumer_phase)
                        sm_stats_barrier.arrive_and_wait_w_index(index=stage * 4 + warp_idx)
                        # cute.copy(tiled_tmem_load_vec, tStScales_t2r[stage], tSrScale_t2r)
                        # cute.arch.fence_view_async_tmem_load()
                        # scale = tSrScale_t2r[0]
                        scale = sScale[tidx + stage * self.m_block_size]
                        should_rescale = cute.arch.vote_ballot_sync(scale < 1.0) != 0
                        # should_rescale = True
                        # if tidx == 0: cute.printf("Correction scale i = %d, for stage %d: %f, should_rescale = %d\n", i, stage, scale, should_rescale)
                        # Don't need O_full anymore, since by the time softmax has signaled the correction
                        # warps, S_i must have been done, so O_i-1 must have been done as well.
                        # pipeline_o_acc.consumer_wait_w_index_phase(stage, o_corr_consumer_phase)
                        if should_rescale:
                            self.correction_rescale(thr_mma_pv, tOtO[None, None, None, stage], tidx, scale)
                        # Notify mma warp that O has been rescaled
                        pipeline_s_p_o.consumer_release_w_index(stage)
                        pipeline_sm_stats.consumer_release_w_index(self.q_stage - 1 - stage)
                    sm_stats_consumer_phase ^= 1
                    # o_corr_consumer_phase ^= 1
                if const_expr(self.q_stage == 2):
                    pipeline_sm_stats.consumer_release_w_index(1)
                # End of seqlen_corr_loop_steps

                # Even in the case of self.overlap_sO_sQ, we can write to stage 0 of sO without
                # additional sync because the MMA in the top half must have been done.
                # Similarly we can write to stage 1 of sO without additional sync.
                learnable_sink_val = [None] * self.q_stage
                if const_expr(learnable_sink is not None):
                    if const_expr(not self.pack_gqa):
                        sink_val = Float32(learnable_sink[head_idx])
                        learnable_sink_val = [sink_val] * self.q_stage
                    else:  # Each thread might have a different sink value due to different q_head
                        for stage in cutlass.range_constexpr(self.q_stage):
                            q_head_idx = (
                                ((m_block * self.q_stage + stage) * self.cta_group_size + mma_tile_coord_v) * self.m_block_size + tidx
                            ) % self.qhead_per_kvhead + head_idx * self.qhead_per_kvhead
                            learnable_sink_val[stage] = Float32(learnable_sink[q_head_idx])
                for stage in cutlass.range_constexpr(self.q_stage):
                    # pipeline_sm_stats.consumer_wait_w_index_phase(stage, sm_stats_consumer_phase)
                    sm_stats_barrier.arrive_and_wait_w_index(index=stage * 4 + warp_idx)
                    # cute.copy(tiled_tmem_load_vec, tStScales_t2r[stage], tSrScale_t2r)
                    # cute.arch.fence_view_async_tmem_load()
                    # scale = tSrScale_t2r[0]
                    row_sum = sScale[tidx + stage * self.m_block_size]
                    if const_expr(mLSE is not None or learnable_sink is not None):
                        row_max = sScale[tidx + stage * self.m_block_size + self.q_stage * self.m_block_size]
                    else:
                        row_max = None
                    pipeline_sm_stats.consumer_release_w_index(stage)
                    if const_expr(learnable_sink is not None):
                        LOG2_E = math.log2(math.e)
                        sink_val = learnable_sink_val[stage]
                        if const_expr(not self.is_split_kv) or split_idx == 0:
                            if row_max == -Float32.inf:
                                # It's possible to have an empty row with splitKV.
                                row_max = sink_val * (LOG2_E / softmax_scale_log2_eff)
                                row_sum = max_offset_scale
                            else:
                                row_sum += cute.math.exp2(
                                    sink_val * LOG2_E - row_max * softmax_scale_log2_eff + max_offset, fastmath=True
                                )
                    acc_O_mn_row_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
                    stats[stage] = (row_sum, row_max, acc_O_mn_row_is_zero_or_nan)
                    scale = cute.arch.rcp_approx(row_sum if not acc_O_mn_row_is_zero_or_nan else 1.0)
                    scale = scale * v_descale
                    # Wait for the last O to be ready from the MMA warp
                    pipeline_o_acc.consumer_wait_w_index_phase(stage, o_corr_consumer_phase)
                    if const_expr(not self.use_correction_warps_for_epi):
                        pipeline_o_epi.producer_acquire_w_index_phase(stage, corr_epi_producer_phase)
                    gO_stage = gO[None, None, stage] if const_expr(gO is not None) else None
                    self.correction_epilogue(
                        thr_mma_pv,
                        tOtO[None, None, None, stage],
                        tidx,
                        stage,
                        m_block,
                        seqlen.seqlen_q,
                        scale,
                        sO[None, None, stage],
                        mO_cur,
                        gO_stage,
                        gmem_tiled_copy_O,
                    )
                    # Signal for the next work tile that O buffers in tmem are already read, so
                    # mma warp can write to them
                    pipeline_s_p_o.consumer_release_w_index(stage)
                    if const_expr(not self.use_correction_warps_for_epi):
                        pipeline_o_epi.producer_commit_w_index(stage)
                    # if tidx == 0: cute.printf("Correction final scale for stage %d: %f\n", stage, scale)

                o_corr_consumer_phase ^= 1
                sm_stats_consumer_phase ^= 1
                corr_epi_producer_phase ^= 1
            else:
                gmem_tiled_copy_O_for_empty_tile = None
                if const_expr(self.use_correction_warps_for_epi):
                    gmem_tiled_copy_O_for_empty_tile = gmem_tiled_copy_O
                if const_expr(self.use_block_sparsity):
                    (
                        sm_stats_consumer_phase,
                        o_corr_consumer_phase,
                        corr_epi_producer_phase,
                    ) = handle_block_sparse_empty_tile_correction_sm100(
                        tidx,
                        self.q_stage,
                        self.m_block_size,
                        self.qhead_per_kvhead,
                        self.pack_gqa,
                        self.is_split_kv,
                        learnable_sink,
                        mLSE,
                        seqlen,
                        m_block,
                        head_idx,
                        batch_idx,
                        split_idx,
                        sScale,
                        stats,
                        self.correction_epilogue,
                        thr_mma_pv,
                        tOtO,
                        sO,
                        pipeline_sm_stats,
                        sm_stats_barrier,
                        pipeline_o_epi,
                        sm_stats_consumer_phase,
                        o_corr_consumer_phase,
                        corr_epi_producer_phase,
                        softmax_scale_log2_eff,
                        max_offset,
                        max_offset_scale,
                        mO_cur,
                        gO,
                        gmem_tiled_copy_O_for_empty_tile,
                    )

            if const_expr(mLSE is not None):
                if const_expr(not seqlen.has_cu_seqlens_q):
                    if const_expr(self.is_split_kv):
                        mLSE_cur = mLSE[None, head_idx, batch_idx, split_idx]
                    else:
                        mLSE_cur = mLSE[None, head_idx, batch_idx]
                else:
                    offset = (
                        seqlen.offset_q if const_expr(not self.pack_gqa) else (0, seqlen.offset_q)
                    )
                    if const_expr(self.is_split_kv):
                        mLSE_cur = cute.domain_offset((offset,), mLSE[None, head_idx, split_idx])
                    else:
                        mLSE_cur = cute.domain_offset((offset,), mLSE[None, head_idx])
                for stage in cutlass.range_constexpr(self.q_stage):
                    m_tile_idx = (m_block * self.q_stage + stage) * self.cta_group_size + mma_tile_coord_v
                    row_sum, row_max, acc_O_mn_row_is_zero_or_nan = stats[stage]
                    # if tidx == 0 and stage <= 1:
                    #     cute.printf("row_sum = {}, row_max = {}, acc_O_mn_row_is_zero_or_nan = {}\n", row_sum, row_max, acc_O_mn_row_is_zero_or_nan)
                    LN2 = math.log(2.0)
                    lse = (
                        (row_max * softmax_scale_log2_eff + (cute.math.log2(row_sum, fastmath=True) - max_offset)) * LN2
                        if not acc_O_mn_row_is_zero_or_nan
                        else -Float32.inf
                    )
                    seqlen_q = (
                        seqlen.seqlen_q
                        if const_expr(not self.pack_gqa)
                        else seqlen.seqlen_q * self.qhead_per_kvhead
                    )
                    if const_expr(not self.pack_gqa or self.m_block_size % self.qhead_per_kvhead == 0):
                        gLSE = cute.local_tile(mLSE_cur, (self.m_block_size,), (m_tile_idx,))
                        if tidx < seqlen_q - m_tile_idx * self.m_block_size:
                            # This actually just works with PackGQA too
                            gLSE[tidx] = lse
                    else:
                        idx = m_tile_idx * self.m_block_size + tidx
                        if idx < seqlen_q:
                            m_idx = idx // self.qhead_per_kvhead
                            h_idx = idx - m_idx * self.qhead_per_kvhead
                            lse_ptr_i64 = utils.elem_pointer(mLSE_cur, ((h_idx, m_idx),)).toint()
                            lse_gmem_ptr = cute.make_ptr(
                                mLSE_cur.element_type, lse_ptr_i64, cute.AddressSpace.gmem, assumed_align=4
                            )
                            cute.make_tensor(lse_gmem_ptr, (1,))[0] = lse

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()
        # End of persistent scheduler loop

        # This is equivalent to pipeline_o_epi.consumer_tail() for the correction warps
        if const_expr(not self.use_correction_warps_for_epi):
            pipeline_o_epi.producer_acquire_w_index_phase(self.q_stage - 1, corr_epi_producer_phase)

    @cute.jit
    def correction_rescale(
        self,
        thr_mma: cute.ThrMma,
        tOtO: cute.Tensor,
        tidx: Int32,
        scale: Float32,
    ):
        """Rescale intermediate attention results based on softmax normalization factor.

        This method performs a crucial correction step in the attention computation pipeline.
        When processing attention in blocks, the softmax normalization factors may change
        as new blocks are processed. This method rescales previously computed partial
        output values to account for updated normalization factors.

        The implementation uses efficient tensor memory operations to:
        1. Load existing partial attention output from tensor memory
        2. Apply the scaling factor to all elements
        3. Store the rescaled results back to tensor memory
        """
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))
        corr_tile_size = 16  # tuneable parameter
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)), self.pv_acc_dtype
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tOtO_i = cute.composition(tOtO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOcO_i = cute.composition(tOcO, cute.make_layout((self.m_block_size, corr_tile_size)))
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tOtO_i).get_slice(tidx)
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tOtO_i).get_slice(tidx)
        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i)
        tOrO_t2r_shape = thr_tmem_load.partition_D(tOcO_i).shape
        tOtO_r2t = thr_tmem_store.partition_D(tOtO_i)

        frg_count = self.head_dim_v_padded // corr_tile_size
        tOrO_frg = cute.make_rmem_tensor((tOrO_t2r_shape, frg_count), self.pv_acc_dtype)
        for i in cutlass.range_constexpr(frg_count):
            tOrO_frg = cute.make_rmem_tensor(tOrO_t2r_shape, self.pv_acc_dtype)
            tOtO_t2r_i = cute.make_tensor(tOtO_t2r.iterator + i * corr_tile_size, tOtO_t2r.layout)
            cute.copy(thr_tmem_load, tOtO_t2r_i, tOrO_frg)
            for j in cutlass.range(0, cute.size(tOrO_frg), 2, unroll_full=True):
                tOrO_frg[j], tOrO_frg[j + 1] = cute.arch.mul_packed_f32x2(
                    (tOrO_frg[j], tOrO_frg[j + 1]), (scale, scale)
                )
            tOtO_r2t_i = cute.make_tensor(tOtO_r2t.iterator + i * corr_tile_size, tOtO_r2t.layout)
            cute.copy(thr_tmem_store, tOrO_frg, tOtO_r2t_i)
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def correction_epilogue(
        self,
        thr_mma: cute.ThrMma,
        tOtO: cute.Tensor,
        tidx: Int32,
        stage: Int32,
        m_block: Int32,
        seqlen_q: Int32,
        scale: Float32,
        sO: cute.Tensor,
        mO_cur: Optional[cute.Tensor] = None,
        gO: Optional[cute.Tensor] = None,
        gmem_tiled_copy_O: Optional[cute.TiledCopy] = None,
    ):
        """Apply final scaling and transformation to attention output before writing to global memory.

        This correction_epilogue function handles the final processing step for attention output values.
        It applies a scaling factor to the accumulated attention results and prepares the
        data for efficient transfer back to global memory.

        The method performs:
        1. Loading of accumulated attention results from tensor memory
        2. Application of the final output scaling factor
        3. Type conversion if necessary (typically from higher precision accumulator to output precision)
        4. Reorganization of data for optimal memory access patterns
        5. Preparation for efficient TMA store operations

        :param thr_mma: Thread MMA operation for the computation
        :type thr_mma: cute.ThrMma
        :param tOtO: Tensor containing accumulated attention output
        :type tOtO: cute.Tensor
        :param scale: Final scaling factor to apply to the output
        :type scale: Float32
        :param sO: Shared memory tensor for the final output
        :type sO: cute.Tensor
        """

        corr_tile_size = 8 * 32 // self.o_dtype.width
        # Use CTA 0 mapping for smem partitioning since sO is per-CTA sized
        tOsO = thr_mma.get_slice(0).partition_C(sO)
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))

        tOtO_i = cute.logical_divide(tOtO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOcO_i = cute.logical_divide(tOcO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOsO_i = cute.logical_divide(tOsO, cute.make_layout((self.m_block_size, corr_tile_size)))

        epi_subtile = (self.epi_tile[0], corr_tile_size)
        tmem_copy_atom = sm100_utils_basic.get_tmem_load_op(
            self.mma_tiler_pv,
            self.o_layout,
            self.o_dtype,
            self.pv_acc_dtype,
            epi_subtile,
            use_2cta_instrs=self.use_2cta_instrs,
        )
        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tOtO_i[(None, None), 0])
        thr_tmem_load = tiled_tmem_load.get_slice(tidx)
        smem_copy_atom = sm100_utils_basic.get_smem_store_op(
            self.o_layout, self.o_dtype, self.pv_acc_dtype, tiled_tmem_load
        )
        tiled_smem_store = cute.make_tiled_copy_D(smem_copy_atom, tiled_tmem_load)

        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i[(None, None), None])
        tOsO_s2r = copy_utils.partition_D_position_independent(thr_tmem_load, tOsO_i[(None, None), None])
        tOcO_t2r = thr_tmem_load.partition_D(tOcO_i[(None, None), None])
        for i in cutlass.range(self.head_dim_v_padded // corr_tile_size, unroll_full=True):
            tOtO_t2r_i = tOtO_t2r[None, 0, 0, i]
            tOsO_r2s_i = tOsO_s2r[None, 0, 0, i]
            tOrO_frg = cute.make_rmem_tensor(tOcO_t2r[None, 0, 0, i].shape, self.pv_acc_dtype)
            cute.copy(tiled_tmem_load, tOtO_t2r_i, tOrO_frg)
            for j in cutlass.range(0, cute.size(tOrO_frg), 2, unroll_full=True):
                tOrO_frg[j], tOrO_frg[j + 1] = cute.arch.mul_packed_f32x2(
                    (tOrO_frg[j], tOrO_frg[j + 1]), (scale, scale)
                )
            copy_utils.cvt_copy(tiled_smem_store, tOrO_frg, tOsO_r2s_i)
        cute.arch.fence_view_async_shared()

        if const_expr(self.use_correction_warps_for_epi):
            assert(not self.use_tma_O)
            assert(gmem_tiled_copy_O is not None)
            cute.arch.barrier(barrier_id=int(NamedBarrierFwdSm100.Epilogue),
                              number_of_threads=len(self.epilogue_warp_ids) * cute.arch.WARP_SIZE)
            mma_tile_coord_v = thr_mma.thr_idx
            m_tile_idx = (m_block * self.q_stage + stage) * self.cta_group_size + mma_tile_coord_v
            self._store_O_to_gmem(
                sO, gO, mO_cur, gmem_tiled_copy_O, tidx, seqlen_q, m_tile_idx
            )

    @cute.jit
    def _store_O_to_gmem(
        self,
        sO_stage: cute.Tensor,
        gO: Optional[cute.Tensor],
        mO_cur: cute.Tensor,
        gmem_tiled_copy_O: cute.TiledCopy,
        tidx: Int32,
        seqlen_q: Int32,
        m_tile_idx: Int32,
    ):
        """Copy a single stage of O from smem to gmem via registers."""
        gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
        tOsO = gmem_thr_copy_O.partition_S(sO_stage)
        cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_v_padded))
        tOcO = gmem_thr_copy_O.partition_S(cO)
        t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
        tOpO = copy_utils.predicate_k(tOcO, limit=mO_cur.shape[1])
        pack_gqa = PackGQA(
            self.m_block_size,
            self.head_dim_v_padded,
            self.check_hdim_v_oob,
            self.qhead_per_kvhead,
        )

        # load acc O from smem to rmem for wider vectorization
        tOrO = cute.make_fragment_like(tOsO, self.o_dtype)
        cute.autovec_copy(tOsO, tOrO)
        # copy acc O from rmem to gmem
        if const_expr(not self.pack_gqa):
            assert gO is not None
            tOgO = gmem_thr_copy_O.partition_D(gO)
            for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
                if (
                    t0OcO[0, rest_m, 0][0] < seqlen_q - m_tile_idx * self.m_block_size - tOcO[0][0]
                ):
                    cute.copy(
                        gmem_tiled_copy_O,
                        tOrO[None, rest_m, None],
                        tOgO[None, rest_m, None],
                        pred=tOpO[None, rest_m, None]
                        if const_expr(self.check_hdim_v_oob)
                        else None,
                    )
        else:
            pack_gqa.store_O(
                mO_cur, tOrO, gmem_tiled_copy_O, tidx, m_tile_idx, seqlen_q
            )

    @cute.jit
    def epilogue_s2g(
        self,
        mO: cute.Tensor,
        sO: cute.Tensor,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        pipeline_o_epi: pipeline.PipelineAsync,
        block_info: BlockInfo,
        num_splits: int,
        SeqlenInfoCls: Callable,
        mma_tile_coord_v: Int32 = 0,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        tile_scheduler=None,
    ):
        epi_consumer_phase = Int32(0)
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block, split_idx, num_splits)
            has_work = const_expr(self.use_block_sparsity or not self.is_split_kv) or n_block_min < n_block_max

            if has_work:
                if const_expr(self.is_split_kv):
                    mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx, split_idx]
                else:
                    mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx]
                gO = None
                if const_expr(self.use_tma_O or not self.pack_gqa):
                    tiler_gO = ((self.mma_tiler_pv[0] * self.q_stage), self.head_dim_v_padded)
                    gO = cute.local_tile(mO_cur, tiler_gO, (m_block, 0))  # (128 * 2, 128)
                    gO = layout_utils.select(
                        cute.flat_divide(gO, (self.mma_tiler_pv[0],)), mode=[0, 2, 1]
                    )  # (128, 128, 2)
                    gO = cute.flat_divide(gO, (self.mma_tiler_pv[0] // self.cta_group_size,))[None, mma_tile_coord_v, None, None]

                if const_expr(self.use_tma_O):
                    store_O, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_O, 0, cute.make_layout(1), sO, gO
                    )
                    for stage in cutlass.range(self.q_stage, unroll_full=True):
                        # wait from corr, issue tma store on smem
                        # 1. wait for O0 / O1 final
                        pipeline_o_epi.consumer_wait_w_index_phase(stage, epi_consumer_phase)
                        # 2. copy O0 / O1 to gmem
                        store_O(src_idx=stage, dst_idx=stage)
                        cute.arch.cp_async_bulk_commit_group()
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # Ensure O0 / O1 buffer is ready to be released
                        cute.arch.cp_async_bulk_wait_group(self.q_stage - 1 - stage, read=True)
                        pipeline_o_epi.consumer_release_w_index(stage)
                else:
                    tidx = cute.arch.thread_idx()[0] % (
                        cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
                    )
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # wait from corr, issue tma store on smem
                        # 1. wait for O0 / O1 final
                        pipeline_o_epi.consumer_wait_w_index_phase(stage, epi_consumer_phase)
                        # 2. copy O0 / O1 to gmem
                        m_tile_idx = (m_block * self.q_stage + stage) * self.cta_group_size + mma_tile_coord_v
                        gO_stage = gO[None, None, stage] if const_expr(gO is not None) else None
                        self._store_O_to_gmem(
                            sO[None, None, stage], gO_stage, mO_cur, gmem_tiled_copy_O,
                            tidx, seqlen.seqlen_q, m_tile_idx,
                        )
                        pipeline_o_epi.consumer_release_w_index(stage)

                epi_consumer_phase ^= 1

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()

    @cute.jit
    def clc_scheduler_warp(
        self,
        tile_scheduler: TileSchedulerProtocol,
    ):
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            tile_scheduler.prefetch_next_work()
            work_tile = tile_scheduler.advance_to_next_work()
            if cute.arch.thread_idx()[0] == self.clc_scheduler_warp_id * cute.arch.WARP_SIZE:
                fa_printf(
                    3,
                    "[CLC] query sm={} cta={} (m_blk={},h={},b={},s={}) valid={}\n",
                    smid(),
                    cute.arch.block_idx()[0],
                    work_tile.tile_idx[0],
                    work_tile.tile_idx[1],
                    work_tile.tile_idx[2],
                    work_tile.tile_idx[3],
                    work_tile.is_valid_tile,
                )
        tile_scheduler.producer_tail()

    @cute.jit
    def empty_warp(
        self,
        tile_scheduler: TileSchedulerProtocol,
    ):
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_tile = tile_scheduler.advance_to_next_work()

    def load_Q(
        self,
        load_Q_fn: Callable,
        pipeline_q: pipeline.PipelineAsync,
        block: Int32,
        stage: int,
        phase: Int32,
    ):
        pipeline_q.producer_acquire_w_index_phase(stage, phase)
        load_Q_fn(src_idx=block, dst_idx=stage, tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(stage))

    def load_Q_non_tma(
        self,
        mQ: cute.Tensor,
        sQ: cute.Tensor,
        gmem_tiled_copy_Q: cute.TiledCopy,
        pipeline_q: pipeline.PipelineAsync,
        tidx: Int32,
        seqlen_q: Int32,
        m_block: Int32,
        block: Int32,
        stage: int,
        phase: Int32,
    ):
        assert self.cta_group_size == 1, "cta_group_size must be 1 for non-tma Q load"
        pipeline_q.producer_acquire_w_index_phase(stage, phase)
        pack_gqa = PackGQA(
            self.m_block_size,
            self.head_dim_padded,
            self.check_hdim_oob,
            self.qhead_per_kvhead,
        )
        sQ_stage = sQ[None, None, None, stage]
        sQ_pi = cute.make_tensor(
            sQ_stage.iterator,
            cute.make_layout(
                (sQ_stage.shape[0][0], (sQ_stage.shape[0][1], sQ_stage.shape[2])),
                stride=(sQ_stage.stride[0][0], (sQ_stage.stride[0][1], sQ_stage.stride[2])),
            ),
        )
        pack_gqa.load_Q(mQ, sQ_pi, gmem_tiled_copy_Q, tidx, m_block * self.q_stage + block, seqlen_q)
        cute.arch.cp_async_commit_group()
        pipeline_q.sync_object_full.arrive_cp_async_mbarrier(stage)

    @cute.jit
    def load_KV(
        self,
        tma_atom: Optional[cute.CopyAtom],
        tXgX: Optional[cute.Tensor],
        tXsX: Optional[cute.Tensor],
        paged_kv_manager: Optional[PagedKVManager],
        sX: cute.Tensor,
        block: Int32,
        pipeline_kv: pipeline.PipelineAsync,
        producer_state: pipeline.PipelineState,
        K_or_V: Literal["K", "V"],
        page_idx: Optional[Int32] = None,
        extra_tx_count: Optional[Int32] = None,
    ):
        assert K_or_V in ("K", "V")
        stage, phase = producer_state.index, producer_state.phase
        extra_tx_count_kv = self.tma_copy_bytes[K_or_V] - self.tma_copy_bytes["K"]
        extra_tx_count = (
            extra_tx_count_kv + (extra_tx_count if extra_tx_count is not None else 0) if const_expr(self.use_tma_KV)
            else None
        )
        extra_kwargs = {"extra_tx_count": extra_tx_count} if const_expr(self.use_tma_KV) else {}
        pipeline_kv.producer_acquire(producer_state, **extra_kwargs)
        if const_expr(K_or_V == "K" and self.uneven_kv_smem):
            # Before this round, the smem location was occupied by V, which is smaller than
            # K. So we need to wait for the stage after that (stage 1) to be empty as well.
            if stage == 0:
                pipeline_kv.sync_object_empty.wait(1, phase)

        if const_expr(self.use_tma_KV):
            assert tXgX is not None and tXsX is not None and tma_atom is not None
            tXsX_cur = tXsX[None, stage]
            if const_expr(self.uneven_kv_smem):
                # Since this is the producer_state, the phase starts at 1, so we have to invert it
                tXsX_cur = self.offset_kv_smem(tXsX_cur, stage, phase ^ 1)
            # Currently we assume that page_size == n_block_size so we index into tXgX with block = 0
            tXgX_cur = tXgX[None, block] if const_expr(page_idx is None) else tXgX[None, 0, page_idx]
            cute.copy(tma_atom, tXgX_cur, tXsX_cur, tma_bar_ptr=pipeline_kv.producer_get_barrier(producer_state))
        else:
            assert paged_kv_manager is not None
            assert extra_tx_count is None
            sX_cur = sX[None, None, None, stage]
            if const_expr(self.uneven_kv_smem):
                sX_cur = self.offset_kv_smem(sX_cur, stage, phase ^ 1)
            paged_kv_manager.load_KV(block, sX_cur, K_or_V)
            cute.arch.cp_async_commit_group()
            pipeline_kv.sync_object_full.arrive_cp_async_mbarrier(stage)

    @cute.jit
    def offset_kv_smem(self, sX: cute.Tensor, stage: Int32, phase: Int32):
        if const_expr(self.uneven_kv_smem):
            # smem layout is [smem_large, smem_small, smem_large], and the current stride is
            # (smem_large + smem_small) // 2. So for stage == 1, move right by offset if
            # phase == 0, or left by offset if phase == 1.
            offset = 0 if stage != 1 else self.uneven_kv_smem_offset * (1 - 2 * phase)
            # Hint that the offset is 128-bit aligned so that
            # ptr + offset preserves the alignment needed by cp.async.
            offset = cute.assume(offset, divby=128 // self.k_dtype.width)
            return cute.make_tensor(sX.iterator + offset, sX.layout)
        else:
            return sX

    # @cute.jit
    # def warp_scheduler_barrier_init(self):
    #     warp_group_idx = utils.canonical_warp_group_idx(sync=False)
    #     if warp_group_idx == 0:
    #         cute.arch.barrier_arrive(
    #             barrier_id=int(NamedBarrierFwdSm100.WarpSchedulerWG1), number_of_threads=2 * 128,
    #         )

    # def warp_scheduler_barrier_sync(self):
    #     cute.arch.barrier(
    #         barrier_id=int(NamedBarrierFwdSm100.WarpSchedulerWG1) + utils.canonical_warp_group_idx(sync=False),
    #         number_of_threads=2 * 128
    #     )

    # def warp_scheduler_barrier_arrive(self):
    #     cur_wg = utils.canonical_warp_group_idx(sync=False)
    #     next_wg = 1 - cur_wg
    #     cute.arch.barrier_arrive(
    #         barrier_id=int(NamedBarrierFwdSm100.WarpSchedulerWG1) + next_wg, number_of_threads=2 * 128,
    #     )

    @cute.jit
    def apply_score_mod(
        self,
        tSrS_t2r,
        thr_tmem_load,
        thr_mma_qk,
        batch_idx,
        head_idx,
        m_block,
        n_block,
        softmax,
        seqlen: SeqlenInfoQK,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        head_divmod=None,
    ):
        """Apply score modification for SM100 (constant q_idx)."""
        # Prepare index tensor with extra partition
        cS = cute.make_identity_tensor((self.m_block_size, self.n_block_size))
        cS = cute.domain_offset((m_block * self.m_block_size, n_block * self.n_block_size), cS)
        tScS = thr_mma_qk.partition_C(cS)
        tScS = tScS[(None, None), 0, 0]
        tScS_t2r = thr_tmem_load.partition_D(tScS)

        # Shared q_idx for all scores
        q_idx_logical = tScS_t2r[0][0]

        # For Pack-GQA, compute the logical head index for this tile
        if cutlass.const_expr(self.pack_gqa):
            assert head_divmod is not None
            # Building up the logical q_head idx: final_q_head = kv_head * qhead_per_kvhead + (q_physical % qhead_per_kvhead)
            q_physical = q_idx_logical
            q_idx_logical, head_offset = divmod(q_physical, head_divmod)
            head_idx = head_idx * self.qhead_per_kvhead + head_offset

        if cutlass.const_expr(aux_data.tensors is not None):
            seqlen_q_divmod, _ = fastdiv_mods
            _, q_idx_logical = divmod(q_idx_logical, seqlen_q_divmod)

        apply_score_mod_inner(
            tSrS_t2r,
            tScS_t2r,
            self.score_mod,
            batch_idx,
            head_idx,
            softmax.softmax_scale,
            self.score_vec_size,
            self.qk_acc_dtype,
            aux_data,
            fastdiv_mods,
            seqlen_info=seqlen,
            constant_q_idx=q_idx_logical,
            qhead_per_kvhead=self.qhead_per_kvhead if cutlass.const_expr(self.pack_gqa) else 1,
        )
