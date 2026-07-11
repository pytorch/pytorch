# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# A reimplementation of https://github.com/Dao-AILab/flash-attention/blob/main/hopper/mainloop_bwd_sm80.hpp
# from Cutlass C++ to Cute-DSL.
import math
from types import SimpleNamespace
from typing import Type, Callable, Optional
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, warp
from cutlass import Int32
import cutlass.utils as utils_basic

from ..quack import layout_utils
from . import ampere_helpers as sm80_utils
from .cute_dsl_utils import assume_tensor_aligned
from . import utils
from .mask import AttentionMask
from .softmax import call_score_mod, call_score_mod_bwd
from .seqlen_info import SeqlenInfoQK
from .block_info import BlockInfo
from ..quack.cute_dsl_utils import ParamsBase
from .tile_scheduler import SingleTileScheduler, SingleTileVarlenScheduler, TileSchedulerArguments
from .block_sparsity import BlockSparseTensors
from .utils import AuxData


class FlashAttentionBackwardSm80:
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: int = 1,
        m_block_size: int = 64,
        n_block_size: int = 128,
        num_stages_Q: int = 2,
        num_stages_dO: int = 2,
        num_threads: int = 256,
        pack_gqa: bool = False,
        is_causal: bool = False,
        is_local: bool = False,
        SdP_swapAB: bool = False,
        dKV_swapAB: bool = False,
        dQ_swapAB: bool = False,
        AtomLayoutMSdP: int = 1,
        AtomLayoutNdKV: int = 8,
        AtomLayoutMdQ: int = 1,
        V_in_regs: bool = False,
        score_mod: cutlass.Constexpr | None = None,
        score_mod_bwd: cutlass.Constexpr | None = None,
    ):
        """Initializes the configuration for a flash attention v2 kernel.

        All contiguous dimensions must be at least 16 bytes aligned which indicates the head dimension
        should be a multiple of 8.

        :param head_dim: head dimension
        :type head_dim: int
        :param m_block_size: m block size
        :type m_block_size: int
        :param n_block_size: n block size
        :type n_block_size: int
        :param num_threads: number of threads
        :type num_threads: int
        :param is_causal: is causal
        """
        self.dtype = dtype
        # padding head_dim to a multiple of 32 (stricter than fwd's 16) due to
        # backward kernel register layout requirements for dQ/dK/dV accumulation
        hdim_multiple_of = 32
        self.head_dim_padded = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.head_dim_v_padded = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        # Can save registers (and hence be faster) if we don't have to check hdim predication
        self.check_hdim_oob = head_dim != self.head_dim_padded
        self.check_hdim_v_oob = head_dim_v != self.head_dim_v_padded
        self.qhead_per_kvhead = qhead_per_kvhead
        self.m_block_size = m_block_size
        self.n_block_size = n_block_size
        self.num_threads = num_threads
        self.pack_gqa = pack_gqa
        self.is_causal = is_causal
        self.is_local = is_local
        self.num_stages_Q = num_stages_Q
        self.num_stages_dO = num_stages_dO
        self.SdP_swapAB = SdP_swapAB
        self.dKV_swapAB = dKV_swapAB
        self.dQ_swapAB = dQ_swapAB
        self.AtomLayoutMSdP = AtomLayoutMSdP
        self.AtomLayoutNdKV = AtomLayoutNdKV
        self.AtomLayoutMdQ = AtomLayoutMdQ
        num_mma_warps = self.num_threads // cute.arch.WARP_SIZE
        self.Mma_dKV_is_RS = AtomLayoutMSdP == 1 and AtomLayoutNdKV == num_mma_warps and SdP_swapAB and not dKV_swapAB
        self.V_in_regs = V_in_regs
        self.share_QV_smem = V_in_regs
        self.score_mod = score_mod
        self.score_mod_bwd = score_mod_bwd

    @staticmethod
    def can_implement(
        dtype, head_dim, head_dim_v, m_block_size, n_block_size, num_stages_Q, num_stages_dO,
        num_threads, is_causal,
        V_in_regs=False
    ) -> bool:
        """Check if the kernel can be implemented with the given parameters.

        :param dtype: data type
        :type dtype: cutlass.Numeric
        :param head_dim: head dimension
        :type head_dim: int
        :param m_block_size: m block size
        :type m_block_size: int
        :param n_block_size: n block size
        :type n_block_size: int
        :param num_threads: number of threads
        :type num_threads: int
        :param is_causal: is causal
        :type is_causal: bool

        :return: True if the kernel can be implemented, False otherwise
        :rtype: bool
        """
        if dtype not in [cutlass.Float16, cutlass.BFloat16]:
            return False
        if head_dim % 8 != 0:
            return False
        if head_dim_v % 8 != 0:
            return False
        if n_block_size % 16 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        # Check if block size setting is out of shared memory capacity
        # Shared memory usage: Q tile + (K tile + V tile) where K and V use the same tile size
        smem_usage_Q = m_block_size * head_dim * num_stages_Q * 2
        smem_usage_dO = m_block_size * head_dim_v * num_stages_dO * 2
        smem_usage_K = n_block_size * head_dim * 2
        smem_usage_V = n_block_size * head_dim_v * 2
        smem_usage_QV = (smem_usage_Q + smem_usage_V) if not V_in_regs else max(smem_usage_Q, smem_usage_V)
        smem_usage = smem_usage_QV + smem_usage_dO + smem_usage_K
        smem_capacity = utils_basic.get_smem_capacity_in_bytes("sm_80")
        if smem_usage > smem_capacity:
            return False
        return True

    def _check_type(
        self,
        mQ_type: Type[cutlass.Numeric],
        mK_type: Type[cutlass.Numeric],
        mV_type: Type[cutlass.Numeric],
        mdO_type: Type[cutlass.Numeric],
        mLSE_type: Type[cutlass.Numeric],
        mdPsum_type: Type[cutlass.Numeric],
        mdQaccum_type: Type[cutlass.Numeric],
        mdK_type: Type[cutlass.Numeric],
        mdV_type: Type[cutlass.Numeric],
        mCuSeqlensQ_type: Type[cutlass.Numeric] | None,
        mCuSeqlensK_type: Type[cutlass.Numeric] | None,
        mSeqUsedQ_type: Type[cutlass.Numeric] | None,
        mSeqUsedK_type: Type[cutlass.Numeric] | None,
    ):
        if cutlass.const_expr(not (mQ_type == mK_type == mV_type == mdO_type)):
            raise TypeError("All tensors must have the same data type")
        if cutlass.const_expr(self.qhead_per_kvhead == 1):
            if cutlass.const_expr(not (mdK_type == mdV_type == mQ_type)):
                raise TypeError("mdK and mdV tensors must have the same data type as mQ")
        else:
            if cutlass.const_expr(not (mdK_type == mdV_type == cutlass.Float32)):
                raise TypeError("mdKaccum and mdVaccum tensors must have the data type Float32")
        if cutlass.const_expr(mQ_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if cutlass.const_expr(mLSE_type not in [cutlass.Float32]):
            raise TypeError("LSE tensor must be Float32")
        if cutlass.const_expr(mdPsum_type not in [cutlass.Float32]):
            raise TypeError("dPsum tensor must be Float32")
        if cutlass.const_expr(mdQaccum_type not in [cutlass.Float32]):
            raise TypeError("dQaccum tensor must be Float32")
        if cutlass.const_expr(mCuSeqlensQ_type not in [None, cutlass.Int32]):
            raise TypeError("cuSeqlensQ tensor must be Int32")
        if cutlass.const_expr(mCuSeqlensK_type not in [None, cutlass.Int32]):
            raise TypeError("cuSeqlensK tensor must be Int32")
        if cutlass.const_expr(mSeqUsedQ_type not in [None, cutlass.Int32]):
            raise TypeError("SeqUsedQ tensor must be Int32")
        if cutlass.const_expr(mSeqUsedK_type not in [None, cutlass.Int32]):
            raise TypeError("SeqUsedK tensor must be Int32")
        assert mQ_type == self.dtype

    def _setup_attributes(self):
        # ///////////////////////////////////////////////////////////////////////////////
        # Shared memory layout: Q/K/V
        # ///////////////////////////////////////////////////////////////////////////////
        sQ_layout_atom = sm80_utils.get_smem_layout_atom(self.dtype, self.head_dim_padded)
        self.sQ_layout = cute.tile_to_shape(
            sQ_layout_atom, (self.m_block_size, self.head_dim_padded, self.num_stages_Q), (0, 1, 2),
        )
        sK_layout_atom = sQ_layout_atom
        self.sK_layout = cute.tile_to_shape(
            sK_layout_atom, (self.n_block_size, self.head_dim_padded), (0, 1),
        )
        sV_layout_atom = sm80_utils.get_smem_layout_atom(self.dtype, self.head_dim_v_padded)
        self.sV_layout = cute.tile_to_shape(
            sV_layout_atom, (self.n_block_size, self.head_dim_v_padded), (0, 1),
        )
        sdO_layout_atom = sV_layout_atom
        self.sdO_layout = cute.tile_to_shape(
            sdO_layout_atom, (self.m_block_size, self.head_dim_v_padded, self.num_stages_dO), (0, 1, 2),
        )
        # TODO: do we set swizzle to be 3 here explicitly?
        sPdS_layout_atom = sm80_utils.get_smem_layout_atom(self.dtype, self.n_block_size)
        self.sPdS_layout = cute.tile_to_shape(
            sPdS_layout_atom, (self.m_block_size, self.n_block_size), (0, 1),
        )
        # We set stride to be multiple of 64 so that if ShuffleLSE, even if threads read from sLSE but out of bounds,
        # it's still a valid smem address.
        self.sLSE_layout = cute.make_layout(
            (self.m_block_size, self.num_stages_Q),
            stride=(1, cute.round_up(self.m_block_size, 64)),
        )
        sLSEMma_layout = cute.make_layout(
            (self.m_block_size, self.n_block_size, self.num_stages_Q),
            stride=(1, 0, cute.round_up(self.m_block_size, 64)),
        )
        sLSEMma_layout_transposed = cute.make_layout(
            (self.n_block_size, self.m_block_size, self.num_stages_Q),
            stride=(0, 1, cute.round_up(self.m_block_size, 64)),
        )
        self.sLSEMma_layout = sLSEMma_layout if not self.SdP_swapAB else sLSEMma_layout_transposed

        # ///////////////////////////////////////////////////////////////////////////////
        # GMEM Tiled copy:
        # ///////////////////////////////////////////////////////////////////////////////
        # Thread layouts for copies
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self.dtype.width
        # atom_async_copy: async copy atom for QKV load
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        # atom_universal_copy: universal copy atom for O store
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=universal_copy_bits,
        )
        # tQK_layout: thread layout for QK load
        tQK_shape_dim_1 = sQ_layout_atom.outer.shape[1] // async_copy_elems
        assert self.num_threads % tQK_shape_dim_1 == 0, "num_threads must be divisible by tQK_shape_dim_1"
        tQK_layout = cute.make_ordered_layout(
            (self.num_threads // tQK_shape_dim_1, tQK_shape_dim_1), order=(1, 0),
        )
        # Do we need to check if we overshot kBlockM when we load Q?
        self.is_even_m_smem_q = self.m_block_size % tQK_layout.shape[0] == 0
        # Do we need to check if we overshot kBlockN when we load K?
        self.is_even_n_smem_k = self.n_block_size % tQK_layout.shape[0] == 0
        tVdO_shape_dim_1 = sV_layout_atom.outer.shape[1] // async_copy_elems
        assert self.num_threads % tVdO_shape_dim_1 == 0, "num_threads must be divisible by tVdO_shape_dim_1"
        tVdO_layout = cute.make_ordered_layout(
            (self.num_threads // tVdO_shape_dim_1, tVdO_shape_dim_1), order=(1, 0),
        )
        # Do we need to check if we overshot kBlockN when we load V?
        self.is_even_n_smem_v = self.n_block_size % tVdO_layout.shape[0] == 0
        self.is_even_m_smem_do = self.m_block_size % tVdO_layout.shape[0] == 0

        # Value layouts for copies
        vQKVdO_layout = cute.make_layout((1, async_copy_elems))

        # gmem_tiled_copy_QK: tiled copy for QK load
        self.gmem_tiled_copy_QK = cute.make_tiled_copy_tv(atom_async_copy, tQK_layout, vQKVdO_layout)
        self.gmem_tiled_copy_VdO = cute.make_tiled_copy_tv(atom_async_copy, tVdO_layout, vQKVdO_layout)
        self.gmem_tiled_copy_dK = cute.make_tiled_copy_tv(atom_universal_copy, tQK_layout, vQKVdO_layout)
        self.gmem_tiled_copy_dV = cute.make_tiled_copy_tv(atom_universal_copy, tVdO_layout, vQKVdO_layout)
        async_copy_elems_accum = universal_copy_bits // cutlass.Float32.width

        # I think we wouldn't require this with smarter padding
        if cutlass.const_expr(not self.varlen_q):
            async_copy_elems_accum = universal_copy_bits // cutlass.Float32.width
            atom_async_copy_accum = cute.make_copy_atom(
                cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                cutlass.Float32,
                num_bits_per_copy=universal_copy_bits,
            )
        else:
            async_copy_elems_accum = 1
            atom_async_copy_accum = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                cutlass.Float32,
                num_bits_per_copy=cutlass.Float32.width,
            )
        self.gmem_tiled_copy_LSE = cute.make_tiled_copy_tv(
            atom_async_copy_accum,
            cute.make_layout(self.num_threads),
            cute.make_layout(async_copy_elems_accum),
        )
        self.gmem_tiled_copy_dQaccum = cute.make_tiled_copy_tv(
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=cutlass.Float32.width
            ),
            cute.make_layout(self.num_threads),
            cute.make_layout(1)
        )
        if cutlass.const_expr(self.qhead_per_kvhead > 1):
            self.gmem_tiled_copy_dK = self.gmem_tiled_copy_dQaccum
            self.gmem_tiled_copy_dV = self.gmem_tiled_copy_dQaccum

    def _get_tiled_mma(self):
        num_mma_warps = self.num_threads // 32
        AtomLayoutSdP = (self.AtomLayoutMSdP, num_mma_warps // self.AtomLayoutMSdP, 1) if cutlass.const_expr(not self.SdP_swapAB) else (num_mma_warps // self.AtomLayoutMSdP, self.AtomLayoutMSdP, 1)
        tiled_mma_sdp = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, cutlass.Float32, (16, 8, 16)),
            AtomLayoutSdP,
            permutation_mnk=(AtomLayoutSdP[0] * 16, AtomLayoutSdP[1] * 16, 16),
        )
        AtomLayoutdKV = (self.AtomLayoutNdKV, num_mma_warps // self.AtomLayoutNdKV, 1) if cutlass.const_expr(not self.dKV_swapAB) else (num_mma_warps // self.AtomLayoutNdKV, self.AtomLayoutNdKV, 1)
        tiled_mma_dkv = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, cutlass.Float32, (16, 8, 16)),
            AtomLayoutdKV,
            permutation_mnk=(AtomLayoutdKV[0] * 16, AtomLayoutdKV[1] * 16, 16),
        )
        AtomLayoutdQ = (self.AtomLayoutMdQ, num_mma_warps // self.AtomLayoutMdQ, 1) if cutlass.const_expr(not self.dQ_swapAB) else (num_mma_warps // self.AtomLayoutMdQ, self.AtomLayoutMdQ, 1)
        tiled_mma_dq = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, cutlass.Float32, (16, 8, 16)),
            AtomLayoutdQ,
            permutation_mnk=(AtomLayoutdQ[0] * 16, AtomLayoutdQ[1] * 16, 16),
        )
        return tiled_mma_sdp, tiled_mma_dkv, tiled_mma_dq

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sV_struct, sdO_struct = [
            cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(layout)], 1024]
            for layout in (self.sQ_layout, self.sK_layout, self.sV_layout, self.sdO_layout)
        ]
        cosize_sQV = max(cute.cosize(self.sQ_layout), cute.cosize(self.sV_layout))
        sQV_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sQV], 1024]
        sLSE_struct, sdPsum_struct = [
            cute.struct.Align[cute.struct.MemRange[cutlass.Float32, cute.cosize(layout)], 128]
            for layout in (self.sLSE_layout, self.sLSE_layout)
        ]
        sP_struct, sdS_struct = [
            cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(layout)], 128]
            for layout in (self.sPdS_layout, self.sPdS_layout)
        ]

        @cute.struct
        class SharedStorageSeparateQV:
            sK: sK_struct
            sV: sV_struct
            sQ: sQ_struct
            sdO: sdO_struct
            sLSE: sLSE_struct
            sdPsum: sdPsum_struct
            sP: sP_struct
            sdS: sdS_struct
            # TODO: the case where there's no sP

        @cute.struct
        class SharedStorageSharedQV:
            sK: sK_struct
            sV: sV_struct
            sQ: sQV_struct
            sdO: sdO_struct
            sLSE: sLSE_struct
            sdPsum: sdPsum_struct
            sP: sP_struct
            sdS: sdS_struct

        return SharedStorageSeparateQV if cutlass.const_expr(not self.share_QV_smem) else SharedStorageSharedQV

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        softmax_scale: cutlass.Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        mdQ_semaphore: Optional[cute.Tensor] = None,
        mdK_semaphore: Optional[cute.Tensor] = None,
        mdV_semaphore: Optional[cute.Tensor] = None,
        aux_data: AuxData = AuxData(),
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        assert mdQ_semaphore is None and mdK_semaphore is None and mdV_semaphore is None, (
            "determinism not supported yet for Sm80"
        )
        # Get the data type and check if it is fp16 or bf16
        self._check_type(*(t.element_type if t is not None else None
                           for t in (mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV, mCuSeqlensQ, mCuSeqlensK, mSeqUsedQ, mSeqUsedK)))
        mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV = [
            assume_tensor_aligned(t) for t in (mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV)
        ]
        self.varlen_q = (mCuSeqlensQ is not None)
        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()
        tiled_mma_sdp, tiled_mma_dkv, tiled_mma_dq = self._get_tiled_mma()

        num_head = mQ.shape[1] if cutlass.const_expr(mCuSeqlensQ is not None) else mQ.shape[2]

        if cutlass.const_expr(mCuSeqlensK is not None):
            TileScheduler = SingleTileVarlenScheduler
            num_batch = mCuSeqlensK.shape[0] - 1
        else:
            TileScheduler = SingleTileScheduler
            num_batch = mK.shape[0]

        # Uses seqlen k, etc. since main bwd kernel's blocks are over n
        tile_sched_args = TileSchedulerArguments(
            num_block=cute.ceil_div(mK.shape[1], self.n_block_size),
            num_head=num_head,
            num_batch=num_batch,
            num_splits=1,
            seqlen_k=0,
            headdim=mK.shape[2],
            headdim_v=mV.shape[2],
            total_q=mK.shape[0],
            tile_shape_mn=(self.n_block_size, self.m_block_size),
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if cutlass.const_expr(self.pack_gqa) else 1,
            mCuSeqlensQ=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedK,
        )

        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        softmax_scale_log2, _ = utils.compute_softmax_scale_log2(
            softmax_scale, self.score_mod
        )
        self.kernel(
            mQ,
            mK,
            mV,
            mdO,
            mLSE,
            mdPsum,
            mdQaccum,
            mdK,
            mdV,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            softmax_scale,
            softmax_scale_log2,
            window_size_left,
            window_size_right,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sdO_layout,
            self.sPdS_layout,
            self.sLSE_layout,
            self.sLSEMma_layout,
            self.gmem_tiled_copy_QK,
            self.gmem_tiled_copy_VdO,
            self.gmem_tiled_copy_dK,
            self.gmem_tiled_copy_dV,
            self.gmem_tiled_copy_LSE,
            self.gmem_tiled_copy_dQaccum,
            tiled_mma_sdp,
            tiled_mma_dkv,
            tiled_mma_dq,
            SharedStorage,
            tile_sched_params,
            TileScheduler,
            aux_data,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        softmax_scale: cutlass.Float32,
        softmax_scale_log2: cutlass.Float32,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sdO_layout: cute.ComposedLayout,
        sPdS_layout: cute.ComposedLayout,
        sLSE_layout: cute.Layout,
        sLSEMma_layout: cute.Layout,
        gmem_tiled_copy_QK: cute.TiledCopy,
        gmem_tiled_copy_VdO: cute.TiledCopy,
        gmem_tiled_copy_dK: cute.TiledCopy,
        gmem_tiled_copy_dV: cute.TiledCopy,
        gmem_tiled_copy_LSE: cute.TiledCopy,
        gmem_tiled_copy_dQaccum: cute.TiledCopy,
        tiled_mma_sdp: cute.TiledMma,
        tiled_mma_dkv: cute.TiledMma,
        tiled_mma_dq: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        aux_data: AuxData = AuxData(),
    ):
        # Thread index, block index
        tidx, _, _ = cute.arch.thread_idx()

        tile_scheduler = TileScheduler.create(tile_sched_params)
        work_tile = tile_scheduler.initial_work_tile_info()

        n_block, head_idx, batch_idx, _ = work_tile.tile_idx

        if work_tile.is_valid_tile:
            seqlen = SeqlenInfoQK.create(
                batch_idx,
                mQ.shape[1],
                mK.shape[1],
                mCuSeqlensQ=mCuSeqlensQ,
                mCuSeqlensK=mCuSeqlensK,
                mSeqUsedQ=mSeqUsedQ,
                mSeqUsedK=mSeqUsedK,
                tile_m=self.m_block_size,
                tile_n=self.n_block_size,
            )

            block_info = BlockInfo(
                self.m_block_size,
                self.n_block_size,
                self.is_causal,
                self.is_local,
                False,
                window_size_left,
                window_size_right,
            )
            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
            # TODO: return early if m_block_max == 0

            # ///////////////////////////////////////////////////////////////////////////////
            # Get the appropriate tiles for this thread block.
            # ///////////////////////////////////////////////////////////////////////////////
            blkQ_shape = (self.m_block_size, self.head_dim_padded)
            blkK_shape = (self.n_block_size, self.head_dim_padded)
            blkV_shape = (self.n_block_size, self.head_dim_v_padded)
            blkdO_shape = (self.m_block_size, self.head_dim_v_padded)

            if cutlass.const_expr(not seqlen.has_cu_seqlens_q):
                mQ_cur = mQ[batch_idx, None, head_idx, None]
                mLSE_cur = mLSE[batch_idx, head_idx, None]
                mdO_cur = mdO[batch_idx, None, head_idx, None]
                mdPsum_cur = mdPsum[batch_idx, head_idx, None]
                mdQaccum_cur = mdQaccum[batch_idx, head_idx, None]
            else:
                padded_offset_q = seqlen.padded_offset_q
                mQ_cur = cute.domain_offset((seqlen.offset_q, 0), mQ[None, head_idx, None])
                mLSE_cur = cute.domain_offset((padded_offset_q,), mLSE[head_idx, None])
                mdO_cur = cute.domain_offset((seqlen.offset_q, 0), mdO[None, head_idx, None])
                mdPsum_cur = cute.domain_offset((padded_offset_q,), mdPsum[head_idx, None])
                mdQaccum_cur = cute.domain_offset((padded_offset_q * self.head_dim_padded,), mdQaccum[head_idx, None])
            head_idx_kv = head_idx // self.qhead_per_kvhead if cutlass.const_expr(not self.pack_gqa) else head_idx

            if cutlass.const_expr(not seqlen.has_cu_seqlens_k):
                mK_cur, mV_cur = [t[batch_idx, None, head_idx_kv, None] for t in (mK, mV)]
            else:
                mK_cur, mV_cur = [cute.domain_offset((seqlen.offset_k, 0), t[None, head_idx_kv, None]) for t in (mK, mV)]

            # (m_block_size, head_dim, m_block)
            gQ = cute.local_tile(mQ_cur, blkQ_shape, (None, 0))
            # (n_block_size, head_dim)
            gK = cute.local_tile(mK_cur, blkK_shape, (n_block, 0))
            # (n_block_size, head_dim_v)
            gV = cute.local_tile(mV_cur, blkV_shape, (n_block, 0))
            # (m_block_size, head_dim_v, m_block)
            gdO = cute.local_tile(mdO_cur, blkdO_shape, (None, 0))
            gLSE = cute.local_tile(mLSE_cur, (self.m_block_size,), (None,))
            gdPsum = cute.local_tile(mdPsum_cur, (self.m_block_size,), (None,))
            gdQaccum = cute.local_tile(mdQaccum_cur, (self.m_block_size * self.head_dim_padded,), (None,))

            # ///////////////////////////////////////////////////////////////////////////////
            # Get shared memory buffer
            # ///////////////////////////////////////////////////////////////////////////////
            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            sQ = storage.sQ.get_tensor(sQ_layout)
            sK = storage.sK.get_tensor(sK_layout)
            if cutlass.const_expr(not self.share_QV_smem):
                sV = storage.sV.get_tensor(sV_layout)
            else:
                sV = cute.make_tensor(cute.recast_ptr(sQ.iterator, dtype=self.dtype), sV_layout)
            sdO = storage.sdO.get_tensor(sdO_layout)
            sP = storage.sP.get_tensor(sPdS_layout)
            sdS = storage.sdS.get_tensor(sPdS_layout)
            sLSE = storage.sLSE.get_tensor(sLSE_layout)
            sdPsum = storage.sdPsum.get_tensor(sLSE_layout)
            sLSEMma = storage.sLSE.get_tensor(sLSEMma_layout)
            sdPsumMma = storage.sdPsum.get_tensor(sLSEMma_layout)

            # Transpose view of tensors for tiled mma
            sQt, sdOt, sKt, sPt, sdSt = [layout_utils.transpose_view(t) for t in (sQ, sdO, sK, sP, sdS)]

            gmem_thr_copy_QK = gmem_tiled_copy_QK.get_slice(tidx)
            gmem_thr_copy_VdO = gmem_tiled_copy_VdO.get_slice(tidx)
            gmem_thr_copy_lse = gmem_tiled_copy_LSE.get_slice(tidx)
            gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_slice(tidx)
            # (CPY_Atom, CPY_M, CPY_K, m_block)
            tQgQ = gmem_thr_copy_QK.partition_S(gQ)
            tQsQ = gmem_thr_copy_QK.partition_D(sQ)
            # (CPY_Atom, CPY_N, CPY_K)
            tKgK = gmem_thr_copy_QK.partition_S(gK)
            tKsK = gmem_thr_copy_QK.partition_D(sK)
            # (CPY_Atom, CPY_N, CPY_K)
            tVgV = gmem_thr_copy_VdO.partition_S(gV)
            tVsV = gmem_thr_copy_VdO.partition_D(sV)
            # (CPY_Atom, CPY_M, CPY_K, m_block)
            tdOgdO = gmem_thr_copy_VdO.partition_S(gdO)
            tdOsdO = gmem_thr_copy_VdO.partition_D(sdO)
            tLSEgLSE = gmem_thr_copy_lse.partition_S(gLSE)
            tLSEsLSE = gmem_thr_copy_lse.partition_D(sLSE)
            tLSEgdPsum = gmem_thr_copy_lse.partition_S(gdPsum)
            tLSEsdPsum = gmem_thr_copy_lse.partition_D(sdPsum)
            tdQgdQaccum = gmem_thr_copy_dQaccum.partition_S(gdQaccum)

            # ///////////////////////////////////////////////////////////////////////////////
            # Tile MMA compute thread partitions and allocate accumulators
            # ///////////////////////////////////////////////////////////////////////////////
            thr_mma_sdp = tiled_mma_sdp.get_slice(tidx)
            thr_mma_dkv = tiled_mma_dkv.get_slice(tidx)
            thr_mma_dq = tiled_mma_dq.get_slice(tidx)
            acc_shape_dK = thr_mma_dkv.partition_shape_C((self.n_block_size, self.head_dim_padded))
            acc_shape_dV = thr_mma_dkv.partition_shape_C((self.n_block_size, self.head_dim_v_padded))
            acc_dK = cute.make_rmem_tensor(acc_shape_dK, cutlass.Float32)
            acc_dV = cute.make_rmem_tensor(acc_shape_dV, cutlass.Float32)
            acc_dK.fill(0.0)
            acc_dV.fill(0.0)

            tSrQ = utils.mma_make_fragment_A(sQ[None, None, 0], thr_mma_sdp, swapAB=self.SdP_swapAB)
            tSrK = utils.mma_make_fragment_B(sK, thr_mma_sdp, swapAB=self.SdP_swapAB)
            tdPrdO = utils.mma_make_fragment_A(sdO[None, None, 0], thr_mma_sdp, swapAB=self.SdP_swapAB)
            tdPrV = utils.mma_make_fragment_B(sV, thr_mma_sdp, swapAB=self.SdP_swapAB)
            tdVrP = utils.mma_make_fragment_A(sPt, thr_mma_dkv, swapAB=self.dKV_swapAB)
            tdVrdO = utils.mma_make_fragment_B(sdOt[None, None, 0], thr_mma_dkv, swapAB=self.dKV_swapAB)
            tdKrdS = utils.mma_make_fragment_A(sdSt, thr_mma_dkv, swapAB=self.dKV_swapAB)
            tdKrQ = utils.mma_make_fragment_B(sQt[None, None, 0], thr_mma_dkv, swapAB=self.dKV_swapAB)
            tdQrdS = utils.mma_make_fragment_A(sdS, thr_mma_dq, swapAB=self.dQ_swapAB)
            tdQrK = utils.mma_make_fragment_B(sKt, thr_mma_dq, swapAB=self.dQ_swapAB)

            LSEslice = (None, 0, None) if cutlass.const_expr(not self.SdP_swapAB) else (0, None, None)
            tSsLSEMma = layout_utils.reshape_acc_to_mn(thr_mma_sdp.partition_C(sLSEMma))[LSEslice]
            tSsdPsumMma = layout_utils.reshape_acc_to_mn(thr_mma_sdp.partition_C(sdPsumMma))[LSEslice]

            # ///////////////////////////////////////////////////////////////////////////////
            # Smem copy atom tiling
            # ///////////////////////////////////////////////////////////////////////////////
            smem_copy_atom = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype,
            )
            smem_copy_atom_transposed = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype,
            )
            smem_thr_copy_QdO = utils.make_tiled_copy_A(
                smem_copy_atom, tiled_mma_sdp, swapAB=self.SdP_swapAB
            ).get_slice(tidx)
            smem_thr_copy_KV = utils.make_tiled_copy_B(
                smem_copy_atom, tiled_mma_sdp, swapAB=self.SdP_swapAB
            ).get_slice(tidx)
            # TODO: should this be smem_copy_atom_transposed?
            smem_thr_copy_PdSt = utils.make_tiled_copy_A(
                smem_copy_atom_transposed, tiled_mma_dkv, swapAB=self.dKV_swapAB
            ).get_slice(tidx)
            smem_thr_copy_QdOt = utils.make_tiled_copy_B(
                smem_copy_atom_transposed, tiled_mma_dkv, swapAB=self.dKV_swapAB
            ).get_slice(tidx)
            smem_thr_copy_dS = utils.make_tiled_copy_A(
                smem_copy_atom, tiled_mma_dq, swapAB=self.dQ_swapAB
            ).get_slice(tidx)
            smem_thr_copy_Kt = utils.make_tiled_copy_B(
                smem_copy_atom_transposed, tiled_mma_dq, swapAB=self.dQ_swapAB
            ).get_slice(tidx)
            # TODO: what's the number of bits? What if SdP_swapAB
            r2s_thr_copy_PdS = cute.make_tiled_copy_C(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=2 * self.dtype.width
                ),
                tiled_mma_sdp,
            ).get_slice(tidx)

            tSsQ = smem_thr_copy_QdO.partition_S(sQ)
            tdPsdO = smem_thr_copy_QdO.partition_S(sdO)
            tSsK = smem_thr_copy_KV.partition_S(sK)
            tdPsV = smem_thr_copy_KV.partition_S(sV)
            tdVsPt = smem_thr_copy_PdSt.partition_S(sPt)
            tdKsdSt = smem_thr_copy_PdSt.partition_S(sdSt)
            tdVsdOt = smem_thr_copy_QdOt.partition_S(sdOt)
            tdKsQt = smem_thr_copy_QdOt.partition_S(sQt)
            tdQsdS = smem_thr_copy_dS.partition_S(sdS)
            tdQsKt = smem_thr_copy_Kt.partition_S(sKt)
            tPsP = r2s_thr_copy_PdS.partition_D(sP)
            tdSsdS = r2s_thr_copy_PdS.partition_D(sdS)

            # ///////////////////////////////////////////////////////////////////////////////
            # Predicate: Mark indices that need to copy when problem_shape isn't a multiple
            # of tile_shape
            # ///////////////////////////////////////////////////////////////////////////////
            # Construct identity layout for KV
            cQ = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
            tQcQ = gmem_thr_copy_QK.partition_S(cQ)
            t0QcQ = gmem_thr_copy_QK.get_slice(0).partition_S(cQ)
            if cutlass.const_expr(self.head_dim_padded == self.head_dim_v_padded):
                tdOcdO = tQcQ
                t0dOcdO = t0QcQ
            else:
                cdO = cute.make_identity_tensor((self.m_block_size, self.head_dim_v_padded))
                tdOcdO = gmem_thr_copy_VdO.partition_S(cdO)
                t0dOcdO = gmem_thr_copy_VdO.get_slice(0).partition_S(cdO)
            cLSE = cute.make_identity_tensor((self.m_block_size,))
            tLSEcLSE = gmem_thr_copy_lse.partition_S(cLSE)

            # Allocate predicate tensors for m and n, here we only allocate the tile of k, and
            # use "if" on the mn dimension.
            # This is to reduce register pressure and gets 2-3% performance gain.

            d_head = mQ.shape[cute.rank(mQ) - 1]
            d_head_v = mdO.shape[cute.rank(mdO) - 1]

            tQpQ = utils.predicate_k(tQcQ, limit=d_head)
            if cutlass.const_expr(self.same_hdim_kv):
                tdOpdO = tQpQ
            else:
                tdOpdO = utils.predicate_k(tdOcdO, limit=d_head_v)

            # group parameters for compute_one_m_block
            mma_params = SimpleNamespace(
                thr_mma_sdp=thr_mma_sdp, thr_mma_dkv=thr_mma_dkv, thr_mma_dq=thr_mma_dq,
                tSrQ=tSrQ, tSrK=tSrK, tdPrdO=tdPrdO, tdPrV=tdPrV,
                tdVrP=tdVrP, tdVrdO=tdVrdO, tdKrdS=tdKrdS, tdKrQ=tdKrQ,
                tdQrdS=tdQrdS, tdQrK=tdQrK,
                acc_dK=acc_dK, acc_dV=acc_dV,
            )
            smem_copy_params = SimpleNamespace(
                smem_thr_copy_QdO=smem_thr_copy_QdO,
                smem_thr_copy_KV=smem_thr_copy_KV,
                smem_thr_copy_PdSt=smem_thr_copy_PdSt,
                smem_thr_copy_QdOt=smem_thr_copy_QdOt,
                smem_thr_copy_dS=smem_thr_copy_dS,
                smem_thr_copy_Kt=smem_thr_copy_Kt,
                r2s_thr_copy_PdS=r2s_thr_copy_PdS,
                tSsQ=tSsQ, tSsK=tSsK, tdPsdO=tdPsdO, tdPsV=tdPsV,
                tSsLSEMma=tSsLSEMma, tSsdPsumMma=tSsdPsumMma,
                tPsP=tPsP, tdSsdS=tdSsdS,
                tdVsPt=tdVsPt, tdVsdOt=tdVsdOt, tdKsdSt=tdKsdSt, tdKsQt=tdKsQt,
                tdQsdS=tdQsdS, tdQsKt=tdQsKt,
            )
            gmem_copy_params = SimpleNamespace(
                gmem_thr_copy_dQaccum=gmem_thr_copy_dQaccum, tdQgdQaccum=tdQgdQaccum
            )
            load_Q_LSE = partial(
                self.load_Q_LSE, gmem_tiled_copy_QK, gmem_tiled_copy_LSE,
                tQgQ, tQsQ, tQcQ, t0QcQ, tQpQ,
                tLSEgLSE, tLSEsLSE, tLSEcLSE, seqlen=seqlen.seqlen_q
            )
            load_dO_dPsum = partial(
                self.load_dO_dPsum, gmem_tiled_copy_VdO, gmem_tiled_copy_LSE,
                tdOgdO, tdOsdO, tdOcdO, t0dOcdO, tdOpdO,
                tLSEgdPsum, tLSEsdPsum, tLSEcLSE, seqlen=seqlen.seqlen_q
            )
            compute_one_m_block = partial(
                self.compute_one_m_block, mma_params=mma_params,
                smem_copy_params=smem_copy_params, gmem_copy_params=gmem_copy_params,
                load_Q_LSE=load_Q_LSE, load_dO_dPsum=load_dO_dPsum,
                m_block_max=m_block_max,
                softmax_scale=softmax_scale,
                softmax_scale_log2=softmax_scale_log2,
                aux_data=aux_data,
            )

            if m_block_min < m_block_max:
                # ///////////////////////////////////////////////////////////////////////////////
                # Prologue
                # ///////////////////////////////////////////////////////////////////////////////
                # Start async loads of the last mn-tile, where we take care of the mn residue
                self.load_V(gmem_thr_copy_VdO, tVgV, tVsV, n_block, seqlen=seqlen.seqlen_k,
                            headdim=d_head_v)
                if cutlass.const_expr(self.V_in_regs):
                    cute.arch.cp_async_commit_group()
                self.load_K(gmem_thr_copy_QK, tKgK, tKsK, n_block, seqlen=seqlen.seqlen_k,
                            headdim=d_head)
                cute.arch.cp_async_commit_group()

                if cutlass.const_expr(self.V_in_regs):
                    cute.arch.cp_async_wait_group(1)
                    cute.arch.barrier()
                    tdPrV_copy_view = smem_thr_copy_KV.retile(tdPrV)
                    cute.copy(smem_thr_copy_KV, tdPsV, tdPrV_copy_view)
                    # Sync to avoid loading Q to smem_q, which overlaps with smem_v
                    cute.arch.barrier()

                m_block = m_block_min
                assert self.num_stages_Q >= self.num_stages_dO
                for stage in cutlass.range_constexpr(self.num_stages_Q):
                    if cutlass.const_expr(self.num_stages_Q == 1 or stage < self.num_stages_Q - 1):
                        if stage == 0 or m_block + stage < m_block_max:
                            load_Q_LSE(m_block + stage, smem_pipe_write_q=stage)
                        cute.arch.cp_async_commit_group()
                    if cutlass.const_expr(stage < self.num_stages_dO):
                        if stage == 0 or m_block + stage < m_block_max:
                            load_dO_dPsum(m_block + stage, smem_pipe_write_q=stage)
                        cute.arch.cp_async_commit_group()

                # ///////////////////////////////////////////////////////////////////////////////
                # Mainloop
                # ///////////////////////////////////////////////////////////////////////////////
                # Start processing of the first n-block.
                mask = AttentionMask(
                    self.m_block_size,
                    self.n_block_size,
                    seqlen,
                    window_size_left,
                    window_size_right,
                )
                mask_fn = partial(
                    mask.apply_mask, n_block=n_block, thr_mma=thr_mma_sdp,
                    batch_idx=batch_idx, head_idx=head_idx,
                    mask_seqlen=True, mask_causal=self.is_causal, mask_local=self.is_local
                )
                smem_pipe_read_q = cutlass.Int32(0)
                smem_pipe_read_do = cutlass.Int32(0)
                smem_pipe_write_q = cutlass.Int32(self.num_stages_Q - 1)
                smem_pipe_write_do = cutlass.Int32(0)
                for m_tile in cutlass.range(m_block_min, m_block_max, unroll=1):
                    compute_one_m_block(
                        m_tile, smem_pipe_read_q, smem_pipe_read_do, smem_pipe_write_q, smem_pipe_write_do,
                        mask_fn=mask_fn,
                    )
                    smem_pipe_read_q = self.advance_pipeline(smem_pipe_read_q, self.num_stages_Q)
                    smem_pipe_read_do = self.advance_pipeline(smem_pipe_read_do, self.num_stages_dO)
                    smem_pipe_write_q = self.advance_pipeline(smem_pipe_write_q, self.num_stages_Q)
                    smem_pipe_write_do = self.advance_pipeline(smem_pipe_write_do, self.num_stages_dO)

            # ///////////////////////////////////////////////////////////////////////////////
            # Epilogue
            # ///////////////////////////////////////////////////////////////////////////////
            # If GQA, we scale dK in the postprocessing kernel instead
            if cutlass.const_expr(self.qhead_per_kvhead == 1):
                acc_dK.store(acc_dK.load() * softmax_scale)
            # reuse sK and sV data iterator
            sdK = cute.make_tensor(sK.iterator, sK_layout)
            sdV = cute.make_tensor(sV.iterator, sV_layout)
            self.epilogue(
                acc_dK, acc_dV, mdK, mdV, sdK, sdV,
                gmem_tiled_copy_dK, gmem_tiled_copy_dV, tiled_mma_dkv,
                tidx, n_block, head_idx, batch_idx, seqlen, d_head, d_head_v
            )

    @cute.jit
    def compute_one_m_block(
        self,
        m_block: cutlass.Int32,
        smem_pipe_read_q: cutlass.Int32,
        smem_pipe_read_do: cutlass.Int32,
        smem_pipe_write_q: cutlass.Int32,
        smem_pipe_write_do: cutlass.Int32,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        gmem_copy_params: SimpleNamespace,
        load_Q_LSE: Callable,
        load_dO_dPsum: Callable,
        m_block_max: cutlass.Int32,
        softmax_scale: cutlass.Float32,
        softmax_scale_log2: cutlass.Float32,
        aux_data: AuxData = AuxData(),
        mask_fn: Optional[Callable] = None,
    ):
        def load_Q_next():
            m_block_next = m_block + (self.num_stages_Q - 1 if cutlass.const_expr(self.num_stages_Q > 1) else 1)
            if m_block_next < m_block_max:
                load_Q_LSE(m_block_next, smem_pipe_write_q)
            cute.arch.cp_async_commit_group()

        def load_dO_next():
            if m_block + self.num_stages_dO < m_block_max:
                load_dO_dPsum(m_block + self.num_stages_dO, smem_pipe_write_do)
            cute.arch.cp_async_commit_group()

        # MMA S
        acc_shape_SdP = mma_params.thr_mma_sdp.partition_shape_C(
            (self.m_block_size, self.n_block_size) if cutlass.const_expr(not self.SdP_swapAB) else (self.n_block_size, self.m_block_size)
        )
        acc_S = cute.make_rmem_tensor(acc_shape_SdP, cutlass.Float32)
        acc_S.fill(0.0)
        cute.arch.cp_async_wait_group(1 if cutlass.const_expr(self.num_stages_Q > 1) else 0)
        cute.arch.barrier()
        sm80_utils.gemm(
            mma_params.thr_mma_sdp, acc_S, mma_params.tSrQ, mma_params.tSrK,
            smem_copy_params.tSsQ[None, None, None, smem_pipe_read_q if cutlass.const_expr(self.num_stages_Q > 1) else 0],
            smem_copy_params.tSsK,
            smem_copy_params.smem_thr_copy_QdO, smem_copy_params.smem_thr_copy_KV,
            swap_AB=self.SdP_swapAB,
        )
        acc_S_pre = cute.make_fragment_like(acc_S)
        acc_S_pre.store(acc_S.load())
        tLSErLSE = cute.make_fragment_like(smem_copy_params.tSsLSEMma[None, 0])
        cute.autovec_copy(
            smem_copy_params.tSsLSEMma[None, smem_pipe_read_q if cutlass.const_expr(self.num_stages_Q > 1) else 0], tLSErLSE
        )
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
        acc_S_pre_mn = layout_utils.reshape_acc_to_mn(acc_S_pre)
        if cutlass.const_expr(self.score_mod is not None):
            for r in cutlass.range(cute.size(acc_S_mn, mode=[0]), unroll_full=True):
                acc_S_mn[r, None].store(
                    call_score_mod(
                        self.score_mod,
                        acc_S_mn[r, None].load() * softmax_scale,
                        0,
                        0,
                        0,
                        0,
                        None,
                        aux_data,
                    )
                )
        if cutlass.const_expr(mask_fn is not None):
            mask_fn(acc_S, m_block=m_block)
        bidx = 0
        # if cute.arch.thread_idx()[0] == 0 and cute.arch.block_idx()[0] == bidx: cute.print_tensor(acc_S_mn)
        # if cute.arch.thread_idx()[0] == 0 and cute.arch.block_idx()[0] == 1: cute.print_tensor(tLSErLSE)
        assert cute.size(acc_S_mn, mode=[0]) == cute.size(tLSErLSE)
        for r in cutlass.range(cute.size(acc_S_mn, mode=[0]), unroll_full=True):
            acc_S_mn[r, None].store(cute.math.exp2(acc_S_mn[r, None].load() * softmax_scale_log2 - tLSErLSE[r], fastmath=True))
        # if cute.arch.thread_idx()[0] == 0 and cute.arch.block_idx()[0] == bidx: cute.print_tensor(acc_S_mn)

        # MMA dP
        acc_dP = cute.make_rmem_tensor(acc_shape_SdP, cutlass.Float32)
        acc_dP.fill(0.0)
        cute.arch.cp_async_wait_group(1 if cutlass.const_expr(self.num_stages_dO > 1) else 0)
        cute.arch.barrier()
        sm80_utils.gemm(
            mma_params.thr_mma_sdp, acc_dP, mma_params.tdPrdO, mma_params.tdPrV,
            smem_copy_params.tdPsdO[None, None, None, smem_pipe_read_do if cutlass.const_expr(self.num_stages_dO > 1) else 0],
            smem_copy_params.tdPsV,
            smem_copy_params.smem_thr_copy_QdO, smem_copy_params.smem_thr_copy_KV,
            hook_fn=load_Q_next if cutlass.const_expr(self.num_stages_Q > 1) else None,
            swap_AB=self.SdP_swapAB,
        )
        tLSErdPsum = cute.make_fragment_like(smem_copy_params.tSsdPsumMma[None, 0])
        cute.autovec_copy(
            smem_copy_params.tSsdPsumMma[None, smem_pipe_read_do if cutlass.const_expr(self.num_stages_dO > 1) else 0], tLSErdPsum
        )
        acc_dP_mn = layout_utils.reshape_acc_to_mn(acc_dP)
        # if cute.arch.thread_idx()[0] == 0 and cute.arch.block_idx()[0] == bidx: cute.print_tensor(acc_dP_mn)
        assert cute.size(acc_dP_mn, mode=[0]) == cute.size(tLSErdPsum)
        for r in cutlass.range(cute.size(acc_dP_mn, mode=[0]), unroll_full=True):
            grad_val = acc_S_mn[r, None].load() * (acc_dP_mn[r, None].load() - tLSErdPsum[r])
            if cutlass.const_expr(self.score_mod_bwd is not None):
                grad_val = call_score_mod_bwd(
                    self.score_mod_bwd,
                    grad_val,
                    acc_S_pre_mn[r, None].load() * softmax_scale,
                    0,
                    0,
                    0,
                    0,
                    None,
                    aux_data,
                )
            acc_dP_mn[r, None].store(grad_val)
        # if cute.arch.thread_idx()[0] == 0 and cute.arch.block_idx()[0] == bidx: cute.print_tensor(acc_dP_mn)
        rP = cute.make_fragment_like(acc_S, self.dtype)
        rP.store(acc_S.load().to(self.dtype))
        if cutlass.const_expr(not self.Mma_dKV_is_RS):
            tPrP = smem_copy_params.r2s_thr_copy_PdS.retile(rP)  # ((Atom,AtomNum), MMA_N, MMA_N)
            cute.copy(smem_copy_params.r2s_thr_copy_PdS, tPrP, smem_copy_params.tPsP)
        rdS = cute.make_fragment_like(acc_dP, self.dtype)
        rdS.store(acc_dP.load().to(self.dtype))
        if cutlass.const_expr(not self.Mma_dKV_is_RS):
            cute.arch.barrier()  # Make sure P is written
        # For hdim 64, It's faster to write to smem_dS first before the dV gemm
        if cutlass.const_expr(not self.Mma_dKV_is_RS):
            tdSrdS = smem_copy_params.r2s_thr_copy_PdS.retile(rdS)
            cute.copy(smem_copy_params.r2s_thr_copy_PdS, tdSrdS, smem_copy_params.tdSsdS)
        if cutlass.const_expr(self.Mma_dKV_is_RS):
            tdVrP = layout_utils.reshape_acc_to_frgA(rP)
        else:
            tdVrP = mma_params.tdVrP

        # MMA dK
        sm80_utils.gemm(
            mma_params.thr_mma_dkv, mma_params.acc_dV, tdVrP, mma_params.tdVrdO,
            smem_copy_params.tdVsPt,
            smem_copy_params.tdVsdOt[None, None, None, smem_pipe_read_do if cutlass.const_expr(self.num_stages_dO > 1) else 0],
            smem_copy_params.smem_thr_copy_PdSt, smem_copy_params.smem_thr_copy_QdOt,
            A_in_regs=self.Mma_dKV_is_RS,
            swap_AB=self.dKV_swapAB,
        )
        # if cute.arch.thread_idx()[0] == 0 and cute.arch.block_idx()[0] == bidx: cute.print_tensor(mma_params.acc_dV)
        cute.arch.barrier()  # Make sure dS is written

        # MMA dQ
        def dQ_mma(hook_fn):
            acc_shape_dQ = mma_params.thr_mma_dq.partition_shape_C(
                (self.m_block_size, self.head_dim_padded) if cutlass.const_expr(not self.dQ_swapAB) else (self.head_dim_padded, self.m_block_size)
            )
            acc_dQ = cute.make_rmem_tensor(acc_shape_dQ, cutlass.Float32)
            acc_dQ.fill(0.0)
            sm80_utils.gemm(
                mma_params.thr_mma_dq, acc_dQ, mma_params.tdQrdS, mma_params.tdQrK,
                smem_copy_params.tdQsdS, smem_copy_params.tdQsKt,
                smem_copy_params.smem_thr_copy_dS, smem_copy_params.smem_thr_copy_Kt,
                swap_AB=self.dQ_swapAB,
                hook_fn=hook_fn
            )
            # ((1, 1), num_elements)
            acc_dQ_atomic = gmem_copy_params.gmem_thr_copy_dQaccum.retile(acc_dQ)
            tdQgdQaccum_atomic = gmem_copy_params.tdQgdQaccum[None, None, m_block]
            assert cute.size(acc_dQ_atomic) == cute.size(tdQgdQaccum_atomic)
            for i in cutlass.range(cute.size(acc_dQ_atomic), unroll_full=True):
                utils.atomic_add_fp32(acc_dQ_atomic[i], utils.elem_pointer(tdQgdQaccum_atomic, i))
                # utils.atomic_add_fp32(acc_dQ[i], tdQgdQaccum_atomic.iterator + i * tdQgdQaccum_atomic.stride[1])
            # if cute.arch.thread_idx()[0] == 64 and cute.arch.block_idx()[0] == bidx: cute.print_tensor(acc_dQ)

        # If num_stages_Q == 1, we want to do Mma_dK first so we can start loading Q for the next iteration
        if cutlass.const_expr(self.num_stages_Q > 1):
            dQ_mma(load_dO_next)

        # MMA dK
        if cutlass.const_expr(self.Mma_dKV_is_RS):
            tdKrdS = layout_utils.reshape_acc_to_frgA(rdS)
        else:
            tdKrdS = mma_params.tdKrdS
        sm80_utils.gemm(
            mma_params.thr_mma_dkv, mma_params.acc_dK, tdKrdS, mma_params.tdKrQ,
            smem_copy_params.tdKsdSt,
            smem_copy_params.tdKsQt[None, None, None, smem_pipe_read_q if cutlass.const_expr(self.num_stages_Q > 1) else 0],
            smem_copy_params.smem_thr_copy_PdSt, smem_copy_params.smem_thr_copy_QdOt,
            A_in_regs=self.Mma_dKV_is_RS,
            swap_AB=self.dKV_swapAB,
            hook_fn=load_dO_next if cutlass.const_expr(self.num_stages_Q == 1) else None,
        )
        # if cute.arch.thread_idx()[0] == 0: cute.print_tensor(mma_params.acc_dK)
        if cutlass.const_expr(self.num_stages_Q == 1):
            cute.arch.barrier()
            dQ_mma(load_Q_next)

    @cute.jit
    def epilogue(
        self,
        acc_dK: cute.Tensor,
        acc_dV: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        sdK: cute.Tensor,
        sdV: cute.Tensor,
        gmem_tiled_copy_dK: cute.TiledCopy,
        gmem_tiled_copy_dV: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        tidx: cutlass.Int32,
        n_block: cutlass.Int32,
        num_head: cutlass.Int32,
        batch_size: cutlass.Int32,
        seqlen: SeqlenInfoQK,
        d_head: cutlass.Int32,
        d_head_v: cutlass.Int32
    ):
        rdV = cute.make_fragment_like(acc_dV, self.dtype)
        rdV.store(acc_dV.load().to(self.dtype))
        rdK = cute.make_fragment_like(acc_dK, self.dtype)
        rdK.store(acc_dK.load().to(self.dtype))
        gmem_thr_copy_dK = gmem_tiled_copy_dK.get_slice(tidx)
        gmem_thr_copy_dV = gmem_tiled_copy_dV.get_slice(tidx)

        batch_idx = batch_size
        head_idx_kv = num_head // self.qhead_per_kvhead if cutlass.const_expr(not self.pack_gqa) else num_head

        if cutlass.const_expr(self.qhead_per_kvhead == 1):
            # Make sure all threads have finished reading K and V, otherwise we get racy dQ
            # because smem_q could be changed.
            cute.arch.barrier()
            # smem copy atom for dKV
            smem_copy_atom_dKV = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=2 * self.dtype.width
            )
            smem_thr_copy_dKV = cute.make_tiled_copy_C(smem_copy_atom_dKV, tiled_mma).get_slice(tidx)
            taccdVrdV = smem_thr_copy_dKV.retile(rdV)
            taccdKrdK = smem_thr_copy_dKV.retile(rdK)
            taccdVsdV = smem_thr_copy_dKV.partition_D(sdV)
            taccdKsdK = smem_thr_copy_dKV.partition_D(sdK)
            # copy acc O from rmem to smem with the smem copy atom
            cute.copy(smem_copy_atom_dKV, taccdVrdV, taccdVsdV)
            cute.copy(smem_copy_atom_dKV, taccdKrdK, taccdKsdK)


            if cutlass.const_expr(not seqlen.has_cu_seqlens_k):
                mdK_cur, mdV_cur = [t[batch_idx, None, head_idx_kv, None] for t in (mdK, mdV)]
            else:
                mdK_cur, mdV_cur = [cute.domain_offset((seqlen.offset_k, 0), t[None, head_idx_kv, None]) for t in (mdK, mdV)]

            blkdK_shape = (self.n_block_size, self.head_dim_padded)
            blkdV_shape = (self.n_block_size, self.head_dim_v_padded)
            gdK = cute.local_tile(mdK_cur, blkdK_shape, (n_block, 0))
            gdV = cute.local_tile(mdV_cur, blkdV_shape, (n_block, 0))
            tdKsdK = gmem_thr_copy_dK.partition_S(sdK)
            tdKgdK = gmem_thr_copy_dK.partition_D(gdK)
            tdVsdV = gmem_thr_copy_dV.partition_S(sdV)
            tdVgdV = gmem_thr_copy_dV.partition_D(gdV)
            tdKrdK = cute.make_fragment_like(tdKgdK, self.dtype)
            tdVrdV = cute.make_fragment_like(tdVgdV, self.dtype)
            # sync before all smem stores are done.
            cute.arch.barrier()
            # load acc dK and dV from smem to rmem for wider vectorization
            # Need to check OOB when reading from smem if kBlockN isn't evenly tiled
            # TODO
            cute.autovec_copy(tdKsdK, tdKrdK)
            cute.autovec_copy(tdVsdV, tdVrdV)

            cdK = cute.make_identity_tensor((self.n_block_size, self.head_dim_padded))
            tdKcdK = gmem_thr_copy_dK.partition_S(cdK)
            t0dKcdK = gmem_tiled_copy_dK.get_slice(0).partition_S(cdK)
            if cutlass.const_expr(self.head_dim_padded == self.head_dim_v_padded):
                tdVcdV = tdKcdK
                t0dVcdV = t0dKcdK
            else:
                cdV = cute.make_identity_tensor((self.n_block_size, self.head_dim_v_padded))
                tdVcdV = gmem_thr_copy_dV.partition_S(cdV)
                t0dVcdV = gmem_tiled_copy_dV.get_slice(0).partition_S(cdV)
            tdKpdK = utils.predicate_k(tdKcdK, limit=d_head)
            if cutlass.const_expr(self.same_hdim_kv):
                tdVpdV = tdKpdK
            else:
                tdVpdV = utils.predicate_k(tdVcdV, limit=d_head_v)
            # copy acc dK and acc_dV from rmem to gmem
            for rest_m in cutlass.range_constexpr(cute.size(tdKrdK.shape[1])):
                if t0dKcdK[0, rest_m, 0][0] < seqlen.seqlen_k - n_block * self.n_block_size - tdKcdK[0][0]:
                    cute.copy(
                        gmem_tiled_copy_dK,
                        tdKrdK[None, rest_m, None],
                        tdKgdK[None, rest_m, None],
                        pred=tdKpdK[None, rest_m, None] if cutlass.const_expr(self.check_hdim_oob) else None,
                    )
            for rest_m in cutlass.range_constexpr(cute.size(tdVrdV.shape[1])):
                if t0dVcdV[0, rest_m, 0][0] < seqlen.seqlen_k - n_block * self.n_block_size - tdVcdV[0][0]:
                    cute.copy(
                        gmem_tiled_copy_dV,
                        tdVrdV[None, rest_m, None],
                        tdVgdV[None, rest_m, None],
                        pred=tdVpdV[None, rest_m, None] if cutlass.const_expr(self.check_hdim_v_oob) else None,
                    )

        else:  # qhead_per_kvhead > 1, do atomic add
            # For Sm90, we need to sync to avoid racy writes to smem_q
            # For Sm80, we don't need to sync since we're not touching smem
            head_idx_kv = num_head // self.qhead_per_kvhead if cutlass.const_expr(not self.pack_gqa) else num_head

            if cutlass.const_expr(not seqlen.has_cu_seqlens_k):
                mdK_cur, mdV_cur = [t[batch_idx, head_idx_kv, None] for t in (mdK, mdV)]
            else:
                padded_offset_k = seqlen.offset_k + batch_idx * self.n_block_size
                mdK_cur = cute.domain_offset((padded_offset_k * self.head_dim_padded,), mdK[head_idx_kv, None])
                mdV_cur = cute.domain_offset((padded_offset_k * self.head_dim_v_padded,), mdV[head_idx_kv, None])

            gdV = cute.local_tile(mdV_cur, (self.n_block_size * self.head_dim_v_padded,), (n_block,))
            gdK = cute.local_tile(mdK_cur, (self.n_block_size * self.head_dim_padded,), (n_block,))
            tdVgdVaccum = gmem_thr_copy_dV.partition_S(gdV)
            tdKgdKaccum = gmem_thr_copy_dK.partition_S(gdK)
            acc_dV_atomic = gmem_thr_copy_dV.retile(acc_dV)
            acc_dK_atomic = gmem_thr_copy_dK.retile(acc_dK)
            assert cute.size(acc_dV_atomic) == cute.size(tdVgdVaccum)
            assert cute.size(acc_dK_atomic) == cute.size(tdKgdKaccum)
            for i in cutlass.range(cute.size(acc_dV_atomic), unroll_full=True):
                utils.atomic_add_fp32(acc_dV_atomic[i], utils.elem_pointer(tdVgdVaccum, i))
            for i in cutlass.range(cute.size(acc_dK_atomic), unroll_full=True):
                utils.atomic_add_fp32(acc_dK_atomic[i], utils.elem_pointer(tdKgdKaccum, i))

    @cute.jit
    def advance_pipeline(self, pipeline_index, num_stages: cutlass.Constexpr):
        return pipeline_index + 1 if pipeline_index < num_stages - 1 else 0

    @cute.jit
    def load_K(
        self,
        gmem_thr_copy: cute.TiledCopy,
        tKgK: cute.Tensor,
        tKsK: cute.Tensor,
        block: cutlass.Int32,
        seqlen: cutlass.Int32,
        headdim: cutlass.Int32,
    ):
        cK = cute.make_identity_tensor((self.n_block_size, self.head_dim_padded))
        tKcK = gmem_thr_copy.partition_S(cK)
        t0KcK = gmem_thr_copy.get_slice(0).partition_S(cK)
        tKpK = utils.predicate_k(tKcK, limit=headdim)
        for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
            # If kBlockN doesn't evenly divide the tiled copy, only the last `n` needs to be checked
            if self.is_even_n_smem_k or n < cute.size(tKsK.shape[1]) - 1 or tKcK[0, n, 0][0] < self.n_block_size:
                # Instead of using tKcK, we using t0KcK and subtract the offset from the limit
                # (seqlen - block * kBlockN). This is because the entries of t0KcK are known at compile time.
                predicate_n = t0KcK[0, n, 0][0] < seqlen - block * self.n_block_size - tKcK[0][0]
                predicate = cute.make_fragment_like(tKpK[None, 0, None])
                for k in cutlass.range_constexpr(cute.size(predicate.shape[1])):
                    for i in cutlass.range_constexpr(cute.size(predicate.shape[0])):
                        predicate[i, k] = (tKpK[i, n, k] if cutlass.const_expr(self.check_hdim_oob) else True) and predicate_n
                cute.copy(
                    gmem_thr_copy, tKgK[None, n, None], tKsK[None, n, None], pred=predicate,
                )
            # We need to clear the sK smem tiles since we'll use sKt for mma_dq

    @cute.jit
    def load_V(
        self,
        gmem_thr_copy: cute.TiledCopy,
        tVgV: cute.Tensor,
        tVsV: cute.Tensor,
        block: cutlass.Int32,
        seqlen: cutlass.Int32,
        headdim: cutlass.Int32,
    ):
        cV = cute.make_identity_tensor((self.n_block_size, self.head_dim_v_padded))
        tVcV = gmem_thr_copy.partition_S(cV)
        t0VcV = gmem_thr_copy.get_slice(0).partition_S(cV)
        tVpV = utils.predicate_k(tVcV, limit=headdim)
        for n in cutlass.range_constexpr(cute.size(tVsV.shape[1])):
            # If kBlockN doesn't evenly divide the tiled copy, only the last `n` needs to be checked
            if self.is_even_n_smem_v or n < cute.size(tVsV.shape[1]) - 1 or tVcV[0, n, 0][0] < self.n_block_size:
                # Instead of using tVcV, we using t0VcV and subtract the offset from the limit
                # (seqlen - block * kBlockN). This is because the entries of t0VcV are known at compile time.
                predicate_n = t0VcV[0, n, 0][0] < seqlen - block * self.n_block_size - tVcV[0][0]
                predicate = cute.make_fragment_like(tVpV[None, 0, None])
                for k in cutlass.range_constexpr(cute.size(predicate.shape[1])):
                    for i in cutlass.range_constexpr(cute.size(predicate.shape[0])):
                        predicate[i, k] = (tVpV[i, n, k] if cutlass.const_expr(self.check_hdim_oob) else True) and predicate_n
                cute.copy(
                    gmem_thr_copy, tVgV[None, n, None], tVsV[None, n, None], pred=predicate,
                )

    @cute.jit
    def load_Q_LSE(
        self,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_LSE: cute.TiledCopy,
        tQgQ: cute.Tensor,
        tQsQ: cute.Tensor,
        tQcQ: cute.Tensor,
        t0QcQ: cute.Tensor,
        tQpQ: cute.Tensor,
        tLSEgLSE: cute.Tensor,
        tLSEsLSE: cute.Tensor,
        tLSEcLSE: cute.Tensor,
        block: cutlass.Int32,
        smem_pipe_write_q: cutlass.Int32,
        seqlen: cutlass.Int32,
    ):
        for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
            # If kBlockM doesn't evenly divide the tiled copy, only the last `m` needs to be checked
            if self.is_even_m_smem_q or m < cute.size(tQsQ.shape[1]) - 1 or tQcQ[0, m, 0][0] < self.m_block_size:
                # Instead of using tQcQ, we using t0QcQ and subtract the offset from the limit
                # (seqlen - block * kBlockM). This is because the entries of t0QcQ are known at compile time.
                predicate_m = t0QcQ[0, m, 0][0] < seqlen - block * self.m_block_size - tQcQ[0][0]
                predicate = cute.make_fragment_like(tQpQ[None, 0, None])
                for k in cutlass.range_constexpr(cute.size(predicate.shape[1])):
                    for i in cutlass.range_constexpr(cute.size(predicate.shape[0])):
                        predicate[i, k] = (tQpQ[i, m, k] if cutlass.const_expr(self.check_hdim_oob) else True) and predicate_m
                cute.copy(
                    gmem_tiled_copy_Q,
                    tQgQ[None, m, None, block],
                    tQsQ[None, m, None, smem_pipe_write_q if cutlass.const_expr(self.num_stages_Q) > 1 else 0],
                    pred=predicate,
                )
            # We need to clear the sQ smem tiles since we'll use sQt for mma_dK
        # We made sure LSE length is padded so we read `kBlockM` elements so that all
        # elements in sLSE are filled. Without this we might have uninitialized sLSE values.
        for m in cutlass.range_constexpr(cute.size(tLSEsLSE.shape[1])):
            if tLSEcLSE[0, m][0] < self.m_block_size:
                cute.copy(
                    gmem_tiled_copy_LSE,
                    tLSEgLSE[None, m, block],
                    tLSEsLSE[None, m, smem_pipe_write_q if cutlass.const_expr(self.num_stages_Q > 1) else 0],
                )

    @cute.jit
    def load_dO_dPsum(
        self,
        gmem_tiled_copy_dO: cute.TiledCopy,
        gmem_tiled_copy_dPsum: cute.TiledCopy,
        tdOgdO: cute.Tensor,
        tdOsdO: cute.Tensor,
        tdOcdO: cute.Tensor,
        t0dOcdO: cute.Tensor,
        tdOpdO: cute.Tensor,
        tdPsumgdPsum: cute.Tensor,
        tdPsumsdPsum: cute.Tensor,
        tdPsumcdPsum: cute.Tensor,
        block: cutlass.Int32,
        smem_pipe_write_q: cutlass.Int32,
        seqlen: cutlass.Int32,
    ):
        for m in cutlass.range_constexpr(cute.size(tdOsdO.shape[1])):
            # If kBlockM doesn't evenly divide the tiled copy, only the last `m` needs to be checked
            if self.is_even_m_smem_do or m < cute.size(tdOsdO.shape[1]) - 1 or tdOcdO[0, m, 0][0] < self.m_block_size:
                # Instead of using tdOcdO, we using t0dOcdO and subtract the offset from the limit
                # (seqlen - block * kBlockM). This is because the entries of t0dOcdO are known at compile time.
                predicate_m = t0dOcdO[0, m, 0][0] < seqlen - block * self.m_block_size - tdOcdO[0][0]
                predicate = cute.make_fragment_like(tdOpdO[None, 0, None])
                for k in cutlass.range_constexpr(cute.size(predicate.shape[1])):
                    for i in cutlass.range_constexpr(cute.size(predicate.shape[0])):
                        predicate[i, k] = (tdOpdO[i, m, k] if cutlass.const_expr(self.check_hdim_oob) else True) and predicate_m
                cute.copy(
                    gmem_tiled_copy_dO,
                    tdOgdO[None, m, None, block],
                    tdOsdO[None, m, None, smem_pipe_write_q if cutlass.const_expr(self.num_stages_dO > 1) else 0],
                    pred=predicate,
                )
            # We need to clear the sQ smem tiles since we'll use sQt for mma_dK
        # We made sure LSE length is padded so we read `kBlockM` elements so that all
        # elements in sLSE are filled. Without this we might have uninitialized sLSE values.
        for m in cutlass.range_constexpr(cute.size(tdPsumgdPsum.shape[1])):
            if tdPsumcdPsum[0, m][0] < self.m_block_size:
                cute.copy(
                    gmem_tiled_copy_dPsum,
                    tdPsumgdPsum[None, m, block],
                    tdPsumsdPsum[None, m, smem_pipe_write_q if cutlass.const_expr(self.num_stages_dO > 1) else 0],
                )
