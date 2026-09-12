# Copyright (c) 2026, Tri Dao.

import enum
import math
from dataclasses import dataclass
from typing import Callable, Dict, Literal, Optional, Sequence, Tuple, Type

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass import Boolean, Int32, const_expr
from cutlass.cute.nvgpu import cpasync
from cutlass.utils import LayoutEnum

import torch._vendor.quack.copy_utils as copy_utils
import torch._vendor.quack.layout_utils as layout_utils
import torch._vendor.quack.utils as utils
from torch._vendor.quack import pipeline_checks
from torch._vendor.quack.cute_dsl_utils import ParamsBase
from torch._vendor.quack.epilogue.ops import DStore, EpiSmemBytes, TileLoad, TileStore, VecReduce
from torch._vendor.quack.gemm_config import SplitKMode
from torch._vendor.quack.pipeline import PipelineTmaAsync, PipelineTmaCpAsync
from torch._vendor.quack.sync import Semaphore
from torch._vendor.quack.tile_scheduler import (
    PersistenceMode,
    TileScheduler,
    TileSchedulerArguments,
    VarlenMTileScheduler,
    VarlenMTileSchedulerArguments,
)
from torch._vendor.quack.varlen_utils import VarlenManager


class NamedBarrierGemm(enum.IntEnum):
    Epilogue = enum.auto()  # starts from 1 as barrier 0 is reserved for sync_threads()
    # For mainloop load warps to signal that the epilogue load warp can start.
    # This is to avoid loading C too early, interfering with loading A and B.
    EpilogueLoad = enum.auto()
    MmaWG0 = enum.auto()
    MmaWG1 = enum.auto()
    EpiWG0 = enum.auto()
    EpiWG1 = enum.auto()
    TmemPtr = enum.auto()
    # CLC-multicast throttle: CTA0 load warp arrives once per tile started,
    # CTA0 scheduler warp syncs once per CLC query (2 warps, 64 threads).
    ClcThrottle = enum.auto()


def reinterpret_packed_fp6(mT, dtype):
    """View a (mn, 3k/4[, l]) Uint8 tensor as a (mn, k[, l]) packed-fp6 tensor.

    The byte tensor holds a little-endian 6-bit stream along K (element i at
    bits [6i, 6i+6) of its row), so the K mode (mode 1, stride 1 - sub-byte
    operands are K-major) scales by 8/6 in extent and keeps stride 1. The K
    EXTENT conversion is exact: K % 128 makes every row whole 96-byte groups.

    The non-K STRIDE conversion (x4//3, floor) is NOT exact for byte pitches
    that aren't multiples of 3 (e.g. a padded 416 B row pitch -> 554.67 fp6
    elements, floored to 554 = 415.5 B). This is safe because the FFI arg
    spec admits only 32 B-aligned pitches and the tensormap encode rounds the
    element stride back to the nearest TMA granule, recovering the true pitch
    exactly: the residue is < 0.75 B either way (verified bit-exact with
    poisoned padding at pitches 416/448/544/4128, both floor and ceil -
    AI/probe_fp6_pitch.py). The 32 B stride validation is load-bearing for
    correctness here, not just for the tensormap's alignment rule.
    """
    shape = tuple(s * 4 // 3 if i == 1 else s for i, s in enumerate(mT.shape))
    stride = tuple(st if i == 1 else st * 4 // 3 for i, st in enumerate(mT.stride))
    ptr = cute.recast_ptr(mT.iterator, dtype=dtype)
    return cute.make_tensor(ptr, cute.make_layout(shape, stride=stride))


class GemmBase:
    """Common non-mainloop pieces shared by GEMM architectures."""

    arch = 0
    # Epilogue mixins that need a reduction over the full accumulator tile
    # BEFORE any subtile is stored (e.g. QK-norm's per-head sum of squares)
    # set this in epi_to_underlying_arguments. The epilogue then runs a
    # prepass over all epi subtiles (tmem loads only, accumulator NOT
    # released) calling epi_prepass_subtile / epi_prepass_end, and the store
    # pass re-reads the accumulator. SM100-only (repeatable tmem loads).
    epi_needs_acc_prepass = False
    # Split-K along the contraction dim. Constexpr (lives on self): split_k == 1 compiles to
    # exactly the non-split kernel. The epilogue (the full epi mixin: alpha, beta*C, bias,
    # activations, aux outputs) runs exactly ONCE per output tile, on the entity that owns
    # the completed f32 sum — CUTLASS-3.x stream-K fixup semantics. Non-finalizing splits
    # run no epilogue at all: they dump raw f32 accumulator fragments into a per-tile
    # workspace region (split_k_partial_commit) and bump the tile's completion flag.
    # SERIAL: commits are turnstile-ordered by split index (bitwise deterministic); the
    #   last split waits flag == S-1, folds the workspace into its accumulator, and runs
    #   the full epilogue.
    # PARALLEL: commits are release-counted in arrival order, no waiting (lowest latency,
    #   NOT deterministic); the last split finalizes identically.
    # SEPARATE: every split runs the (op-less) epilogue storing raw f32 partials to its own
    #   workspace slice; a separate reduction kernel (quack/split_k_reduce.py) sums them
    #   and applies the full epilogue math.
    split_k = 1
    split_k_mode = SplitKMode.SERIAL
    # mB arrives (l, k, n) and is transposed to (n, k, l) at trace time when set
    # (see rotate_batch_last); compile_gemm_kernel sets these per compiled
    # variant. All three are trace-time relabels replacing per-call host
    # views: b_transposed for B crossing as (l, k, n); a_transposed for the
    # swap-at-trace kernel-A slot (caller B crossing as (l, k, m_kernel));
    # cd_transposed for swap-at-trace D/C/tile-epi tensors crossing in caller
    # (n_kernel, m_kernel) orientation.
    b_transposed = False
    a_transposed = False
    cd_transposed = False
    # dgated packed-native: D/C cross the FFI boundary in their RAW 16-bit
    # dtype with 2 lanes packed per f32 element; rotate_batch_last recasts
    # them to the f32 view the epilogue expects. "n": lanes pack along the
    # contiguous N dim; "m": along the contiguous M dim (the AB-swapped-caller
    # layout). Replaces the per-call torch .view(float32) host views.
    cd_packed = None

    def _recast_packed_cd(self, mT):
        """Trace-time f32 view of a kernel-order 16-bit packed tensor: halve
        the packed extent and every dynamic stride (element counts halve when
        re-typed 32-bit; the contiguous dim keeps its static stride 1). Host
        validation guarantees evenness; the divby assume preserves the 16 B
        alignment knowledge the old f32 host views carried (raw strides are
        8-element-divisible fakes, so halves are 4-divisible)."""
        packed_dim = 1 if self.cd_packed == "n" else 0
        shape = tuple(s // 2 if i == packed_dim else s for i, s in enumerate(mT.shape))
        # Halve every stride except the packed dim's unit stride. Static
        # strides halve too (a fully-static tensor, e.g. a direct-compiled
        # prototype, carries its real row stride statically); only the
        # interface's dynamic fakes hit the assume branch.
        stride = tuple(
            s
            if const_expr(i == packed_dim)
            else (s // 2 if const_expr(cute.is_static(s)) else cute.assume(s // 2, divby=4))
            for i, s in enumerate(mT.stride)
        )
        return cute.make_tensor(
            cute.recast_ptr(mT.iterator, dtype=cutlass.Float32),
            cute.make_layout(shape, stride=stride),
        )

    def rotate_batch_last(self, mA, mB, mD, mC, epilogue_args, append_batch_if_2d=False):
        """Rotate all batched inputs from caller order (l, x, y) to kernel order (x, y, l).

        Batched tensors cross the FFI boundary in caller order; __call__ rotates
        them at trace time via this method. That replaces per-call torch
        .permute() host views (~0.7us each) with a free compile-time layout
        rewrite, so hosts pass torch tensors as-is. Fake tensors must be built
        batch-first to match (see gemm_tvm_ffi_utils.fake_batched).

        ``append_batch_if_2d`` (dense calls only, i.e. no varlen args): rank-2
        operands are unbatched (m, k) etc. and get a static size-1 stride-0
        batch mode appended, so hosts can pass 2D tensors without per-call
        .unsqueeze() views. Varlen calls must leave it False — their rank-2
        operands are flattened, not unbatched.

        ``self.b_transposed`` (dense only): mB crossed the boundary in the
        caller's (k, n[, l]) orientation and is transposed to kernel order
        (n, k, l) here, saving the host a per-call .mT view.
        """
        mA, mB, mD, mC = (self.permute_batch_last(t, append_batch_if_2d) for t in (mA, mB, mD, mC))
        if const_expr(self.a_transposed):
            mA = layout_utils.select(mA, [1, 0, 2])
        if const_expr(self.b_transposed):
            mB = layout_utils.select(mB, [1, 0, 2])
        if const_expr(self.cd_transposed):
            if const_expr(mD is not None):
                mD = layout_utils.select(mD, [1, 0, 2])
            if const_expr(mC is not None):
                mC = layout_utils.select(mC, [1, 0, 2])
        if const_expr(self.cd_packed is not None):
            if const_expr(mD is not None):
                mD = self._recast_packed_cd(mD)
            if const_expr(mC is not None):
                mC = self._recast_packed_cd(mC)
        if const_expr(mA is not None and mB is not None):
            kernel_m, kernel_n = cute.size(mA, mode=[0]), cute.size(mB, mode=[0])
            self.caller_m, self.caller_n = (
                (kernel_n, kernel_m) if const_expr(self.a_transposed) else (kernel_m, kernel_n)
            )
        return mA, mB, mD, mC, self.permute_batch_last_epi_args(epilogue_args, append_batch_if_2d)

    def permute_batch_last(
        self, mT: Optional[cute.Tensor], append_batch_if_2d=False
    ) -> Optional[cute.Tensor]:
        """Trace-time (l, x, y) -> (x, y, l) permute of a batched tensor.

        Rank-2 tensors (the varlen flattened operands, which are never
        batch-permuted — or dense 2D operands when ``append_batch_if_2d``,
        which get a trivial batch mode appended instead) and None pass through.
        """
        if const_expr(mT is not None and cute.rank(mT) == 3):
            return layout_utils.select(mT, [1, 2, 0])
        if const_expr(mT is not None and append_batch_if_2d and cute.rank(mT) == 2):
            return layout_utils.expand(mT, 2, 1)
        return mT

    def permute_batch_last_epi_args(self, epilogue_args, append_batch_if_2d=False):
        """Rotate the tile-shaped epilogue tensors from (l, m, n) to (m, n, l).

        Exactly the TileStore/TileLoad fields of ``_epi_ops`` are GEMM-tile
        shaped and consumed in kernel order (m, n, l) — PostAct/PreAct/aux
        outputs and tile loads. Everything else keeps its host layout: vec
        broadcasts are rank 1/2, and reduce outputs (e.g. a (l, m, n_tiles)
        mColVecReduce) are batch-first natively. Rank-2 tile fields pass
        through like the main operands (varlen-flattened), or — for dense
        ``append_batch_if_2d`` calls — get the trivial batch mode appended.
        Unbatched rank-2 VecReduce outputs get it PREPENDED (they are
        batch-first, and their rank-2 form otherwise means varlen).
        """
        if const_expr(epilogue_args is None):
            return epilogue_args
        epi_ops = getattr(self, "_epi_ops", ())
        tile_fields = {op.name for op in epi_ops if isinstance(op, (TileLoad, TileStore))}
        reduce_fields = {op.name for op in epi_ops if isinstance(op, VecReduce)}
        rotated = {}
        for name, v in zip(epilogue_args._fields, epilogue_args):
            if not isinstance(v, cute.Tensor):
                continue
            if name in tile_fields:
                if cute.rank(v) == 3:
                    rotated[name] = layout_utils.select(v, [1, 2, 0])
                elif append_batch_if_2d and cute.rank(v) == 2:
                    rotated[name] = layout_utils.expand(v, 2, 1)
                if const_expr(self.cd_transposed) and name in rotated:
                    rotated[name] = layout_utils.select(rotated[name], [1, 0, 2])
            elif name in reduce_fields and append_batch_if_2d and cute.rank(v) == 2:
                rotated[name] = layout_utils.expand(v, 0, 1)
        return epilogue_args._replace(**rotated) if rotated else epilogue_args

    @dataclass
    class EpilogueArguments:
        pass

    EpilogueParams = ParamsBase

    def epi_smem_warp_shape_mnk(self):
        return (self.num_epi_warps, 1, 1)

    def epi_r2s_pair_xor(self) -> bool:
        """Whether the D r2s copy uses the conflict-free pair-XOR STS.32 split
        (see copy_utils.cvt_copy_pair_xor_sts32) instead of vectorized STS.64.
        Only valid where the acc fragment holds contiguous column pairs and the
        epi smem swizzle makes the vectorized store 2-way conflicted."""
        return False

    def _init_split_k(self, split_k: int, split_k_mode: int):
        """Validate and store the constexpr split-K configuration. Call after self.gather_A."""
        assert split_k >= 1, "split_k must be >= 1"
        assert split_k_mode in tuple(SplitKMode), f"invalid split_k_mode: {split_k_mode}"
        self.split_k = split_k
        self.split_k_mode = SplitKMode(split_k_mode)
        if split_k > 1:
            assert not self.gather_A, "split_k does not support gather_A"

    @cute.jit
    def epilogue(
        self,
        params: EpilogueParams,
        epi_smem_tensors: Dict[str, cute.Tensor],
        epi_pipeline: Optional[cutlass.pipeline.PipelineAsync],
        epi_store_pipeline: Optional[cutlass.pipeline.PipelineAsync],
        epi_read_state: Optional[cutlass.pipeline.PipelineState],
        epi_producer_state: Optional[cutlass.pipeline.PipelineState],
        epi_tile: cute.Tile,
        load_acc_subtile: Callable,
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor],
        tiled_copy_t2r: Optional[cute.TiledCopy],  # Only for Sm100
        tiled_copy_r2s: cute.TiledCopy,
        tRS_sD: cute.Tensor,
        tiled_copy_s2r: Optional[cute.ThrCopy],
        tSR_rC: Optional[cute.Tensor],
        tSR_sC: Optional[cute.Tensor],
        copy_D: Optional[Callable],
        copy_C: Optional[Callable],
        tile_coord_mnkl: cute.Coord,
        varlen_manager: VarlenManager,
        epilogue_barrier: cutlass.pipeline.NamedBarrier,
        tile_scheduler,
        tidx: Int32,
        is_tma_warp: cutlass.Boolean,
    ) -> Tuple[cutlass.pipeline.PipelineState, cutlass.pipeline.PipelineState]:
        from cutlass.cute.experimental import iket

        has_C = const_expr(tRS_rC is not None)
        has_epi_load = const_expr(self.epi_c_stage > 0)
        has_D = const_expr(copy_D is not None)
        use_tma_epi = const_expr(epi_store_pipeline is not None)
        use_tma_c = const_expr(epi_pipeline is not None)
        inline_epi_load = const_expr(copy_C is not None)

        # Store contexts — one per stored output, D first (when present), then
        # the aux TileStore outputs. Each is ``(op, quant, tiled_copy, tRS_s,
        # copy_fn, store_pred)``: the op owns the storage-dtype convert
        # (store_convert) and the register->smem copy (store_r2s); ``quant``
        # is the output's optional quantize codec (BlockScaleFactorStore),
        # run by this driver on the final fragment right before the convert;
        # ``store_pred`` is None for an unconditional store, else a
        # per-CTA-tile Boolean (e.g. GemmSymmetric skips the mirrored write
        # on diagonal tiles). D's host pieces stay kernel-built (see DStore).
        store_ctxs = self.epi_setup_aux_out(
            params,
            epi_smem_tensors,
            tiled_copy_r2s,
            tiled_copy_t2r,
            tile_coord_mnkl,
            varlen_manager,
            tidx,
        )
        if const_expr(has_D):
            store_ctxs = (
                (DStore(), self._epi_store_quant("D"), tiled_copy_r2s, tRS_sD, copy_D, None),
            ) + store_ctxs

        epi_tile_shape = cute.zipped_divide(
            cute.make_layout(self.cta_tile_shape_mnk[:2]), epi_tile
        ).shape[1]
        epi_tile_layout = cute.make_ordered_layout(
            epi_tile_shape, order=(0, 1) if const_expr(self.epi_m_major) else (1, 0)
        )
        epi_tile_num = cute.size(epi_tile_shape)
        num_prev_subtiles = tile_scheduler.num_tiles_executed * epi_tile_num

        epi_tensors = self.epi_begin(
            params,
            epi_smem_tensors,
            epi_tile,
            tiled_copy_t2r,
            tiled_copy_r2s,
            tile_coord_mnkl,
            varlen_manager,
            epilogue_barrier,
            tidx,
            tRS_rD.layout,
        )

        if const_expr(self.epi_needs_acc_prepass):
            assert self.arch in (90, 100, 120), (
                "acc prepass needs a re-readable accumulator (SM90/SM120 registers / SM100 tmem)"
            )
            # Pingpong is safe (SM90 and SM120 share the protocol): the two
            # warpgroups' epilogues are strictly exclusive (the leaving WG
            # drains its TMA stores via producer_tail before arriving the
            # peer's epi barrier), so epi smem — including prepass
            # statistics — is only temporally shared.
            # SERIAL/PARALLEL split-K compose with the prepass: only the finalizing
            # split runs the epilogue, and its load_acc_subtile is the folding
            # wrapper (epilogue_split_k), so the prepass statistics see the
            # completed accumulator. SEPARATE would run the prepass on every
            # split's raw partial — reject it.
            assert const_expr(self.split_k == 1 or self.split_k_mode != SplitKMode.SEPARATE), (
                "acc prepass + SEPARATE split-k would read raw partials"
            )
            for epi_idx in cutlass.range_constexpr(epi_tile_num):
                epi_coord = epi_tile_layout.get_hier_coord(epi_idx)
                load_acc_subtile(tRS_rD, epi_coord, no_release=True)
                self.epi_prepass_subtile(params, epi_tensors, tRS_rD, epi_coord, epi_idx)
            self.epi_prepass_end(params, epi_tensors)

        if const_expr(inline_epi_load):
            for epi_idx in cutlass.range(min(epi_tile_num, self.epi_c_stage), unroll=1):
                epi_coord_C = epi_tile_layout.get_hier_coord(epi_idx)
                if const_expr(use_tma_c):
                    if is_tma_warp:
                        epi_pipeline.producer_acquire(epi_producer_state)
                        copy_C(src_idx=epi_coord_C, producer_state=epi_producer_state)
                        epi_pipeline.producer_commit(epi_producer_state)
                    epi_producer_state.advance()
                else:
                    # TODO: turn this to cp.async instead of direct G2R copy
                    copy_C(src_idx=epi_coord_C, dst_idx=epi_idx % self.epi_c_stage)
            if const_expr(use_tma_c):
                epilogue_barrier.arrive_and_wait()

        for epi_idx in cutlass.range_constexpr(epi_tile_num):
            epi_coord = epi_tile_layout.get_hier_coord(epi_idx)  # (epi_m, epi_n)
            # Copy from acc to D registers. Under split-K this callable is the
            # folding wrapper built in epilogue_split_k, so tRS_rD holds the
            # completed accumulator — the epilogue itself is split-K-free.
            load_acc_subtile(tRS_rD, epi_coord)
            if const_expr(has_epi_load):
                if const_expr(use_tma_c):
                    iket.range_push("epi_c_wait")
                    epi_pipeline.consumer_wait(epi_read_state)
                    iket.range_pop()
                    iket.range_push("epi_c_s2r")
                    if const_expr(has_C):
                        cute.copy(
                            tiled_copy_s2r, tSR_sC[None, None, None, epi_read_state.index], tSR_rC
                        )
                    self.epi_tile_load_s2r(params, epi_tensors, epi_read_state.index)
                    cute.arch.fence_view_async_shared()
                    epi_pipeline.consumer_release(epi_read_state)
                    epi_read_state.advance()
                    iket.range_pop()
                else:
                    c_buffer = epi_idx % self.epi_c_stage
                    cute.copy(tiled_copy_s2r, tSR_sC[None, None, None, c_buffer], tSR_rC)
                    # TODO: cp.async wait once we switch to cp.async
                    epilogue_barrier.arrive_and_wait()
            epi_loop_tensors = self.epi_begin_loop(params, epi_tensors, epi_coord)
            if const_expr(inline_epi_load and epi_idx + self.epi_c_stage < epi_tile_num):
                epi_coord_C = epi_tile_layout.get_hier_coord(epi_idx + self.epi_c_stage)
                if const_expr(use_tma_c):
                    if is_tma_warp:
                        epi_pipeline.producer_acquire(epi_producer_state)
                        copy_C(src_idx=epi_coord_C, producer_state=epi_producer_state)
                        epi_pipeline.producer_commit(epi_producer_state)
                    epi_producer_state.advance()
                else:
                    epilogue_barrier.arrive_and_wait()
                    copy_C(
                        src_idx=epi_coord_C,
                        dst_idx=(epi_idx + self.epi_c_stage) % self.epi_c_stage,
                    )
            # Returns a tuple of register tensors — one per aux output.
            # Length matches ``aux_out_ctxs``. ``()`` for the default
            # epilogue (no aux output).
            iket.range_push("epi_math")
            tRS_rAuxOuts = self.epi_visit_subtile(params, epi_loop_tensors, tRS_rD, tRS_rC)
            iket.range_pop()
            # end_loop sees the final D fragment and may mutate it in place
            # before the dtype convert below.
            self.epi_end_loop(
                params,
                epi_tensors,
                epi_coord,
                tRS_rD,
                epi_tile,
                tiled_copy_t2r,
                tiled_copy_r2s,
                tile_coord_mnkl,
                varlen_manager,
                tidx,
            )
            # Quantize (optional per-output codec) + convert each stored
            # output to its storage dtype. The quantize rescales the final
            # fragment in place and collects the SF bytes (flushed in the
            # codec's end), so the convert below emits the quantized values.
            store_frags = ((tRS_rD,) if has_D else ()) + tRS_rAuxOuts
            store_frags_out = []
            # range_constexpr: store_ctxs/store_frags are static Python
            # tuples — a staged loop var cannot index them.
            for i in cutlass.range_constexpr(len(store_ctxs)):
                op, quant, _, _, _, _ = store_ctxs[i]
                if const_expr(quant is not None):
                    quant.quantize(self, epi_loop_tensors[quant.name], store_frags[i])
                store_frags_out.append(
                    op.store_convert(
                        self,
                        store_frags[i],
                        epi_loop_tensors.get("sr_seed"),
                        tidx,
                        tile_coord_mnkl,
                        num_prev_subtiles,
                        epi_idx,
                    )
                )
            iket.range_push("epi_store_acq")
            if const_expr(use_tma_epi):
                if is_tma_warp:
                    epi_store_pipeline.producer_acquire()
            else:
                epilogue_barrier.arrive_and_wait()
            if const_expr(use_tma_epi):
                epilogue_barrier.arrive_and_wait()
            iket.range_pop()
            epi_buffer = (num_prev_subtiles + epi_idx) % self.epi_stage
            # Copy each output from registers to shared memory. All share the
            # same ``epi_buffer`` index so the s2g TMA stores below happen in
            # lockstep after the fence.
            iket.range_push("epi_r2s")
            for i in cutlass.range_constexpr(len(store_ctxs)):
                op, _, tiled_copy_st, tRS_s, _, _ = store_ctxs[i]
                op.store_r2s(
                    self,
                    tiled_copy_st,
                    store_frags_out[i],
                    tRS_s[None, None, None, epi_buffer],
                    tidx,
                )
            iket.range_pop()
            if const_expr(use_tma_epi):
                iket.range_push("epi_store_issue")
                cute.arch.fence_view_async_shared()
                epilogue_barrier.arrive_and_wait()
                if is_tma_warp:
                    for i in cutlass.range_constexpr(len(store_ctxs)):
                        _, _, _, _, copy_out, store_pred = store_ctxs[i]
                        if const_expr(store_pred is None):
                            copy_out(src_idx=epi_buffer, dst_idx=epi_coord)
                        else:
                            if store_pred:
                                copy_out(src_idx=epi_buffer, dst_idx=epi_coord)
                    epi_store_pipeline.producer_commit()
                iket.range_pop()
            else:
                epilogue_barrier.arrive_and_wait()
                for i in cutlass.range_constexpr(len(store_ctxs)):
                    _, _, _, _, copy_out, store_pred = store_ctxs[i]
                    if const_expr(store_pred is None):
                        copy_out(src_idx=epi_buffer, dst_idx=epi_coord)
                    else:
                        if store_pred:
                            copy_out(src_idx=epi_buffer, dst_idx=epi_coord)
                epilogue_barrier.arrive_and_wait()

        self.epi_end(
            params,
            epi_tensors,
            epi_tile,
            tiled_copy_t2r,
            tiled_copy_r2s,
            tile_coord_mnkl,
            varlen_manager,
            tidx,
        )

        return epi_read_state, epi_producer_state

    @cute.jit
    def split_k_partial_commit(
        self,
        load_acc_subtile: Callable,
        tRS_rD: cute.Tensor,
        epi_tile: cute.Tile,
        ws_ptr: cute.Pointer,
        split_k_sem: Semaphore,
        split_idx: Int32,
        tidx: Int32,
    ) -> None:
        """Non-finalizing split: commit the raw f32 accumulator partial, run no epilogue.

        The tile's workspace region is a flat (epi_subtile, vector, thread, 4) stripe of
        accumulator fragments — no (m, n) semantics, no predication; the finalizer reads
        it back with the identical partitioning (the load_acc_and_fold closure in
        epilogue_split_k).
        SERIAL: a turnstile (flag == number of committed splits) orders the f32 adds in
        split order — bitwise deterministic; split 0's plain store initializes the
        region, later splits red.add into it. PARALLEL: the workspace is host-zeroed and
        every split red.adds immediately in arrival order (no waiting, NOT
        deterministic), then release-increments the flag.

        ``split_k_sem`` carries the epi-group barrier as its sync policy, so every
        epi thread must call this; the sem broadcasts the acquire after wait_eq and
        collects the group's commits before the release.
        """
        epi_tile_shape = cute.zipped_divide(
            cute.make_layout(self.cta_tile_shape_mnk[:2]), epi_tile
        ).shape[1]
        epi_tile_layout = cute.make_ordered_layout(
            epi_tile_shape, order=(0, 1) if const_expr(self.epi_m_major) else (1, 0)
        )
        num_epi_threads = self.num_epi_warps * cute.arch.WARP_SIZE
        frag_elems = cute.size(tRS_rD)
        if const_expr(self.split_k_mode == SplitKMode.SERIAL):
            # Wait until all preceding splits have committed.
            split_k_sem.wait_eq(split_idx, skip_zero=True)
        for epi_idx in cutlass.range_constexpr(cute.size(epi_tile_shape)):
            epi_coord = epi_tile_layout.get_hier_coord(epi_idx)
            load_acc_subtile(tRS_rD, epi_coord)
            frag_base = ws_ptr + epi_idx * num_epi_threads * frag_elems
            if const_expr(self.split_k_mode == SplitKMode.SERIAL):
                if split_idx == 0:
                    # First split initializes the (uninitialized) region in-order.
                    self._frag_stripe_op("store", frag_base, tidx, num_epi_threads, tRS_rD)
                else:
                    self._frag_stripe_op("red_add", frag_base, tidx, num_epi_threads, tRS_rD)
            else:
                # PARALLEL: no initializing store (the host zero-fills the workspace),
                # every split reduces — in arrival order, hence not deterministic.
                self._frag_stripe_op("red_add", frag_base, tidx, num_epi_threads, tRS_rD)
        if const_expr(self.split_k_mode == SplitKMode.SERIAL):
            split_k_sem.release_store(split_idx + 1)
        else:
            split_k_sem.release_add()

    @cute.jit
    def _frag_stripe_op(
        self,
        op: cutlass.Constexpr[Literal["store", "red_add", "load_add"]],
        frag_base: cute.Pointer,
        tidx: Int32,
        num_threads: cutlass.Constexpr,
        tRS_rD: cute.Tensor,
    ) -> None:
        """One accumulator fragment vs the tile's flat workspace stripe.

        The stripe is (vector, thread, 4)-interleaved: at a fixed vector index,
        adjacent threads access adjacent 16-byte chunks — vectorized v4 when the
        fragment allows, scalar (thread-strided) otherwise. The three ops share
        this addressing exactly, which is what makes the write and read sides
        provably consistent:

        - "store":    plain store of the fragment (split 0's in-order init)
        - "red_add":  one-way L2 red.add, no read-back (later splits' commits)
        - "load_add": read the stripe and add it INTO the fragment (finalizer fold)
        """
        frag_elems = cute.size(tRS_rD)
        if const_expr(op == "load_add"):
            tRS_rWs = cute.make_rmem_tensor(tRS_rD.shape, self.acc_dtype)
            frag = tRS_rWs
        else:
            frag = tRS_rD
        if const_expr(frag_elems % 4 == 0 and self.acc_dtype == cutlass.Float32):
            gWs = cute.make_tensor(frag_base, cute.make_layout(num_threads * frag_elems))
            thr_copy, tCgWs = copy_utils.vectorized_thread_partition(
                gWs, tidx, num_threads, 4, is_source=const_expr(op == "load_add")
            )
            for v in cutlass.range_constexpr(frag_elems // 4):
                chunk = cute.make_tensor(frag.iterator + 4 * v, cute.make_layout(4))
                if const_expr(op == "store"):
                    cute.copy(thr_copy, chunk, tCgWs[None, v])
                elif const_expr(op == "red_add"):
                    cute.arch.atomic_add(tCgWs[None, v].iterator, chunk.load())
                else:
                    cute.copy(thr_copy, tCgWs[None, v], chunk)
        else:
            frag_layout = cute.make_layout((num_threads, frag_elems), stride=(1, num_threads))
            tRS_gWs = cute.make_tensor(frag_base, frag_layout)
            for v in cutlass.range_constexpr(frag_elems):
                if const_expr(op == "store"):
                    copy_utils.store(utils.elem_pointer(tRS_gWs, (tidx, v)), frag[v])
                elif const_expr(op == "red_add"):
                    cute.arch.atomic_add(utils.elem_pointer(tRS_gWs, (tidx, v)), frag[v])
                else:
                    frag[v] = tRS_gWs[tidx, v]
        if const_expr(op == "load_add"):
            tRS_rD.store(tRS_rD.load() + tRS_rWs.load())

    @cute.jit
    def epilogue_split_k(
        self,
        params: EpilogueParams,
        epi_fn: Callable,
        load_acc_subtile: Callable,
        tRS_rD: cute.Tensor,
        epi_tile: cute.Tile,
        epi_read_state: Optional[cutlass.pipeline.PipelineState],
        epi_producer_state: Optional[cutlass.pipeline.PipelineState],
        epi_store_pipeline: Optional[cutlass.pipeline.PipelineAsync],
        tile_coord_mnkl: cute.Coord,
        epilogue_barrier: cutlass.pipeline.NamedBarrier,
        tidx: Int32,
        is_tma_warp: cutlass.Boolean,
    ) -> Tuple[cutlass.pipeline.PipelineState, cutlass.pipeline.PipelineState]:
        """The split-K finalization protocol around an epilogue closure.

        ``epi_fn`` is self.epilogue with everything bound except its accumulator
        source: ``epi_fn(load_acc) -> (epi_read_state, epi_producer_state)``.
        This wrapper owns ALL split-K knowledge: the commit/finalize semaphore
        protocol, and the workspace fold — injected behind the load_acc_subtile
        contract, because the finalizer's true accumulator is its own partial
        plus the workspace. The epilogue data path stays split-K-free, and any
        consumer of the accumulator (store pass, acc prepass) sees the completed
        value.

        split_k == 1 and SEPARATE pass straight through (SEPARATE splits store raw f32
        partials to disjoint workspace slices via the normal TMA path; the host strips
        the epi-op arguments so the visit is a no-op, and the separate reduction kernel
        applies the full epilogue). SERIAL/PARALLEL: non-finalizing splits commit raw
        partials and skip the epilogue entirely — including the C loads — while the
        last split waits for the tile's completion flag, folds the workspace into its
        accumulator, and runs the full epi mixin exactly once.

        ``epi_read_state``/``epi_producer_state`` are the same values already bound
        into ``epi_fn``; they are passed here only so the partial-commit branch can
        return them unchanged.
        """
        finalizer_load_acc = load_acc_subtile
        if const_expr(self.split_k > 1 and self.split_k_mode != SplitKMode.SEPARATE):
            # The flag and workspace are CuTe tensors over the (cluster-rounded) tile
            # domain — their layouts own the address computation.
            assert self.acc_dtype == cutlass.Float32, "split_k workspace is f32"
            batch_idx, split_idx = tile_coord_mnkl[3], tile_coord_mnkl[2]
            # Per-tile flag; the epi-group barrier is the sem's sync policy, so all
            # epi threads make the sem calls (thread_idx is epi-group-relative).
            split_k_sem = Semaphore(
                utils.elem_pointer(
                    params.split_k_semaphore, (tile_coord_mnkl[0], tile_coord_mnkl[1], batch_idx)
                ),
                tidx,
                sync=epilogue_barrier,
            )
            ws_ptr = utils.elem_pointer(
                params.split_k_workspace, (0, tile_coord_mnkl[0], tile_coord_mnkl[1], batch_idx)
            )
            # The stripe must tile the host-allocated region exactly.
            epi_tile_shape = cute.zipped_divide(
                cute.make_layout(self.cta_tile_shape_mnk[:2]), epi_tile
            ).shape[1]
            epi_tile_num = cute.size(epi_tile_shape)
            num_epi_threads = self.num_epi_warps * cute.arch.WARP_SIZE
            assert (
                cute.size(tRS_rD) * num_epi_threads * epi_tile_num
                == self.cta_tile_shape_mnk[0] * self.cta_tile_shape_mnk[1]
            ), "split-K workspace stripe does not tile cta_tile_m * cta_tile_n"
            # Same subtile ordering as split_k_partial_commit's write side, so
            # crd2idx recovers the flat stripe index from the epilogue's coord.
            epi_tile_layout = cute.make_ordered_layout(
                epi_tile_shape, order=(0, 1) if const_expr(self.epi_m_major) else (1, 0)
            )

            def load_acc_and_fold(tRS_rD_, epi_coord, **kwargs):
                load_acc_subtile(tRS_rD_, epi_coord, **kwargs)
                epi_idx = cute.crd2idx(epi_coord, epi_tile_layout)
                frag_base = ws_ptr + epi_idx * num_epi_threads * cute.size(tRS_rD_)
                self._frag_stripe_op("load_add", frag_base, tidx, num_epi_threads, tRS_rD_)

            finalizer_load_acc = load_acc_and_fold
        else:
            split_k_sem, ws_ptr = None, None
        # Mixed static/dynamic test (quack.dsl.mixed_constexpr_if): the const_expr
        # prefix folds at trace time, so split_k == 1 / SEPARATE codegen is a bare
        # epilogue call with no dynamic if, while SERIAL/PARALLEL non-finalizing
        # splits commit raw partials and only the last split runs the epilogue.
        if (
            const_expr(self.split_k > 1 and self.split_k_mode != SplitKMode.SEPARATE)
            and split_idx < self.split_k - 1
        ):
            self.split_k_partial_commit(
                load_acc_subtile, tRS_rD, epi_tile, ws_ptr, split_k_sem, split_idx, tidx
            )
        else:
            if const_expr(self.split_k > 1 and self.split_k_mode != SplitKMode.SEPARATE):
                # Finalizer (fixed: the last split, so e.g. the SM100 epi-load warp can
                # gate C loads statically). Wait for all S-1 sibling commits; the flag
                # counts committed splits in both modes. The finalizer folds the
                # workspace with generic loads, so no async-proxy fence is needed
                # after the acquire.
                split_k_sem.wait_eq(self.split_k - 1)
            epi_read_state, epi_producer_state = epi_fn(finalizer_load_acc)
            if const_expr(self.split_k > 1 and self.split_k_mode != SplitKMode.SEPARATE):
                # Self-clean the flag (fresh-zeros allocation also works, but this keeps
                # the tensor reusable if the host ever caches it).
                split_k_sem.release_store(0)
                # Drain this finalizer's TMA stores before this persistent CTA advances.
                # Non-finalizing work issues no TMA stores, but still increments
                # num_tiles_executed, so the CTA's next finalizer may start at a
                # non-consecutive epi smem buffer. producer_acquire only limits the
                # number of outstanding groups; it does not wait for that specific
                # buffer. With no stores outstanding, any next buffer is safe.
                if const_expr(epi_store_pipeline is not None):
                    if is_tma_warp:
                        epi_store_pipeline.producer_tail()
        return epi_read_state, epi_producer_state

    def get_scheduler_class(self, varlen_m: bool = False):
        """Return the scheduler class to use. Override in subclasses for custom schedulers."""
        return TileScheduler if not varlen_m else VarlenMTileScheduler

    def resolve_epi_m_major(self, epilogue_args: EpilogueArguments):
        return True

    def get_scheduler_arguments(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mD: Optional[cute.Tensor],
        scheduler_args,
        varlen_args,
        epilogue_args,
    ):
        """Create scheduler arguments. Override in subclasses for custom schedulers."""
        if const_expr(not self.is_persistent):
            persistence_mode = PersistenceMode.NONE
        else:
            if const_expr(self.arch >= 100 and self.use_clc_persistence):
                persistence_mode = PersistenceMode.CLC
            elif const_expr(scheduler_args.tile_count_semaphore is not None):
                persistence_mode = PersistenceMode.DYNAMIC
            else:
                persistence_mode = PersistenceMode.STATIC
        if const_expr(varlen_args.mCuSeqlensM is None):
            num_problems = (
                mD.shape[2]
                if mD is not None
                else (
                    mB.shape[2]
                    if varlen_args.mCuSeqlensK is None
                    else varlen_args.mCuSeqlensK.shape[0] - 1
                )
            )
            if const_expr(self.split_k > 1 and self.split_k_mode == SplitKMode.SEPARATE):
                # mD is the f32 partials workspace whose batch extent is L * split_k; the
                # scheduler needs the true L (it scales the work-id space by num_split_k
                # itself). B always carries the true L here (varlen is rejected).
                num_problems = mB.shape[2]
            # Layout-owning transform_a: mA is not an (M, K) operand (e.g. a
            # repacked blob), so the M extent comes from D instead.
            a_owned = (
                getattr(self, "transform_a", None) is not None and self.transform_a.owns_a_layout
            )
            m_size = cute.size(mA, mode=[0]) if const_expr(not a_owned) else cute.size(mD, mode=[0])
            problem_shape_ntile_mnl = (
                cute.ceil_div(m_size, self.cta_tile_shape_mnk[0]),
                cute.ceil_div(cute.size(mB, mode=[0]), self.cta_tile_shape_mnk[1]),
                num_problems,
            )
            tile_sched_args = TileSchedulerArguments(
                problem_shape_ntile_mnl=problem_shape_ntile_mnl,
                raster_order=scheduler_args.raster_order,
                group_size=scheduler_args.max_swizzle_size,
                cluster_shape_mnk=self.cluster_shape_mnk,
                tile_count_semaphore=scheduler_args.tile_count_semaphore,
                batch_idx_permute=scheduler_args.batch_idx_permute,
                persistence_mode=persistence_mode,
                num_split_k=self.split_k,
                ag=getattr(scheduler_args, "ag", None),
            )
        else:
            assert getattr(scheduler_args, "ag", None) is None, (
                "AllGather+GEMM does not support varlen_m"
            )
            assert self.split_k == 1, "split_k does not support varlen_m"
            has_epi_tile_store = any(
                getattr(epilogue_args, op.name, None) is not None
                for op in getattr(type(self), "_epi_ops", ())
                if op.is_tile_store()
            )
            assert (mD is not None) or has_epi_tile_store or (not self.gather_A)
            problem_shape_ntile_mnl = (
                None,
                cute.ceil_div(cute.size(mB, mode=[0]), self.cta_tile_shape_mnk[1]),
                varlen_args.mCuSeqlensM.shape[0] - 1,
            )
            tile_sched_args = VarlenMTileSchedulerArguments(
                problem_shape_ntile_mnl=problem_shape_ntile_mnl,
                total_m=(
                    mD.shape[0]
                    if mD is not None
                    else (
                        varlen_args.mAIdx.shape[0]
                        if varlen_args.mAIdx is not None
                        else cute.size(mA, mode=[0])
                    )
                ),
                cu_seqlens_m=varlen_args.mCuSeqlensM,
                raster_order=scheduler_args.raster_order,
                group_size=scheduler_args.max_swizzle_size,
                tile_shape_mn=self.cta_tile_shape_mnk[:2],
                cluster_shape_mnk=self.cluster_shape_mnk,
                tile_count_semaphore=scheduler_args.tile_count_semaphore,
                persistence_mode=persistence_mode,
            )
        return tile_sched_args

    @cute.jit
    def epi_load_acc_subtile(
        self,
        tRS_rAcc: cute.Tensor,
        tRS_rD: cute.Tensor,
        epi_coord,  # (int, int)
        no_release: cutlass.Constexpr[bool] = False,
    ):
        # no_release is the prepass flag (epi_needs_acc_prepass); the register
        # accumulator has nothing to release, so re-reads are always safe here.
        cute.autovec_copy(tRS_rAcc[None, None, None, epi_coord], tRS_rD)

    @cute.jit
    def epi_begin(
        self,
        params: EpilogueParams,
        epi_smem_tensors: Dict[str, cute.Tensor],
        epi_tile: cute.Tile,
        tiled_copy_t2r: Optional[cute.TiledCopy],
        tiled_copy_r2s: cute.TiledCopy,
        tile_coord_mnkl: cute.Coord,
        varlen_manager: VarlenManager,
        epilogue_barrier: cutlass.pipeline.NamedBarrier,
        tidx: Int32,
        tRS_rD_layout=None,
    ) -> Tuple[cute.Tensor, ...]:
        return ()

    def epi_begin_loop(
        self, params: EpilogueParams, epi_tensors: Tuple[cute.Tensor, ...], epi_coord: cute.Coord
    ) -> Tuple[cute.Tensor, ...]:
        return ()

    def epi_visit_subtile(
        self,
        params: EpilogueParams,
        epi_loop_tensors: Tuple[cute.Tensor, ...],
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor] = None,
    ) -> Tuple[cute.Tensor, ...]:
        return ()

    def epi_visit_acc(
        self,
        params: EpilogueParams,
        acc: cute.Tensor,
        tiled_mma: cute.TiledMma,
        tile_coord_mnkl: cute.Coord,
        tidx: Int32,
    ) -> None:
        pass

    @cute.jit
    def epi_end_loop(
        self,
        params: EpilogueParams,
        epi_tensors: Tuple[cute.Tensor, ...],
        epi_coord: cute.Coord,
        tRS_rD: cute.Tensor,
        epi_tile: cute.Tile,
        tiled_copy_t2r: Optional[cute.TiledCopy],
        tiled_copy_r2s: cute.TiledCopy,
        tile_coord_mnkl: cute.Coord,
        varlen_manager,
        tidx,
    ) -> None:
        pass

    @cute.jit
    def epi_end(
        self,
        params: EpilogueParams,
        epi_tensors: Tuple[cute.Tensor, ...],
        epi_tile: cute.Tile,
        tiled_copy_t2r: Optional[cute.TiledCopy],
        tiled_copy_r2s: cute.TiledCopy,
        tile_coord_mnkl: cute.Coord,
        varlen_manager,
        tidx,
    ) -> None:
        pass

    def epi_to_underlying_arguments(
        self, args: EpilogueArguments, *, loc=None, ip=None
    ) -> EpilogueParams:
        return self.EpilogueParams()

    def epi_get_tma_atoms(
        self, params: EpilogueParams, *, loc=None, ip=None
    ) -> list[cute.CopyAtom]:
        """Subclasses can override this."""
        return []

    def epi_tile_load_g2s_copy_fns(
        self,
        params,
        epi_smem_tensors,
        tile_coord_mnkl,
        varlen_manager,
        epi_pipeline,
    ):
        return ()

    @cute.jit
    def epi_tile_load_s2r(self, params, epi_tensors, stage_idx):
        pass

    @staticmethod
    def epi_smem_bytes(
        args: Optional[EpilogueArguments],
        cta_tile_shape_mnk: Tuple[int, int, int],
        epi_tile: cute.Tile,
        warp_shape_mnk: Tuple[int, int, int] | None = None,
    ) -> EpiSmemBytes:
        return EpiSmemBytes()

    def epi_get_smem_struct(self, params: EpilogueParams):
        return cute.struct.MemRange[Int32, 0]  # Dummy struct

    def epi_get_smem_tensors(self, params: EpilogueParams, storage) -> Dict[str, cute.Tensor]:
        return {}

    def epi_setup_aux_out(
        self,
        params,
        epi_smem_tensors,
        tiled_copy_r2s,
        tiled_copy_t2r,
        tile_coord_mnkl,
        varlen_manager,
        tidx,
    ):
        """Return a tuple of ``(op, quant, tiled_copy, tRS_sAuxOut,
        copy_aux_out, store_pred)`` store contexts — one per aux output (see
        TileStore.store_setup; ComposableEpiMixin provides the generic
        op-driven implementation). The default epilogue has no aux output, so
        the tuple is empty.
        """
        return ()

    def _epi_store_quant(self, output_name):
        """The quantize codec op attached to the named stored output ("D" =
        the main output), or None (ComposableEpiMixin provides the generic
        implementation)."""
        return None


class GemmTmaBase(GemmBase):
    """Common TMA descriptor and pipeline helpers for SM90+ GEMM paths."""

    @cute.jit
    def load_tma(
        self,
        pipeline: cutlass.pipeline.PipelineAsync,
        producer_state: cutlass.pipeline.PipelineState,
        copy_fns: Sequence[Optional[Callable]],
        k_tile_cnt: Int32,
        k_tile_start: Int32 | int = 0,
    ) -> cutlass.pipeline.PipelineState:
        # Peek (try_wait) AB buffer empty for k_block = prefetch_k_tile_cnt.
        peek_empty_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_empty_status = pipeline.producer_try_acquire(producer_state)
        # TMA load
        for k_tile in cutlass.range(k_tile_cnt, unroll=1):
            # Wait for A/B buffers to be empty before loading into them.
            # Also sets the transaction barrier for the A/B buffers.
            pipeline.producer_acquire(producer_state, peek_empty_status)
            tma_bar_ptr = pipeline.producer_get_barrier(producer_state)
            smem_idx = producer_state.index
            for copy_fn in copy_fns:
                if const_expr(copy_fn is not None):
                    copy_fn(k_tile_start + k_tile, smem_idx, tma_bar_ptr=tma_bar_ptr)
            # Mainloop pipeline's producer commit is a NOP for TMA pipelines.
            pipeline.producer_commit(producer_state)
            producer_state.advance()
            peek_empty_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_empty_status = pipeline.producer_try_acquire(producer_state)
        return producer_state

    def _make_gmem_tiled_copy_A(self, dtype, major_mode, num_threads, copy_bits=128):
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            dtype,
            num_bits_per_copy=copy_bits,
        )
        copy_elems = copy_bits // dtype.width
        loads_per_cache_line = 128 * 8 // copy_bits  # 128 bytes per cache line
        shape_dim_1 = cute.size(self.cta_tile_shape_mnk[2]) // copy_elems
        if shape_dim_1 > loads_per_cache_line:
            shape_dim_1 = math.gcd(shape_dim_1, loads_per_cache_line)
        # thread layout for copy
        thread_layout = cute.make_layout(
            (num_threads // shape_dim_1, shape_dim_1), stride=(shape_dim_1, 1)
        )
        if major_mode != LayoutEnum.ROW_MAJOR:
            shape_dim_0 = cute.size(self.cta_tile_shape_mnk[0]) // copy_elems
            if shape_dim_0 > loads_per_cache_line:
                shape_dim_0 = math.gcd(shape_dim_0, loads_per_cache_line)
            thread_layout = cute.make_layout(
                (shape_dim_0, num_threads // shape_dim_0), stride=(1, shape_dim_0)
            )
        # Value layout for copy
        value_layout = (
            cute.make_layout((1, copy_elems))
            if major_mode == LayoutEnum.ROW_MAJOR
            else cute.make_layout((copy_elems, 1))
        )
        return cute.make_tiled_copy_tv(atom_async_copy, thread_layout, value_layout)

    def epilogue_zero_fill_varlen_m(self, epilogue_args):
        """True when an active M-fold (dim==1) reduce sink demands zero-filled
        OOB rows from every varlen_m TMA load (A, C): the fold across a tile's
        M rows is unpredicated, so rows past a sequence end must read as zero
        (identity for add), not as the next sequence's data. Without such a
        sink, garbage rows die at the (predicated) stores and the loads keep
        their plain 2-D descriptors."""
        return any(
            getattr(op, "fn_port", None) == "sink"
            and getattr(op, "dim", None) == 1
            and getattr(epilogue_args, op.name, None) is not None
            for op in getattr(self, "_epi_ops", ())
        )

    def make_tma_load_atoms_and_tensors(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        varlen_k: bool,
        a_internal_type: Optional[Type[cutlass.Numeric]] = None,
        b_internal_type: Optional[Type[cutlass.Numeric]] = None,
        varlen_m_zero_fill: bool = False,
    ):
        tma_atom_a, tma_tensor_a = None, None
        if const_expr(not self.gather_A):
            if const_expr(varlen_m_zero_fill):
                # varlen_m + M-fold sink: zero-fill rows past the sequence end
                # (see epilogue_zero_fill_varlen_m). 2-extra-dim wraparound —
                # ptr_shift is store-only.
                mA_tma = copy_utils.create_ragged_tensor_for_tma(mA, ragged_dim=0, ptr_shift=False)
            elif const_expr(varlen_k):
                mA_tma = copy_utils.create_ragged_tensor_for_tma(mA, ragged_dim=1)
            else:
                mA_tma = mA
            tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
                mA_tma,
                a_smem_layout,
                (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[2]),
                self.cluster_shape_mnk[1],
                internal_type=a_internal_type,
            )
        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            copy_utils.create_ragged_tensor_for_tma(mB, ragged_dim=1) if varlen_k else mB,
            b_smem_layout,
            (self.cta_tile_shape_mnk[1], self.cta_tile_shape_mnk[2]),
            self.cluster_shape_mnk[0],
            internal_type=b_internal_type,
        )
        return tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b

    def make_tma_epilogue_atoms_and_tensors(
        self,
        mD: Optional[cute.Tensor],
        mC: Optional[cute.Tensor],
        epilogue_args,
        varlen_m: bool,
    ):
        add_to_output = const_expr(
            hasattr(epilogue_args, "add_to_output") and epilogue_args.add_to_output
        )
        # Split-K needs no special D atom: only the finalizing entity stores D (a plain
        # store, or the reduce-add atom with add_to_output, exactly like the non-split
        # kernel); partials travel through the f32 workspace, not through D.
        tma_atom_d, tma_tensor_d = None, None
        if const_expr(mD is not None):
            tma_atom_d, tma_tensor_d = self._make_tma_epi_atoms_and_tensors(
                copy_utils.create_ragged_tensor_for_tma(mD, ragged_dim=0, ptr_shift=True)
                if varlen_m
                else mD,
                self.epi_smem_layout_staged,
                self.epi_tile,
                op_type="store" if not add_to_output else "add",
            )
        tma_atom_c, tma_tensor_c = None, None
        if const_expr(mC is not None):
            # Same zero-fill contract as A: with an active M-fold sink, C rows
            # past a sequence end must load as zero, not the next sequence's.
            tma_atom_c, tma_tensor_c = self._make_tma_epi_atoms_and_tensors(
                copy_utils.create_ragged_tensor_for_tma(mC, ragged_dim=0, ptr_shift=False)
                if varlen_m and self.epilogue_zero_fill_varlen_m(epilogue_args)
                else mC,
                self.epi_c_smem_layout_staged,
                self.epi_tile,
                op_type="load",
            )
        return (
            tma_atom_d,
            tma_tensor_d,
            tma_atom_c,
            tma_tensor_c,
        )

    def epilog_gmem_copy_and_partition(
        self,
        atom: cute.CopyAtom | cute.TiledCopy,
        mD_mn: cute.Tensor,
        tile_shape_mn: cute.Tile,
        epi_tile: cute.Tile,
        sD: cute.Tensor,
        tile_coord_mnkl: cute.Coord,
    ) -> Tuple[cute.Tensor, cute.Tensor]:
        gD = cute.local_tile(mD_mn, tile_shape_mn, tile_coord_mnkl[:2])  # (bM, bN)
        tDgD_for_tma_partition = cute.zipped_divide(gD, epi_tile)
        is_s2g = isinstance(
            atom.op, (cpasync.CopyBulkTensorTileS2GOp, cpasync.CopyReduceBulkTensorTileS2GOp)
        )
        src_tensor, dst_tensor = (
            (sD, tDgD_for_tma_partition) if is_s2g else (tDgD_for_tma_partition, sD)
        )
        # NOTE(l2-hints, tried July 2026): a cache_policy kwarg (PTX
        # createpolicy Int64) flows through tma_get_copy_fn -> block_copy ->
        # cute.copy if ever needed — we wired D stores as evict_first here and
        # B loads as evict_last in the mainloop and measured a net REGRESSION
        # (-0.9% plain / -3.5% AG at TP4 16384x4096x8192 settled). Don't
        # retry without profiling data showing D/B L2 residency is the
        # binding constraint (see gemm_sm100 load-warp note for details).
        return copy_utils.tma_get_copy_fn(
            atom,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            src_tensor=src_tensor,
            dst_tensor=dst_tensor,
        )

    def make_ab_pipeline(
        self,
        tiled_mma: cute.TiledMma,
        cluster_layout_vmnk: cute.Layout,
    ):
        # Threads/warps participating in this pipeline
        producer_cnt = 1 if const_expr(not self.gather_A) else 1 + self.num_ab_load_warps * 32
        # B's TMA warp contributes 1 (elect_one arrive_and_expect_tx); with gather_A the
        # cp.async producer warps contribute all 32 lanes each (cp.async.mbarrier.arrive).
        pipeline_checks.check_arrive_count(
            "sm90 ab_pipeline.producer",
            producer_cnt,
            pipeline_checks.tma_producer_arrives(
                num_tma_warps=1,
                cpasync_warps=0 if const_expr(not self.gather_A) else self.num_ab_load_warps,
            ),
            gather_A=self.gather_A,
            num_ab_load_warps=self.num_ab_load_warps,
        )
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, producer_cnt)
        # Each warp will contribute to the arrive count with the number of mcast size
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        consumer_arrive_cnt = mcast_size * tiled_mma.size // cute.arch.WARP_SIZE
        # One arrive per consumer (mma) warp, delivered to every CTA in this CTA's A- and
        # B-multicast groups (self counted once).
        pipeline_checks.check_arrive_count(
            "sm90 ab_pipeline.consumer",
            consumer_arrive_cnt,
            pipeline_checks.async_thread_arrives(
                tiled_mma.size // cute.arch.WARP_SIZE,
                per_warp=True,
                ctas_routed=pipeline_checks.mcast_peer_ctas(
                    num_mcast_ctas_a=self.num_mcast_ctas_a,
                    num_mcast_ctas_b=self.num_mcast_ctas_b,
                ),
            ),
            num_mma_warps=tiled_mma.size // cute.arch.WARP_SIZE,
            num_mcast_ctas_a=self.num_mcast_ctas_a,
            num_mcast_ctas_b=self.num_mcast_ctas_b,
        )
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )
        pipeline_cls = pipeline.PipelineTmaAsync if not self.gather_A else PipelineTmaCpAsync
        return pipeline_cls.create(
            num_stages=self.ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

    def make_epi_pipeline(
        self,
        tx_count: int,
    ):
        epi_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        # Each warp will contribute 1 to the arrive count
        consumer_arrive_cnt = self.num_epi_warps
        # One arrive per epilogue warp (elected lane), CTA-local barrier; per_warp bound
        # to the elect_one_release flag passed to create below.
        elect_one_release = True
        pipeline_checks.check_arrive_count(
            "epi_pipeline.consumer",
            consumer_arrive_cnt,
            pipeline_checks.async_thread_arrives(self.num_epi_warps, per_warp=elect_one_release),
            num_epi_warps=self.num_epi_warps,
        )
        epi_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )
        return PipelineTmaAsync.create(
            num_stages=self.epi_c_stage,
            producer_group=epi_pipeline_producer_group,
            consumer_group=epi_pipeline_consumer_group,
            tx_count=tx_count,
            defer_sync=True,
            elect_one_release=elect_one_release,
            syncwarp_before_release=True,
        )

    def make_epi_store_pipeline(self):
        num_epi_threads = self.num_epi_warps * cute.arch.WARP_SIZE
        epi_store_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_epi_threads)
        return pipeline.PipelineTmaStore.create(
            num_stages=self.epi_stage, producer_group=epi_store_producer_group
        )

    @staticmethod
    def _make_tma_epi_atoms_and_tensors(
        tensor_d: cute.Tensor,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_tile: Tuple[int, int],
        op_type: Literal["store", "load", "add"],
    ) -> Tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for storing D or loading C."""
        assert op_type in ["load", "store", "add"]
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        d_cta_v_layout = cute.composition(cute.make_identity_layout(tensor_d.shape), epi_tile)
        op = {
            "load": cpasync.CopyBulkTensorTileG2SOp(),
            "store": cpasync.CopyBulkTensorTileS2GOp(),
            "add": cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.ADD),
        }[op_type]
        tma_atom_d, tma_tensor_d = cpasync.make_tiled_tma_atom(
            op, tensor_d, epi_smem_layout, d_cta_v_layout
        )
        return tma_atom_d, tma_tensor_d

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor: cute.Tensor,
        smem_layout: cute.ComposedLayout,
        smem_tile: Tuple[int, int],
        mcast_dim: int,
        internal_type: Optional[Type[cutlass.Numeric]] = None,
    ) -> Tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for input tensors.

        ``internal_type=Int8`` with a packed-fp4 gmem tensor selects the
        16U4_ALIGN8B tensormap (SM120 mixed fp4 x fp8): each 16-element group
        lands as 8 packed-nibble bytes + 8 pad bytes of smem footprint, ready
        for the ldsm.b4x16_p64 unpacking ldmatrix."""
        # block_copy takes compiler-driven multicast metadata at the copy site,
        # so the TMA atom itself must stay the non-multicast variant here.
        op = cpasync.CopyBulkTensorTileG2SOp()
        tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
            op, tensor, smem_layout, smem_tile, internal_type=internal_type
        )
        return tma_atom, tma_tensor
