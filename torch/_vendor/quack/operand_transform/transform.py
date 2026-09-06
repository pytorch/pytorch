# Copyright (c) 2026, Tri Dao.
"""A-operand transforms for the register-sourced GEMM mainloops: SM90 RS
(WGMMA) and SM120 (warp MMA).

Both mainloops (``GemmSm90.mma_rs_interleaved``, CUTLASS rs_warpspecialized
scheme; ``GemmSm120.mma``, whose warp-level MMA always consumes A from
registers) produce the A fragment one k16 block at a time through an
abstract seam: ``copy_block(stage_idx, b, k_tile)``. The default produce is
the canonical ldmatrix s2r load (``gemm.canonical_a_load``); a transform
substitutes its own — a dequant of packed weights, or a value fn applied on
the way — while the mainloop keeps owning the MMA issue, the commit-group
discipline (SM90) and the pipeline waits. The A fragment atom is
``((2, 2, 2), MMA_M, MMA_K)`` with identical slot semantics on both archs (a
WGMMA m64 block is four warps' m16n8k16 fragments stacked in M), so the
per-block transform bodies are arch-independent; only the m64-block/thread
slot mapping of layout-owning decodes branches on ``gemm.arch``.

The kernel is agnostic to what the transform computes. It only consumes the
declarative contract below: A's storage layout (possibly transform-owned),
the required tile_K, and the in-kernel ``make_copy_block`` hook.

Ported from the transformA branch, restructured for the interleaved mainloop
(the branch's whole-tile ``make_mma_fn`` / double-buffered fragment scheme is
gone — main has a single tile-wide fragment and per-block produce). Runtime
operands of value fns are a KIND taxonomy over the transform's (M, K) index
space (see quack.operand_transform.kinds.ARG_KINDS), delivered via the aux
A-side TMA slot bundled into the mA argument
(:class:`TransformAOperand`), not extra kernel parameters. Shipped: the
strip family at 2-D (gran_m, gran_k) granularity — ``colvec_ktile`` /
``colvec_k64/k32/k16`` (per-(row, k-group), e.g. the linear-CE dx pow2
rescale) and ``kvec_m64`` (per-(m64 block, k-element), the LCE dw strip);
plus dropout (:class:`TransformADropout` — seed rides the bundle RAW via
``aux_raw``, per-tile coordinates via the ``on_work_tile`` hook and the
seam's global ``k_tile``). There is deliberately no k-invariant colvec kind
(per-row scales commute to the epilogue). NOT ported yet: the fp8 m-major
layout transform.
"""

from typing import NamedTuple, Optional

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr

from torch._vendor.quack.cute_dsl_utils import mlir_namedtuple
from torch._vendor.quack.operand_transform.kinds import ARG_KINDS


@mlir_namedtuple
class TransformAOperand(NamedTuple):
    """The A operand of a layout-owning transform, crossing the kernel
    boundary as ONE bundled argument in the mA slot — the mainloop analogue
    of EpilogueArguments: the host layer never learns the bundle's anatomy,
    and the GEMM signature arity stays fixed for plain GEMMs. ``blob`` is
    the repacked storage; ``sf`` the optional aux strip (TMA'd per k-tile
    alongside A under the same mbarrier — see AuxOperandA). Future transform
    runtime operands (colvec scales, dropout seeds) become new optional
    fields here, not new kernel parameters."""

    blob: cute.Tensor
    sf: Optional[cute.Tensor] = None


class AuxOperandA:
    """An extra A-side operand riding the AB pipeline: one TMA box per k-tile
    into its own per-stage smem buffer, arriving under the same mbarrier as A
    and B. GemmSm90 consumes this protocol (duck-typed); anything — a
    transform, or a future standalone feature — can install one.

    Contract:
      - ``dtype``: cutlass numeric type of the smem buffer.
      - ``bytes_per_stage()``: smem bytes per pipeline stage (stage-count
        heuristic input; must match the staged layout below).
      - ``make_smem_layout_staged(ab_stage)``: the (…, ab_stage) smem layout.
      - ``make_tma(mAux)``: TMA atom + tma tensor for the gmem operand.
      - ``gmem_slice(mAux, tile_coord_mnkl, batch_idx)``: the per-CTA-tile
        gmem view whose last mode walks k-tiles (source for the block copy).
      - ``multicast`` (optional, default True): False opts out of the A-side
        cluster multicast — small boxes (e.g. 128 B scale strips) load a full
        copy per CTA instead of splitting the box.
    """

    dtype = cutlass.Uint8


class TransformA:
    """A-operand transform: produce the WGMMA A fragment from smem in
    registers each k16 block, instead of the canonical ldmatrix s2r load.

    The contract with GemmSm90 is declarative — the kernel never learns what
    the transform computes, only:
      - The MMA compute dtype is the GEMM's own ``mma_a_dtype`` (its a_dtype
        constructor arg) — a transform never changes it, it must PRODUCE
        fragments in it (validate support in ``__init__``). ``a_major_mode``:
        the fragment major declared by layout-owning transforms (a storage
        blob has no natural major; must be canonical in K — B stays SS behind
        a canonical descriptor, so no K reorder is absorbable). ``tile_k``:
        required tile_K, or None.
      - ``owns_a_layout``: mA is not an (M, K) operand (e.g. a repacked blob);
        the transform then owns A's smem layout (``make_a_smem_layout_staged``,
        ``a_bytes_per_stage``), TMA (``make_a_tma``) and gmem slicing
        (``a_gmem_slice``), and the kernel skips (M, K)-based checks, batch
        rotation and length derivation (M comes from D).
      - ``aux``: optional :class:`AuxOperandA` installed by this transform;
        its smem arrives in ``make_copy_block`` as ``sAux``. The aux facility
        is transform-agnostic (per-row scales for a plain bf16 GEMM could
        ride it without any transform); a transform merely *installs* one
        (W4's scale-factor strip).
      - ``__init__(gemm)`` validates the config and may adjust register
        budgets / occupancy (runs after the gemm's defaults, before
        _setup_attributes).
      - ``make_copy_block(tiled_mma, sA, tCrA, tidx, warp_group_idx, sAux,
        mAux)``: called in-kernel by each MMA warpgroup; returns
        ``copy_block(stage_idx, b, k_tile)`` which produces k16 block ``b``
        (a static Python int: register indexing) of pipeline stage
        ``stage_idx`` into the fragment ``tCrA``; ``k_tile`` is the GLOBAL
        k-tile index the block belongs to (split-k correct — needed only by
        coordinate-dependent transforms like dropout). The mainloop calls it
        under the rs_warpspecialized schedule (produce of block b+1 between
        WGMMA(b) and WGMMA(b+1), slot-0 preload of the next k-tile during
        the last block) — per-block work only; the schedule is never the
        transform's.
      - ``aux_raw``: the bundle's ``sf`` tensor is NOT an AuxOperandA TMA
        operand but a small raw gmem tensor (e.g. a dropout seed) handed to
        ``make_copy_block`` as ``mAux`` untouched — no smem, no pipeline.
      - ``uses_work_tile``: the mainloop calls ``on_work_tile(tile_coord_mnkl)``
        at each work-tile start (MMA warps, before the k-tile loop) so the
        transform can refresh per-tile register state (e.g. per-row RNG
        coordinates). ``mma_rs_interleaved`` runs once per work tile, so all
        ``copy_block`` calls between two hooks — including the next-k-tile
        slot-0 preload — belong to the hooked tile.
      - ``promote``: slow-accum transform (W4A8): the mainloop zero-inits the
        WGMMA accumulator at each k-tile's block 0, drains (wait_group(0))
        after its last block, and calls ``promote_acc(acc_slow, acc_wave,
        zero_init)`` — the transform's fp32 fold of the k-tile's wave into
        the persistent accumulator (e.g. ``acc += scale_row * wave``). The
        drain kills cross-tile WGMMA overlap by design: with 2 cooperative
        warpgroups the other WG's WGMMAs fill the tensor pipe during a WG's
        drain+promote, and single-fresh-buffer drain-promote measured ahead
        of every double-buffered variant on SM90 (see memory: kscale).
    """

    a_major_mode = cute.nvgpu.OperandMajorMode.K
    tile_k = None  # None -> kernel default
    owns_a_layout = False
    aux = None
    aux_raw = False
    uses_work_tile = False
    promote = False


class AuxKTileStrip(AuxOperandA):
    """Byte-granular per-(row-block, k-tile) aux strip: ``sf_bytes`` per m64
    block per k-tile, plain (sfb, tm64, stage) smem, one (sfb, tm64) box per
    k-tile arriving with A under the AB mbarrier. Semantically this carries
    a colvec-per-k-chunk operand in a PACKED encoding — W4's SF words are
    such an instance (repack-ordered per m64 atom for the pair_slot LDS,
    consumed inside decode_k16). The dense canonical form of the same
    concept is :class:`_StripAux` (element-typed boxes, no m64 structure
    needed)."""

    def __init__(self, gemm, sf_bytes):
        self.gemm = gemm
        self.sf_bytes = sf_bytes

    def _tm64(self):
        return self.gemm.cta_tile_shape_mnk[0] // 64

    def bytes_per_stage(self):
        return self.sf_bytes * self._tm64()

    def make_smem_layout_staged(self, ab_stage):
        return cute.make_ordered_layout((self.sf_bytes, self._tm64(), ab_stage), order=(0, 1, 2))

    def make_tma(self, mAux):
        gemm = self.gemm
        sf_smem_layout = cute.make_ordered_layout((self.sf_bytes, self._tm64()), order=(0, 1))
        return gemm._make_tma_atoms_and_tensors(
            mAux, sf_smem_layout, (self.sf_bytes, self._tm64()), gemm.cluster_shape_mnk[1]
        )

    def gmem_slice(self, mAux, tile_coord_mnkl, batch_idx):
        # (sfb, tm64, Gt, RestK, L) -> (sfb, tm64, RestK)
        return mAux[None, None, tile_coord_mnkl[0], None, batch_idx]


class TransformAW4(TransformA):
    """Packed 4-bit / 8-bit weights as operand A, decoded to bf16 in
    registers and fed to RS WGMMA (W4A16; Hopper has no fp4 tensor cores).

    mA is the offline-repacked blob (see blockscaled/nvfp4_utils.repack_*):
    per (m64, k-tile) block each thread's 16 B (32 B for 8-bit weights or
    tile_k=128 formats) LDS lands values directly in WGMMA A-fragment order,
    so the decode is shuffle-free. ``copy_block(stage, 0)`` LDSes the k-tile's
    raw words and decodes block 0; blocks 1.. decode from the same registers
    (a tile's raw words are dead before the next tile's slot-0 produce, so a
    single register set suffices).

    Scale factors ride the aux-operand slot (a per-stage strip TMA'd next to
    A under the same mbarrier). Formats: nvfp4 (e2m1 + e4m3 SF per 16, scale
    folded in the decode — exact), int4 (u4b8 + bf16 group scale), int4awq
    (scale + zero, one HFMA2), mxfp4/mxfp8 (e8m0 per 32), int8/fp8 (no strip;
    per-channel scale left to the epilogue), qtip* (no strip; per-tensor
    scale rides alpha).

    This transform is requested explicitly rather than layout-detected: mA's
    shape alone does not identify the format. D is typically written
    (N_w, M_act) m-major (out = act @ W^T row-major).
    """

    owns_a_layout = True

    def __init__(self, gemm, w4_format):
        # Lazy: this module is imported by GemmSm90 — resolving the format
        # here keeps the W4 registry (and qtip) out of the kernel import path.
        from torch._vendor.quack.operand_transform.formats import decode_format

        self.fmt = decode_format(w4_format)
        assert gemm.mma_a_dtype == self.fmt.mma_dtype, (
            f"w4 format {self.fmt.name!r} decodes to {self.fmt.mma_dtype}, "
            f"but the GEMM was built for {gemm.mma_a_dtype}"
        )
        self.promote = self.fmt.promote
        self.gemm = gemm
        self.w4_format = self.fmt.name
        self.tile_k = self.fmt.tile_k
        # raw i32 words per thread per m64 block per stage
        self._nw = (8 if self.fmt.w8 else 4) * (self.tile_k // 64)
        # split_k measured working + winning on grid-starved decode shapes
        # (N/tile_m CTAs < machine): serial split-k 1.2-1.5x there.
        assert not gemm.gather_A
        assert not gemm.pingpong, "w4 only supports cooperative for now"
        assert gemm.cta_tile_shape_mnk[0] % 64 == 0, "w4 requires tile_M % 64 == 0"
        assert gemm.cluster_shape_mnk[0] == 1 and gemm.cluster_shape_mnk[2] == 1, (
            "w4 supports (1, cluster_N, 1) clusters"
        )
        if gemm.arch == 120:
            # Warp-MMA mainloop, atom_n == 1 (the gemm ctor picks (4,1,1) or
            # (8,1,1) for layout-owning transforms): an MMA_M step covers
            # atom_m * 16 rows = atom_m/4 m64 blocks, one per 4-warp group —
            # the same m64-block/warp-group split as a WGMMA warpgroup stack,
            # so the mapping below mirrors SM90's (m64 = m * groups + wg).
            assert gemm.atom_layout_mnk[0] in (4, 8), "w4 on SM120 needs 4- or 8-warp MMA_M steps"
            assert gemm.atom_layout_mnk[1] == 1, "w4 requires atom_layout_n == 1"
            assert not self.promote, (
                "W4A8 promote (int4sm) needs the per-k-tile promote seam; SM120's mma() "
                "does not implement it yet — use int4smf (folded, fast-accum) instead"
            )
        else:
            assert gemm.atom_layout_mnk[1] == 1, "w4 requires atom_layout_n == 1"
        if self.fmt.sf_words > 0:
            # the format's SF words: a compressed colvec-per-k-group instance
            # of the k-tile strip geometry, consumed inside decode_k16
            self.aux = AuxKTileStrip(gemm, self.fmt.sf_bytes)
        if gemm.arch == 120:
            # Small-N decode shapes stall at occupancy 1 (RTX 5090 measured:
            # ~500 GB/s of weight BW at (64, 32) vs ~1.5 TB/s machine peak;
            # split-k sweeps 1-8 move it < 1.3x, so it's consumer-side stall
            # latency, not grid coverage). Same fix as the SM90 rule below:
            # 2 CTAs/SM, with the budget arithmetic keyed on the CTA's warp
            # group count (2 mma WGs -> 384 threads, launch cap 80; 1 mma WG
            # ((4,1,1) decode layout) -> 256 threads, launch cap 128).
            if gemm.cta_tile_shape_mnk[1] <= 32:
                gemm.occupancy = 2
                if gemm.mma_warp_groups == 2:
                    gemm.num_regs_load, gemm.num_regs_mma = 32, 104
                else:
                    gemm.num_regs_load, gemm.num_regs_mma = 40, 152
        elif const_expr(gemm.cta_tile_shape_mnk[1] <= 32):
            # Small-N (decode) shapes are latency-bound: consumers need few
            # regs (small acc + A frag), so shrink budgets to fit 2 CTAs/SM
            # and double the warps available to hide LDS/decode latency.
            # Budget: with min_blocks_per_mp=2 ptxas caps launch regs at
            # floor(65536 / (2*threads) / 8) * 8, and setmaxnreg deadlocks if
            # the inc demand exceeds what the producer's dec released, so
            # keep 128*load + 256*mma (2 WG) within threads*launch_regs.
            gemm.occupancy = 2
            if gemm.mma_warp_groups == 2:
                gemm.num_regs_load, gemm.num_regs_mma = 32, 104  # 384thr @ 80
            else:
                # 2 WGs @ 256 threads, launch cap 128: math can take up to
                # (256*128 - 128*40)/128 = 216; give it slack so ptxas
                # keeps decode LUT constants resident instead of UR->R
                # rematerializing them in the mainloop.
                gemm.num_regs_load, gemm.num_regs_mma = 40, 152
        elif self.promote:
            # Slow accum doubles the accumulator (persistent + per-tile
            # wave); redo the ctor's heavy-pressure rule with the doubled
            # acc + the (8-bit) A fragment.
            tile_m, tile_n = gemm.cta_tile_shape_mnk[:2]
            acc_regs = tile_m * tile_n // (gemm.atom_layout_mnk[0] * 128)
            frag_regs = (tile_m // gemm.atom_layout_mnk[0]) * self.tile_k // (128 * 4)
            if 2 * acc_regs + frag_regs >= 208:
                gemm.num_regs_load, gemm.num_regs_mma = 24, 240

    # ---- A layout ownership -------------------------------------------------

    def a_bytes_per_stage(self):
        gemm = self.gemm
        tm64 = gemm.cta_tile_shape_mnk[0] // 64
        return 256 * (2 * self._nw) * tm64

    def make_a_smem_layout_staged(self, ab_stage):
        """A smem holds the repacked blob, 4 * nw B per thread slot per m64
        block, no swizzle; TMA-facing shape has a 256 B inner run."""
        gemm = self.gemm
        tm64 = gemm.cta_tile_shape_mnk[0] // 64
        return cute.make_ordered_layout(
            (256, 2 * self._nw, tm64, ab_stage),
            order=(0, 1, 2, 3),
        )

    def make_a_tma(self, mA):
        """mA is the blob (256, 2*nw, tm64, Gt, Kt, L); one (256, 2*nw, tm64)
        box per k-tile."""
        gemm = self.gemm
        tm64 = gemm.cta_tile_shape_mnk[0] // 64
        araw_smem_layout = cute.slice_(gemm.a_smem_layout_staged, (None, None, None, 0))
        return gemm._make_tma_atoms_and_tensors(
            mA,
            araw_smem_layout,
            (256, 2 * self._nw, tm64),
            gemm.cluster_shape_mnk[1],
        )

    def a_gmem_slice(self, mA, tile_coord_mnkl, batch_idx):
        # (256, 8|16, tm64, Gt, RestK, L) -> (256, 8|16, tm64, RestK)
        return mA[None, None, None, tile_coord_mnkl[0], None, batch_idx]

    # ---- the per-block produce ----------------------------------------------

    @cute.jit
    def _decode_block(self, xw, sfw, frag_i32, b, mma_m, consts):
        """Decode k16 block b (all MMA_M atoms) from preloaded raw words: the
        format's decode_k16 produces the 4 packed bf16x2 registers per m-atom
        in fragment slot order; the slot assignment here is format-agnostic."""
        for m in cutlass.range_constexpr(mma_m):
            r0, r1, r2, r3 = self.fmt.decode_k16(xw[None, m], sfw[None, m], b, consts)
            frag_i32[(0, 0, 0), m, b] = r0
            frag_i32[(0, 1, 0), m, b] = r1
            frag_i32[(0, 0, 1), m, b] = r2
            frag_i32[(0, 1, 1), m, b] = r3

    @cute.jit
    def make_copy_block(self, tiled_mma, sA, tCrA, tidx, warp_group_idx, sAux=None, mAux=None):
        gemm = self.gemm
        tm64 = gemm.cta_tile_shape_mnk[0] // 64
        nw = self._nw
        sA_i32 = cute.make_tensor(
            cute.recast_ptr(sA.iterator, dtype=Int32),
            cute.make_ordered_layout((nw, 128, tm64, gemm.ab_stage), order=(0, 1, 2, 3)),
        )
        sAux_i32 = cute.recast_tensor(sAux, Int32) if const_expr(sAux is not None) else None
        if const_expr(gemm.arch == 120):
            # Warp-MMA fragment ownership (atom_n == 1): an MMA_M step covers
            # atom_m/4 m64 blocks, one per 4-warp group — warp w covers rows
            # 16*(w%4)..+15 of its group's block, the same (warp, lane) ->
            # fragment-row map as a WGMMA warpgroup, so the repacked blob's
            # thread slots carry over with t128 = (w%4) * 32 + lane and
            # m64 = m * groups + warp_group_idx (groups = atom_m/4; the
            # (4,1,1) decode layout has one group, so wg is always 0 there).
            t128 = ((tidx // 32) % 4) * 32 + tidx % 32
            atom_m, wg = gemm.atom_layout_mnk[0] // 4, warp_group_idx
        else:
            t128 = tidx % 128
            atom_m, wg = gemm.atom_layout_mnk[0], warp_group_idx
        # this thread's SF word slot within the m64 block's 32 fragment rows:
        # (warp, quad) -> row pair (matches repack_*_sf's word order)
        pair_slot = (t128 // 32) * 8 + (t128 % 32) // 4
        consts = self.fmt.make_consts()
        frag_i32 = cute.recast_tensor(tCrA, Int32)
        mma_m = const_expr(cute.size(tCrA.shape[1]))
        sf_words = self.fmt.sf_words
        ts_words = const_expr(self.fmt.tile_state_words)
        xw = cute.make_rmem_tensor((nw, mma_m), Int32)
        sfw = cute.make_rmem_tensor((2, mma_m), Int32)
        tstate = None
        if const_expr(ts_words > 0):
            # per-tile register state derived from the strip words at each
            # tile's block-0 produce (e.g. folded-W4A8 scaled LUTs); decode
            # reads it in place of the raw strip words
            tstate = cute.make_rmem_tensor((ts_words, mma_m), Int32)
        if const_expr(self.promote):
            # promote reads the k-tile's scale words AFTER the next tile's
            # slot-0 preload has overwritten sfw (the preload is issued
            # before this tile's drain); move them to a second buffer at
            # b == 1 — by then this tile's words are in sfw (loaded at its
            # own b == 0, i.e. during the previous tile), and b == 1 always
            # precedes the preload.
            self._sfc = cute.make_rmem_tensor((sf_words, mma_m), Int32)

        def copy_block(stage_idx, b, k_tile=None):
            if const_expr(b == 0):
                for m in cutlass.range_constexpr(mma_m):
                    m64 = m * atom_m + wg
                    cute.autovec_copy(sA_i32[None, t128, m64, stage_idx], xw[None, m])
                    for w in cutlass.range_constexpr(sf_words):
                        sfw[w, m] = sAux_i32[sf_words * pair_slot + w, m64, stage_idx]
                    if const_expr(ts_words > 0):
                        self.fmt.build_tile_state(sfw[None, m], consts, tstate[None, m])
            if const_expr(self.promote and b == 1):
                for m in cutlass.range_constexpr(mma_m):
                    for w in cutlass.range_constexpr(sf_words):
                        self._sfc[w, m] = sfw[w, m]
            self._decode_block(
                xw, tstate if const_expr(ts_words > 0) else sfw, frag_i32, b, mma_m, consts
            )

        return copy_block

    @cute.jit
    def promote_acc(self, acc_slow, acc, zero_init: cutlass.Constexpr[bool] = False):
        """The k-tile's fp32 promotion: acc_slow (+)= scale_row * acc, the
        scale pair (row r, row r+8) from the strip word staged by
        copy_block. Called by the mainloop after the tile's WGMMA drain."""
        mma_m = const_expr(cute.size(acc.shape[1]))
        for m in cutlass.range_constexpr(mma_m):
            sf0, sf1 = self.fmt.promote_scale_pair(self._sfc[0, m])
            for r in cutlass.range_constexpr(2):
                acc_sl = acc_slow[(None, r, None), m, None]
                wave_sl = acc[(None, r, None), m, None]
                sf = sf0 if r == 0 else sf1
                if const_expr(zero_init):
                    acc_sl.store(wave_sl.load() * sf)
                else:
                    acc_sl.store(acc_sl.load() + wave_sl.load() * sf)


# Runtime operands of value-fn transforms are a KIND taxonomy over the
# transform's (M, K) index space — see quack.operand_transform.kinds
# (ARG_KINDS): each kind owns its geometry, device staging, and host
# view/fake in one object.


class TransformAValue(TransformA):
    """Value transform on an unpacked 16-bit A: the canonical ldmatrix s2r
    load, then the mod's fn applied in-place over the block's fragment
    elements in ``vec_size`` chunks (running in the WGMMA shadow under the
    interleaved schedule). The fn contract (see frontend.py): one lane's
    ``vec_size`` fragment elements as a TensorSSA vector in the MMA dtype,
    FRAGMENT-SLOT-ORDERED (2 adjacent k x 2 rows x 2 k-halves per block —
    not k-contiguous), same-length vector out; chunks are pair-aligned, so
    vec_size in {2, 4, 8}.

    ``mod.args`` ((param_name, kind) pairs): runtime operands — the fn's
    parameters between x and consts, each staged per element by its kind
    (see quack.operand_transform.kinds.ARG_KINDS). At
    most one aux-delivered operand for now (the bundle has a single aux
    slot); the host passes A as a ``TransformAOperand(A, view)`` bundle
    built by :func:`quack.operand_transform.host.transform_a_operand`."""

    def __init__(self, gemm, mod):
        self.gemm = gemm
        self.mod = mod
        assert gemm.mma_a_dtype.width == 16, (
            "value transforms ride the canonical ldmatrix s2r load (16-bit only)"
        )
        if getattr(mod, "regs", None) is not None:
            gemm.num_regs_load, gemm.num_regs_mma = mod.regs
        self._arg_impls = [
            ARG_KINDS[kind].device_arg(gemm) for _name, kind in getattr(mod, "args", ()) or ()
        ]
        aux_impls = [impl for impl in self._arg_impls if getattr(impl, "aux", None) is not None]
        assert len(aux_impls) <= 1, "one aux-delivered operand per transform (single aux slot)"
        if aux_impls:
            self.aux = aux_impls[0].aux

    @cute.jit
    def _apply_block(self, buf, b, mma_m, consts):
        """Apply the fn to k16 block b, vec_size at a time. The scalar
        staging copies are register moves that fold away in SSA; element
        order i0-fastest keeps chunks pair-aligned."""
        vec = self.mod.vec_size
        s0 = buf.shape[0]  # ((2, 2, 2)) fragment slot mode
        coords = [
            (i2, i1, i0)
            for i0 in range(cute.size(s0[2]))
            for i1 in range(cute.size(s0[1]))
            for i2 in range(cute.size(s0[0]))
        ]
        for m in cutlass.range_constexpr(mma_m):
            for c in cutlass.range_constexpr(len(coords) // vec):
                tmp = cute.make_rmem_tensor((vec,), buf.element_type)
                for i in cutlass.range_constexpr(vec):
                    tmp[i] = buf[coords[c * vec + i], m, b]
                args = [tmp.load()]
                for impl in self._arg_impls:
                    # per-element operand values, staged from the kind's
                    # register cache (folds to selects; fragment dtype so
                    # the fn math stays packed)
                    sv = cute.make_rmem_tensor((vec,), buf.element_type)
                    for i in cutlass.range_constexpr(vec):
                        sv[i] = impl.element(coords[c * vec + i], m, b)
                    args.append(sv.load())
                if const_expr(self.mod.consts is not None):
                    args.append(consts)
                y = self.mod.fn(*args)
                if const_expr(y.dtype != buf.element_type):
                    # fn math may promote (e.g. TensorSSA * python float ->
                    # f32); convert back to the MMA dtype — packed cvt.rn,
                    # free (see memory: dsl-to-packed-cvt).
                    y = y.to(buf.element_type)
                tmp.store(y)
                for i in cutlass.range_constexpr(vec):
                    buf[coords[c * vec + i], m, b] = tmp[i]

    @cute.jit
    def make_copy_block(self, tiled_mma, sA, tCrA, tidx, warp_group_idx, sAux=None, mAux=None):
        consts = None
        if const_expr(self.mod.consts is not None):
            consts = self.mod.consts()  # hoisted: once per kernel
        # the gemm owns the canonical produce (WGMMA RS ldmatrix on SM90,
        # warp-MMA ldmatrix on SM120 — same fragment atoms either way)
        load_block = self.gemm.canonical_a_load(tiled_mma, sA, tidx, tCrA)
        mma_m = const_expr(cute.size(tCrA.shape[1]))
        for impl in self._arg_impls:
            impl.setup(tiled_mma, tidx, mma_m, sAux)

        def copy_block(stage_idx, b, k_tile=None):
            for impl in self._arg_impls:
                impl.on_block(stage_idx, b, mma_m)
            load_block(stage_idx, b)
            self._apply_block(tCrA, b, mma_m, consts)

        return copy_block


class TransformADropout(TransformA):
    """Dropout: a philox-derived keep-mask ANDed onto the fragment. MASK-ONLY
    — no 1/(1-p) multiply in the mainloop; fold the scale into the epilogue
    (an alpha, or an epi-mod multiply).

    Scheme (see quack/operand_transform/rng.py): the mask of element (m, k)
    is a pure function of (m, k, seed, offset) — one philox4x32 call per
    canonical group (row-pair x 32-k quad-strided set), which is exactly a
    lane's fragment ownership, so generation is thread-local (no shuffles,
    no redundancy) and any kernel (dgrad epilogue, wgrad) regenerates the
    same mask. Block parity p, row v1 consume philox word v1 + 2p whole; the
    per-register apply is PRMT (bytes -> constant-exponent b16 lanes) +
    set.ge.u32.b16x2 (0xFFFF lane mask) + AND: 3 SASS per 2 elements, no
    float math. The keep threshold int(round(p_drop * 256)) is a host
    constant baked into the mod's semantic key.

    Delivery: the (2,) int64 [seed, offset] tensor rides the
    :class:`TransformAOperand` bundle's sf slot RAW (``aux_raw`` — no
    TMA/smem); per-row coordinates refresh at each work tile via
    ``on_work_tile``; the mask is split-k invariant because the seam's
    ``k_tile`` is global (k_tile_start included)."""

    aux_raw = True
    uses_work_tile = True

    def __init__(self, gemm, mod):
        self.gemm = gemm
        self.mod = mod
        assert gemm.mma_a_dtype in (cutlass.BFloat16, cutlass.Float16), (
            "dropout masks 16-bit A fragments"
        )
        # tile_K may be unresolved (0) at ctor time for (M, N) ctor tile
        # shapes; geometry checks live in make_copy_block (lazy, like strips)

    @cute.jit
    def on_work_tile(self, tile_coord_mnkl):
        m_base = tile_coord_mnkl[0] * self.gemm.cta_tile_shape_mnk[0]
        for ma in cutlass.range_constexpr(self._mma_m):
            r0 = m_base + self._row0[ma]
            self._gm[ma] = (r0 // 16) * 8 + r0 % 8

    @cute.jit
    def _gen_pair(self, xw, span, mma_m):
        """philox words for global 32-k span ``span``, all m-atoms."""
        from cutlass import Uint64

        from torch._vendor.quack.operand_transform import rng

        kg = (span << 2) | self._q
        for ma in cutlass.range_constexpr(mma_m):
            cnt = (Uint64(Int32(kg)) << Uint64(32)) | Uint64(self._gm[ma])
            x0, x1, x2, x3 = rng.philox(cnt, self._key, n_rounds=self.mod.rounds)
            xw[0, ma] = Int32(x0)
            xw[1, ma] = Int32(x1)
            xw[2, ma] = Int32(x2)
            xw[3, ma] = Int32(x3)

    @cute.jit
    def _mask_block(self, frag_i32, xw, b, mma_m):
        """AND the keep-mask onto k16 block b: block parity p, row v1 consume
        philox word v1 + 2p; per register one PRMT + SET + AND."""
        from torch._vendor.quack.blockscaled.nvfp4_utils import prmt

        from torch._vendor.quack.operand_transform import rng

        p = b % 2
        for ma in cutlass.range_constexpr(mma_m):
            for v1 in cutlass.range_constexpr(2):
                word = xw[v1 + 2 * p, ma]
                for h in cutlass.range_constexpr(2):
                    sel = rng.PRMT_SEL_H0 if h == 0 else rng.PRMT_SEL_H1
                    lanes = prmt(word, self._base_bytes, Int32(sel))
                    mask = rng.set_ge_u32_b16x2(Int32(lanes), self._thr_pair, self.gemm.mma_a_dtype)
                    frag_i32[(0, v1, h), ma, b] = frag_i32[(0, v1, h), ma, b] & mask

    @cute.jit
    def make_copy_block(self, tiled_mma, sA, tCrA, tidx, warp_group_idx, sAux=None, mAux=None):
        from cutlass import Int64, Uint64

        from torch._vendor.quack.operand_transform import rng

        gemm = self.gemm
        tile_m, tile_k = gemm.cta_tile_shape_mnk[0], gemm.cta_tile_shape_mnk[2]
        assert tile_k % 32 == 0, "dropout groups span 32 k"
        assert mAux is not None, "dropout needs the (2,) int64 [seed, offset] via mA.sf"
        load_block = gemm.canonical_a_load(tiled_mma, sA, tidx, tCrA)
        base = rng.b16_base_pattern(gemm.mma_a_dtype)
        t = self.mod.threshold
        self._thr_pair = Int32((base | t) | ((base | t) << 16))
        self._base_bytes = Int32((base >> 8) * 0x01010101)
        # per-THREAD fragment coordinates: logical (m, k) of slot (e, v1, h)
        cA = cute.make_identity_tensor((tile_m, tile_k))
        tCcA = tiled_mma.get_slice(tidx).partition_A(cA)
        mma_m = const_expr(cute.size(tCcA.shape[1]))
        self._mma_m = mma_m
        # tile-relative row of each m-atom's first slot (static per lane) and
        # the quad class from the k coord of slot (e=0, v1=0, h=0): k = 2q
        self._row0 = [tCcA[(0, 0, 0), ma, 0][0] for ma in range(mma_m)]
        self._q = (tCcA[(0, 0, 0), 0, 0][1] % 8) // 2
        self._gm = cute.make_rmem_tensor((mma_m,), Int32)
        seed, offset = Int64(mAux[0]), Int64(mAux[1])
        # offset stream folded into the key (counter words carry coordinates)
        self._key = (seed + offset * Int64(rng.PHILOX_OFFSET_MIX)).to(Uint64)
        xw = cute.make_rmem_tensor((4, mma_m), Int32)
        frag_i32 = cute.recast_tensor(tCrA, Int32)
        spans = const_expr(tile_k // 32)

        def copy_block(stage_idx, b, k_tile):
            load_block(stage_idx, b, k_tile)
            # regenerate at each 32-k span boundary; span s+1 lands in the
            # WGMMA shadow like every other produce
            if const_expr(b % 2 == 0):
                self._gen_pair(xw, k_tile * spans + b // 2, mma_m)
            self._mask_block(frag_i32, xw, b, mma_m)

        return copy_block
