# Copyright (c) 2026, Tri Dao.
"""Quantized-output (SFD) epilogue op: blocked scale factors for fp8/fp4 D.

BlockScaleFactorStore writes the (L, rm, rk, 32, 4, 4) blocked scale-factor
tensor for an mxfp8 / mxfp4 / nvfp4 D output and rescales the accumulator
fragments in place so the ordinary f32 -> d_dtype store produces the quantized
values. Semantics are bit-exact with cuBLAS and the CUTLASS C++
Sm100BlockScaleFactorRow/ColStore (see quack/blockscaled/quantize_utils.py for the
contract); the quantize core itself is shared with the fused RMSNorm /
LayerNorm quantized forward through quack.blockscaled.quantize_utils.

Direction is a parameter: "row" (default) puts SF vectors along N — the output
feeds the next GEMM's K — and "col" puts them along M for backward consumers
that contract over this output's M.

Arches: both directions run on the SM100 tmem epilogue and the SM90-style
register epilogue (SM120 warp MMA) — the lane/warp geometry is derived from
tiled_copy_t2r or, when that is None, the r2s copy's source TV layout. On
SM120 a vec-32 row SFD normally widens the tiled-MMA warp run to 32
contiguous N columns (GemmSm120.mma_n_warp_run) so the whole SF vector is
warp-local; when the permutation cannot widen (tile_n not a multiple of
32 * atom_n), and always in the col direction (warp M stripes are the 16-row
mma instruction tiles), vectors spanning warp neighbors combine their amaxes
through a small smem exchange instead, mirroring the CUTLASS C++
Sm120BlockScaleFactorRow/ColStore. SM100's col direction keeps its dedicated
redux.sync path (lane == row). Aux-target (gated postact) quantization runs on
both arches too; on SM120 it keeps the 16-column warp run (the halved postact
store's dummy-MMA STSM contract encodes it) and quantizes via the exchange.
"""

import math
from typing import NamedTuple, Optional

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Int32, const_expr

from torch._vendor.quack.epilogue.ops import EpiOp, EpiSmemBytes, _get_lane_warp_layouts
from torch._vendor.quack.gemm_tvm_ffi_utils import make_fake_sf_tensor
from torch._vendor.quack.reduce import warp_reduce
from torch._vendor.quack.blockscaled.quantize_utils import (
    FLT_MAX,
    QUANT_DTYPE_MAX,
    quantize_sf_slots,
    sf_vec_size_for,
)
import torch._vendor.quack.layout_utils as layout_utils
import torch._vendor.quack.utils as utils


def active_row_sfd_reqs(epi_ops, epilogue_args):
    """Static epi-tile requirements of the active row-direction SFD codecs in
    ``epi_ops``, as ``(vec_acc, epi_n_min)`` (both 0 when none is active).
    Consulted by the arch kernels' _setup_attributes (SM100 widens the epi
    tile N, SM120 additionally widens the tiled-MMA warp N run), which run
    BEFORE arg filtering — pass the class-level op tuple and the raw args.

    vec_acc: the widest SF vector in ACCUMULATOR columns — sf_vec, doubled
    for a gated aux target (postact column j comes from acc columns
    {2j, 2j+1}).
    epi_n_min: the minimum epilogue-tile N run in acc columns — whole SF
    vectors, and for a sub-byte aux target at least 128 contiguous bits of
    values per row (the smem/TMA atom granule; D's own epi tile already
    respects its granule).
    """
    vec_acc, epi_n_min = 0, 0
    for op in epi_ops:
        if not (isinstance(op, BlockScaleFactorStore) and op.direction == "row"):
            continue
        arg = getattr(epilogue_args, op.name, None)
        if arg is None:
            continue
        vec = sf_vec_size_for(arg.element_type)
        min_vals, mult = vec, 1
        if op.quant_output != "D":
            store = next(o for o in epi_ops if o.is_tile_store() and o.name == op.quant_output)
            mult = 2 if store.gated else 1
            aux_width = getattr(epilogue_args, op.quant_output).element_type.width
            min_vals = max(vec, 128 // aux_width)
        vec_acc = max(vec_acc, vec * mult)
        epi_n_min = max(epi_n_min, min_vals * mult)
    return vec_acc, epi_n_min


class _SfdRowState(NamedTuple):
    tDgSFD: cute.Tensor  # partitioned gmem SF view, (CPY, CPY_M, CPY_N, EPI_M, EPI_N)
    tDrSFD: cute.Tensor  # whole-tile SF-byte registers, same partitioning
    tDrAmax: cute.Tensor  # per-subtile fp32 amax scratch (broadcast slot layout)
    tDrScale: cute.Tensor  # per-subtile fp32 rescale scratch
    limit_m: Int32
    limit_n: Int32
    lane_span: int  # static: lane_N subgroups sharing one SF vector (in-warp part)
    lane_stride: int  # static: lane stride along the vector's axis
    norm_const: Optional[Float32]
    # Cross-warp part (SM120 warp-MMA layouts, where a 32-wide SF vector can
    # span two warps along its axis — 16-column warp N runs for row, 16-row
    # warp M stripes for col): the vector's amax is combined across
    # `warp_span` warp neighbors through smem.
    warp_span: int  # static: warps sharing one SF vector (1 = in-warp)
    warp_member: Int32  # this warp's index within its exchange group
    sExch: Optional[cute.Tensor]  # (sf slots of the subtile, warp_span, 2) f32 smem
    tDcD: Optional[cute.Tensor]  # partitioned (m, n) coords for smem slot indexing


class _SfdColState(NamedTuple):
    tDgSFD: cute.Tensor
    tDcSFD: cute.Tensor  # partitioned (m, n) coordinate tensor for bounds
    limit_m: Int32
    limit_n: Int32
    norm_const: Optional[Float32]


class _SfdRowLoopState(NamedTuple):
    """Per-subtile quantize state (begin_loop slices the whole-tile state so
    the driver's store loop, via :meth:`BlockScaleFactorStore.quantize`, works
    on this subtile directly)."""

    tDrAmax: cute.Tensor
    tDrScale: cute.Tensor
    tDrSFD_cur: cute.Tensor
    lane_span: int
    lane_stride: int
    norm_const: Optional[Float32]
    xwarp: Optional[tuple]  # cross-warp amax exchange bundle (SM120), or None


class _SfdColLoopState(NamedTuple):
    tDgSFD_cur: cute.Tensor
    tDcSFD_cur: cute.Tensor
    limit_m: Int32
    limit_n: Int32
    norm_const: Optional[Float32]


class BlockScaleFactorStore(EpiOp):
    """Output-quantization scale factors (SFD) for mxfp8 / mxfp4 / nvfp4 D.

    Per epilogue subtile, computes the absolute maximum of every SF vector
    (sf_vec_size = 32 for e8m0 scales, 16 for e4m3 scales, contiguous along N),
    quantizes ``scale = amax / d_dtype_max * norm_const`` to the SF dtype
    (f32->e8m0 rounds up, f32->e4m3 rounds to nearest even — hardware cvt
    semantics, same as the CUTLASS C++ Sm100BlockScaleFactorRowStore), and
    rescales tRS_rD in place by ``min(norm_const * rcp(dequant(SF)), FLT_MAX)``
    so the downstream fp32 -> d_dtype conversion produces the quantized values.

    SF bytes for the whole CTA tile are kept in registers and written to gmem
    once per tile in ``end`` with plain vectorized stores (the 128x4 SF atom
    puts 4 consecutive SF slots of a row at consecutive bytes, so a full row
    flushes as 4-byte stores). No TMA; smem only for the cross-warp amax
    exchange on SM120 warp-MMA layouts (see the smem hooks below).

    The gmem tensor is the blocked (L, rm, rk, 32, 4, 4) scale layout shared
    with SFA/SFB; ``to_params`` re-views it with the logical (M_pad, N_pad, L)
    layout whose intra-vector mode has stride 0, so it can be tiled and
    partitioned exactly like D. Register/gmem slot pairing needs no assumption
    on the fragment order: SF slots live in a zero-stride-broadcast register
    tensor partitioned the same way as tRS_rD (the VecReduce trick), and
    ``filter_zeros`` on both sides yields matching slot orders.

    Bounds: SF storage is padded to 128-row x 4-slot atoms, so M is either
    fully covered or fully out for a CTA (tile_M and the padded extent are both
    multiples of 128 per CTA). Along N: when tile_N covers whole SF atom
    columns (tile_N % (4 * vec) == 0), the gmem view is CTA-tiled and a fully
    in-bounds tile flushes with one autovec copy; otherwise (or for a partial
    last N tile) the flush runs per epi subtile with a uniform bounds check —
    every epi subtile N width divides the atom coverage, so each subtile is
    either fully inside or fully outside the padded SF extent.
    """

    def __init__(self, name, direction="row", output="D"):
        """direction selects the SF vector orientation: "row" = vec contiguous
        elements along N (output feeds the next GEMM's K = this N), "col" =
        vec contiguous elements along M (for backward, where the consumer
        contracts over this output's M). "col" quantization reduces across
        lanes: on SM100, with the (4,1) epilogue warp shape and the 32dp32b
        tmem load, lane == row within a warp's 32-row stripe, so an SF vector
        of 32 (16) rows is exactly one warp (half-warp) — a butterfly-shuffle
        max per column, no smem, SF bytes stored per byte by a designated
        lane (adjacent columns' bytes are 16 B apart in the transposed atom,
        so no packing is possible — CUTLASS ColStore behaves the same). On
        SM120 col runs the generic slot machinery with the vector axis
        along M (stride-4 lane groups, warp_m-pair smem exchange).

        output: which stored output this codec quantizes. "D" (default) is
        the main output; a minted-mod output name (e.g. "postact") quantizes
        that aux output instead. Either way the driver's store loop calls
        :meth:`quantize` on the output's final fragment right before its
        store op converts to the storage dtype (see gemm_base.epilogue).
        For gated (acc_pair) aux outputs the SF slots live in ACCUMULATOR
        space — postact column j comes from acc columns {2j, 2j+1}, so one
        SF vector of ``vec`` postact values spans ``2 * vec`` acc columns
        (vec_acc below) and the fragment indexes its slot at acc_stride * i.
        The gmem SF atom layout is vec-agnostic, so only the broadcast factor
        changes.
        """
        super().__init__(name)
        assert direction in ("row", "col")
        assert output == "D" or direction == "row", "aux quantization is row-direction only"
        self.direction = direction
        self.quant_output = output

    def config_key(self):
        return (self.direction, self.quant_output)

    def _col_redux_path(self, arch):
        """Whether the col direction takes SM100's dedicated implementation
        instead of the generic slot machinery. Justified fork, not incidental:
        the SM100 tmem-load geometry puts lane == row, so each column's amax
        is ONE redux.sync.max.abs.NaN.f32 (SM100-family instruction) and SF
        bytes store inline per subtile — whole-tile register accumulation
        would need one slot byte per fragment element (every column is its
        own slot; nothing aliases). SM120's warp-MMA fragments spread each
        column over stride-4 lane groups and 16-row warp stripes, so col
        rides the generic butterfly + sExch path there. ``arch`` may be None
        (arch-less smem budgeting): treated as non-SM100."""
        return self.direction == "col" and arch == 100

    def host_fake_arg(self, key, fctx):
        # Blocked (L, rm, rk, 32, 4, 4) fake; varlen_m SFD is one M-padded
        # buffer with a static batch dim of 1 (same convention as input SFA).
        dtype, ndim = key
        assert ndim == 6, f"SFD expects a 6-D blocked scale tensor, got ndim {ndim}"
        return make_fake_sf_tensor(dtype, 1 if fctx.varlen_m else fctx.l)

    def _vec_attr(self):
        return f"_sf_vec_{self.name}"

    def _sf_dtype_attr(self):
        return f"_sf_dtype_{self.name}"

    def _val_dtype_attr(self):
        return f"_sf_val_dtype_{self.name}"

    def _acc_mult_attr(self):
        return f"_sf_acc_mult_{self.name}"

    def to_params(self, gemm, args):
        mSFD = getattr(args, self.name)
        # The optional fp32 norm constant (reciprocal of the nvfp4 per-tensor
        # scale) is THIS op's parameter: pair it into the param so the device
        # hooks are self-contained. The EpilogueArguments field stays flat
        # (host FFI schemas carry no tuples).
        norm_const = getattr(args, "sfd_norm_const", None)
        sf_dtype = mSFD.element_type
        if self.quant_output == "D":
            val_dtype, val_layout = gemm.d_dtype, gemm.d_layout
            acc_mult = 1
        else:
            # Aux-output quantization: the value tensor is the owning
            # TileStore's arg; a gated (acc_pair) output pairs adjacent
            # accumulator columns, so one postact column covers 2 acc columns.
            val_tensor = getattr(args, self.quant_output)
            val_dtype = val_tensor.element_type
            val_layout = cutlass.utils.LayoutEnum.from_tensor(val_tensor)
            store = next(
                op for op in gemm._epi_ops if op.is_tile_store() and op.name == self.quant_output
            )
            acc_mult = 2 if store.gated else 1
            assert not getattr(gemm, "_epi_mod_packed_cd", False), (
                "aux quantization does not support packed_cd mods"
            )
        # Both directions and aux-output (postact) quantization run on the
        # SM100 tmem epilogue and the SM90-style register epilogue (SM120
        # warp MMA).
        assert gemm.arch in (100, 120), (
            "Quantized output (SFD) is only supported on SM100/SM120 for now"
        )
        assert sf_dtype in (cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN), (
            f"SFD dtype must be e8m0 (mx) or e4m3 (nvfp4), got {sf_dtype}"
        )
        vec = sf_vec_size_for(sf_dtype)
        assert val_dtype in QUANT_DTYPE_MAX, (
            f"Quantized output requires fp8/fp4 values, got {val_dtype}"
        )
        assert not (sf_dtype is cutlass.Float8E4M3FN and val_dtype is not cutlass.Float4E2M1FN), (
            "e4m3 scale factors (nvfp4) require Float4E2M1FN output"
        )
        assert val_layout.is_n_major_c(), "Quantized output requires row-major (N-major) values"
        assert gemm.acc_dtype is Float32, "Quantized output requires fp32 accumulation"
        assert not (gemm.cta_tile_shape_mnk[0] == 64 and getattr(gemm, "use_2cta_instrs", False)), (
            "Quantized output (SFD) requires the (4,1) epilogue warp shape: the 64-row "
            "2-CTA tile splits SF vectors across threads (use tile_m 256 or cluster_m 1)"
        )
        if self.direction == "col":
            return self._to_params_col(gemm, mSFD, sf_dtype, vec, norm_const)
        # The op works entirely in accumulator-N space: identity for D, and
        # acc_mult * vec consecutive acc columns per SF vector for a gated
        # aux target (postact interleaves gate/up at pair granularity).
        vec_acc = vec * acc_mult
        assert gemm.cta_tile_shape_mnk[1] % vec_acc == 0
        epi_tile_n = gemm.epi_tile[1]
        # Each SF vector must be produced within a single epilogue subtile: the
        # contiguous N run of a subtile must cover whole SF vectors. epi_tile_n
        # is an int shape normally, or a strided layout from the non-pow2
        # tile_n fixup in which case mode 0 is the contiguous run.
        if isinstance(epi_tile_n, cute.Layout):
            shape = epi_tile_n.shape
            epi_n_run = shape if isinstance(shape, int) else shape[0]
        else:
            epi_n_run = cute.size(epi_tile_n)
        assert epi_n_run % vec_acc == 0, (
            f"epi tile N run ({epi_n_run}) must cover whole SF vectors of {vec_acc} acc columns"
        )
        setattr(gemm, self._vec_attr(), vec_acc)
        setattr(gemm, self._sf_dtype_attr(), sf_dtype)
        setattr(gemm, self._val_dtype_attr(), val_dtype)
        setattr(gemm, self._acc_mult_attr(), acc_mult)
        # (L, rm, rk, 32, 4, 4) blocked scale tensor -> logical (M_pad, N_pad, L)
        # view with the hardware 128x4 SF atom and stride-0 intra-vector mode
        # (N_pad in ACC columns: rk * 4 slots * vec_acc — the 512 B atom is
        # vec-agnostic, only the broadcast factor differs for gated aux).
        # For varlen_m the buffer is one M-padded (1, total_padded_rm, rk)
        # region (tile-aligned per-batch padding, same convention as input
        # SFA); build a rank-2 view so offset_batch_SFA's compound-coord tile
        # shift applies.
        m_pad = mSFD.shape[1] * 128
        n_pad = mSFD.shape[2] * vec_acc * 4
        if getattr(gemm, "varlen_m", False):
            sfd_shape = (m_pad, n_pad)
        else:
            sfd_shape = (m_pad, n_pad, mSFD.shape[0])
        sfd_layout = layout_utils.tile_atom_to_shape_SF_strided(sfd_shape, vec_acc, mSFD.stride)
        return {self.name: (cute.make_tensor(mSFD.iterator, sfd_layout), norm_const)}

    def _to_params_col(self, gemm, mSFD, sf_dtype, vec, norm_const):
        """Column-direction SF: vectors of `vec` rows along M. The blocked
        tensor is (L, rn, rm_k, 32, 4, 4) — 128-column atoms with rm_k
        4-slot groups along M. Build a logical (M_pad, N_pad, L) view whose
        M mode broadcasts over the vec rows sharing one SF byte."""
        assert not getattr(gemm, "varlen_m", False) and not getattr(gemm, "varlen_k", False), (
            "Column-direction SFD does not support varlen yet"
        )
        if gemm.arch == 120:
            # The slot machinery produces each SF vector within one epilogue
            # subtile; its M extent must cover whole vec-row vectors.
            epi_m = cute.size(cute.shape(gemm.epi_tile[0]))
            assert epi_m % vec == 0, (
                f"epi tile M ({epi_m}) must cover whole SF vectors of {vec} rows"
            )
        setattr(gemm, self._vec_attr(), vec)
        setattr(gemm, self._sf_dtype_attr(), sf_dtype)
        setattr(gemm, self._val_dtype_attr(), gemm.d_dtype)
        setattr(gemm, self._acc_mult_attr(), 1)
        s = mSFD.stride
        # (N, M, L): N = ((32, 4), rn) column atoms; M = ((vec, 4), rm_k)
        # with the vec rows broadcast (stride 0) and 4 consecutive M slots at
        # consecutive bytes. M is all-or-nothing per CTA because the host
        # allocates rm_k in whole 128-row stripes (multiples of 128/(4*vec)).
        layout_nml = cute.make_layout(
            (((32, 4), mSFD.shape[1]), ((vec, 4), mSFD.shape[2]), mSFD.shape[0]),
            stride=(((16, 4), s[1]), ((0, 1), s[2]), s[0]),
        )
        sfd_layout = cute.select(layout_nml, [1, 0, 2])  # (M_pad, N_pad, L)
        return {self.name: (cute.make_tensor(mSFD.iterator, sfd_layout), norm_const)}

    def param_fields(self):
        return [(self.name, object, None)]

    # --- Cross-warp amax exchange smem (SM120 warp-MMA epilogues only) ---
    #
    # A 32-wide SF vector can span warp neighbors along its axis: row
    # direction when the tiled-MMA warp N run is 16 (interleaved 16-column
    # runs; the run is normally widened to 32 instead, see
    # GemmSm120.mma_n_warp_run), col direction always (warp M stripes are the
    # 16-row mma instruction tiles). The neighbors' partial amaxes meet
    # through this buffer: one f32 per (epi-subtile SF slot, member warp),
    # double-buffered by subtile parity so one epilogue-barrier per subtile is
    # WAR-safe. 16-wide vectors (nvfp4) never allocate. The host-side
    # condition intentionally matches the trace-time warp_span > 1 geometry;
    # begin() asserts if they ever disagree.

    def _xwarp_smem_shape(self, epi_tile, warps):
        epi_m = cute.size(cute.shape(epi_tile[0]))
        epi_n = cute.size(cute.shape(epi_tile[1]))
        if self.direction == "col":
            return (max(epi_m // 32, 1), epi_n, warps, 2)
        return (epi_m, max(epi_n // 32, 1), warps, 2)

    def _xwarp_smem_needed(self, vec_eff, warp_shape_mnk, n_warp_run=16, arch=None):
        # vec_eff: the SF vector width in acc elements (sf_vec, doubled for a
        # gated aux target). n_warp_run: consecutive N columns per warp in the
        # tiled-MMA permutation (GemmSm120.mma_n_warp_run) — a vector no wider
        # than the run is warp-local; smem_bytes budgets with the conservative
        # default 16 (the run is chosen later), the struct hooks pass the real
        # value.
        if warp_shape_mnk is None:
            return False
        if self.direction == "col":
            # Slot-machinery col only (SM120 warp MMA: 16-row warp M stripes
            # vs the 32-row vector). SM100's dedicated redux path needs no
            # smem; its (4,1) epi warp shape is also excluded by
            # warp_shape[1] > 1 for the arch-less smem_bytes budgeting, the
            # struct hooks pass arch.
            return (
                not self._col_redux_path(arch)
                and warp_shape_mnk[0] > 1
                and warp_shape_mnk[1] > 1
                and vec_eff > 16
            )
        return warp_shape_mnk[1] > 1 and vec_eff > n_warp_run

    def _xwarp_warps(self, warp_shape_mnk):
        return warp_shape_mnk[0] if self.direction == "col" else warp_shape_mnk[1]

    def smem_bytes(self, arg_tensor, cta_tile_shape_mnk, epi_tile, warp_shape_mnk=None):
        # Budget conservatively for an aux target: gated-ness is unknown here,
        # assume the doubled acc-space vector.
        vec_eff = sf_vec_size_for(arg_tensor.element_type) * (1 if self.quant_output == "D" else 2)
        if not self._xwarp_smem_needed(vec_eff, warp_shape_mnk):
            return EpiSmemBytes()
        return EpiSmemBytes(
            unstaged=math.prod(self._xwarp_smem_shape(epi_tile, self._xwarp_warps(warp_shape_mnk)))
            * 4
        )

    def _xwarp_smem_shape_for(self, gemm):
        """The exchange smem shape for this compiled config, or None when the
        exchange is not needed — the shared guard of the two smem hooks (the
        exact geometry, unlike smem_bytes' conservative budget)."""
        vec_acc = getattr(gemm, self._vec_attr())
        run = getattr(gemm, "mma_n_warp_run", 16)
        warp_shape = gemm.epi_smem_warp_shape_mnk()
        if not self._xwarp_smem_needed(vec_acc, warp_shape, run, arch=gemm.arch):
            return None
        return self._xwarp_smem_shape(gemm.epi_tile, self._xwarp_warps(warp_shape))

    def smem_struct_field(self, gemm, params):
        shape = self._xwarp_smem_shape_for(gemm)
        if shape is None:
            return None
        return (
            f"s_{self.name}_xw",
            cute.struct.Align[cute.struct.MemRange[Float32, math.prod(shape)], 16],
        )

    def get_smem_tensor(self, gemm, params, storage_epi):
        shape = self._xwarp_smem_shape_for(gemm)
        if shape is None:
            return None
        return getattr(storage_epi, f"s_{self.name}_xw").get_tensor(cute.make_layout(shape))

    @cute.jit
    def _sfd_gmem_view(self, gemm, param, ctx, atom_cols):
        """Partitioned CTA-tile view of the logical (M_pad, N_pad, L) SF
        tensor plus the padded-extent limits — shared by the slot-machinery
        ``begin`` and the SM100 col ``_begin_col``. CTA-tiles directly when
        the tile covers whole SF atoms along N (``atom_cols``: 4 * vec for
        row, 128 transposed-atom columns for col) so a fully in-bounds tile
        can flush with one autovec copy; otherwise (a fractional atom, e.g.
        tile_n 192 vs 128 columns) shifts the origin (domain_offset only
        evaluates the layout, no division) and lets partition_for_epilogue
        tile by epi_tile, with per-subtile bounds checks at the stores.
        Limits are vs the padded SF extents: M is all-or-nothing per CTA
        (tile grid and pad are both 128-quantized; for varlen_m the extent is
        this batch's tile-aligned padded region, same convention as input
        SFA); N can be partial when tile_N doesn't divide the atom coverage.
        """
        tile_M, tile_N = ctx.tile_M, ctx.tile_N
        if const_expr(ctx.varlen_manager.varlen_m):
            # Rank-2 M-padded view; shift to this batch's tile-aligned region
            # (cu_seqlens_m[b] // 128 + b tiles), same formula as input SFA.
            mSFD_mn = ctx.varlen_manager.offset_batch_SFA(param, ctx.batch_idx)
        else:
            mSFD_mn = param[None, None, ctx.batch_idx]
        if const_expr(tile_N % atom_cols == 0):
            gSFD = cute.local_tile(
                mSFD_mn, (tile_M, tile_N), (ctx.tile_coord_mnkl[0], ctx.tile_coord_mnkl[1])
            )
        else:
            gSFD = cute.domain_offset(
                (ctx.tile_coord_mnkl[0] * tile_M, ctx.tile_coord_mnkl[1] * tile_N), mSFD_mn
            )
        # (CPY, CPY_M, CPY_N, EPI_M, EPI_N); un-retiled t2r partitions index
        # element-wise compatibly with tRS_rD (same TV layout; see VecReduce).
        tDgSFD = ctx.partition_for_epilogue_fn(gSFD)
        if const_expr(ctx.varlen_manager.varlen_m):
            len_m = ctx.varlen_manager.len_m(ctx.batch_idx)
            m_extent = (len_m + 127) // 128 * 128
        else:
            m_extent = cute.size(mSFD_mn, mode=[0])
        limit_m = Int32(m_extent - ctx.tile_coord_mnkl[0] * tile_M)
        limit_n = Int32(cute.size(mSFD_mn, mode=[1]) - ctx.tile_coord_mnkl[1] * tile_N)
        return tDgSFD, limit_m, limit_n

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        param, norm_const_arg = param
        # Load the norm constant once per tile (immediate or device pointer).
        norm_const = None
        if const_expr(norm_const_arg is not None):
            norm_const = utils.load_scalar_or_pointer(norm_const_arg)
        if const_expr(self._col_redux_path(gemm.arch)):
            return self._begin_col(gemm, param, ctx, norm_const)
        # Slot-machinery path: row direction on both arches, col on SM120.
        # ``ax`` is the SF vector's axis (1 = along N, 0 = along M); the
        # broadcast layout, lane/warp geometry and exchange slot indexing are
        # the same construction with M and N swapped.
        ax = const_expr(0 if self.direction == "col" else 1)
        vec = getattr(gemm, self._vec_attr())
        tile_M, tile_N = ctx.tile_M, ctx.tile_N
        tDgSFD, limit_m, limit_n = self._sfd_gmem_view(
            gemm, param, ctx, atom_cols=const_expr(128 if ax == 0 else 4 * vec)
        )
        # Register SF-slot tensor: same partitioning of a broadcast layout whose
        # codomain is the per-tile SF slot index, so vec elements alias one slot.
        if const_expr(ax == 0):
            num_sf_m = tile_M // vec
            sf_bcast_layout = cute.make_layout(((vec, num_sf_m), tile_N), stride=((0, 1), num_sf_m))
        else:
            num_sf_n = tile_N // vec
            sf_bcast_layout = cute.make_layout((tile_M, (vec, num_sf_n)), stride=(num_sf_n, (0, 1)))
        tDrSFD_layout = ctx.partition_for_epilogue_fn(
            cute.make_rmem_tensor(sf_bcast_layout, Float32)
        ).layout
        tDrSFD = cute.make_rmem_tensor(tDrSFD_layout, getattr(gemm, self._sf_dtype_attr()))
        # SF vectors may be split across lanes along their axis (e.g. the
        # 16dp256b tmem load of tile_m 64 gives each lane 8-column runs, with
        # lanes {l, l+16} sharing every row; SM120 col vectors span the 8-lane
        # stride-4 column groups of the mma fragment): each thread then owns
        # vec / lanes elements of every vector it touches, and the per-thread
        # partial amaxes are combined with butterfly shuffles across the lane
        # group. On the SM90-style register epilogue (SM120 warp MMA,
        # tiled_copy_t2r is None) the geometry comes from the r2s copy's
        # SOURCE TV layout — same tiled_copy selection as VecReduce.
        tiled_copy = (
            ctx.tiled_copy_t2r if const_expr(ctx.tiled_copy_t2r is not None) else ctx.tiled_copy_r2s
        )
        lane_layout_MN, warp_layout_MN = _get_lane_warp_layouts(
            tiled_copy, reference_src=const_expr(ctx.tiled_copy_t2r is None)
        )
        lanes_in_ax = cute.size(lane_layout_MN[ax])
        warps_in_ax = cute.size(warp_layout_MN[ax])
        lane_stride = lane_layout_MN[ax].stride if lanes_in_ax > 1 else 0
        # Data-driven containment: elems_per_slot says how much of each SF
        # vector this thread holds. Thread-local vectors (elems_per_slot ==
        # vec) need no cross-lane work. Otherwise the vector spans `span`
        # consecutive lane subgroups, combined by butterfly shuffles (and only
        # the subgroup leader stores); if it exceeds the warp's lane group
        # (SM120 warp MMA: 16-column warp N runs or 16-row warp M stripes vs
        # a 32-wide SF vector), the remainder spans warp_span warp neighbors
        # along the axis, combined through the sExch smem buffer.
        tDrSFD_sub0 = cute.group_modes(tDrSFD, 3, cute.rank(tDrSFD))[None, None, None, 0]
        n_slots_sub = cute.size(cute.filter_zeros(tDrSFD_sub0))
        elems_per_slot = cute.size(tDrSFD_sub0) // n_slots_sub
        assert cute.size(tDrSFD_sub0) == elems_per_slot * n_slots_sub
        assert vec % elems_per_slot == 0, (
            f"SFD: thread holds {elems_per_slot} elements per SF slot, not a divisor of "
            f"the vector size {vec}; check tile_m / epi tile"
        )
        span = vec // elems_per_slot
        if const_expr(span <= lanes_in_ax):
            lane_span = span
            warp_span = 1
            assert lanes_in_ax % lane_span == 0, (
                f"SF vector spans {lane_span} lanes but the lane group is {lanes_in_ax}"
            )
        else:
            assert span % lanes_in_ax == 0 and lanes_in_ax == (1 << int(math.log2(lanes_in_ax)))
            lane_span = lanes_in_ax
            warp_span = span // lanes_in_ax
            assert warp_span <= warps_in_ax and warps_in_ax % warp_span == 0, (
                f"SF vector spans {warp_span} warps along its axis but only {warps_in_ax} exist"
            )
        warp_member = Int32(0)
        tDcD = None
        sExch = None
        if const_expr(warp_span > 1):
            assert smem_tensor is not None, (
                "SFD cross-warp amax exchange needs its smem buffer (smem_struct_field "
                "and the trace-time geometry disagree)"
            )
            warp_idx = cute.arch.make_warp_uniform(ctx.tidx // cute.arch.WARP_SIZE)
            warp_member = Int32(warp_layout_MN.get_hier_coord(warp_idx)[ax] % warp_span)
            tDcD = ctx.partition_for_epilogue_fn(cute.make_identity_tensor((tile_M, tile_N)))
            sExch = smem_tensor
        # Per-subtile fp32 scratch (amax, then dequantized-scale); all subtiles
        # share the same sliced layout so one buffer is reused across the loop.
        sub_layout = tDrSFD_sub0.layout
        tDrAmax = cute.make_rmem_tensor(sub_layout, Float32)
        tDrScale = cute.make_rmem_tensor(sub_layout, Float32)
        return _SfdRowState(
            tDgSFD,
            tDrSFD,
            tDrAmax,
            tDrScale,
            limit_m,
            limit_n,
            lane_span,
            lane_stride,
            norm_const,
            warp_span,
            warp_member,
            sExch,
            tDcD,
        )

    @cute.jit
    def _begin_col(self, gemm, param, ctx, norm_const):
        vec = getattr(gemm, self._vec_attr())
        tile_M, tile_N = ctx.tile_M, ctx.tile_N
        # lane == row within the warp's row stripe: required so an SF vector
        # of `vec` rows is a contiguous lane group (whole warp / half warp for
        # 32dp loads; for 16dp256b loads the 16-lane halves each cover all
        # rows of their own column segment, so vec <= 16 still reduces within
        # a contiguous lane group). Static check via the tiled copy TV layout.
        lane_layout_MN, _ = _get_lane_warp_layouts(ctx.tiled_copy_t2r, reference_src=False)
        lanes_in_m = cute.size(lane_layout_MN[0])
        assert lane_layout_MN[0].stride == 1, (
            "Column-direction SFD requires lane == row (unit lane stride along M)"
        )
        assert lanes_in_m % vec == 0, (
            f"Column-direction SF vector ({vec} rows) exceeds the {lanes_in_m}-lane row "
            "stripe; reducing across warps would need smem (unsupported — use a larger "
            "tile_m or a 16-row SF vector)"
        )
        tDgSFD, limit_m, limit_n = self._sfd_gmem_view(gemm, param, ctx, atom_cols=128)
        tDcSFD = ctx.partition_for_epilogue_fn(cute.make_identity_tensor((tile_M, tile_N)))
        return _SfdColState(tDgSFD, tDcSFD, limit_m, limit_n, norm_const)

    @cute.jit
    def begin_loop(self, gemm, state, epi_coord):
        """Pre-slice the per-subtile quantize state: the driver's store loop
        consumes it via epi_loop_tensors (no epi_coord there) in
        :meth:`quantize`."""
        if const_expr(self._col_redux_path(gemm.arch)):
            tDgSFD_cur = cute.group_modes(state.tDgSFD, 3, cute.rank(state.tDgSFD))[
                None, None, None, epi_coord
            ]
            tDcSFD_cur = cute.group_modes(state.tDcSFD, 3, cute.rank(state.tDcSFD))[
                None, None, None, epi_coord
            ]
            return _SfdColLoopState(
                tDgSFD_cur, tDcSFD_cur, state.limit_m, state.limit_n, state.norm_const
            )
        tDrSFD_cur = cute.group_modes(state.tDrSFD, 3, cute.rank(state.tDrSFD))[
            None, None, None, epi_coord
        ]
        xwarp = None
        if const_expr(state.warp_span > 1):
            # Cross-warp amax exchange for this subtile: parity-picked half
            # of the double-buffered sExch (consecutive subtiles alternate
            # buffers, so one barrier per subtile is WAR-safe — writes to a
            # buffer are separated from the previous reads of that same
            # buffer by the intervening subtile's barrier).
            epi_tile = gemm.epi_tile
            epi_m_size = cute.size(epi_tile[0])
            epi_n_size = cute.size(epi_tile[1])
            cnt_m = gemm.cta_tile_shape_mnk[0] // epi_m_size
            cnt_n = gemm.cta_tile_shape_mnk[1] // epi_n_size
            if const_expr(gemm.epi_m_major):
                linear = epi_coord[1] * cnt_m + epi_coord[0]
            else:
                linear = epi_coord[0] * cnt_n + epi_coord[1]
            tDcD_cur = cute.group_modes(state.tDcD, 3, cute.rank(state.tDcD))[
                None, None, None, epi_coord
            ]
            # Exchange slot = the SF vector's coordinate within the subtile:
            # the vector axis is divided by the vector width, the other axis
            # is per-element.
            vec = getattr(gemm, self._vec_attr())
            slot_div = (vec, 1) if const_expr(self.direction == "col") else (1, vec)
            xwarp = (
                state.sExch[None, None, None, linear % 2],
                tDcD_cur,
                (epi_m_size, epi_n_size),
                slot_div,
                state.warp_member,
                state.warp_span,
                gemm.epilogue_barrier,
            )
        return _SfdRowLoopState(
            state.tDrAmax,
            state.tDrScale,
            tDrSFD_cur,
            state.lane_span,
            state.lane_stride,
            state.norm_const,
            xwarp,
        )

    @cute.jit
    def quantize(self, gemm, loop_state, frag):
        """Quantize one subtile of the owning output in place (SF generation +
        rescale). The driver's store loop calls this on the output's final
        fragment — after epi_visit_subtile and epi_end_loop, so every consumer
        of the un-rescaled values has run — right before the owning store op's
        storage-dtype convert, which then emits the quantized values. ``frag``
        element i maps to acc column acc_stride * i (a gated postact
        interleaves gate/up pairs)."""
        value_dtype = getattr(gemm, self._val_dtype_attr())
        if const_expr(self._col_redux_path(gemm.arch)):
            sfd_quantize_subtile_col(
                gemm,
                loop_state.tDgSFD_cur,
                loop_state.tDcSFD_cur,
                getattr(gemm, self._vec_attr()),
                loop_state.limit_m,
                loop_state.limit_n,
                frag,
                norm_const=loop_state.norm_const,
                value_dtype=value_dtype,
            )
        else:
            sfd_quantize_subtile(
                gemm,
                loop_state.tDrAmax,
                loop_state.tDrScale,
                loop_state.tDrSFD_cur,
                frag,
                loop_state.lane_span,
                loop_state.lane_stride,
                norm_const=loop_state.norm_const,
                value_dtype=value_dtype,
                acc_stride=getattr(gemm, self._acc_mult_attr()),
                xwarp=loop_state.xwarp,
            )

    @staticmethod
    def _sf_pack_width(tDg_flt, max_pack=4):
        """Static store width (bytes) for the filtered SF gmem view: the
        leading contiguous run after coalescing. 1 when a thread's slots are
        not adjacent (e.g. 16dp loads interleave lane halves byte-by-byte)."""
        lay = cute.coalesce(tDg_flt).layout
        shape, stride = lay.shape, lay.stride
        if isinstance(shape, int):
            shape, stride = (shape,), (stride,)
        if not isinstance(stride[0], int) or stride[0] != 1 or not isinstance(shape[0], int):
            return 1
        return math.gcd(shape[0], max_pack)

    @staticmethod
    @cute.jit
    def _sf_packed_copy(tDr_flt, tDg_flt, pack):
        """Store SF bytes packed as one u32/u16 per `pack` consecutive slots.

        autovec_copy alone emits byte stores here: the per-thread gmem offset
        (16*(m%32) + 4*(m//32%4) + slot) defeats its alignment analysis, so
        widen explicitly — all strides and the base are multiples of `pack`.
        """
        if const_expr(pack == 4):
            wide = Int32
        elif const_expr(pack == 2):
            wide = cutlass.Int16
        else:
            wide = None
        if const_expr(wide is not None):
            # SF slot addresses are pack-aligned by construction (16 B base,
            # strides 16/4/1 within an atom, 512 B across atoms), but the
            # compiler's alignment analysis can lose that through tile
            # divisions (e.g. 64-row tiles) — assert it explicitly.
            tDg_flt = cute.coalesce(tDg_flt)
            tDg_flt = cute.make_tensor(tDg_flt.iterator.align(pack), tDg_flt.layout)
            cute.autovec_copy(
                cute.coalesce(cute.recast_tensor(cute.coalesce(tDr_flt), wide)),
                cute.coalesce(cute.recast_tensor(tDg_flt, wide)),
            )
        else:
            cute.autovec_copy(tDr_flt, tDg_flt)

    @cute.jit
    def _flush_per_subtile(self, gemm, tDgSFD, tDrSFD, epi_tile, limit_m, limit_n):
        """Per-subtile flush with a uniform bounds check. Every epi subtile
        width divides the SF atom coverage along both axes (epi_m divides the
        128-row pad, epi tile N divides the 4*vec atom columns), so each
        subtile is fully inside or fully outside the padded SF extent."""
        epi_m_size = cute.size(epi_tile[0])
        epi_n_size = cute.size(epi_tile[1])
        cnt_m = gemm.cta_tile_shape_mnk[0] // epi_m_size
        cnt_n = gemm.cta_tile_shape_mnk[1] // epi_n_size
        tDrSFD_g = cute.group_modes(tDrSFD, 3, cute.rank(tDrSFD))
        for em in cutlass.range_constexpr(cnt_m):
            for en in cutlass.range_constexpr(cnt_n):
                if (em + 1) * epi_m_size <= limit_m and (en + 1) * epi_n_size <= limit_n:
                    tDrSFD_cur = tDrSFD_g[None, None, None, (em, en)]
                    tDgSFD_cur = cute.filter_zeros(tDgSFD[None, None, None, em, en])
                    self._sf_packed_copy(
                        cute.filter_zeros(tDrSFD_cur),
                        tDgSFD_cur,
                        self._sf_pack_width(tDgSFD_cur),
                    )

    @cute.jit
    def end(
        self,
        gemm,
        param,
        state,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tile_coord_mnkl,
        varlen_manager,
        tidx,
    ):
        if const_expr(self._col_redux_path(gemm.arch)):
            # Column-direction SF bytes are stored inline per subtile in
            # sfd_quantize_subtile_col (per-byte designated-lane stores).
            return
        tDgSFD, tDrSFD = state.tDgSFD, state.tDrSFD
        limit_m, limit_n = state.limit_m, state.limit_n
        lane_span, lane_stride = state.lane_span, state.lane_stride
        vec = getattr(gemm, self._vec_attr())
        tile_N = gemm.cta_tile_shape_mnk[1]
        # When SF vectors are lane-split, every lane in the subgroup holds
        # identical SF bytes; only the subgroup leader stores. Warp-split
        # vectors (SM120): every member warp holds identical bytes after the
        # sExch fold; only member 0 stores.
        is_leader = Boolean(True)
        if const_expr(lane_span > 1):
            lane = cute.arch.lane_idx()
            is_leader = (lane // lane_stride) % lane_span == 0
        if const_expr(state.warp_span > 1):
            is_leader = is_leader & (state.warp_member == 0)
        # Whole-tile autovec flush only when begin() CTA-tiled the gmem view
        # (row direction, tile_N covering whole atoms); the col transposed
        # atom stores per byte anyway, so per-subtile costs nothing extra.
        whole_tile_ok = const_expr(self.direction == "row" and tile_N % (4 * vec) == 0)
        if is_leader and limit_m > 0:
            if const_expr(whole_tile_ok):
                if limit_m >= gemm.cta_tile_shape_mnk[0] and limit_n >= tile_N:
                    # Whole tile in bounds: vectorized stores (4 consecutive SF
                    # slots of a row are consecutive bytes -> 4-byte stores,
                    # when this thread's slots are actually adjacent).
                    tDgSFD_flt = cute.filter_zeros(tDgSFD)
                    self._sf_packed_copy(
                        cute.filter_zeros(tDrSFD), tDgSFD_flt, self._sf_pack_width(tDgSFD_flt)
                    )
                else:
                    self._flush_per_subtile(gemm, tDgSFD, tDrSFD, epi_tile, limit_m, limit_n)
            else:
                self._flush_per_subtile(gemm, tDgSFD, tDrSFD, epi_tile, limit_m, limit_n)


@cute.jit
def sfd_quantize_subtile(
    gemm,
    tDrAmax,
    tDrScale,
    tDrSFD_cur,
    tRS_rD,
    lane_span,
    lane_stride,
    norm_const=None,
    value_dtype=None,
    acc_stride=1,
    xwarp=None,
):
    """Compute + quantize SF for one subtile and rescale tRS_rD in place.

    Called from BlockScaleFactorStore.quantize (the driver's store loop),
    after all element-wise epilogue ops, so the scale factors reflect the
    final output values.

    The SF slot tensors live in accumulator space. ``acc_stride`` maps
    fragment element i to acc element acc_stride * i: 1 for full-tile values,
    2 for a halved gated postact whose element i is act(acc[2i], acc[2i+1]).

    ``xwarp`` (SF vectors spanning warps along their axis, SM120 warp MMA): a
    ``(sExch, tDcD_cur, (epi_m, epi_n), (div_m, div_n), warp_member,
    warp_span, barrier)`` bundle. After the in-warp butterfly, each warp's
    per-slot partial amax is published to sExch[(m % epi_m) // div_m,
    (n % epi_n) // div_n, warp_member] — the divisor is the vector width on
    the vector's axis and 1 on the other — the barrier syncs the epilogue
    warps, and every lane folds in the other members' partials, so the
    quantize and rescale below stay lane-local and every warp stores
    identical SF bytes (a single member does, see end()).
    """
    if const_expr(value_dtype is None):
        value_dtype = gemm.d_dtype
    tDrAmax_flt = cute.filter_zeros(tDrAmax)
    tDrScale_flt = cute.filter_zeros(tDrScale)
    tDrSFD_flt = cute.filter_zeros(tDrSFD_cur)
    tDrAmax_flt.fill(0.0)
    # Zero-stride broadcast layout accumulates each element into its SF slot,
    # independent of the fragment's register order. Off SM100, the whole fold
    # — element accumulate, lane butterfly, cross-warp exchange — runs on
    # two-input max.xorsign.abs (fmax abs=True): it preserves the maximum
    # MAGNITUDE while XORing operand signs, so no per-element absf is needed
    # and the arbitrary accumulated sign is cleared once per slot at the end
    # (same pattern as the VecReduce max_abs epilogue). On SM100 the plain
    # fmax(acc, |x|) chain is kept: ptxas fuses it into 3-input FMNMX3.ABS
    # (abs is an operand modifier there, and FMNMX3 is SM100-only), which
    # the two-input xorsign form would defeat.
    xorsign = const_expr(gemm.arch != 100)
    for i in cutlass.range(cute.size(tRS_rD), unroll_full=True):
        if const_expr(xorsign):
            tDrAmax[acc_stride * i] = cute.arch.fmax(tDrAmax[acc_stride * i], tRS_rD[i], abs=True)
        else:
            tDrAmax[acc_stride * i] = cute.arch.fmax(
                tDrAmax[acc_stride * i], cute.math.absf(tRS_rD[i])
            )
    if const_expr(lane_span > 1):
        # SF vectors split across lane_span consecutive lane subgroups along
        # the vector's axis: butterfly-combine the per-lane partial amaxes;
        # every lane ends up with the full-vector amax (with xorsign: the
        # magnitude — signs diverge per lane and are cleared below) so the
        # rescale below stays lane-local.
        for vi in cutlass.range_constexpr(cute.size(tDrAmax_flt)):
            v = tDrAmax_flt[vi]
            step = lane_span // 2
            while step > 0:
                v = cute.arch.fmax(
                    v, cute.arch.shuffle_sync_bfly(v, offset=step * lane_stride), abs=xorsign
                )
                step //= 2
            tDrAmax_flt[vi] = v
    if const_expr(xwarp is not None):
        sExch, tDcD_cur, epi_mn, slot_div, warp_member, warp_span, barrier = xwarp
        epi_m_size, epi_n_size = epi_mn
        div_m, div_n = slot_div
        # Publish this warp's per-slot partials (sign-garbled, magnitudes
        # correct). Aliased elements (and the lanes of a subgroup) write the
        # same value to the same slot — benign. The coord partition lives in
        # ACC space, so a halved gated fragment indexes it at acc_stride * i,
        # same as the slot tensors.
        for i in cutlass.range(cute.size(tRS_rD), unroll_full=True):
            c = tDcD_cur[acc_stride * i]
            sExch[(c[0] % epi_m_size) // div_m, (c[1] % epi_n_size) // div_n, warp_member] = (
                tDrAmax[acc_stride * i]
            )
        barrier.arrive_and_wait()
        # Fold in every member's partial (including our own — idempotent).
        for i in cutlass.range(cute.size(tRS_rD), unroll_full=True):
            c = tDcD_cur[acc_stride * i]
            v = tDrAmax[acc_stride * i]
            for j in cutlass.range_constexpr(warp_span):
                v = cute.arch.fmax(
                    v,
                    sExch[(c[0] % epi_m_size) // div_m, (c[1] % epi_n_size) // div_n, j],
                    abs=True,
                )
            tDrAmax[acc_stride * i] = v
    if const_expr(xorsign):
        # Clear the XOR-accumulated sign once per slot.
        for vi in cutlass.range_constexpr(cute.size(tDrAmax_flt)):
            tDrAmax_flt[vi] = cute.math.abs(tDrAmax_flt[vi])
    # Shared quantize core: SF bytes + rescale factors from the amaxes (see
    # quack.blockscaled.quantize_utils for the cuBLAS/CUTLASS semantics contract).
    quantize_sf_slots(tDrAmax_flt, tDrSFD_flt, tDrScale_flt, value_dtype, norm_const=norm_const)
    for i in cutlass.range(cute.size(tRS_rD), unroll_full=True):
        tRS_rD[i] = tRS_rD[i] * tDrScale[acc_stride * i]


@cute.jit
def sfd_quantize_subtile_col(
    gemm, tDgSFD_cur, tDcSFD_cur, vec, limit_m, limit_n, tRS_rD, norm_const=None, value_dtype=None
):
    """Column-direction variant of sfd_quantize_subtile: one SF byte per
    (32/16-row block, column). Each column's amax is one
    ``redux.sync.max.abs.NaN.f32`` across the `vec` lanes of the row group
    (lane == row within the warp stripe; SM100a, same instruction the CUTLASS
    C++ ColStore uses — a 5-round shuffle butterfly otherwise). Every lane
    then rescales its own value and a designated lane per group stores the SF
    byte. All per-byte stores — the transposed atom puts adjacent columns'
    bytes 16 B apart, so there is nothing to pack.

    NaN semantics: redux propagates (a NaN anywhere in the vector poisons the
    SF byte, matching CUTLASS); the row direction's thread-local FMNMX chain
    currently drops NaNs instead.
    """
    if const_expr(value_dtype is None):
        value_dtype = gemm.d_dtype
    rcp_dmax = 1.0 / QUANT_DTYPE_MAX[value_dtype]
    if const_expr(norm_const is not None):
        norm_scaled = norm_const * rcp_dmax
    else:
        norm_scaled = Float32(rcp_dmax)
    sf_dtype = tDgSFD_cur.element_type
    is_e8m0 = const_expr(sf_dtype is cutlass.Float8E8M0FNU)
    # 1-slot scratch for the exact e8m0 integer reciprocal (254 - exponent
    # byte); scalar setitem to dodge the degenerate-view store bug.
    tScratchB = cute.make_rmem_tensor(cute.make_layout(1), cutlass.Uint8)
    tScratch_sf = cute.recast_tensor(tScratchB, sf_dtype)
    tScratch_e8 = cute.recast_tensor(tScratchB, cutlass.Float8E8M0FNU)
    lane = cute.arch.lane_idx()
    m_ok = limit_m > 0  # all-or-nothing per CTA (128-row stripes)
    for i in cutlass.range_constexpr(cute.size(tRS_rD)):
        # Amax over the vec-lane row group; every lane gets it. One
        # redux.sync.max.abs.NaN on the SM100 family, shuffle butterfly on
        # arches without fp32 redux (SM120).
        amax = warp_reduce(tRS_rD[i], cute.arch.fmax, threads_in_group=vec, abs=True, nan=True)
        tScratch_sf[0] = sf_dtype(amax * norm_scaled)
        # One lane per group stores this column's byte; distribute columns
        # round-robin over the group so stores go out in parallel.
        if m_ok and (lane % vec) == (i % vec) and tDcSFD_cur[i][1] < limit_n:
            tDgSFD_cur[i] = tScratch_sf[0]
        # Rescale by the reciprocal of the quantized scale.
        if const_expr(is_e8m0):
            tScratchB[0] = cutlass.Uint8((254 - tScratchB[0]) & 0xFF)
            acc_scale = tScratch_e8[0].to(Float32)
        else:
            acc_scale = cute.arch.rcp_approx(tScratch_sf[0].to(Float32))
        if const_expr(norm_const is not None):
            acc_scale = acc_scale * norm_const
        tRS_rD[i] = tRS_rD[i] * cute.arch.fmin(acc_scale, FLT_MAX)
