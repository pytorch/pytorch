# Copyright (c) 2025, Tri Dao.
"""Composable epilogue operations (EpiOps) for GEMM kernels.

Each EpiOp encapsulates a single tensor kind's behavior across the epilogue lifecycle:
smem allocation, begin (one-time per-tile setup), begin_loop (per-subtile extraction),
end (cleanup).

The ops are composed via ComposableEpiMixin which iterates over a static _epi_ops tuple
to generate epi_smem_bytes_per_stage, epi_get_smem_struct, epi_get_smem_tensors,
epi_begin, and epi_begin_loop automatically.
"""

import math
import operator
from functools import partial

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, const_expr

from .epi_utils import assume_stride_divisibility, setup_epi_tensor
from .sm90_utils import partition_for_epilogue
from . import utils
from . import copy_utils
from . import layout_utils


class EpiContext:
    """Shared context passed to EpiOp.begin methods. Bundles common arguments."""

    __slots__ = (
        "epi_tile",
        "tiled_copy_t2r",
        "tiled_copy_r2s",
        "tile_coord_mnkl",
        "varlen_manager",
        "epilogue_barrier",
        "tidx",
        "partition_for_epilogue_fn",
        "num_epi_threads",
        "batch_idx",
        "tile_M",
        "tile_N",
    )

    def __init__(
        self,
        gemm,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tile_coord_mnkl,
        varlen_manager,
        epilogue_barrier,
        tidx,
    ):
        self.epi_tile = epi_tile
        self.tiled_copy_t2r = tiled_copy_t2r
        self.tiled_copy_r2s = tiled_copy_r2s
        self.tile_coord_mnkl = tile_coord_mnkl
        self.varlen_manager = varlen_manager
        self.epilogue_barrier = epilogue_barrier
        self.tidx = tidx
        self.tile_M = gemm.cta_tile_shape_mnk[0]
        self.tile_N = gemm.cta_tile_shape_mnk[1]
        self.batch_idx = tile_coord_mnkl[3]
        self.num_epi_threads = gemm.num_epi_warps * cute.arch.WARP_SIZE
        self.partition_for_epilogue_fn = partial(
            partition_for_epilogue,
            epi_tile=epi_tile,
            tiled_copy=tiled_copy_t2r if tiled_copy_t2r is not None else tiled_copy_r2s,
            tidx=tidx,
            reference_src=tiled_copy_t2r is None,
        )


def _get_lane_warp_layouts(tiled_copy, reference_src=True):
    """Derive lane and warp layouts along M and N from the epilogue tiled_copy.

    Follows the CUTLASS Sm90RowReduction / Sm90ColReduction pattern.
    Uses layout_src_tv_tiled (SM90, reference_src=True) or
    layout_dst_tv_tiled (SM100, reference_src=False), matching the C++ impl's
    get_layoutS_TV / get_layoutD_TV selection.

    Returns (lane_layout_MN, warp_layout_MN) where each is a 2D layout (M, N):
      lane_layout_MN[0] = lane_M: (lanes_in_M):(lane_stride_M) — e.g. 8:4
      lane_layout_MN[1] = lane_N: (lanes_in_N):(lane_stride_N) — e.g. 4:1
      warp_layout_MN[0] = warp_M: (warps_in_M):(warp_stride_M) — e.g. 4:1
      warp_layout_MN[1] = warp_N: (warps_in_N):(warp_stride_N) — e.g. 1:0

    For RowVecReduce (reduce along M): shuffle across lane_M, smem reduce across warp_M.
    For ColVecReduce (reduce along N): shuffle across lane_N, direct write (warps_in_N == 1).
    """
    # right_inverse of the TV layout gives tile_element_idx -> tv_idx.
    # SM90: use src (register) layout; SM100: use dst (smem) layout.
    layout_tv = tiled_copy.layout_src_tv_tiled if reference_src else tiled_copy.layout_dst_tv_tiled
    ref_layout = cute.right_inverse(layout_tv)
    tile_M_size, tile_N_size = cute.size(tiled_copy.tiler_mn[0]), cute.size(tiled_copy.tiler_mn[1])
    ref_layout_MN = cute.composition(
        ref_layout, cute.make_layout((tile_M_size, tile_N_size))
    )  # (tile_M, tile_N) -> tv_idx

    num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE

    # tv2lane: tv_idx -> lane_idx  (lane = tv_idx % 32)
    tv2lane = cute.make_layout((cute.arch.WARP_SIZE, num_warps, 1), stride=(1, 0, 0))
    ref2lane = cute.composition(tv2lane, ref_layout_MN)  # (tile_M, tile_N) -> lane_idx
    # select mode [0] = M part, [1] = N part; filter removes stride-0
    lane_M = cute.filter(cute.select(ref2lane, [0]))  # lane_m -> lane_idx
    lane_N = cute.filter(cute.select(ref2lane, [1]))  # lane_n -> lane_idx
    lane_layout_MN = layout_utils.concat_layout(lane_M, lane_N)  # (lane_M, lane_N) -> lane_idx

    # tv2warp: tv_idx -> warp_idx  (warp = tv_idx / 32)
    tv2warp = cute.make_layout((cute.arch.WARP_SIZE, num_warps, 1), stride=(0, 1, 0))
    ref2warp = cute.composition(tv2warp, ref_layout_MN)  # (tile_M, tile_N) -> warp_idx
    warp_M = cute.filter(cute.select(ref2warp, [0]))  # warp_m -> warp_idx
    warp_N = cute.filter(cute.select(ref2warp, [1]))  # warp_n -> warp_idx
    warp_layout_MN = layout_utils.concat_layout(warp_M, warp_N)  # (warp_M, warp_N) -> warp_idx

    return lane_layout_MN, warp_layout_MN


class EpiOp:
    """Base class for composable epilogue operations."""

    def __init__(self, name):
        self.name = name

    # --- Host-side: args → params ---
    def param_fields(self):
        """Return [(field_name, type, default), ...] for auto-generating EpilogueParams.
        Must match the keys returned by to_params()."""
        return []

    def to_params(self, gemm, args):
        """Convert this op's arg field(s) to param dict entries.
        Returns dict of {param_name: value}. Like EVT's to_underlying_arguments."""
        return {}

    def epi_m_major_score(self, arg_tensor, gemm):
        """Preference for epilogue subtile order. Positive prefers M-major, negative N-major."""
        return 0

    # --- Host-side: smem allocation ---
    def smem_bytes(self, arg_tensor, cta_tile_shape_mnk, epi_tile, warp_shape_mnk=None):
        """Bytes of smem needed per stage. arg_tensor is the EpilogueArguments field."""
        return 0

    def smem_struct_field(self, gemm, params):
        """Return (field_name, field_type) for @cute.struct, or None if no smem needed.
        params is the full EpilogueParams object."""
        return None

    def get_smem_tensor(self, gemm, params, storage_epi):
        """Extract smem tensor from storage.epi. Returns tensor or None.
        params is the full EpilogueParams object."""
        return None

    def tma_atoms(self, gemm, params):
        """Return list of TMA atoms for this op."""
        return []

    # --- Device-side: kernel execution ---
    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        """One-time per-tile setup. Returns state for begin_loop."""
        return None

    def begin_loop(self, gemm, state, epi_coord):
        """Per-subtile extraction. Returns value for epi_visit_subtile."""
        return state

    def end_loop(
        self,
        gemm,
        param,
        state,
        epi_coord,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tile_coord_mnkl,
        varlen_manager,
        tidx,
    ):
        """Per-subtile cleanup after epi_visit_subtile."""
        pass

    def needs_async_fence(self):
        """Whether this op issues async copies that need a fence."""
        return False

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
        """Cleanup after all subtiles (reductions, direct writes)."""
        pass


class Scalar(EpiOp):
    """Loads a scalar value or device pointer once per tile. No smem."""

    def __init__(self, name, dtype=None):
        super().__init__(name)
        self.dtype = dtype

    def param_fields(self):
        return [(self.name, object, None)]

    def to_params(self, gemm, args):
        return {self.name: getattr(args, self.name)}

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        result = None
        if const_expr(param is not None):
            result = (
                utils.load_scalar_or_pointer(param, dtype=self.dtype)
                if const_expr(self.dtype is not None)
                else utils.load_scalar_or_pointer(param)
            )
        return result


class VecLoad(EpiOp):
    """Base class for broadcast vector loads (row or col) via cp_async.

    Subclasses set `dim` to 0 (M/col) or 1 (N/row) and override `_get_gmem_vec`
    for varlen handling.
    """

    dim = None  # 0 for col (M), 1 for row (N)

    def param_fields(self):
        return [(self.name, object, None)]

    def to_params(self, gemm, args):
        return {self.name: assume_stride_divisibility(getattr(args, self.name))}

    def _tile_size(self, cta_tile_shape_mnk):
        return cta_tile_shape_mnk[self.dim]

    def _broadcast_stride(self):
        # Row: stride (0,1) — broadcast along M. Col: stride (1,0) — broadcast along N.
        return (0, 1) if self.dim == 1 else (1, 0)

    def _tile_dim(self, ctx):
        return ctx.tile_N if self.dim == 1 else ctx.tile_M

    def _coord_idx(self):
        return 1 if self.dim == 1 else 0

    def smem_bytes(self, arg_tensor, cta_tile_shape_mnk, epi_tile, warp_shape_mnk=None):
        if arg_tensor is None:
            return 0
        return self._tile_size(cta_tile_shape_mnk) * (arg_tensor.element_type.width // 8)

    def smem_struct_field(self, gemm, params):
        tensor = getattr(params, self.name, None)
        if tensor is None:
            size, dtype = 0, Float32
        else:
            size = self._tile_size(gemm.cta_tile_shape_mnk)
            dtype = tensor.element_type
        return (f"s_{self.name}", cute.struct.Align[cute.struct.MemRange[dtype, size], 16])

    def get_smem_tensor(self, gemm, params, storage_epi):
        if getattr(params, self.name, None) is None:
            return None
        return getattr(storage_epi, f"s_{self.name}").get_tensor(
            cute.make_layout(self._tile_size(gemm.cta_tile_shape_mnk))
        )

    def needs_async_fence(self):
        return True

    def epi_m_major_score(self, arg_tensor, gemm):
        if arg_tensor is None:
            return 0
        # It costs more registers (say 4x) to keep rowvec in register vs keeping colvec in register
        return 4 if self.dim == 1 else -1

    def _get_gmem_vec(self, param, ctx):
        """Get the global memory vector for this tile. Override for varlen."""
        return param[ctx.batch_idx, None]

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        tDsV = None
        tDrV_cvt = None
        if const_expr(param is not None):
            dtype = param.element_type
            num_copy_elems = const_expr(max(32, dtype.width)) // dtype.width
            thr_copy = copy_utils.tiled_copy_1d(
                dtype, ctx.num_epi_threads, num_copy_elems, is_async=True
            ).get_slice(ctx.tidx)
            mVec = self._get_gmem_vec(param, ctx)
            tile_dim = self._tile_dim(ctx)
            coord_idx = ctx.tile_coord_mnkl[self._coord_idx()]
            gVec = cute.local_tile(mVec, (tile_dim,), (coord_idx,))
            tVgV = thr_copy.partition_S(gVec)
            tVsV = thr_copy.partition_D(smem_tensor)
            tVcV = thr_copy.partition_S(cute.make_identity_tensor(tile_dim))
            limit = min(cute.size(mVec, mode=[0]) - coord_idx * tile_dim, tile_dim)
            pred = cute.make_rmem_tensor((1, cute.size(tVsV.shape[1])), Boolean)
            for m in cutlass.range(cute.size(tVsV.shape[1]), unroll_full=True):
                pred[0, m] = tVcV[0, m] < limit
            cute.copy(thr_copy, tVgV, tVsV, pred=pred)
            tDsV = ctx.partition_for_epilogue_fn(
                cute.make_tensor(
                    smem_tensor.iterator,
                    cute.make_layout((ctx.tile_M, ctx.tile_N), stride=self._broadcast_stride()),
                )
            )
            if const_expr(ctx.tiled_copy_t2r is not None):
                tDsV = ctx.tiled_copy_r2s.retile(tDsV)
            # Pre-allocate register tensor reused across begin_loop calls
            tDsV_sub = cute.group_modes(tDsV, 3, cute.rank(tDsV))[None, None, None, 0]
            tDrV_cvt = cute.make_rmem_tensor(tDsV_sub.layout, gemm.acc_dtype)
        return [tDsV, tDrV_cvt]

    @cute.jit
    def begin_loop(self, gemm, state, epi_coord):
        tDsV, tDrV_cvt = state[0], state[1]
        if const_expr(tDsV is not None):
            should_load = Boolean(True)
            if const_expr(self.dim == 1):
                if const_expr(gemm.epi_m_major):
                    should_load = epi_coord[0] == 0
            else:
                if const_expr(not gemm.epi_m_major):
                    should_load = epi_coord[1] == 0
            if should_load:
                tDsV_cur = cute.group_modes(tDsV, 3, cute.rank(tDsV))[None, None, None, epi_coord]
                tDrV = cute.make_rmem_tensor(tDsV_cur.layout, tDsV_cur.element_type)
                cute.autovec_copy(cute.filter_zeros(tDsV_cur), cute.filter_zeros(tDrV))
                tDrV_cvt.store(tDrV.load().to(gemm.acc_dtype))
        return tDrV_cvt


class RowVecLoad(VecLoad):
    """Loads a row vector (N,) via cp_async, broadcasts along M with stride (0,1)."""

    dim = 1


class ColVecLoad(VecLoad):
    """Loads a col vector (M,) via cp_async, broadcasts along N with stride (1,0).

    Optimization: with N-major subtile loop, consecutive epi_n iterations for the same
    epi_m share the same column data. The smem→register copy only runs when epi_n == 0.
    Supports varlen_m via domain_offset.
    """

    dim = 0

    @cute.jit
    def _get_gmem_vec(self, param, ctx):
        if const_expr(not ctx.varlen_manager.varlen_m):
            mVec = param[ctx.batch_idx, None]
        else:
            mVec = cute.domain_offset(
                (ctx.varlen_manager.params.cu_seqlens_m[ctx.batch_idx],), param
            )
        return mVec

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        tDsV = None
        tDrV_cvt = None
        if const_expr(param is not None):
            dtype = param.element_type
            num_copy_elems = const_expr(max(32, dtype.width)) // dtype.width
            thr_copy = copy_utils.tiled_copy_1d(
                dtype, ctx.num_epi_threads, num_copy_elems, is_async=True
            ).get_slice(ctx.tidx)
            mVec = self._get_gmem_vec(param, ctx)
            tile_dim = self._tile_dim(ctx)
            coord_idx = ctx.tile_coord_mnkl[self._coord_idx()]
            gVec = cute.local_tile(mVec, (tile_dim,), (coord_idx,))
            tVgV = thr_copy.partition_S(gVec)
            tVsV = thr_copy.partition_D(smem_tensor)
            tVcV = thr_copy.partition_S(cute.make_identity_tensor(tile_dim))
            # ColVec uses varlen-aware limit
            limit = min(
                ctx.varlen_manager.len_m(ctx.batch_idx) - coord_idx * tile_dim,
                tile_dim,
            )
            pred = cute.make_rmem_tensor((1, cute.size(tVsV.shape[1])), Boolean)
            for m in cutlass.range(cute.size(tVsV.shape[1]), unroll_full=True):
                pred[0, m] = tVcV[0, m] < limit
            cute.copy(thr_copy, tVgV, tVsV, pred=pred)
            tDsV = ctx.partition_for_epilogue_fn(
                cute.make_tensor(
                    smem_tensor.iterator,
                    cute.make_layout((ctx.tile_M, ctx.tile_N), stride=self._broadcast_stride()),
                )
            )
            if const_expr(ctx.tiled_copy_t2r is not None):
                tDsV = ctx.tiled_copy_r2s.retile(tDsV)
            # Pre-allocate register tensor reused across begin_loop calls
            tDsV_sub = cute.group_modes(tDsV, 3, cute.rank(tDsV))[None, None, None, 0]
            tDrV_cvt = cute.make_rmem_tensor(tDsV_sub.layout, gemm.acc_dtype)
        return [tDsV, tDrV_cvt]


class TileStore(EpiOp):
    """Tile-sized output tensor stored via TMA (e.g. postact).

    Args:
        name: field name in EpilogueArguments/Params (e.g. "mAuxOut")
        epi_tile_fn: optional (gemm, epi_tile) -> epi_tile for half-tile (GemmGated)
    """

    def __init__(self, name, epi_tile_fn=None):
        super().__init__(name)
        self.epi_tile_fn = epi_tile_fn

    def _tma_atom_key(self):
        return f"tma_atom_{self.name}"

    def _smem_layout_key(self):
        return f"epi_{self.name}_smem_layout_staged"

    def _epi_tile_key(self):
        return f"epi_tile_{self.name}"

    def param_fields(self):
        from dataclasses import MISSING

        return [
            (self._tma_atom_key(), object, MISSING),
            (self.name, object, MISSING),
            (self._smem_layout_key(), object, MISSING),
            (self._epi_tile_key(), object, MISSING),
        ]

    def to_params(self, gemm, args):
        tensor = getattr(args, self.name, None)
        if tensor is None:
            return {
                self._tma_atom_key(): None,
                self.name: None,
                self._smem_layout_key(): None,
                self._epi_tile_key(): None,
            }
        epi_tile = self.epi_tile_fn(gemm, gemm.epi_tile) if self.epi_tile_fn else None
        tma_atom, tma_tensor, smem_layout, epi_tile_out = setup_epi_tensor(
            gemm, tensor, epi_tile=epi_tile
        )
        return {
            self._tma_atom_key(): tma_atom,
            self.name: tma_tensor,
            self._smem_layout_key(): smem_layout,
            self._epi_tile_key(): epi_tile_out,
        }

    def smem_bytes(self, arg_tensor, cta_tile_shape_mnk, epi_tile, warp_shape_mnk=None):
        if arg_tensor is None:
            return 0
        if self.epi_tile_fn is not None:
            epi_tile = self.epi_tile_fn(None, epi_tile)
        return cute.size(cute.shape(epi_tile)) * (arg_tensor.element_type.width // 8)

    def smem_struct_field(self, gemm, params):
        smem_layout_key = self._smem_layout_key()
        smem_layout = getattr(params, smem_layout_key, None)
        if smem_layout is None:
            return (f"s_{self.name}", cute.struct.MemRange[Float32, 0])
        return (
            f"s_{self.name}",
            cute.struct.Align[
                cute.struct.MemRange[
                    gemm.aux_out_dtype,
                    cute.cosize(smem_layout),
                ],
                gemm.buffer_align_bytes,
            ],
        )

    def get_smem_tensor(self, gemm, params, storage_epi):
        smem_layout = getattr(params, self._smem_layout_key(), None)
        if smem_layout is None:
            return None
        return getattr(storage_epi, f"s_{self.name}").get_tensor(
            smem_layout.outer,
            swizzle=smem_layout.inner,
        )

    def tma_atoms(self, gemm, params):
        atom = getattr(params, self._tma_atom_key(), None)
        return [] if atom is None else [atom]
        return []


@cute.jit
def vec_multiply(gemm, tRS_rD, tDrColVec, tDrRowVec):
    """Multiply tRS_rD by colvec and/or rowvec in-place. Uses packed f32x2 on SM100."""
    if const_expr(tDrColVec is not None):
        if const_expr(gemm.arch != 100):
            for i in cutlass.range(cute.size(tDrColVec), unroll_full=True):
                tRS_rD[i] *= tDrColVec[i]
        else:
            for i in cutlass.range(cute.size(tRS_rD) // 2, unroll_full=True):
                tRS_rD[2 * i], tRS_rD[2 * i + 1] = cute.arch.mul_packed_f32x2(
                    (tRS_rD[2 * i], tRS_rD[2 * i + 1]),
                    (tDrColVec[2 * i], tDrColVec[2 * i + 1]),
                )
    if const_expr(tDrRowVec is not None):
        if const_expr(gemm.arch != 100):
            for i in cutlass.range(cute.size(tDrRowVec), unroll_full=True):
                tRS_rD[i] *= tDrRowVec[i]
        else:
            for i in cutlass.range(cute.size(tRS_rD) // 2, unroll_full=True):
                tRS_rD[2 * i], tRS_rD[2 * i + 1] = cute.arch.mul_packed_f32x2(
                    (tRS_rD[2 * i], tRS_rD[2 * i + 1]),
                    (tDrRowVec[2 * i], tDrRowVec[2 * i + 1]),
                )


@cute.jit
def colvec_reduce_accumulate(gemm, tDrReduce, tRS_rInput, transform_fn=None, rScale=None):
    """Accumulate transform_fn(input) or input * rScale into a ColVecReduce buffer.

    If transform_fn is provided, accumulates transform_fn(input[i]).
    If rScale is provided, accumulates input[i] * rScale[i] (uses packed mul/fma for SM100).
    If neither, accumulates input directly (identity).
    """
    if const_expr(tDrReduce is not None):
        if const_expr(transform_fn is None):
            transform_fn = lambda x: x
        if const_expr(gemm.arch != 100):
            for i in cutlass.range(cute.size(tDrReduce), unroll_full=True):
                val = transform_fn(tRS_rInput[i])
                tDrReduce[i] += val * rScale[i] if const_expr(rScale is not None) else val
        else:
            tDrReduce_mn = layout_utils.convert_layout_zero_stride(tDrReduce, tDrReduce.layout)
            tRS_rInput_mn = layout_utils.convert_layout_zero_stride(tRS_rInput, tDrReduce.layout)
            if const_expr(rScale is not None):
                rScale_mn = layout_utils.convert_layout_zero_stride(rScale, tDrReduce.layout)
            for m in cutlass.range(cute.size(tDrReduce_mn, mode=[0]), unroll_full=True):
                inp = lambda n: (tRS_rInput_mn[m, 2 * n], tRS_rInput_mn[m, 2 * n + 1])
                val0 = transform_fn(inp(0))
                assert cute.size(tDrReduce_mn, mode=[1]) % 2 == 0
                if const_expr(rScale is not None):
                    row_sum = cute.arch.mul_packed_f32x2(val0, (rScale_mn[m, 0], rScale_mn[m, 1]))
                else:
                    row_sum = val0
                for n in cutlass.range(1, cute.size(tDrReduce_mn, mode=[1]) // 2, unroll_full=True):
                    val = transform_fn(inp(n))
                    if const_expr(rScale is not None):
                        row_sum = cute.arch.fma_packed_f32x2(
                            val, (rScale_mn[m, 2 * n], rScale_mn[m, 2 * n + 1]), row_sum
                        )
                    else:
                        row_sum = cute.arch.add_packed_f32x2(val, row_sum)
                tDrReduce_mn[m, 0] += row_sum[0] + row_sum[1]


@cute.jit
def rowvec_reduce_accumulate(gemm, tDrReduce, tRS_rInput, transform_fn=None, rScale=None):
    """Accumulate transform_fn(input) or input * rScale into a RowVecReduce buffer.

    Reduces along M dimension, keeping N. The zero-stride layout on M ensures
    elements at different M positions but same N column accumulate correctly.
    """
    if const_expr(tDrReduce is not None):
        if const_expr(transform_fn is None):
            transform_fn = lambda x: x
        if const_expr(gemm.arch != 100):
            for i in cutlass.range(cute.size(tDrReduce), unroll_full=True):
                val = transform_fn(tRS_rInput[i])
                tDrReduce[i] += val * rScale[i] if const_expr(rScale is not None) else val
        else:
            # Keep CUTLASS's linear fragment indexing, but use packed f32x2 arithmetic
            # for any transform that accepts and returns an f32x2 tuple.
            # We have to be careful to avoid tDrReduce[2 * i] and tDrReduce[2 * i + 1] aliasing
            # each other. For SM100, tDrReduce has layout ((32,1),1,1):((1,0),0,0) or
            # (((2,2,4),1),2,1):(((1,0,8),0),0,0), so this works. But it's error-prone.
            for i in cutlass.range(cute.size(tRS_rInput) // 2, unroll_full=True):
                acc = (tDrReduce[2 * i], tDrReduce[2 * i + 1])
                val = (tRS_rInput[2 * i], tRS_rInput[2 * i + 1])
                val = transform_fn(val)
                if const_expr(rScale is not None):
                    scale = (rScale[2 * i], rScale[2 * i + 1])
                    tDrReduce[2 * i], tDrReduce[2 * i + 1] = cute.arch.fma_packed_f32x2(
                        val, scale, acc
                    )
                else:
                    tDrReduce[2 * i], tDrReduce[2 * i + 1] = cute.arch.add_packed_f32x2(val, acc)
            if const_expr(cute.size(tRS_rInput) % 2 != 0):
                i = cute.size(tRS_rInput) - 1
                val = transform_fn(tRS_rInput[i])
                tDrReduce[i] += val * rScale[i] if const_expr(rScale is not None) else val


class VecReduce(EpiOp):
    """Base class for row/column vector reductions."""

    dim = 0  # 0 for colvec output along M, 1 for rowvec output along N
    epi_m_major_preference = 0

    def param_fields(self):
        return [(self.name, object, None)]

    def to_params(self, gemm, args):
        return {self.name: assume_stride_divisibility(getattr(args, self.name))}

    def epi_m_major_score(self, arg_tensor, gemm):
        return self.epi_m_major_preference if arg_tensor is not None else 0

    def _tile_size(self, cta_tile_shape_mnk):
        return cta_tile_shape_mnk[self.dim]

    def _broadcast_stride(self):
        # Col: stride (1,0) broadcasts along N. Row: stride (0,1) broadcasts along M.
        return (1, 0) if self.dim == 0 else (0, 1)

    def _reduce_dim(self):
        return 1 - self.dim

    def _smem_warps(self, warp_shape_mnk):
        warps = warp_shape_mnk[self._reduce_dim()] if warp_shape_mnk is not None else 1
        return max(warps - 1, 0)

    def smem_bytes(self, arg_tensor, cta_tile_shape_mnk, epi_tile, warp_shape_mnk=None):
        smem_warps = self._smem_warps(warp_shape_mnk)
        if arg_tensor is None or smem_warps == 0:
            return 0
        return self._tile_size(cta_tile_shape_mnk) * smem_warps * (Float32.width // 8)

    def smem_struct_field(self, gemm, params):
        tensor = getattr(params, self.name, None)
        smem_warps = self._smem_warps(gemm.epi_smem_warp_shape_mnk())
        if tensor is None or smem_warps == 0:
            return None
        size = self._tile_size(gemm.cta_tile_shape_mnk) * smem_warps
        return (f"s_{self.name}", cute.struct.Align[cute.struct.MemRange[Float32, size], 16])

    def get_smem_tensor(self, gemm, params, storage_epi):
        smem_warps = self._smem_warps(gemm.epi_smem_warp_shape_mnk())
        if getattr(params, self.name, None) is None or smem_warps == 0:
            return None
        return getattr(storage_epi, f"s_{self.name}").get_tensor(
            cute.make_layout((self._tile_size(gemm.cta_tile_shape_mnk), smem_warps))
        )

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        result = None
        if const_expr(param is not None):
            vec_mma_layout = cute.make_layout(
                (ctx.tile_M, ctx.tile_N), stride=self._broadcast_stride()
            )
            tDrReduce_layout = ctx.partition_for_epilogue_fn(
                cute.make_rmem_tensor(vec_mma_layout, Float32)
            ).layout
            tDrReduce = cute.make_rmem_tensor(tDrReduce_layout, Float32)
            result = (tDrReduce, smem_tensor)
        return result

    @cute.jit
    def begin_loop(self, gemm, state, epi_coord):
        result = None
        if const_expr(state is not None):
            tDrReduce = state[0]
            result = tDrReduce[None, None, None, epi_coord[0], epi_coord[1]]
            if const_expr(epi_coord[self._reduce_dim()] == 0):
                cute.filter_zeros(result).fill(0.0)
        return result


class ColVecReduce(VecReduce):
    """Column vector reduction: accumulates across N subtiles in registers,
    then reduces across N lanes/warps and writes to gmem per completed M stripe.

    The accumulation itself happens in epi_visit_subtile (user code).
    This op handles the register allocation (begin), per-subtile slicing (begin_loop),
    and reduction + gmem write (end_loop).
    """

    dim = 0
    epi_m_major_preference = -1

    @cute.jit
    def end_loop(
        self,
        gemm,
        param,
        state,
        epi_coord,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tile_coord_mnkl,
        varlen_manager,
        tidx,
    ):
        """Flush the current M stripe when the last N subtile has accumulated."""
        if const_expr(param is not None):
            epi_tile_shape = cute.zipped_divide(
                cute.make_layout(gemm.cta_tile_shape_mnk[:2]), epi_tile
            ).shape[1]
            if const_expr(epi_coord[1] == epi_tile_shape[1] - 1):
                tDrReduce, sDrReduce = state[0], state[1]
                tDrReduce_cur = tDrReduce[None, None, None, epi_coord[0], epi_coord[1]]
                tiled_copy = tiled_copy_t2r if tiled_copy_t2r is not None else tiled_copy_r2s
                reference_src = tiled_copy_t2r is None

                # ── Derive lane layout from tiled_copy ──
                lane_layout_MN, warp_layout_MN = _get_lane_warp_layouts(tiled_copy, reference_src)
                # For ColVecReduce: reduce across N lanes (lanes_in_N threads share same M row)
                lanes_in_N = cute.size(lane_layout_MN, mode=[1])
                is_lane_n_leader = cute.arch.lane_idx() % lanes_in_N == 0
                # Typically lanes_in_N is 4 for Sm90
                assert lanes_in_N == 1 << int(math.log2(lanes_in_N)), (
                    "lanes_in_N must be a power of 2 for butterfly reduction"
                )

                # Intra-warp shuffle reduction across N lanes
                if const_expr(lanes_in_N > 1):
                    # Assumes threads for each M row are contiguous along N, so
                    # warp_reduction over groups of lanes_in_N matches lane_layout_MN.
                    assert lane_layout_MN.stride[1] == 1
                    tDrReduce_flt = cute.filter_zeros(tDrReduce_cur)
                    for i in cutlass.range(cute.size(tDrReduce_flt), unroll_full=True):
                        tDrReduce_flt[i] = cute.arch.warp_reduction(
                            tDrReduce_flt[i], operator.add, threads_in_group=lanes_in_N
                        )

                warp_N = warp_layout_MN[1]
                warps_in_N = const_expr(cute.size(warp_N))
                partition_for_epilogue_fn = partial(
                    partition_for_epilogue,
                    epi_tile=epi_tile,
                    tiled_copy=tiled_copy,
                    tidx=tidx,
                    reference_src=tiled_copy_t2r is None,
                )
                tile_M, tile_N = gemm.cta_tile_shape_mnk[:2]
                tDcD = partition_for_epilogue_fn(cute.make_identity_tensor((tile_M, tile_N)))
                tDcD_cur = tDcD[None, None, None, epi_coord[0], epi_coord[1]]
                tDrReduce_m = layout_utils.convert_layout_zero_stride(
                    tDrReduce_cur, tDrReduce_cur.layout
                )[None, 0]
                tDcD_m = layout_utils.convert_layout_zero_stride(tDcD_cur, tDrReduce_cur.layout)[
                    None, 0
                ]

                # Inter-warp reduction through smem
                warp_idx = cute.arch.make_warp_uniform(tidx // cute.arch.WARP_SIZE)
                warp_n_idx = warp_layout_MN.get_hier_coord(warp_idx)[1]
                if const_expr(warps_in_N > 1):
                    if warp_n_idx > 0 and is_lane_n_leader:
                        for m in cutlass.range(cute.size(tDcD_m, mode=[0])):
                            row_idx = tDcD_m[m][0]
                            sDrReduce[row_idx, warp_n_idx - 1] = tDrReduce_m[m]
                    gemm.epilogue_barrier.arrive_and_wait()
                    if warp_n_idx == 0 and is_lane_n_leader:
                        for m in cutlass.range(cute.size(tDcD_m, mode=[0])):
                            row_idx = tDcD_m[m][0]
                            for warp_n in cutlass.range_constexpr(1, warps_in_N):
                                tDrReduce_m[m] += sDrReduce[row_idx, warp_n - 1]

                # Write to gmem
                batch_idx = tile_coord_mnkl[3]
                limit_m = min(varlen_manager.len_m(batch_idx) - tile_coord_mnkl[0] * tile_M, tile_M)
                limit_n_tiles = param.shape[2] if not varlen_manager.varlen_m else param.shape[1]
                if const_expr(not varlen_manager.varlen_m):
                    mColVec = param[batch_idx, None, tile_coord_mnkl[1]]
                else:
                    mColVec = cute.domain_offset(
                        (varlen_manager.params.cu_seqlens_m[batch_idx],),
                        param[None, tile_coord_mnkl[1]],
                    )
                gColVec = cute.local_tile(mColVec, (tile_M,), (tile_coord_mnkl[0],))
                should_write_gmem = (
                    is_lane_n_leader
                    if const_expr(warps_in_N == 1)
                    else warp_n_idx == 0 and is_lane_n_leader
                )
                if tile_coord_mnkl[1] < limit_n_tiles and should_write_gmem:
                    for m in cutlass.range(cute.size(tDcD_m, mode=[0])):
                        row_idx = tDcD_m[m][0]
                        if row_idx < limit_m:
                            gColVec[row_idx] = tDrReduce_m[m]


class RowVecReduce(VecReduce):
    """Row vector reduction: accumulates across M subtiles in registers,
    then reduces across M lanes/warps and writes to gmem per completed N stripe.

    Output shape is (L, ceildiv(M, tile_M), N): one partial sum per CTA-M tile per
    N column. This mirrors ColVecReduce with M/N swapped.
    """

    dim = 1
    epi_m_major_preference = 4

    @cute.jit
    def end_loop(
        self,
        gemm,
        param,
        state,
        epi_coord,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tile_coord_mnkl,
        varlen_manager,
        tidx,
    ):
        """Flush the current N stripe when the last M subtile has accumulated."""
        if const_expr(param is not None):
            epi_tile_shape = cute.zipped_divide(
                cute.make_layout(gemm.cta_tile_shape_mnk[:2]), epi_tile
            ).shape[1]
            if const_expr(epi_coord[0] == epi_tile_shape[0] - 1):
                tDrReduce, sDrReduce = state[0], state[1]
                tDrReduce_cur = tDrReduce[None, None, None, epi_coord[0], epi_coord[1]]
                tiled_copy = tiled_copy_t2r if tiled_copy_t2r is not None else tiled_copy_r2s
                reference_src = tiled_copy_t2r is None

                # ── Derive lane layout from tiled_copy ──
                lane_layout_MN, warp_layout_MN = _get_lane_warp_layouts(tiled_copy, reference_src)
                # For RowVecReduce: reduce across M lanes (lanes_in_M threads share same N col)
                lanes_in_M = cute.size(lane_layout_MN, mode=[0])
                lanes_in_N = cute.size(lane_layout_MN, mode=[1])
                is_lane_m_leader = cute.arch.lane_idx() < lanes_in_N
                assert lanes_in_M == 1 << int(math.log2(lanes_in_M)), (
                    "lanes_in_M must be a power of 2 for butterfly reduction"
                )
                if const_expr(lanes_in_N > 1):
                    assert lane_layout_MN.stride[1] == 1, (
                        "RowVecReduce assumes contiguous N lanes when lanes_in_N > 1"
                    )

                # Intra-warp shuffle reduction across M lanes. M lanes may be either contiguous
                # (SM100 N-major output) or strided by N lanes (SM100 M-major output).
                tDrReduce_n = layout_utils.convert_layout_zero_stride(
                    tDrReduce_cur, tDrReduce_cur.layout
                )[None, 0]
                if const_expr(lanes_in_M > 1):
                    for n in cutlass.range(cute.size(tDrReduce_n), unroll_full=True):
                        reduction_rows = lanes_in_M // 2
                        while reduction_rows > 0:
                            tDrReduce_n[n] += cute.arch.shuffle_sync_bfly(
                                tDrReduce_n[n],
                                offset=cute.crd2idx((reduction_rows, 0), lane_layout_MN),
                            )
                            reduction_rows = reduction_rows // 2

                warp_M = warp_layout_MN[0]
                warps_in_M = const_expr(cute.size(warp_M))
                partition_for_epilogue_fn = partial(
                    partition_for_epilogue,
                    epi_tile=epi_tile,
                    tiled_copy=tiled_copy,
                    tidx=tidx,
                    reference_src=tiled_copy_t2r is None,
                )
                tile_M, tile_N = gemm.cta_tile_shape_mnk[:2]
                tDcD = partition_for_epilogue_fn(cute.make_identity_tensor((tile_M, tile_N)))
                tDcD_cur = tDcD[None, None, None, epi_coord[0], epi_coord[1]]
                tDcD_n = layout_utils.convert_layout_zero_stride(tDcD_cur, tDrReduce_cur.layout)[
                    None, 0
                ]

                # Inter-warp reduction through smem
                warp_idx = cute.arch.make_warp_uniform(tidx // cute.arch.WARP_SIZE)
                warp_m_idx = warp_layout_MN.get_hier_coord(warp_idx)[0]
                if const_expr(warps_in_M > 1):
                    if warp_m_idx > 0 and is_lane_m_leader:
                        for n in cutlass.range(cute.size(tDcD_n, mode=[0])):
                            col_idx = tDcD_n[n][1]
                            sDrReduce[col_idx, warp_m_idx - 1] = tDrReduce_n[n]
                    gemm.epilogue_barrier.arrive_and_wait()
                    if warp_m_idx == 0 and is_lane_m_leader:
                        for n in cutlass.range(cute.size(tDcD_n, mode=[0])):
                            col_idx = tDcD_n[n][1]
                            for warp_m in cutlass.range_constexpr(1, warps_in_M):
                                tDrReduce_n[n] += sDrReduce[col_idx, warp_m - 1]

                # Write to gmem
                batch_idx = tile_coord_mnkl[3]
                limit_m_tiles = param.shape[1] if not varlen_manager.varlen_m else param.shape[0]
                if const_expr(not varlen_manager.varlen_m):
                    mRowVec = param[batch_idx, tile_coord_mnkl[0], None]
                else:
                    mRowVec = param[tile_coord_mnkl[0], None]
                gRowVec = cute.local_tile(mRowVec, (tile_N,), (tile_coord_mnkl[1],))
                limit_n = min(
                    cute.size(mRowVec, mode=[0]) - tile_coord_mnkl[1] * tile_N,
                    tile_N,
                )
                should_write_gmem = (
                    is_lane_m_leader
                    if const_expr(warps_in_M == 1)
                    else warp_m_idx == 0 and is_lane_m_leader
                )
                if tile_coord_mnkl[0] < limit_m_tiles and should_write_gmem:
                    for n in cutlass.range(cute.size(tDcD_n, mode=[0])):
                        col_idx = tDcD_n[n][1]
                        if col_idx < limit_n:
                            gRowVec[col_idx] = tDrReduce_n[n]
