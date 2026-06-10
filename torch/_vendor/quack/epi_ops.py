# Copyright (c) 2025, Tri Dao.
"""Composable epilogue operations (EpiOps) for GEMM kernels.

Each EpiOp encapsulates a single tensor kind's behavior across the epilogue lifecycle:
smem allocation, begin (one-time per-tile setup), begin_loop (per-subtile extraction),
end (cleanup).

The ops are composed via ComposableEpiMixin. Class-level `_epi_ops` is the
static schema; `_epi_ops_to_params_dict` (called from each subclass's
`epi_to_underlying_arguments`) shadows it with an instance-level tuple of only
the active ops (those whose arg tensor is non-None). All EpiOp hook methods
below therefore assume their `param` / `arg_tensor` is non-None — the
framework guarantees inactive ops are never iterated.
"""

import math
import operator
from functools import partial
from typing import NamedTuple

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, const_expr

from .epi_utils import assume_stride_divisibility, setup_epi_tensor
from .sm90_utils import partition_for_epilogue
from . import utils
from . import copy_utils
from . import layout_utils


class EpiContext:
    """Shared context passed to EpiOp.begin methods. Bundles common arguments.

    `tRS_rD_layout` is only populated by callers that need TileLoad — it's the
    register layout of the matmul output tile, which TileLoad uses to shape its
    own register tile so it lines up element-wise with tRS_rD in epi_visit_subtile.
    """

    __slots__ = (
        "epi_tile",
        "tiled_copy_t2r",
        "tiled_copy_r2s",
        "tile_coord_mnkl",
        "varlen_manager",
        "epilogue_barrier",
        "tidx",
        "tRS_rD_layout",
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
        tRS_rD_layout=None,
    ):
        self.epi_tile = epi_tile
        self.tiled_copy_t2r = tiled_copy_t2r
        self.tiled_copy_r2s = tiled_copy_r2s
        self.tile_coord_mnkl = tile_coord_mnkl
        self.varlen_manager = varlen_manager
        self.epilogue_barrier = epilogue_barrier
        self.tidx = tidx
        self.tRS_rD_layout = tRS_rD_layout
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


class EpiSmemBytes(NamedTuple):
    """Shared-memory accounting for one epilogue op.

    unstaged: allocated once per CTA tile.
    d_stage: allocated per D/store epilogue stage.
    c_stage: allocated per C/load epilogue stage.
    """

    unstaged: int = 0
    d_stage: int = 0
    c_stage: int = 0

    def __add__(self, other):
        return EpiSmemBytes(
            self.unstaged + other.unstaged,
            self.d_stage + other.d_stage,
            self.c_stage + other.c_stage,
        )

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)


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
        """Bytes of smem needed by unstaged / D-stage / C-stage storage."""
        return EpiSmemBytes()

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

    def is_tile_load(self):
        """Whether this op is a tile-sized epilogue input loaded through the C pipeline."""
        return False

    def load_g2s_copy_fn(
        self,
        gemm,
        params,
        smem_tensor,
        tile_coord_mnkl,
        varlen_manager,
        epi_pipeline,
    ):
        """Return a per-subtile gmem->smem copy function, or None."""
        return None

    # --- Device-side: kernel execution ---
    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        """One-time per-tile setup. Returns state for begin_loop."""
        return None

    def begin_loop(self, gemm, state, epi_coord):
        """Per-subtile extraction. Returns value for epi_visit_subtile."""
        return state

    @cute.jit
    def load_s2r(self, gemm, param, state, stage_idx):
        """Issue this op's tile-load smem->register copy for one epilogue stage."""
        pass

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
        if const_expr(self.dtype is not None):
            return utils.load_scalar_or_pointer(param, dtype=self.dtype)
        return utils.load_scalar_or_pointer(param)


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
        return EpiSmemBytes(
            unstaged=self._tile_size(cta_tile_shape_mnk) * (arg_tensor.element_type.width // 8)
        )

    def smem_struct_field(self, gemm, params):
        tensor = getattr(params, self.name)
        size = self._tile_size(gemm.cta_tile_shape_mnk)
        return (
            f"s_{self.name}",
            cute.struct.Align[cute.struct.MemRange[tensor.element_type, size], 16],
        )

    def get_smem_tensor(self, gemm, params, storage_epi):
        return getattr(storage_epi, f"s_{self.name}").get_tensor(
            cute.make_layout(self._tile_size(gemm.cta_tile_shape_mnk))
        )

    def needs_async_fence(self):
        return True

    def epi_m_major_score(self, arg_tensor, gemm):
        # It costs more registers (say 4x) to keep rowvec in register vs keeping colvec in register
        return 4 if self.dim == 1 else -1

    def _get_gmem_vec(self, param, ctx):
        """Get the global memory vector for this tile. Override for varlen."""
        return param[ctx.batch_idx, None]

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
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
        for m in cutlass.range(cute.size(tVsV.shape[1]), unroll_full=True):
            if tVcV[0, m] < tile_dim:  # Guard to avoid writing beyond the smem we've allocated
                pred = cute.make_rmem_tensor(1, Boolean)
                pred[0] = tVcV[0, m] < limit
                cute.copy(thr_copy, tVgV[None, m], tVsV[None, m], pred=pred)
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
        for m in cutlass.range(cute.size(tVsV.shape[1]), unroll_full=True):
            if tVcV[0, m] < tile_dim:  # Guard to avoid writing beyond the smem we've allocated
                pred = cute.make_rmem_tensor(1, Boolean)
                pred[0] = tVcV[0, m] < limit
                cute.copy(thr_copy, tVgV[None, m], tVsV[None, m], pred=pred)
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


class VecTupleLoad(EpiOp):
    vec_op_cls = VecLoad

    def param_fields(self):
        return [(self.name, object, None)]

    def to_params(self, gemm, args):
        tensors = getattr(args, self.name)
        if tensors is None:
            return {self.name: None}
        return {self.name: tuple(assume_stride_divisibility(tensor) for tensor in tensors)}

    def smem_bytes(self, arg_tensor, cta_tile_shape_mnk, epi_tile, warp_shape_mnk=None):
        return sum(
            (
                self.vec_op_cls(f"{self.name}{i}").smem_bytes(
                    tensor, cta_tile_shape_mnk, epi_tile, warp_shape_mnk
                )
                for i, tensor in enumerate(arg_tensor)
            ),
            EpiSmemBytes(),
        )

    def smem_struct_field(self, gemm, params):
        annotations = {}
        for i, tensor in enumerate(getattr(params, self.name)):
            size = self.vec_op_cls(f"{self.name}{i}")._tile_size(gemm.cta_tile_shape_mnk)
            annotations[f"v{i}"] = cute.struct.Align[
                cute.struct.MemRange[tensor.element_type, size], 16
            ]
        storage = type(f"{self.name}Storage", (), {"__annotations__": annotations})
        return (f"s_{self.name}", cute.struct(storage))

    def get_smem_tensor(self, gemm, params, storage_epi):
        storage = getattr(storage_epi, f"s_{self.name}")
        return tuple(
            getattr(storage, f"v{i}").get_tensor(
                cute.make_layout(
                    self.vec_op_cls(f"{self.name}{i}")._tile_size(gemm.cta_tile_shape_mnk)
                )
            )
            for i, _ in enumerate(getattr(params, self.name))
        )

    def needs_async_fence(self):
        return True

    def epi_m_major_score(self, arg_tensor, gemm):
        return sum(
            self.vec_op_cls(f"{self.name}{i}").epi_m_major_score(tensor, gemm)
            for i, tensor in enumerate(arg_tensor)
        )

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        states = []
        for i, tensor in enumerate(param):
            states.append(
                self.vec_op_cls(f"{self.name}{i}").begin(gemm, tensor, smem_tensor[i], ctx)
            )
        return tuple(states)

    def begin_loop(self, gemm, state, epi_coord):
        values = []
        for i, tensor_state in enumerate(state):
            values.append(
                self.vec_op_cls(f"{self.name}{i}").begin_loop(gemm, tensor_state, epi_coord)
            )
        return tuple(values)


class RowVecTupleLoad(VecTupleLoad):
    vec_op_cls = RowVecLoad


class ColVecTupleLoad(VecTupleLoad):
    vec_op_cls = ColVecLoad


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
        # Defaults are None so EpilogueParams can be constructed when this op is
        # filtered out (inactive). Active calls always set all four via to_params.
        return [
            (self._tma_atom_key(), object, None),
            (self.name, object, None),
            (self._smem_layout_key(), object, None),
            (self._epi_tile_key(), object, None),
        ]

    def to_params(self, gemm, args):
        tensor = getattr(args, self.name)
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
        if self.epi_tile_fn is not None:
            epi_tile = self.epi_tile_fn(None, epi_tile)
        # epi_tile may contain Layout entries (from SM100's compute_epilogue_tile_shape
        # fixup path), so extract the int shape first.
        return EpiSmemBytes(
            d_stage=cute.size(cute.shape(epi_tile)) * (arg_tensor.element_type.width // 8)
        )

    def smem_struct_field(self, gemm, params):
        smem_layout = getattr(params, self._smem_layout_key())
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
        smem_layout = getattr(params, self._smem_layout_key())
        return getattr(storage_epi, f"s_{self.name}").get_tensor(
            smem_layout.outer,
            swizzle=smem_layout.inner,
        )

    def tma_atoms(self, gemm, params):
        return [getattr(params, self._tma_atom_key())]


class _TileLoadState(NamedTuple):
    """Per-tile register state produced by TileLoad.begin and consumed by load_s2r /
    begin_loop. tRS_rTile is the register tile partitioned to match tRS_rD's layout;
    tSR_sTile / tSR_rTile drive the per-stage smem→register copy."""

    tiled_copy_s2r: object
    tRS_rTile: object
    tSR_rTile: object
    tSR_sTile: object


class TileLoad(EpiOp):
    """Tile-sized auxiliary input loaded through the epilogue load pipeline.

    TileLoad uses the same staged gmem->smem->register pipeline as GEMM's C operand,
    but it is exposed to the epilogue as ``epi_loop_tensors[name]`` instead of as
    ``tRS_rC``. That lets custom epilogues consume extra MxN tensors without using
    the GEMM C argument.

    Its shared memory is accounted as ``EpiSmemBytes.c_stage``, so it is allocated
    per epilogue load stage. Multiple TileLoads are supported: each has its own TMA
    descriptor and smem buffer, and the pipeline transaction count includes C plus
    all enabled TileLoad buffers. Supported on SM90, SM100, and SM120.
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

    # The original LayoutEnum and element_type can't be recovered from the
    # TMA-prepared tensor that ends up in params (`from_tensor` returns a typing
    # annotation post-TMA, not a Numeric class). We stash both on the gemm at
    # to_params time and read them back in begin(). The dtype is also exposed on
    # the params dataclass for smem_struct_field.
    def _layout_gemm_attr(self):
        return f"_tile_load_layout_{self.name}"

    def _dtype_gemm_attr(self):
        return f"_tile_load_dtype_{self.name}"

    def _dtype_field(self):
        return f"{self.name}_dtype"

    def param_fields(self):
        # Defaults are None so EpilogueParams can be constructed when this op is
        # filtered out (inactive). Active calls always set all five via to_params.
        return [
            (self._tma_atom_key(), object, None),
            (self.name, object, None),
            (self._smem_layout_key(), object, None),
            (self._epi_tile_key(), object, None),
            (self._dtype_field(), object, None),
        ]

    def to_params(self, gemm, args):
        tensor = getattr(args, self.name)
        setattr(gemm, self._layout_gemm_attr(), cutlass.utils.LayoutEnum.from_tensor(tensor))
        setattr(gemm, self._dtype_gemm_attr(), tensor.element_type)
        epi_tile = self.epi_tile_fn(gemm, gemm.epi_tile) if self.epi_tile_fn else None
        tma_atom, tma_tensor, smem_layout, epi_tile_out = setup_epi_tensor(
            gemm, tensor, epi_tile=epi_tile, op_type="load", stage=gemm.epi_c_stage
        )
        return {
            self._tma_atom_key(): tma_atom,
            self.name: tma_tensor,
            self._smem_layout_key(): smem_layout,
            self._epi_tile_key(): epi_tile_out,
            self._dtype_field(): tensor.element_type,
        }

    def is_tile_load(self):
        return True

    def smem_bytes(self, arg_tensor, cta_tile_shape_mnk, epi_tile, warp_shape_mnk=None):
        if self.epi_tile_fn is not None:
            epi_tile = self.epi_tile_fn(None, epi_tile)
        # epi_tile may contain Layout entries from SM100's compute_epilogue_tile_shape
        # fixup; extract the int shape first.
        return EpiSmemBytes(
            c_stage=cute.size(cute.shape(epi_tile)) * (arg_tensor.element_type.width // 8)
        )

    def smem_struct_field(self, gemm, params):
        smem_layout = getattr(params, self._smem_layout_key())
        dtype = getattr(params, self._dtype_field())
        return (
            f"s_{self.name}",
            cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(smem_layout)],
                gemm.buffer_align_bytes,
            ],
        )

    def get_smem_tensor(self, gemm, params, storage_epi):
        smem_layout = getattr(params, self._smem_layout_key())
        return getattr(storage_epi, f"s_{self.name}").get_tensor(
            smem_layout.outer,
            swizzle=smem_layout.inner,
        )

    def tma_atoms(self, gemm, params):
        return [getattr(params, self._tma_atom_key())]

    def load_g2s_copy_fn(
        self,
        gemm,
        params,
        smem_tensor,
        tile_coord_mnkl,
        varlen_manager,
        epi_pipeline,
    ):
        tensor = getattr(params, self.name)
        batch_idx = tile_coord_mnkl[3]
        copy_tile_fn, _, _ = gemm.epilog_gmem_copy_and_partition(
            getattr(params, self._tma_atom_key()),
            varlen_manager.offset_batch_epi(tensor, batch_idx),
            gemm.cta_tile_shape_mnk[:2],
            getattr(params, self._epi_tile_key()),
            smem_tensor,
            tile_coord_mnkl,
        )
        return copy_utils.tma_producer_copy_fn(copy_tile_fn, epi_pipeline)

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        assert gemm.arch in (90, 100, 120), "TileLoad requires the SM90/SM100/SM120 epilogue path"
        assert ctx.tRS_rD_layout is not None
        smem_load_ref = ctx.tiled_copy_t2r if const_expr(gemm.arch == 100) else gemm.tiled_mma
        tiled_copy_s2r, tRS_rTile, tSR_rTile, tSR_sTile = gemm.epilog_smem_load_and_partition(
            smem_load_ref,
            getattr(gemm, self._layout_gemm_attr()),
            getattr(gemm, self._dtype_gemm_attr()),
            smem_tensor,
            ctx.tRS_rD_layout,
            ctx.tidx,
        )
        # Shape: (s2r-copy-handle, register-tile-as-rD-layout, smem→r retile target,
        # smem→r staged source). begin_loop returns tRS_rTile; load_s2r uses the rest.
        return _TileLoadState(tiled_copy_s2r, tRS_rTile, tSR_rTile, tSR_sTile)

    @cute.jit
    def load_s2r(self, gemm, param, state, stage_idx):
        cute.copy(
            state.tiled_copy_s2r,
            state.tSR_sTile[None, None, None, stage_idx],
            state.tSR_rTile,
        )

    @cute.jit
    def begin_loop(self, gemm, state, epi_coord):
        return state.tRS_rTile


class TileTupleLoad(EpiOp):
    def _tma_atoms_key(self):
        return f"tma_atoms_{self.name}"

    def _smem_layouts_key(self):
        return f"epi_{self.name}_smem_layouts_staged"

    def _epi_tiles_key(self):
        return f"epi_tiles_{self.name}"

    def _layout_gemm_attr(self, index):
        return f"_tile_tuple_load_layout_{self.name}_{index}"

    def _dtype_gemm_attr(self, index):
        return f"_tile_tuple_load_dtype_{self.name}_{index}"

    def param_fields(self):
        return [
            (self._tma_atoms_key(), object, None),
            (self.name, object, None),
            (self._smem_layouts_key(), object, None),
            (self._epi_tiles_key(), object, None),
        ]

    def to_params(self, gemm, args):
        tensors = getattr(args, self.name)
        tma_atoms, tma_tensors, smem_layouts, epi_tiles = [], [], [], []
        for i, tensor in enumerate(tensors):
            setattr(gemm, self._layout_gemm_attr(i), cutlass.utils.LayoutEnum.from_tensor(tensor))
            setattr(gemm, self._dtype_gemm_attr(i), tensor.element_type)
            tma_atom, tma_tensor, smem_layout, epi_tile = setup_epi_tensor(
                gemm, tensor, epi_tile=None, op_type="load", stage=gemm.epi_c_stage
            )
            tma_atoms.append(tma_atom)
            tma_tensors.append(tma_tensor)
            smem_layouts.append(smem_layout)
            epi_tiles.append(epi_tile)
        return {
            self._tma_atoms_key(): tuple(tma_atoms),
            self.name: tuple(tma_tensors),
            self._smem_layouts_key(): tuple(smem_layouts),
            self._epi_tiles_key(): tuple(epi_tiles),
        }

    def is_tile_load(self):
        return True

    def smem_bytes(self, arg_tensor, cta_tile_shape_mnk, epi_tile, warp_shape_mnk=None):
        return sum(
            (
                EpiSmemBytes(
                    c_stage=cute.size(cute.shape(epi_tile)) * tensor.element_type.width // 8
                )
                for tensor in arg_tensor
            ),
            EpiSmemBytes(),
        )

    def smem_struct_field(self, gemm, params):
        annotations = {}
        for i, smem_layout in enumerate(getattr(params, self._smem_layouts_key())):
            annotations[f"v{i}"] = cute.struct.Align[
                cute.struct.MemRange[
                    getattr(gemm, self._dtype_gemm_attr(i)), cute.cosize(smem_layout)
                ],
                gemm.buffer_align_bytes,
            ]
        storage = type(f"{self.name}Storage", (), {"__annotations__": annotations})
        return (f"s_{self.name}", cute.struct(storage))

    def get_smem_tensor(self, gemm, params, storage_epi):
        storage = getattr(storage_epi, f"s_{self.name}")
        return tuple(
            getattr(storage, f"v{i}").get_tensor(smem_layout.outer, swizzle=smem_layout.inner)
            for i, smem_layout in enumerate(getattr(params, self._smem_layouts_key()))
        )

    def tma_atoms(self, gemm, params):
        return list(getattr(params, self._tma_atoms_key()))

    def load_g2s_copy_fn(self, gemm, params, smem_tensor, tile_coord_mnkl, varlen_manager, epi_pipeline):
        copy_fns = []
        for tma_atom, tensor, epi_tile, smem in zip(
            getattr(params, self._tma_atoms_key()),
            getattr(params, self.name),
            getattr(params, self._epi_tiles_key()),
            smem_tensor,
        ):
            copy_tile_fn, _, _ = gemm.epilog_gmem_copy_and_partition(
                tma_atom,
                varlen_manager.offset_batch_epi(tensor, tile_coord_mnkl[3]),
                gemm.cta_tile_shape_mnk[:2],
                epi_tile,
                smem,
                tile_coord_mnkl,
            )
            copy_fns.append(copy_utils.tma_producer_copy_fn(copy_tile_fn, epi_pipeline))
        return copy_utils.chain_tma_producer_copy_fns(tuple(copy_fns))

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        assert gemm.arch in (90, 100, 120), "TileTupleLoad requires the SM90/SM100/SM120 epilogue path"
        states = []
        smem_load_ref = ctx.tiled_copy_t2r if const_expr(gemm.arch == 100) else gemm.tiled_mma
        for i, smem in enumerate(smem_tensor):
            tiled_copy_s2r, tRS_rTile, tSR_rTile, tSR_sTile = gemm.epilog_smem_load_and_partition(
                smem_load_ref,
                getattr(gemm, self._layout_gemm_attr(i)),
                getattr(gemm, self._dtype_gemm_attr(i)),
                smem,
                ctx.tRS_rD_layout,
                ctx.tidx,
            )
            states.append(_TileLoadState(tiled_copy_s2r, tRS_rTile, tSR_rTile, tSR_sTile))
        return tuple(states)

    @cute.jit
    def load_s2r(self, gemm, param, state, stage_idx):
        for tile_state in state:
            cute.copy(
                tile_state.tiled_copy_s2r,
                tile_state.tSR_sTile[None, None, None, stage_idx],
                tile_state.tSR_rTile,
            )

    @cute.jit
    def begin_loop(self, gemm, state, epi_coord):
        return tuple(tile_state.tRS_rTile for tile_state in state)


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
        return self.epi_m_major_preference

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
        if smem_warps == 0:
            return EpiSmemBytes()
        return EpiSmemBytes(
            unstaged=self._tile_size(cta_tile_shape_mnk) * smem_warps * (Float32.width // 8)
        )

    def smem_struct_field(self, gemm, params):
        smem_warps = self._smem_warps(gemm.epi_smem_warp_shape_mnk())
        if smem_warps == 0:
            return None
        size = self._tile_size(gemm.cta_tile_shape_mnk) * smem_warps
        return (f"s_{self.name}", cute.struct.Align[cute.struct.MemRange[Float32, size], 16])

    def get_smem_tensor(self, gemm, params, storage_epi):
        smem_warps = self._smem_warps(gemm.epi_smem_warp_shape_mnk())
        if smem_warps == 0:
            return None
        return getattr(storage_epi, f"s_{self.name}").get_tensor(
            cute.make_layout((self._tile_size(gemm.cta_tile_shape_mnk), smem_warps))
        )

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        vec_mma_layout = cute.make_layout((ctx.tile_M, ctx.tile_N), stride=self._broadcast_stride())
        tDrReduce_layout = ctx.partition_for_epilogue_fn(
            cute.make_rmem_tensor(vec_mma_layout, Float32)
        ).layout
        tDrReduce = cute.make_rmem_tensor(tDrReduce_layout, Float32)
        return (tDrReduce, smem_tensor)

    @cute.jit
    def begin_loop(self, gemm, state, epi_coord):
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
