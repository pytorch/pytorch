# Copyright (c) 2025-2026, Han Guo, Tri Dao.
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
import hashlib
import inspect
from functools import partial
from typing import NamedTuple, Optional

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as blackwell_helpers
from cutlass import Boolean, Float32, Int32, Uint32, const_expr
from cutlass.cute.nvgpu import warp
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

import torch

from torch._vendor.quack.compile_utils import div_for_dtype, fake_batched, make_fake_tensor
from torch._vendor.quack.cute_dsl_utils import torch2cute_dtype_map
import torch._vendor.quack.sm90_utils as sm90_utils
from torch._vendor.quack.rounding import (
    SR_STORE_DTYPES,
    RoundingMode,
    convert_f32_frag_sr,
    epilogue_aux_out_sr_seed,
    epilogue_sr_seed,
)
from torch._vendor.quack.reduce import swap_shuffle_reduce
from torch._vendor.quack.sm90_utils import partition_for_epilogue
import torch._vendor.quack.utils as utils
import torch._vendor.quack.copy_utils as copy_utils
import torch._vendor.quack.layout_utils as layout_utils


def assume_stride_divisibility(tensor):
    """Assume all strides are divisible by 32 bits (except static strides).

    Used for broadcast vectors and similar tensors where stride alignment is guaranteed.
    Returns a new tensor with the assumed strides.
    """
    if tensor is None:
        return None
    divby = 32 // tensor.element_type.width
    if divby <= 1:  # >= 32-bit elements: nothing to assume
        return tensor
    new_stride = tuple(
        cute.assume(s, divby=divby) if not cute.is_static(s) else s for s in tensor.stride
    )
    return cute.make_tensor(tensor.iterator, cute.make_layout(tensor.shape, stride=new_stride))


def setup_epi_tensor(gemm, tensor, epi_tile=None, op_type="store", stage=None):
    """Create copy metadata + smem layout for a supplemental epilogue tensor.

    Args:
        gemm: The GEMM object (provides arch, epi_stage, and epilogue layout helpers).
        tensor: The global memory tensor to set up for the epilogue.
        epi_tile: Epilogue tile shape. Defaults to gemm.epi_tile.
        op_type: "store" or "load".

    Returns:
        (copy_atom, tensor, smem_layout_staged, epi_tile). copy_atom is None for pre-TMA archs.
    """
    if epi_tile is None:
        epi_tile = gemm.epi_tile
    if stage is None:
        stage = gemm.epi_stage
    dtype = tensor.element_type
    layout = cutlass.utils.LayoutEnum.from_tensor(tensor)
    utils_cls = blackwell_helpers if gemm.arch >= 100 else sm90_utils
    smem_layout_staged = utils_cls.make_smem_layout_epi(dtype, layout, epi_tile, stage)
    # Ragging-for-TMA is for varlen_m stores that need a per-batch row offset baked
    # into the TMA descriptor. Loads don't currently support varlen_m, so skip the
    # ragging conversion.
    tma_input = (
        copy_utils.create_ragged_tensor_for_tma(tensor, ragged_dim=0, ptr_shift=True)
        if op_type != "load" and cute.rank(tensor) == 2
        else tensor
    )
    tma_atom, tma_tensor = gemm._make_tma_epi_atoms_and_tensors(
        tma_input,
        smem_layout_staged,
        epi_tile,
        op_type=op_type,
    )
    return tma_atom, tma_tensor, smem_layout_staged, epi_tile


def _callable_config_key(fn):
    """Stable, picklable identity for a callable stored in an EpiOp config."""
    if fn is None:
        return None
    try:
        source = inspect.getsource(fn).encode()
    except (OSError, TypeError):
        code = getattr(fn, "__code__", None)
        source = code.co_code if code is not None else repr(fn).encode()
    return (
        getattr(fn, "__module__", ""),
        getattr(fn, "__qualname__", repr(fn)),
        hashlib.sha256(source).hexdigest(),
    )


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


def _mode_flat_stride(layout, mode):
    """Trace-time introspection: the constant stride ``s`` if ``layout``
    mode ``mode`` enumerates the flat arithmetic progression ``r -> r*s``
    (the lane geometry ``swap_shuffle_reduce`` needs), else None. A nested
    mode qualifies when its (shape, stride) pairs chain column-major
    (stride[i+1] == stride[i] * shape[i])."""

    def flat(x):
        return sum((flat(e) for e in x), []) if isinstance(x, (tuple, list)) else [x]

    try:
        shapes = [int(v) for v in flat(layout.shape[mode])]
        strides = [int(v) for v in flat(layout.stride[mode])]
    except (TypeError, ValueError):  # dynamic layout: no static answer
        return None
    base, expected = None, None
    for sh, st in zip(shapes, strides):
        if sh == 1:
            continue
        if base is None:
            base = st
        elif st != expected:
            return None
        expected = st * sh
    return base


@cute.jit
def _lane_warp_info_n(tiled_copy, reference_src, tidx):
    """(lanes_in_N, warps_in_N, warp_n_idx, is_lane_n_leader) for N-direction
    reduces and exchanges (ColVecReduce, OnlineLSEReduce, GroupedColStatsBase).
    Asserts the contiguous power-of-2 N-lane group the butterfly protocols
    assume."""
    lane_layout_MN, warp_layout_MN = _get_lane_warp_layouts(tiled_copy, reference_src)
    lanes_in_N = const_expr(cute.size(lane_layout_MN, mode=[1]))
    warps_in_N = const_expr(cute.size(warp_layout_MN, mode=[1]))
    assert lanes_in_N == 1 << int(math.log2(lanes_in_N)), (
        "lanes_in_N must be a power of 2 for butterfly reduction"
    )
    if const_expr(lanes_in_N > 1):
        assert lane_layout_MN.stride[1] == 1, (
            "N-direction reduce needs contiguous N lanes (lane_layout stride[1] == 1)"
        )
    warp_idx = cute.arch.make_warp_uniform(tidx // cute.arch.WARP_SIZE)
    warp_n_idx = warp_layout_MN.get_hier_coord(warp_idx)[1]
    is_lane_n_leader = cute.arch.lane_idx() % lanes_in_N == 0
    return lanes_in_N, warps_in_N, warp_n_idx, is_lane_n_leader


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

    # --- Value-port protocol (quack.epilogue.frontend fn frontend). fn_port is
    # THE declaration of how an op joins the fn's per-element dataflow (the
    # frontend never isinstance-dispatches); the resource lifecycle below
    # (begin/begin_loop/end_loop/end) stays the smem/TMA/flush protocol.
    #   "row" / "col" / "tile" / "scalar": built-in fragment kinds — the fn
    #            receives op.name as a per-element value (or the scalar), with
    #            the frontend's standard shape checks and swap-relabeling.
    #   "value": custom value-source op: the fn receives op.name per element;
    #            begin_loop's fragment must be elementwise congruent with the
    #            accumulator tile and DENSE (fn_prepare may densify).
    #   "apply": the fn receives op.name as a CALLABLE — `y = rope(acc)` — so the
    #            op's math slots into the fn's dataflow at a user-chosen point.
    #            fn_apply runs inside the (possibly vectorized) loop: index only
    #            dense per-loop-index state prepared in fn_prepare, and speak the
    #            scalar/F2/Pair value vocabulary.
    #   "sink":  the fn returns op.name; the frontend collects the values into a
    #            dense fragment and hands it to fn_sink_flush once per subtile
    #            (fragment-level, so sinks can do numerically smart things like
    #            one rescale per subtile instead of per element).
    #   None: not usable from the fn frontend (hand-written mixins only).
    fn_port = None
    supports_swap_ab = False

    def fn_prepare(self, gemm, state, paired):
        """Per-subtile port state derived from this op's begin_loop result.
        ``paired``: the fn loop runs per adjacent-N pair (values are Pairs)."""
        return state

    def fn_apply(self, gemm, pstate, i, value):
        raise NotImplementedError

    def fn_sink_flush(self, gemm, state, frag):
        """Fold a fragment of fn-produced values into this op's accumulator.
        ``state`` is the begin_loop result; ``frag`` is elementwise-congruent
        with the accumulator tile fragment."""
        raise NotImplementedError

    def __init__(self, name):
        self.name = name

    def config_key(self):
        """Picklable static configuration that affects generated code.

        Stateless ops inherit the empty key. Stateful ops must opt in
        explicitly: silently omitting an instance attribute would alias two
        semantically different epilogues in the persistent JIT cache.
        """
        extra = tuple(sorted(set(vars(self)) - {"name"}))
        if extra:
            raise NotImplementedError(
                f"{type(self).__name__} has static configuration {extra}; implement config_key()"
            )
        return ()

    def cache_key(self):
        return (
            type(self).__module__,
            type(self).__qualname__,
            self.name,
            self.config_key(),
        )

    def __quack_semantic_key__(self):
        # Fail-closed semantic-key protocol (quack.epilogue.frontend): op instances
        # captured by epilogue fns fingerprint as their cache identity.
        return self.cache_key()

    # --- Host-side: torch-arg schema (drives the generic plan/compile layer in
    # quack.gemm_runtime.host). Each op describes its own argument in three steps:
    # host_arg_key extracts a small picklable descriptor from the caller's torch
    # value (part of the jit_cache disk key), host_fake_arg rebuilds the fake
    # trace-time argument from that descriptor alone, and host_call_arg converts
    # the per-call torch value into what the compiled signature expects. ---
    def host_arg_key(self, value):
        """Picklable compile-key descriptor of the caller's value; None = absent
        (the op is filtered out of the compiled epilogue)."""
        if value is None:
            return None

        return (torch2cute_dtype_map[value.dtype], value.ndim)

    def host_fake_arg(self, key, fctx):
        """Fake trace-time argument reconstructed from ``host_arg_key``'s
        descriptor. ``fctx`` is a quack.gemm_runtime.host.FakeArgCtx with the shared
        (m, n, k, l) sym ints and the batched/varlen_m flags."""
        return None

    def host_call_arg(self, value, key):
        """Per-call runtime argument matching the compiled signature."""
        return value

    def arg_spec_type(self, const=False):
        """Type annotation for this op's EpilogueArguments field. ``const``
        reflects host_arg_form: a Constexpr[...] annotation makes the TVM-FFI
        converter emit NO runtime argument (value baked at trace, None at
        call — see the converter patch in quack.cute_dsl_utils)."""
        return Optional[cute.Tensor]

    def host_arg_form(self, value):
        """Mint-key suffix for per-call arg FORMS that change the compiled
        signature (e.g. constexpr vs tensor); "" when there is one form."""
        return ""

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

    def is_tile_store(self):
        """Whether this op is a tile-sized epilogue output on the aux store path."""
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

    def end_loop_stage(
        self,
        gemm,
        param,
        state,
        epi_coord,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tidx,
    ):
        """Per-subtile flush, phase 1: intra-warp reduce + smem staging.

        Returns None when this op has nothing to flush at this epi_coord,
        else ``(needs_barrier, finish_state)``. The driver (epi_end_loop)
        issues ONE shared epilogue barrier per subtile covering every op
        that staged — a multi-sink epilogue syncs once per flush, not once
        per sink (each sink stages into its own disjoint smem) — then calls
        ``end_loop_finish(finish_state)``."""
        return None

    def end_loop_finish(self, gemm, param, staged, tile_coord_mnkl, varlen_manager):
        """Per-subtile flush, phase 2 (after the driver's shared barrier):
        merge the smem-staged partials + write gmem."""
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

    fn_port = "scalar"
    supports_swap_ab = True

    def __init__(self, name, dtype=None):
        super().__init__(name)
        self.dtype = dtype

    def config_key(self):
        return (self.dtype,)

    def _target_dtype(self):
        return self.dtype if self.dtype is not None else Float32

    def arg_spec_type(self, const=False):
        return Optional[(self.dtype or Float32) | cute.Tensor]

    def _validate_pointer_value(self, value):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"scalar '{self.name}' pointer value must be a torch.Tensor")
        if value.numel() != 1:
            raise ValueError(f"scalar '{self.name}' tensor must contain exactly one element")
        if not value.is_cuda:
            raise ValueError(f"scalar '{self.name}' tensor must be on CUDA")
        if not value.is_contiguous():
            raise ValueError(f"scalar '{self.name}' tensor must be contiguous")
        actual = torch2cute_dtype_map.get(value.dtype)
        target = self._target_dtype()
        if actual != target:
            raise TypeError(
                f"scalar '{self.name}' tensor must have dtype {target}, got {value.dtype}"
            )

    def host_key_for_mode(self, mode):
        return (("absent", "immediate", "pointer")[mode], self._target_dtype())

    def _decode_host_key(self, key):
        # Integer keys are accepted for the existing hand-written wrappers;
        # new callers use the self-describing (mode, dtype) form.
        return self.host_key_for_mode(key) if isinstance(key, int) else key

    # Scalar keys are the compile-time *mode*: 0 = absent (op compiled out),
    # 1 = host constant, 2 = device pointer. Variants with a non-trivial
    # neutral-folding rule (e.g. alpha == 1.0 -> absent) pass the mode as an
    # epi_key_overrides entry instead of relying on this default.
    def host_arg_key(self, value):
        if value is None:
            return self.host_key_for_mode(0)
        if hasattr(value, "data_ptr"):
            self._validate_pointer_value(value)
            return self.host_key_for_mode(2)
        return self.host_key_for_mode(1)

    def host_fake_arg(self, key, fctx):
        mode, dtype = self._decode_host_key(key)
        if mode == "absent":
            return None
        if mode == "immediate":
            return dtype(0)
        from cutlass.cute.runtime import make_ptr

        return make_ptr(dtype, 0, cute.AddressSpace.gmem, assumed_align=4)

    def host_call_arg(self, value, key):
        mode, dtype = self._decode_host_key(key)
        if mode == "absent":
            return None
        if mode == "immediate":
            return dtype(value)
        self._validate_pointer_value(value)
        return value.data_ptr()

    def param_fields(self):
        return [(self.name, object, None)]

    def to_params(self, gemm, args):
        return {self.name: getattr(args, self.name)}

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        if const_expr(self.dtype is not None):
            value = utils.load_scalar_or_pointer(param, dtype=self.dtype)
            if const_expr(self.dtype in (cutlass.Float16, cutlass.BFloat16)):
                return Float32(value)
            return value
        return utils.load_scalar_or_pointer(param)


def is_floating_dtype(dtype):
    """Return whether a static CuTeDSL numeric type is floating-point."""
    return hasattr(dtype, "__tvm_ffi_float__")


class VecLoad(EpiOp):
    """Base class for broadcast vector loads (row or col) via cp_async.

    Subclasses set `dim` to 0 (M/col) or 1 (N/row) and override `_get_gmem_vec`
    for varlen handling.
    """

    dim = None  # 0 for col (M), 1 for row (N)

    def __init__(self, name, dtype=None):
        super().__init__(name)
        self.dtype = dtype

    def config_key(self):
        return (self.dtype,)

    def host_fake_arg(self, key, fctx):
        dtype, ndim = key
        vec_dim = fctx.n if self.dim == 1 else fctx.m
        shape = (fctx.l, vec_dim) if ndim == 2 else (vec_dim,)
        return make_fake_tensor(dtype, shape, leading_dim=ndim - 1, divisibility=4)

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

    @cute.jit
    def _valid_extent(self, vector, coord_idx, tile_dim, ctx):
        """Return the in-bounds vector extent for the current tile."""
        return min(cute.size(vector, mode=[0]) - coord_idx * tile_dim, tile_dim)

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
        if cute.rank(param) == 1:
            return param  # rank-1 (vec,): one vector shared across the batch
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
        limit = self._valid_extent(mVec, coord_idx, tile_dim, ctx)
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
        # Pre-allocate register tensor reused across begin_loop calls.
        tDsV_sub = cute.group_modes(tDsV, 3, cute.rank(tDsV))[None, None, None, 0]
        register_dtype = gemm.acc_dtype if is_floating_dtype(dtype) else dtype
        tDrV_cvt = cute.make_rmem_tensor(tDsV_sub.layout, register_dtype)
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
            tDrV_cvt.store(tDrV.load().to(tDrV_cvt.element_type))
        return tDrV_cvt


class RowVecLoad(VecLoad):
    """Loads a row vector (N,) via cp_async, broadcasts along M with stride (0,1)."""

    dim = 1
    fn_port = "row"
    supports_swap_ab = True


class ColVecLoad(VecLoad):
    """Loads a col vector (M,) via cp_async, broadcasts along N with stride (1,0).

    Optimization: with N-major subtile loop, consecutive epi_n iterations for the same
    epi_m share the same column data. The smem→register copy only runs when epi_n == 0.
    Supports varlen_m via domain_offset.
    """

    dim = 0
    fn_port = "col"
    supports_swap_ab = True

    @cute.jit
    def _get_gmem_vec(self, param, ctx):
        if const_expr(ctx.varlen_manager.varlen_m):
            # varlen: rank-1 (total_m,) concatenated vector, offset per sequence
            mVec = cute.domain_offset(
                (ctx.varlen_manager.params.cu_seqlens_m[ctx.batch_idx],), param
            )
        elif const_expr(cute.rank(param) == 2):
            mVec = param[ctx.batch_idx, None]
        else:
            mVec = param  # dense rank-1 (m,): one vector shared across the batch
        return mVec

    @cute.jit
    def _valid_extent(self, vector, coord_idx, tile_dim, ctx):
        """Use the active varlen-M extent for column-vector tail predicates."""
        return min(
            ctx.varlen_manager.len_m(ctx.batch_idx) - coord_idx * tile_dim,
            tile_dim,
        )


def _contract_epi_tile_n(epi_tile, group):
    """Contract an epilogue tile's N extent by a static group size."""
    if isinstance(epi_tile[1], cute.Layout):
        return (epi_tile[0], cute.recast_layout(group, 1, epi_tile[1]))
    return (epi_tile[0], epi_tile[1] // group)


def _gated_epi_tile_fn(gemm, epi_tile):
    """Halve the N dimension of the epi_tile for gated postact."""
    return _contract_epi_tile_n(epi_tile, 2)


def _grouped_main_epi_tile_2(gemm, epi_tile):
    """Contract a grouped-main output tile by two adjacent N lanes."""
    return _contract_epi_tile_n(epi_tile, 2)


def _grouped_main_epi_tile_4(gemm, epi_tile):
    """Contract a grouped-main output tile by four adjacent N lanes."""
    return _contract_epi_tile_n(epi_tile, 4)


class TileStore(EpiOp):
    """Tile-sized output tensor stored via TMA (e.g. postact).

    Owns the whole device store path for its tensor: the arch-specific
    register-to-smem tiled copy, dtype conversion with per-op rounding, the
    gated halved-tile machinery (epi tile, STSM register permute, SM120 copy
    override, SM120 epi-tile override), the store predicate, and the
    smem-to-gmem TMA copy. The driver (gemm_base.epilogue) only sequences the
    hooks: ``store_setup`` once per CTA tile, ``store_convert`` once per
    subtile; each op derives everything from its own tensor, so multiple
    TileStores with mixed dtypes compose.

    Args:
        name: field name in EpilogueArguments/Params (e.g. "postact")
        epi_tile_fn: optional (gemm, epi_tile) -> epi_tile override
        gated: half-of-GEMM-N output paired over adjacent accumulator N lanes
            (implies the halved epi tile; 16-bit n-major only; tile_N % 32 on
            SM90)
        rounding: per-op RoundingMode override; None = the kernel-global
            ``gemm.rounding_mode`` (the legacy mixin behavior)
        store_pred_fn: optional ``(gemm, tile_coord_mnkl) -> Boolean``
            evaluated once per CTA tile; False skips this op's gmem store
            (e.g. GemmSymmetric skips the mirrored write on diagonal tiles)
        quant: optional quantize codec (BlockScaleFactorStore) for this
            output, declared where the output is declared. Declaration sugar:
            the EpiMod frontend lifts it into the op set (extra_ops), and the
            driver runs it on the final fragment right before store_convert
            (see gemm_base.epilogue and ComposableEpiMixin._epi_store_quant).
    """

    supports_swap_ab = True

    def __init__(
        self, name, epi_tile_fn=None, gated=False, rounding=None, store_pred_fn=None, quant=None
    ):
        super().__init__(name)
        if gated and epi_tile_fn is None:
            epi_tile_fn = _gated_epi_tile_fn
        self.epi_tile_fn = epi_tile_fn
        self.gated = gated
        self.rounding = rounding
        self.store_pred_fn = store_pred_fn
        if quant is not None:
            assert getattr(quant, "quant_output", None) == name, (
                f"quantize codec for output {name!r} must declare output={name!r}"
            )
        self.quant = quant

    def config_key(self):
        return (
            _callable_config_key(self.epi_tile_fn),
            self.gated,
            self.rounding,
            _callable_config_key(self.store_pred_fn),
            self.quant.cache_key() if self.quant is not None else None,
        )

    def is_tile_store(self):
        return True

    def _tma_atom_key(self):
        return f"tma_atom_{self.name}"

    def _smem_layout_key(self):
        return f"epi_{self.name}_smem_layout_staged"

    def _epi_tile_key(self):
        return f"epi_tile_{self.name}"

    # Same gemm-stash pattern as TileLoad: LayoutEnum/dtype can't be recovered
    # from the TMA-prepared tensor in params, so to_params saves them for the
    # device-side hooks. The dtype is also a params field for smem_struct_field.
    def _layout_gemm_attr(self):
        return f"_tile_store_layout_{self.name}"

    def _dtype_gemm_attr(self):
        return f"_tile_store_dtype_{self.name}"

    def _dtype_field(self):
        return f"{self.name}_dtype"

    def host_arg_key(self, value):
        if value is None:
            return None

        major = "n" if value.stride(-1) == 1 else "m"
        return (torch2cute_dtype_map[value.dtype], major)

    def host_fake_arg(self, key, fctx):
        dtype, major = key
        # A halved/reshaped tile has an N extent unrelated to the GEMM's n,
        # while only sub-byte storage requires static packing divisibility.
        if dtype.width < 8:
            n = cute.sym_int(divisibility=div_for_dtype(dtype))
        elif self.epi_tile_fn is not None:
            n = cute.sym_int()
        else:
            n = fctx.n
        leading = 1 if (major == "n" or self.epi_tile_fn is not None) else 0
        batch = fctx.l if (fctx.batched and not fctx.varlen_m) else None
        if fctx.swapped:
            # Crossing order is caller-oriented (n_k, m_k); the caller-stride
            # major label flips with the shape order, so ``leading`` is
            # unchanged (both flips cancel). Transposed at trace time.
            assert self.epi_tile_fn is None, "swap_ab: reshaped tiles unsupported"
            return fake_batched(dtype, fctx.n, fctx.m, batch, leading, div_for_dtype(dtype))
        return fake_batched(dtype, fctx.m, n, batch, leading, div_for_dtype(dtype))

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
        layout = cutlass.utils.LayoutEnum.from_tensor(tensor)
        if self.gated:
            # The smem store path degrades to a universal SIMT copy for
            # narrow dtypes (get_smem_store_atom / SM100's get_smem_store_op),
            # so fp8/fp4 gated postact (quantized output) works on SM100 and
            # SM120 (the SM120 halved retile lays the narrow atom over the
            # 16-bit C-atom geometry, same as the fp8/fp4 D store); SM90's
            # STSM path remains 16-bit only.
            assert tensor.element_type.width == 16 or (
                gemm.arch in (100, 120) and tensor.element_type.width in (4, 8)
            ), "gated aux output must be 16-bit (or fp8/fp4 on SM100/SM120)"
            assert gemm.d_layout is None or gemm.d_layout.is_n_major_c()
            assert layout.is_n_major_c()
            if gemm.arch == 90:
                assert gemm.cta_tile_shape_mnk[1] % 32 == 0, (
                    "gated epilogue on SM90 requires tile_N divisible by 32"
                )
        setattr(gemm, self._layout_gemm_attr(), layout)
        setattr(gemm, self._dtype_gemm_attr(), tensor.element_type)
        epi_tile = self.epi_tile_fn(gemm, gemm.epi_tile) if self.epi_tile_fn else None
        tma_atom, tma_tensor, smem_layout, epi_tile_out = setup_epi_tensor(
            gemm, tensor, epi_tile=epi_tile
        )
        return {
            self._tma_atom_key(): tma_atom,
            self.name: tma_tensor,
            self._smem_layout_key(): smem_layout,
            self._epi_tile_key(): epi_tile_out,
            self._dtype_field(): tensor.element_type,
        }

    def smem_bytes(self, arg_tensor, cta_tile_shape_mnk, epi_tile, warp_shape_mnk=None):
        if self.epi_tile_fn is not None:
            epi_tile = self.epi_tile_fn(None, epi_tile)
        # epi_tile may contain Layout entries (from SM100's compute_epilogue_tile_shape
        # fixup path), so extract the int shape first.
        return EpiSmemBytes(
            # multiply before dividing: sub-byte dtypes (fp4) would floor to 0
            d_stage=cute.size(cute.shape(epi_tile)) * arg_tensor.element_type.width // 8
        )

    def smem_struct_field(self, gemm, params):
        smem_layout = getattr(params, self._smem_layout_key())
        return (
            f"s_{self.name}",
            cute.struct.Align[
                cute.struct.MemRange[
                    getattr(params, self._dtype_field()),
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

    def epi_tile_shape_override(self, arch, cta_tile_shape_mnk, atom_layout_mnk):
        """Static epi-tile override consulted from _setup_attributes (before
        params exist). SM120 gated: each N warp needs 32 elems so the halved
        postact keeps 16 per warp; tile_m may shrink to the M warp extent."""
        if not (self.gated and arch == 120):
            return None
        tile_m = math.gcd(atom_layout_mnk[0] * 16, cute.size(cta_tile_shape_mnk, mode=[0]))
        tile_n = math.gcd(atom_layout_mnk[1] * 8 * 4, cute.size(cta_tile_shape_mnk, mode=[1]))
        return (tile_m, tile_n)

    # --- Device-side store path (driven by gemm_base.epilogue) ---

    def _make_copy_atom_r2s(self, gemm, params, tiled_copy_t2r, dtype_override=None):
        """Build the register-to-shared copy atom for this output."""
        dtype = (
            dtype_override if dtype_override is not None else getattr(gemm, self._dtype_gemm_attr())
        )
        layout = getattr(gemm, self._layout_gemm_attr())
        if gemm.arch == 100:
            return blackwell_helpers.get_smem_store_op(
                layout, dtype, gemm.acc_dtype, tiled_copy_t2r
            )
        else:
            return copy_utils.get_smem_store_atom(
                dtype,
                transpose=layout != cutlass.utils.LayoutEnum.ROW_MAJOR,
                major_mode_size=cute.size(getattr(params, self._epi_tile_key()), mode=[1])
                // gemm.atom_layout_mnk[1],
            )

    def _make_tiled_copy_r2s(self, gemm, params, tiled_copy_r2s, tiled_copy_t2r):
        """Build the register-to-shared tiled copy for this output."""
        copy_atom_r2s = self._make_copy_atom_r2s(gemm, params, tiled_copy_t2r)
        if self.gated and gemm.arch == 120:
            # SM120 halved postact: retile through an N-doubled permuted MMA so
            # each warp's STSM lanes cover the halved tile contiguously. The
            # C-side atom is always the 16-bit STSM one — for narrow (fp8/fp4)
            # quantized postact it only provides the source-layout geometry
            # while copy_atom_r2s is the universal narrow atom, exactly like
            # the D path's tiled_copy_C_atom (see epilog_smem_copy_atom).
            copy_atom_postact_c = self._make_copy_atom_r2s(
                gemm, params, cutlass.Float16, dtype_override=cutlass.Float16
            )
            # dummy tiled mma: only its C-side (M, N) fragment geometry is
            # consumed, which is identical for every mma.sync inst K and
            # operand width — so build it 16-bit even for fp8/blockscaled
            # GEMMs (MmaF16BF16Op rejects fp8 dtypes and inst K 32)
            dummy_dtype = gemm.a_dtype if gemm.a_dtype.width == 16 else cutlass.BFloat16
            op = warp.MmaF16BF16Op(dummy_dtype, gemm.acc_dtype, (16, 8, 16))
            tC = cute.make_layout(gemm.atom_layout_mnk)
            atom_m, atom_n, atom_k = gemm.atom_layout_mnk
            permutation_mnk = (
                gemm.mma_inst_mnk[0] * atom_m,
                gemm.mma_inst_mnk[1] * atom_n * 2,
                16 * atom_k,
            )
            tiled_mma_gated_postact = cute.make_tiled_mma(op, tC, permutation_mnk=permutation_mnk)
            tiled_copy_c_atom = cute.make_tiled_copy_C_atom(
                copy_atom_postact_c, tiled_mma_gated_postact
            )
            return cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_c_atom)
        return cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_r2s)

    def store_tile_shape_mn(self, gemm):
        """Return the logical CTA tile shape written by this store."""
        if self.gated:
            return (gemm.cta_tile_shape_mnk[0], gemm.cta_tile_shape_mnk[1] // 2)
        return gemm.cta_tile_shape_mnk[:2]

    def store_setup(
        self,
        gemm,
        params,
        smem_tensor,
        tiled_copy_r2s,
        tiled_copy_t2r,
        tile_coord_mnkl,
        varlen_manager,
        tidx,
    ):
        """Per-CTA-tile setup. Returns the tail of the driver's store context
        ``(tiled_copy_r2s, tRS_sAux, copy_fn, store_pred)`` where store_pred
        is None (always store) or a per-tile Boolean (the mixin prepends the
        op itself and its quantize codec — see gemm_base.epilogue)."""
        tiled_copy_aux_r2s = self._make_tiled_copy_r2s(gemm, params, tiled_copy_r2s, tiled_copy_t2r)
        tRS_sAux = tiled_copy_aux_r2s.get_slice(tidx).partition_D(smem_tensor)
        batch_idx = tile_coord_mnkl[3]
        copy_aux, _, _ = gemm.epilog_gmem_copy_and_partition(
            getattr(params, self._tma_atom_key()),
            varlen_manager.offset_batch_epi(getattr(params, self.name), batch_idx),
            self.store_tile_shape_mn(gemm),
            getattr(params, self._epi_tile_key()),
            smem_tensor,
            tile_coord_mnkl,
        )
        pred = self.store_pred_fn(gemm, tile_coord_mnkl) if self.store_pred_fn else None
        return (tiled_copy_aux_r2s, tRS_sAux, copy_aux, pred)

    @cute.jit
    def store_convert(
        self, gemm, tRS_rAuxOut, sr_seed, tidx, tile_coord_mnkl, num_prev_subtiles, epi_idx
    ):
        """Convert one subtile's values from acc_dtype to this op's storage
        dtype (per-op rounding), plus the gated STSM register permute."""
        dtype = getattr(gemm, self._dtype_gemm_attr())
        rounding = self.rounding if self.rounding is not None else gemm.rounding_mode
        if const_expr(self.gated and gemm.arch in (90, 120) and dtype.width < 16):
            # The store TV follows the 16-bit STSM C-atom contract (see
            # _make_tiled_copy_r2s); narrow (fp8/fp4) quantized postact
            # applies the same register permute at fp32 granularity
            # BEFORE the convert (after quantize — placement only).
            layout_utils.permute_gated_Cregs_f32(tRS_rAuxOut)
        if const_expr(
            rounding == RoundingMode.RS
            and tRS_rAuxOut.element_type == cutlass.Float32
            and dtype in SR_STORE_DTYPES
        ):
            seed = epilogue_aux_out_sr_seed(sr_seed, tile_coord_mnkl, num_prev_subtiles + epi_idx)
            tRS_rAuxOut_out = convert_f32_frag_sr(tRS_rAuxOut, dtype, seed, tidx)
        else:
            tRS_rAuxOut_out = tRS_rAuxOut.to(dtype)
        if const_expr(self.gated and gemm.arch in (90, 120) and dtype.width == 16):
            # The STSM store contract's register permute (16-bit prmt form).
            layout_utils.permute_gated_Cregs_b16(tRS_rAuxOut_out)
        return tRS_rAuxOut_out

    @cute.jit
    def store_r2s(self, gemm, tiled_copy, frag_out, tRS_s_stage, tidx):
        """Copy one converted subtile from registers to this op's smem stage."""
        # Need contiguous for Sm80 and Sm120 where acc layout is ((2, 2), MMA_M, MMA_N)
        cute.copy(tiled_copy, tiled_copy.retile(frag_out).contiguous(), tRS_s_stage)


class DStore(EpiOp):
    """The main D output's device store path, as a store op.

    D's host plumbing stays kernel-owned — the TMA atom, the staged smem
    layout, and the ``sD`` struct field feed tile/stage sizing, split-K's
    workspace re-pointing, and ``add_to_output`` — so unlike TileStore this op
    has no host hooks and does not live in ``_epi_ops``. The driver
    (gemm_base.epilogue) assembles its store context directly from the
    kernel-built pieces (tiled_copy_r2s, tRS_sD, copy_D); this op owns the
    convert (kernel-global rounding, D's stochastic-rounding seed) and the
    register-to-smem copy (including SM90's fp32 pair-XOR STS.32 path), so
    every stored output — D included — flows through the same
    store_convert / store_r2s hooks and the same quantize seam.
    """

    def __init__(self):
        super().__init__("D")

    @cute.jit
    def store_convert(
        self, gemm, tRS_rD, sr_seed, tidx, tile_coord_mnkl, num_prev_subtiles, epi_idx
    ):
        """Convert one subtile of tRS_rD from acc_dtype to d_dtype."""
        dtype = gemm.d_dtype
        if const_expr(
            gemm.rounding_mode == RoundingMode.RS
            and tRS_rD.element_type == cutlass.Float32
            and dtype in SR_STORE_DTYPES
        ):
            seed = epilogue_sr_seed(sr_seed, tile_coord_mnkl, num_prev_subtiles + epi_idx)
            tRS_rD_out = convert_f32_frag_sr(tRS_rD, dtype, seed, tidx)
        elif const_expr(tRS_rD.element_type != dtype):
            tRS_rD_out = tRS_rD.to(dtype)
        else:
            tRS_rD_out = tRS_rD
        return tRS_rD_out

    @cute.jit
    def store_r2s(self, gemm, tiled_copy, frag_out, tRS_s_stage, tidx):
        if const_expr(gemm.epi_r2s_pair_xor()):
            # fp32 n-major D whose smem swizzle 2-way-conflicts vectorized
            # STS.64: pair-exchanged STS.32 (frag_out is the unconverted f32
            # fragment — pair_xor implies d_dtype == acc_dtype).
            copy_utils.cvt_copy_pair_xor_sts32(frag_out, tRS_s_stage, tidx)
        else:
            # frag_out is tRS_rD (already retiled by the kernel) or its
            # converted same-layout copy: no retile needed.
            cute.copy(tiled_copy, frag_out, tRS_s_stage)


class GroupedMainStore(TileStore):
    """Store one value per adjacent-N group from a direct TensorSSA callback.

    The callback owns the logical lane contraction; this op owns the contracted
    epilogue tile, output buffer schema, physical store geometry, and QuACK
    config legality. It intentionally remains an ordinary output EpiOp rather
    than introducing a grouped-main GEMM mode.
    """

    supports_swap_ab = False

    def __init__(self, name, group, paired=False):
        if group not in (2, 4):
            raise ValueError("GroupedMainStore supports group 2 or 4")
        if paired and group != 2:
            raise ValueError("paired GroupedMainStore supports group 2 only")
        epi_tile_fn = _grouped_main_epi_tile_2 if group == 2 else _grouped_main_epi_tile_4
        super().__init__(name, epi_tile_fn=epi_tile_fn)
        self.group = group
        self.paired = paired
        self.paired_output_bytes = (1, 2) if paired else None

    def config_key(self):
        return (self.group, self.paired, *super().config_key())

    def output_n(self, n):
        """Return the contracted logical output N extent."""
        if n % self.group:
            raise ValueError(
                f"grouped main output requires GEMM N divisible by {self.group}, got {n}"
            )
        return n // self.group

    def supports_config(self, config):
        """Return whether a config has validated grouped-main store ownership."""
        supported_arch = (
            config.device_capacity in (10, 11) if self.group == 2 else config.device_capacity == 10
        )
        supported_m_cluster = (config.tile_m in (128, 256) and config.cluster_m == 1) or (
            config.tile_m == 256 and config.cluster_m == 2
        )
        min_tile_n = 64 if self.group == 2 else 128
        return (
            supported_arch
            and not config.swap_ab
            and supported_m_cluster
            and config.cluster_n == 1
            and config.tile_n >= min_tile_n
            and config.tile_n % self.group == 0
        )

    def supports_problem(self, config, m, n):
        """Apply problem-size legality not expressible from config fields alone."""
        return n % self.group == 0 and config.tile_n <= n

    def config_support_error(self, configs):
        if self.group == 4:
            return "group-4 grouped main outputs require an SM100 config"
        return "group-2 grouped main outputs require an SM100 or SM110 config"

    def to_params(self, gemm, args):
        tensor = getattr(args, self.name)
        layout = cutlass.utils.LayoutEnum.from_tensor(tensor)
        if not layout.is_n_major_c():
            raise ValueError("grouped main output must be N-major")
        if self.paired:
            assert gemm.d_layout is None or gemm.d_layout.is_n_major_c()
        setattr(gemm, self._layout_gemm_attr(), layout)
        setattr(gemm, self._dtype_gemm_attr(), tensor.element_type)
        epi_tile = self.epi_tile_fn(gemm, gemm.epi_tile)
        tma_atom, tma_tensor, smem_layout, epi_tile_out = setup_epi_tensor(
            gemm, tensor, epi_tile=epi_tile
        )
        return {
            self._tma_atom_key(): tma_atom,
            self.name: tma_tensor,
            self._smem_layout_key(): smem_layout,
            self._epi_tile_key(): epi_tile_out,
            self._dtype_field(): tensor.element_type,
        }

    def store_tile_shape_mn(self, gemm):
        return (gemm.cta_tile_shape_mnk[0], gemm.cta_tile_shape_mnk[1] // self.group)


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

    fn_port = "tile"
    supports_swap_ab = True

    def __init__(self, name, epi_tile_fn=None, dtype=None):
        super().__init__(name)
        self.epi_tile_fn = epi_tile_fn
        self.dtype = dtype

    def config_key(self):
        return (_callable_config_key(self.epi_tile_fn), self.dtype)

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

    # Same host schema as TileStore: an (m, n[, l]) tile keyed by dtype + major.
    host_arg_key = TileStore.host_arg_key
    host_fake_arg = TileStore.host_fake_arg

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
            # multiply before dividing: sub-byte dtypes (fp4) would floor to 0
            c_stage=cute.size(cute.shape(epi_tile)) * arg_tensor.element_type.width // 8
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


@cute.jit
def _vec_reduce_combine(a, b, combine):
    if const_expr(combine == "add"):
        return a + b
    return cute.arch.fmax(a, b, abs=combine == "max_abs")


@cute.jit
def colvec_reduce_accumulate(
    gemm, tDrReduce, tRS_rInput, transform_fn=None, rScale=None, combine="add"
):
    """Accumulate transform_fn(input) or input * rScale into a ColVecReduce buffer.

    If transform_fn is provided, accumulates transform_fn(input[i]).
    If rScale is provided, accumulates input[i] * rScale[i] (uses packed mul/fma for SM100).
    If neither, accumulates input directly (identity).
    ``combine="max"`` folds with fmax and ``"max_abs"`` with max.abs instead
    of add (plain input only): the aliased-lane assignment is order-free, so
    one scalar loop serves all archs.
    """
    if const_expr(combine != "add"):
        assert transform_fn is None and rScale is None, "max combines take the input directly"
        if const_expr(tDrReduce is not None):
            for i in cutlass.range(cute.size(tDrReduce), unroll_full=True):
                tDrReduce[i] = _vec_reduce_combine(tDrReduce[i], tRS_rInput[i], combine)
        return
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
def rowvec_reduce_accumulate(
    gemm, tDrReduce, tRS_rInput, transform_fn=None, rScale=None, combine="add"
):
    """Accumulate transform_fn(input) or input * rScale into a RowVecReduce buffer.

    Reduces along M dimension, keeping N. The zero-stride layout on M ensures
    elements at different M positions but same N column accumulate correctly.
    ``combine="max"`` folds with fmax and ``"max_abs"`` with max.abs instead
    of add (plain input only).
    """
    if const_expr(combine != "add"):
        assert transform_fn is None and rScale is None, "max combines take the input directly"
        if const_expr(tDrReduce is not None):
            for i in cutlass.range(cute.size(tDrReduce), unroll_full=True):
                tDrReduce[i] = _vec_reduce_combine(tDrReduce[i], tRS_rInput[i], combine)
        return
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
    """Base class for row/column vector reductions.

    ``combine`` selects the reduction: "add" (default), "max", or "max_abs".
    max_abs folds raw inputs with PTX max.abs and clears its XOR-derived sign
    only at the final store. Out-of-bounds accumulator elements are zero
    (predicated loads), the identity for add and max_abs. They are NOT the
    identity for max, so max reduces mask OOB elements to -inf via a
    per-element coordinate select (``check_oob``, on by default; the select
    is an ALU op, see OnlineLSEReduce for the measured cost). Pass
    ``check_oob=False`` to compile it out when the reduce dim is known
    tile-divisible; the frontend host rejects ragged shapes then.
    """

    dim = 0  # 0 for colvec output along M, 1 for rowvec output along N
    epi_m_major_preference = 0
    fn_port = "sink"
    # f32 values exchanged through smem per (row, warp) in the inter-warp
    # merge: 1 for plain reduces, 2 for coupled accumulators (OnlineLSE).
    reduce_planes = 1

    def __init__(self, name, combine="add", scaled=False, check_oob=None):
        super().__init__(name)
        if combine not in ("add", "max", "max_abs"):
            raise ValueError(f"unsupported combine {combine!r}")
        if scaled and combine != "add":
            raise ValueError("scaled reduces only support combine='add'")
        if check_oob is not None and combine != "max":
            raise ValueError("check_oob applies only to combine='max'")
        # add/max_abs OOB zeros ARE the fold identity: normalized to True so
        # the frontend's ragged-shape reject never trips (no mask is emitted
        # — codegen gates on combine == "max").
        self.check_oob = check_oob if check_oob is not None else True
        self.combine = combine
        # scaled=True: the fn returns the two FACTORS ``(val, scale)`` under
        # this op's name and the fold is one fused ``fma(val, scale, acc)`` —
        # the product is never rounded on its own. This keeps a reduce of a
        # product (sq-sums, postact*dout dots) bitwise-equal to folding the
        # product directly into the accumulator, and one FFMA instead of
        # FMUL+FADD per pair.
        self.scaled = scaled

    def config_key(self):
        return (self.combine, self.scaled, self.check_oob)

    def sink_alloc_shape(self, lead, n, tile_m, tile_n, num_seqs=None):
        """Buffer shape for a (lead=(batch?, m) or (total_m,), n) problem at
        (tile_m, tile_n): per-CTA-tile partials along the reduce dim. THE
        single statement of the sink tiling rule — validation (EpiMod.gemm),
        eager allocation (_alloc_sinks), the torch-op fakes, and the autotune
        worst-case/slicing all call this. ``tile_m`` is the PER-CTA M tile
        (``cta_tile_shape_m``: half the config tile under the SM100 2-CTA
        MMA) — the kernel indexes partial slots by the per-CTA tile coord.

        ``num_seqs`` (varlen_m only, dim==1): partial rows are per-sequence
        tile-prefix slots — sum(ceil(len_b / tile_m)) is bounded by
        total_m // tile_m + num_seqs (the host can't see device seqlens);
        rows past the live prefix are never read by the cu_tiles-segment
        finalize (host_finalize_varlen)."""
        if self.dim == 0:
            return (*lead, -(-n // tile_n))
        if num_seqs is not None:
            return (lead[-1] // tile_m + num_seqs, n)
        return (*lead[:-1], -(-lead[-1] // tile_m), n)

    def host_finalize(self, partials):
        """Fold the per-tile partial buffer into the user-visible reduce value
        (host side, after the kernel): colvec partials are (..., m, n_tiles),
        rowvec partials (..., m_tiles, n). Subclasses whose partials need a
        non-trivial fold (coupled accumulators) set ``host_finalize = None``
        and the interface layer returns the raw buffer."""
        axis = -1 if self.dim == 0 else -2
        return partials.sum(dim=axis) if self.combine == "add" else partials.amax(dim=axis)

    def _mask_oob_active(self):
        return self.combine == "max" and self.check_oob

    @cute.jit
    def _mask_oob(self, frag, coords, limit, tile_shape_mn):
        """Mask elements past the ragged reduce-dim boundary to the max fold
        identity (-inf), in place — the frag is this sink's scratch (one
        flush per fragment). Rebased compare, same idiom as
        OnlineLSEReduce._fold: the static per-element offset (an ISETP
        immediate) compares against limit - base, deleting the per-element
        coordinate materialization."""
        rd = self._reduce_dim()
        proj_stride = (1, 0) if rd == 0 else (0, 1)
        lay_r = cute.composition(cute.make_layout(tile_shape_mn, stride=proj_stride), coords.layout)
        limit_rel = limit - coords[0][rd]
        for i in cutlass.range(cute.size(frag), unroll_full=True):
            off_r = cute.crd2idx(i, lay_r)
            frag[i] = frag[i] if off_r < limit_rel else -math.inf

    @cute.jit
    def fn_sink_flush(self, gemm, state, frag, scale=None):
        if const_expr(self._mask_oob_active()):
            acc, coords, limit = state
            self._mask_oob(frag, coords, limit, gemm.cta_tile_shape_mnk[:2])
            state = acc
        if const_expr(self.dim == 0):
            colvec_reduce_accumulate(gemm, state, frag, rScale=scale, combine=self.combine)
        else:
            rowvec_reduce_accumulate(gemm, state, frag, rScale=scale, combine=self.combine)

    def host_fake_arg(self, key, fctx):
        dtype, ndim = key
        # Reduce outputs are partial per CTA tile along the reduced dim:
        # ColVecReduce (l, m, n_tiles), RowVecReduce (l, m_tiles, n); rank 2
        # drops the batch mode (varlen_m / dense-2D calls).
        tiles = cute.sym_int()
        inner = (fctx.m, tiles) if self.dim == 0 else (tiles, fctx.n)
        shape = (fctx.l, *inner) if ndim == 3 else inner
        return make_fake_tensor(dtype, shape, leading_dim=ndim - 1, divisibility=1)

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
            unstaged=self._tile_size(cta_tile_shape_mnk)
            * smem_warps
            * self.reduce_planes
            * (Float32.width // 8)
        )

    def smem_struct_field(self, gemm, params):
        smem_warps = self._smem_warps(gemm.epi_smem_warp_shape_mnk())
        if smem_warps == 0:
            return None
        size = self._tile_size(gemm.cta_tile_shape_mnk) * smem_warps * self.reduce_planes
        return (f"s_{self.name}", cute.struct.Align[cute.struct.MemRange[Float32, size], 16])

    def get_smem_tensor(self, gemm, params, storage_epi):
        smem_warps = self._smem_warps(gemm.epi_smem_warp_shape_mnk())
        if smem_warps == 0:
            return None
        return getattr(storage_epi, f"s_{self.name}").get_tensor(
            cute.make_layout(
                (self._tile_size(gemm.cta_tile_shape_mnk), smem_warps, self.reduce_planes)
            )
        )

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        vec_mma_layout = cute.make_layout((ctx.tile_M, ctx.tile_N), stride=self._broadcast_stride())
        tDrReduce_layout = ctx.partition_for_epilogue_fn(
            cute.make_rmem_tensor(vec_mma_layout, Float32)
        ).layout
        tDrReduce = cute.make_rmem_tensor(tDrReduce_layout, Float32)
        state = (tDrReduce, smem_tensor)
        if const_expr(self._mask_oob_active()):
            # OOB accumulator zeros are not a max identity: the fold masks on
            # the per-element reduce-dim coordinate against the ragged tile
            # boundary. Same partitioning as the accumulators, so linear
            # indices line up in fn_sink_flush.
            tDcD = ctx.partition_for_epilogue_fn(
                cute.make_identity_tensor((ctx.tile_M, ctx.tile_N))
            )
            if const_expr(self._reduce_dim() == 1):
                limit = ctx.varlen_manager.len_n() - ctx.tile_coord_mnkl[1] * ctx.tile_N
            else:
                limit = (
                    ctx.varlen_manager.len_m(ctx.tile_coord_mnkl[3])
                    - ctx.tile_coord_mnkl[0] * ctx.tile_M
                )
            state = (tDrReduce, smem_tensor, tDcD, limit)
        return state

    @cute.jit
    def begin_loop(self, gemm, state, epi_coord):
        tDrReduce = state[0]
        result = tDrReduce[None, None, None, epi_coord[0], epi_coord[1]]
        if const_expr(epi_coord[self._reduce_dim()] == 0):
            cute.filter_zeros(result).fill(-math.inf if const_expr(self.combine == "max") else 0.0)
        if const_expr(self._mask_oob_active()):
            c_cur = state[2][None, None, None, epi_coord[0], epi_coord[1]]
            return (result, c_cur, state[3])
        return result


class ColVecReduce(VecReduce):
    """Column vector reduction: accumulates across N subtiles in registers,
    then reduces across N lanes/warps and writes to gmem per completed M stripe.

    The accumulation itself happens in epi_visit_subtile (user code).
    This op handles the register allocation (begin), per-subtile slicing (begin_loop),
    and reduction + gmem write (end_loop).

    end_loop is a generic TUPLE-VALUED exchange engine: subclasses with
    coupled accumulators (OnlineLSEReduce's (max, sum)) override
    ``reduce_planes`` and the ``_end_loop_values`` / ``_end_loop_smem`` /
    ``_merge`` / ``_finalize`` hooks; the butterfly, smem exchange, and gmem
    write protocol are shared.
    """

    dim = 0
    epi_m_major_preference = -1

    @cute.jit
    def _end_loop_values(self, state, epi_coord):
        """Tuple of per-stripe register accumulators (zero-strided slices)."""
        return (state[0][None, None, None, epi_coord[0], epi_coord[1]],)

    @cute.jit
    def _end_loop_smem(self, state):
        return state[1]

    @cute.jit
    def _merge(self, vals, others):
        """Combine two value tuples (same-row partials from different lanes/warps)."""
        return (_vec_reduce_combine(vals[0], others[0], self.combine),)

    @cute.jit
    def _finalize(self, vals):
        """Value tuple -> the scalar written to gmem."""
        # Two-input max.xorsign.abs preserves the maximum magnitude but XORs
        # operand signs, so clear that arbitrary sign once after the full fold.
        return cute.math.abs(vals[0]) if const_expr(self.combine == "max_abs") else vals[0]

    @cute.jit
    def end_loop_stage(
        self,
        gemm,
        param,
        state,
        epi_coord,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tidx,
    ):
        """Stage the current M stripe (intra-warp reduce + smem write) when
        its last N subtile has accumulated; merge/gmem-write run in
        end_loop_finish after the driver's shared barrier."""
        staged = None
        epi_tile_shape = cute.zipped_divide(
            cute.make_layout(gemm.cta_tile_shape_mnk[:2]), epi_tile
        ).shape[1]
        if const_expr(epi_coord[1] == epi_tile_shape[1] - 1):
            vals_cur = self._end_loop_values(state, epi_coord)
            sExch = self._end_loop_smem(state)
            tiled_copy = tiled_copy_t2r if tiled_copy_t2r is not None else tiled_copy_r2s
            reference_src = tiled_copy_t2r is None
            lanes_in_N, warps_in_N, warp_n_idx, is_lane_n_leader = _lane_warp_info_n(
                tiled_copy, reference_src, tidx
            )
            num_vals = const_expr(len(vals_cur))

            partition_for_epilogue_fn = partial(
                partition_for_epilogue,
                epi_tile=epi_tile,
                tiled_copy=tiled_copy,
                tidx=tidx,
                reference_src=reference_src,
            )
            tile_M, tile_N = gemm.cta_tile_shape_mnk[:2]
            tDcD = partition_for_epilogue_fn(cute.make_identity_tensor((tile_M, tile_N)))
            tDcD_cur = tDcD[None, None, None, epi_coord[0], epi_coord[1]]
            ref_layout = vals_cur[0].layout
            vals_m = tuple(
                layout_utils.convert_layout_zero_stride(v, ref_layout)[None, 0] for v in vals_cur
            )
            tDcD_m = layout_utils.convert_layout_zero_stride(tDcD_cur, ref_layout)[None, 0]

            # Intra-warp reduction across N lanes, tuple-valued. Swap shuffle
            # when the geometry allows: ~E shuffles+merges instead of the
            # butterfly's E*log2(lanes), and results end DISTRIBUTED (N-lane
            # g owns slice g of this thread-group's rows) so the smem/gmem
            # stores below spread across lanes instead of leader-serializing.
            E = const_expr(cute.size(tDcD_m, mode=[0]))
            num_slices = const_expr(min(lanes_in_N, E))
            use_swap_shuffle = const_expr(
                lanes_in_N > 1
                and E % num_slices == 0
                and num_slices == 1 << int(math.log2(num_slices))
            )
            slice_elems = const_expr(E // num_slices if use_swap_shuffle else E)
            lane_g = cute.arch.lane_idx() % lanes_in_N
            if const_expr(use_swap_shuffle):
                swap_shuffle_reduce(
                    vals_m,
                    self._merge,
                    num_lanes=lanes_in_N,
                    lane_stride=1,
                    slice_elems=slice_elems,
                )
            elif const_expr(lanes_in_N > 1):
                flts = tuple(cute.filter_zeros(v) for v in vals_cur)
                for i in cutlass.range(cute.size(flts[0]), unroll_full=True):
                    off = lanes_in_N // 2
                    while off > 0:
                        others = tuple(cute.arch.shuffle_sync_bfly(f[i], offset=off) for f in flts)
                        merged = self._merge(tuple(f[i] for f in flts), others)
                        # range_constexpr: k indexes Python TUPLES, which
                        # need trace-time ints (staged range vars can't).
                        for k in cutlass.range_constexpr(num_vals):
                            flts[k][i] = merged[k]
                        off = off // 2

            # Stage the partials for the inter-warp merge (rows are absolute
            # CTA-tile rows, so stripes write disjoint slots). The driver's
            # SHARED barrier — one per epi_coord across ALL reduce ops —
            # orders these writes before end_loop_finish.
            if const_expr(warps_in_N > 1):
                if const_expr(use_swap_shuffle):
                    if warp_n_idx > 0 and lane_g < num_slices:
                        for j in cutlass.range_constexpr(slice_elems):
                            row_idx = tDcD_m[lane_g * slice_elems + j][0]
                            for k in cutlass.range_constexpr(num_vals):
                                sExch[row_idx, warp_n_idx - 1, k] = vals_m[k][j]
                else:
                    if warp_n_idx > 0 and is_lane_n_leader:
                        for m in cutlass.range(cute.size(tDcD_m, mode=[0])):
                            row_idx = tDcD_m[m][0]
                            for k in cutlass.range_constexpr(num_vals):
                                sExch[row_idx, warp_n_idx - 1, k] = vals_m[k][m]
            staged = (
                warps_in_N > 1,
                (
                    vals_m,
                    tDcD_m,
                    sExch,
                    warps_in_N,
                    warp_n_idx,
                    is_lane_n_leader,
                    use_swap_shuffle,
                    num_slices,
                    slice_elems,
                    lane_g,
                ),
            )
        return staged

    @cute.jit
    def end_loop_finish(self, gemm, param, staged, tile_coord_mnkl, varlen_manager):
        """Inter-warp merge from smem + gmem write for a stripe staged by
        end_loop_stage (runs after the driver's shared barrier). Under swap
        shuffle each lane merges/writes only its OWNED slice."""
        vals_m, tDcD_m, sExch = staged[0], staged[1], staged[2]
        warps_in_N, warp_n_idx, is_lane_n_leader = staged[3], staged[4], staged[5]
        use_swap_shuffle, num_slices, slice_elems, lane_g = (
            staged[6],
            staged[7],
            staged[8],
            staged[9],
        )
        num_vals = const_expr(len(vals_m))
        if const_expr(warps_in_N > 1):
            if const_expr(use_swap_shuffle):
                if warp_n_idx == 0 and lane_g < num_slices:
                    for j in cutlass.range_constexpr(slice_elems):
                        row_idx = tDcD_m[lane_g * slice_elems + j][0]
                        for warp_n in cutlass.range_constexpr(1, warps_in_N):
                            others = tuple(sExch[row_idx, warp_n - 1, k] for k in range(num_vals))
                            merged = self._merge(tuple(v[j] for v in vals_m), others)
                            for k in cutlass.range_constexpr(num_vals):
                                vals_m[k][j] = merged[k]
            else:
                if warp_n_idx == 0 and is_lane_n_leader:
                    for m in cutlass.range(cute.size(tDcD_m, mode=[0])):
                        row_idx = tDcD_m[m][0]
                        for warp_n in cutlass.range_constexpr(1, warps_in_N):
                            others = tuple(sExch[row_idx, warp_n - 1, k] for k in range(num_vals))
                            merged = self._merge(tuple(v[m] for v in vals_m), others)
                            for k in cutlass.range_constexpr(num_vals):
                                vals_m[k][m] = merged[k]

        # Write to gmem
        tile_M = gemm.cta_tile_shape_mnk[0]
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
        if const_expr(use_swap_shuffle):
            in_warp0 = True if const_expr(warps_in_N == 1) else warp_n_idx == 0
            if tile_coord_mnkl[1] < limit_n_tiles and in_warp0 and lane_g < num_slices:
                for j in cutlass.range_constexpr(slice_elems):
                    row_idx = tDcD_m[lane_g * slice_elems + j][0]
                    if row_idx < limit_m:
                        gColVec[row_idx] = self._finalize(tuple(v[j] for v in vals_m))
        else:
            should_write_gmem = (
                is_lane_n_leader
                if const_expr(warps_in_N == 1)
                else warp_n_idx == 0 and is_lane_n_leader
            )
            if tile_coord_mnkl[1] < limit_n_tiles and should_write_gmem:
                for m in cutlass.range(cute.size(tDcD_m, mode=[0])):
                    row_idx = tDcD_m[m][0]
                    if row_idx < limit_m:
                        gColVec[row_idx] = self._finalize(tuple(v[m] for v in vals_m))


class RowVecReduce(VecReduce):
    """Row vector reduction: accumulates across M subtiles in registers,
    then reduces across M lanes/warps and writes to gmem per completed N stripe.

    Output shape is (L, ceildiv(M, tile_M), N): one partial sum per CTA-M tile per
    N column. This mirrors ColVecReduce with M/N swapped. Under varlen_m the
    partial rows are per-sequence tile-prefix slots (see sink_alloc_shape's
    num_seqs form and host_finalize_varlen).
    """

    dim = 1
    epi_m_major_preference = 4

    def host_finalize_varlen(self, partials, cu_seqlens_m, tile_m_cta):
        """Per-sequence fold of a varlen partial buffer into (num_seqs, n).

        Rows are grouped by the cu_tiles_m prefix (sequence b owns rows
        [cu_tiles[b], cu_tiles[b+1])); rows past the prefix total (the buffer
        is an upper bound) are never read. Device-only ops, graph-safe.
        Non-add combines are rejected under varlen at plan time (zero-filled
        OOB rows are not a max identity, and a segment-max is not graph-safe),
        so only the add fold appears here.
        """
        import torch

        assert self.combine == "add"
        seqlens = cu_seqlens_m[1:] - cu_seqlens_m[:-1]
        tiles = torch.div(seqlens + (tile_m_cta - 1), tile_m_cta, rounding_mode="floor").long()
        cu_tiles = torch.cat([tiles.new_zeros(1), tiles.cumsum(0)])
        csum = torch.cat([partials.new_zeros((1, partials.shape[-1])), partials.cumsum(0)])
        return csum[cu_tiles[1:]] - csum[cu_tiles[:-1]]

    @cute.jit
    def end_loop_stage(
        self,
        gemm,
        param,
        state,
        epi_coord,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tidx,
    ):
        """Stage the current N stripe (intra-warp reduce + smem write) when
        its last M subtile has accumulated; merge/gmem-write run in
        end_loop_finish after the driver's shared barrier."""
        staged = None
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

            tDrReduce_n = layout_utils.convert_layout_zero_stride(
                tDrReduce_cur, tDrReduce_cur.layout
            )[None, 0]

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

            # Intra-warp reduction across M lanes (contiguous on SM100
            # N-major output, strided by N lanes otherwise). Swap shuffle
            # when the M lanes form a flat pow2-strided run and the slice
            # count is pow2: ~E shuffles+merges instead of E*log2(lanes),
            # results DISTRIBUTED (M-lane g owns col slice g) so the smem/
            # gmem stores below spread across lanes.
            E = const_expr(cute.size(tDcD_n, mode=[0]))
            s_m_flat = _mode_flat_stride(lane_layout_MN, 0)
            s_m = const_expr(s_m_flat if s_m_flat is not None else 1)
            num_slices = const_expr(min(lanes_in_M, E))
            use_swap_shuffle = const_expr(
                lanes_in_M > 1
                and s_m_flat is not None
                and s_m == 1 << int(math.log2(s_m))
                and E % num_slices == 0
                and num_slices == 1 << int(math.log2(num_slices))
            )
            slice_elems = const_expr(E // num_slices if use_swap_shuffle else E)
            lane_g = (cute.arch.lane_idx() // s_m) % lanes_in_M
            if const_expr(use_swap_shuffle):
                swap_shuffle_reduce(
                    (tDrReduce_n,),
                    lambda vals, others: (_vec_reduce_combine(vals[0], others[0], self.combine),),
                    num_lanes=lanes_in_M,
                    lane_stride=s_m,
                    slice_elems=slice_elems,
                )
            elif const_expr(lanes_in_M > 1):
                for n in cutlass.range(cute.size(tDrReduce_n), unroll_full=True):
                    reduction_rows = lanes_in_M // 2
                    while reduction_rows > 0:
                        tDrReduce_n[n] = _vec_reduce_combine(
                            tDrReduce_n[n],
                            cute.arch.shuffle_sync_bfly(
                                tDrReduce_n[n],
                                offset=cute.crd2idx((reduction_rows, 0), lane_layout_MN),
                            ),
                            self.combine,
                        )
                        reduction_rows = reduction_rows // 2

            # Stage the partials for the inter-warp merge (cols are absolute
            # CTA-tile cols, so stripes write disjoint slots). The driver's
            # SHARED barrier — one per epi_coord across ALL reduce ops —
            # orders these writes before end_loop_finish.
            warp_idx = cute.arch.make_warp_uniform(tidx // cute.arch.WARP_SIZE)
            warp_m_idx = warp_layout_MN.get_hier_coord(warp_idx)[0]
            if const_expr(warps_in_M > 1):
                if const_expr(use_swap_shuffle):
                    if warp_m_idx > 0 and lane_g < num_slices:
                        for j in cutlass.range_constexpr(slice_elems):
                            col_idx = tDcD_n[lane_g * slice_elems + j][1]
                            sDrReduce[col_idx, warp_m_idx - 1, 0] = tDrReduce_n[j]
                else:
                    if warp_m_idx > 0 and is_lane_m_leader:
                        for n in cutlass.range(cute.size(tDcD_n, mode=[0])):
                            col_idx = tDcD_n[n][1]
                            sDrReduce[col_idx, warp_m_idx - 1, 0] = tDrReduce_n[n]
            staged = (
                warps_in_M > 1,
                (
                    tDrReduce_n,
                    tDcD_n,
                    sDrReduce,
                    warps_in_M,
                    warp_m_idx,
                    is_lane_m_leader,
                    use_swap_shuffle,
                    num_slices,
                    slice_elems,
                    lane_g,
                ),
            )
        return staged

    @cute.jit
    def end_loop_finish(self, gemm, param, staged, tile_coord_mnkl, varlen_manager):
        """Inter-warp merge from smem + gmem write for a stripe staged by
        end_loop_stage (runs after the driver's shared barrier)."""
        tDrReduce_n, tDcD_n, sDrReduce = staged[0], staged[1], staged[2]
        warps_in_M, warp_m_idx, is_lane_m_leader = staged[3], staged[4], staged[5]
        use_swap_shuffle, num_slices, slice_elems, lane_g = (
            staged[6],
            staged[7],
            staged[8],
            staged[9],
        )
        if const_expr(warps_in_M > 1):
            if const_expr(use_swap_shuffle):
                if warp_m_idx == 0 and lane_g < num_slices:
                    for j in cutlass.range_constexpr(slice_elems):
                        col_idx = tDcD_n[lane_g * slice_elems + j][1]
                        for warp_m in cutlass.range_constexpr(1, warps_in_M):
                            tDrReduce_n[j] = _vec_reduce_combine(
                                tDrReduce_n[j],
                                sDrReduce[col_idx, warp_m - 1, 0],
                                self.combine,
                            )
            else:
                if warp_m_idx == 0 and is_lane_m_leader:
                    for n in cutlass.range(cute.size(tDcD_n, mode=[0])):
                        col_idx = tDcD_n[n][1]
                        for warp_m in cutlass.range_constexpr(1, warps_in_M):
                            tDrReduce_n[n] = _vec_reduce_combine(
                                tDrReduce_n[n],
                                sDrReduce[col_idx, warp_m - 1, 0],
                                self.combine,
                            )

        # Write to gmem
        tile_N = gemm.cta_tile_shape_mnk[1]
        batch_idx = tile_coord_mnkl[3]
        if const_expr(not varlen_manager.varlen_m):
            limit_m_tiles = param.shape[1]
            mRowVec = param[batch_idx, tile_coord_mnkl[0], None]
        else:
            # The scheduler's m-tile index is sequence-local: offset by the
            # per-sequence tile prefix (cu_tiles_m) so sequences' partial rows
            # stay disjoint, and bound by this sequence's own tile count. The
            # host sizes the buffer as total_m // tile_M_cta + num_seqs (upper
            # bound on the prefix total) and finalizes per cu_tiles_m segment.
            assert varlen_manager.params.cu_tiles_m is not None, (
                "varlen_m RowVecReduce requires the cu_tiles_m prefix (host passes it "
                "whenever an M-fold sink is active)"
            )
            limit_m_tiles = varlen_manager.len_m_tiles(batch_idx)
            mRowVec = param[varlen_manager.tile_m_offset(batch_idx) + tile_coord_mnkl[0], None]
        gRowVec = cute.local_tile(mRowVec, (tile_N,), (tile_coord_mnkl[1],))
        limit_n = min(
            cute.size(mRowVec, mode=[0]) - tile_coord_mnkl[1] * tile_N,
            tile_N,
        )
        # Two-input max.xorsign.abs leaves an XOR-derived sign; clear it once
        # after the full reduction.
        if const_expr(use_swap_shuffle):
            in_warp0 = True if const_expr(warps_in_M == 1) else warp_m_idx == 0
            if tile_coord_mnkl[0] < limit_m_tiles and in_warp0 and lane_g < num_slices:
                for j in cutlass.range_constexpr(slice_elems):
                    col_idx = tDcD_n[lane_g * slice_elems + j][1]
                    if col_idx < limit_n:
                        gRowVec[col_idx] = (
                            cute.math.abs(tDrReduce_n[j])
                            if const_expr(self.combine == "max_abs")
                            else tDrReduce_n[j]
                        )
        else:
            should_write_gmem = (
                is_lane_m_leader
                if const_expr(warps_in_M == 1)
                else warp_m_idx == 0 and is_lane_m_leader
            )
            if tile_coord_mnkl[0] < limit_m_tiles and should_write_gmem:
                for n in cutlass.range(cute.size(tDcD_n, mode=[0])):
                    col_idx = tDcD_n[n][1]
                    if col_idx < limit_n:
                        gRowVec[col_idx] = (
                            cute.math.abs(tDrReduce_n[n])
                            if const_expr(self.combine == "max_abs")
                            else tDrReduce_n[n]
                        )


class GroupedColStatsBase(EpiOp):
    """Deterministic per-(tile row, N-group) prepass statistics.

    The prepass folds values into register accumulators derived from the actual
    M/N register layouts, including interleaved layouts. At prepass end each
    local group is reduced across its contiguous N-lane subgroup and written
    once to the corresponding (row, group, warp_n) shared-memory plane. The
    prepass barrier publishes those raw planes; consumers combine warp_n planes
    in fixed order, so the statistic is bitwise reproducible run to run.

    The default host schema accepts either a group width integer or a 1-D tensor
    whose length is the group width. Stats-only subclasses can set ``combine``
    and override ``stat_value``. Tensorless or resource-carrying subclasses can
    override the host methods and ``fn_prepare`` while retaining the stats state
    as element 0 of their begin/begin_loop state.
    """

    fn_port = "value"
    combine = "add"

    def stats_identity(self):
        """Identity value for register and shared-memory statistic slots."""
        return {"add": 0.0, "max": -math.inf}[self.combine]

    def stats_combine_fn(self):
        """Binary operator used for every local, lane, and warp-plane fold."""
        return {"add": operator.add, "max": cute.arch.fmax}[self.combine]

    def _combine_op(self):
        return self.stats_combine_fn()

    def _fold_identity(self):
        return self.stats_identity()

    def host_arg_key(self, value):
        if isinstance(value, int):
            # Plain group width: a true constexpr — baked at trace, no kernel
            # argument at all (distinct key from the tensor form, which
            # carries a runtime pointer).
            return ("width", value)
        return (torch2cute_dtype_map[value.dtype], value.shape[0])

    def host_fake_arg(self, key, fctx):
        if key[0] == "width":
            # Baked at trace through the Constexpr[int]-annotated Args field.
            return key[1]
        dtype, group_cols = key
        return make_fake_tensor(
            dtype, (group_cols,), leading_dim=0, divisibility=128 // dtype.width
        )

    def host_call_arg(self, value, key):
        # Constexpr-annotated fields carry no runtime argument: the traced
        # value is baked; pass None at call time (converter emits ConstNone).
        return None if isinstance(value, int) else value

    def host_validate(self, value, *, n, tile_N, **_):
        """Validate that complete, globally aligned groups cover N and tile_N."""
        if isinstance(value, int):
            group_cols = value
        elif value.ndim != 1:
            raise ValueError(
                f"'{self.name}': grouped-stats descriptor must be an int width or a "
                f"1-D tensor, got shape {tuple(value.shape)}"
            )
        else:
            group_cols = value.shape[0]
        if group_cols <= 0:
            raise ValueError(f"'{self.name}': grouped-stats width must be positive")
        if n % group_cols or tile_N % group_cols:
            raise ValueError(
                f"'{self.name}': stats group width {group_cols} must divide "
                f"N={n} and tile_N={tile_N}"
            )

    def param_fields(self):
        return [(self.name, object, None)]

    def arg_spec_type(self, const=False):
        return cutlass.Constexpr[int] if const else Optional[cute.Tensor]

    def host_arg_form(self, value):
        return "Const" if isinstance(value, int) else ""

    def _group_cols(self, arg):
        return arg if isinstance(arg, int) else arg.shape[0]

    def _stats_shape_attr(self):
        return f"_{self.name}_stats_smem_shape"

    def to_params(self, gemm, args):
        tensor = getattr(args, self.name)
        # One statistics slot per (tile row, group, warp_n). Sizing by tile_M
        # matters on SM90, whose epi tiles can be only 64 rows.
        rows = gemm.cta_tile_shape_mnk[0]
        groups = gemm.cta_tile_shape_mnk[1] // self._group_cols(tensor)
        setattr(gemm, self._stats_shape_attr(), (rows, groups, gemm.epi_smem_warp_shape_mnk()[1]))
        return {self.name: tensor}

    def smem_bytes(self, arg_tensor, cta_tile_shape_mnk, epi_tile, warp_shape_mnk=None):
        rows = cta_tile_shape_mnk[0]
        groups = cta_tile_shape_mnk[1] // self._group_cols(arg_tensor)
        warps_n = warp_shape_mnk[1] if warp_shape_mnk is not None else 1
        return EpiSmemBytes(unstaged=rows * groups * warps_n * (Float32.width // 8))

    def smem_struct_field(self, gemm, params):
        size = math.prod(getattr(gemm, self._stats_shape_attr()))
        return (f"s_{self.name}", cute.struct.Align[cute.struct.MemRange[Float32, size], 16])

    def get_smem_tensor(self, gemm, params, storage_epi):
        return getattr(storage_epi, f"s_{self.name}").get_tensor(
            cute.make_layout(getattr(gemm, self._stats_shape_attr()))
        )

    @cute.jit
    def stats_begin(self, gemm, smem_tensor, ctx, group_cols):
        """Build coordinate/layout geometry and initialize statistic storage."""
        assert gemm.arch in (90, 100, 120), (
            f"{type(self).__name__} needs the acc prepass (SM90/SM100/SM120)"
        )
        tDcC = ctx.partition_for_epilogue_fn(cute.make_identity_tensor((ctx.tile_M, ctx.tile_N)))
        tDrRefs = tuple(
            ctx.partition_for_epilogue_fn(
                cute.make_rmem_tensor(
                    cute.make_layout((ctx.tile_M, ctx.tile_N), stride=stride), Float32
                )
            )
            for stride in ((1, 0), (0, 1))
        )
        if const_expr(ctx.tiled_copy_t2r is not None):
            tDcC = ctx.tiled_copy_r2s.retile(tDcC)
            tDrRefs = tuple(ctx.tiled_copy_r2s.retile(ref) for ref in tDrRefs)
        tDcC = cute.group_modes(tDcC, 3, cute.rank(tDcC))
        ref_layouts = tuple(
            cute.group_modes(ref, 3, cute.rank(ref))[None, None, None, 0].layout for ref in tDrRefs
        )
        ref_layout = ref_layouts[0]
        tiled_copy = ctx.tiled_copy_t2r if ctx.tiled_copy_t2r is not None else ctx.tiled_copy_r2s
        reference_src = ctx.tiled_copy_t2r is None
        lanes_in_N, warps_in_N, warp_n_idx, is_lane_n_leader = _lane_warp_info_n(
            tiled_copy, reference_src, ctx.tidx
        )

        # Warp-N coordinates contribute partials to the same row/group, while
        # every other epilogue warp must partition M and own disjoint rows.
        _, warp_layout_MN = _get_lane_warp_layouts(tiled_copy, reference_src)
        warps_in_M = const_expr(cute.size(warp_layout_MN, mode=[0]))
        num_epi_warps = const_expr(ctx.num_epi_threads // cute.arch.WARP_SIZE)
        assert warps_in_M * warps_in_N == num_epi_warps, (
            "grouped stats require every non-N epilogue warp to partition M"
        )
        assert warps_in_N == const_expr(cute.size(smem_tensor, mode=[2])), (
            "grouped-stats smem plane count must match the tiled-copy warp-N layout"
        )

        # A warp whose layout misses a group leaves the identity in that plane;
        # persistent tiles also reuse this shared storage.
        identity = const_expr(self.stats_identity())
        total = const_expr(cute.size(smem_tensor.shape))
        sFlat = cute.make_tensor(smem_tensor.iterator, cute.make_layout(total))
        for i0 in cutlass.range(0, total, ctx.num_epi_threads, unroll_full=True):
            i = i0 + ctx.tidx
            if i < total:
                sFlat[i] = Float32(identity)
        ctx.epilogue_barrier.arrive_and_wait()
        lane_info = (lanes_in_N, warps_in_N, warp_n_idx, is_lane_n_leader)

        # Register accumulators for the sweep. ``visit`` identifies an N
        # subtile set; ``group_slots`` maps each row/group to its actual flat
        # register indices, including interleaved M64 layouts.
        epi_shape = cute.zipped_divide(
            cute.make_layout((ctx.tile_M, ctx.tile_N)), ctx.epi_tile
        ).shape
        n_e = const_expr(cute.size(epi_shape[0][1]))
        epi_m_cnt = const_expr(cute.size(epi_shape[1][0]))
        assert max(group_cols, n_e) % min(group_cols, n_e) == 0, (
            "grouped stats need the epi tile N extent and group width to nest"
        )
        slot_coords = tuple(
            (cute.crd2idx(i, ref_layouts[0]), cute.crd2idx(i, ref_layouts[1]))
            for i in range(cute.size(ref_layout))
        )
        row_offsets = tuple(sorted({m for m, _ in slot_coords}))
        group_slots = tuple(
            tuple(
                tuple(
                    i
                    for i, (slot_m, slot_n) in enumerate(slot_coords)
                    if slot_m == row and slot_n // group_cols == group_id
                )
                for group_id in sorted(
                    {slot_n // group_cols for slot_m, slot_n in slot_coords if slot_m == row}
                )
            )
            for row in row_offsets
        )
        rows_sub = const_expr(len(group_slots))
        groups_per_run = const_expr(len(group_slots[0]))
        cols_per_group = const_expr(len(group_slots[0][0]))
        assert all(
            len(row_groups) == groups_per_run
            and all(len(slots) == cols_per_group for slots in row_groups)
            for row_groups in group_slots
        ), "grouped stats need every row to have the same local group geometry"
        lanes_per_group = const_expr(max(1, min(group_cols, n_e) // cols_per_group))
        assert lanes_per_group == 1 << int(math.log2(lanes_per_group)), (
            "grouped stats need each group to occupy a power-of-two N-lane subgroup"
        )
        assert lanes_in_N % lanes_per_group == 0, (
            "grouped stats need N-lane subgroups to partition the warp layout"
        )
        n_visits = const_expr(ctx.tile_N // max(group_cols, n_e))
        rAcc = cute.make_rmem_tensor((rows_sub * epi_m_cnt, n_visits, groups_per_run), Float32)
        rAcc.fill(identity)
        geom = (
            rows_sub,
            n_e,
            epi_m_cnt,
            n_visits,
            group_slots,
            groups_per_run,
            lanes_per_group,
        )
        return (smem_tensor, tDcC, ref_layout, group_cols, lane_info, rAcc, geom)

    @cute.jit
    def stats_slice(self, state, epi_coord):
        smem_tensor, tDcC, ref_layout, group_cols, lane_info, rAcc, geom = state
        rows_sub, n_e = geom[0], geom[1]
        row_base = const_expr(epi_coord[0] * rows_sub)
        visit_n = const_expr((epi_coord[1] * n_e) // max(group_cols, n_e))
        return (
            smem_tensor,
            tDcC[None, None, None, epi_coord],
            ref_layout,
            group_cols,
            lane_info,
            rAcc,
            row_base,
            visit_n,
            geom,
        )

    @cute.jit
    def fn_sink_flush(self, gemm, state, frag):
        """Fold one prepass fragment into static row/visit/group registers."""
        stats = state[0]
        rAcc, row_base, visit_n, geom = stats[5], stats[6], stats[7], stats[8]
        combine_fn = const_expr(self.stats_combine_fn())
        identity = const_expr(self.stats_identity())
        num_rows, group_slots, groups_per_run = geom[0], geom[4], geom[5]
        for r in cutlass.range_constexpr(num_rows):
            for g in cutlass.range_constexpr(groups_per_run):
                slots = const_expr(group_slots[r][g])
                partial = Float32(identity)
                for j in cutlass.range_constexpr(len(slots)):
                    partial = combine_fn(partial, frag[slots[j]])
                rAcc[row_base + r, visit_n, g] = combine_fn(rAcc[row_base + r, visit_n, g], partial)

    @cute.jit
    def fn_prepass_end(self, gemm, state):
        """Reduce lane subgroups and publish raw warp-N planes to shared memory."""
        smem_tensor, tDcC, _, group_cols, lane_info, rAcc, geom = state[0]
        _, warps_in_N, warp_n_idx, _ = lane_info
        rows_sub, n_e, epi_m_cnt, n_visits = geom[:4]
        groups_per_run, lanes_per_group = geom[5], geom[6]
        combine_fn = const_expr(self.stats_combine_fn())
        e_per_visit = const_expr(max(1, group_cols // n_e))
        for em in cutlass.range_constexpr(epi_m_cnt):
            for visit in cutlass.range_constexpr(n_visits):
                coords = tDcC[None, None, None, (em, visit * e_per_visit)]
                for r in cutlass.range_constexpr(rows_sub):
                    for g in cutlass.range_constexpr(groups_per_run):
                        total = rAcc[em * rows_sub + r, visit, g]
                        if const_expr(lanes_per_group > 1):
                            total = cute.arch.warp_reduction(
                                total, combine_fn, threads_in_group=lanes_per_group
                            )
                        if const_expr(warps_in_N == 1):
                            rAcc[em * rows_sub + r, visit, g] = self.stat_value(total, group_cols)
                        coord = coords[const_expr(geom[4][r][g][0])]
                        if cute.arch.lane_idx() % lanes_per_group == 0:
                            smem_tensor[coord[0], coord[1] // group_cols, warp_n_idx] = total

    def prepass_resolve_needed(self, gemm):
        """Whether raw warp-N planes need a post-barrier register resolution."""
        return getattr(gemm, self._stats_shape_attr())[2] > 1

    @cute.jit
    def fn_prepass_resolve(self, gemm, state):
        """Resolve cross-warp statistics into each consumer's register slots."""
        stats = state[0]
        _, tDcC, _, group_cols, _, rAcc, geom = stats
        rows_sub, n_e, epi_m_cnt, n_visits = geom[:4]
        groups_per_run = geom[5]
        e_per_visit = const_expr(max(1, group_cols // n_e))
        for em in cutlass.range_constexpr(epi_m_cnt):
            for visit in cutlass.range_constexpr(n_visits):
                coords = tDcC[None, None, None, (em, visit * e_per_visit)]
                for r in cutlass.range_constexpr(rows_sub):
                    for g in cutlass.range_constexpr(groups_per_run):
                        coord = coords[const_expr(geom[4][r][g][0])]
                        rAcc[em * rows_sub + r, visit, g] = self.stat_value(
                            self.stat_total(stats, coord[0], coord[1] // group_cols),
                            group_cols,
                        )

    @cute.jit
    def stat_total(self, stats, row, group):
        """Combine raw warp-N planes for one (row, group) in fixed order."""
        smem_tensor, lane_info = stats[0], stats[4]
        combine_fn = const_expr(self.stats_combine_fn())
        warps_in_N = lane_info[1]
        total = smem_tensor[row, group, 0]
        for w in cutlass.range_constexpr(1, warps_in_N):
            total = combine_fn(total, smem_tensor[row, group, w])
        return total

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        return (self.stats_begin(gemm, smem_tensor, ctx, const_expr(self._group_cols(param))),)

    @cute.jit
    def begin_loop(self, gemm, state, epi_coord):
        return [self.stats_slice(state[0], epi_coord)]

    def stat_value(self, total, group_cols):
        """Finalize one statistic before broadcasting or writing it."""
        raise NotImplementedError

    def out(self, name):
        """Return a companion global-memory writer for finalized statistics."""
        return GroupedColStatsOut(name, self)

    @cute.jit
    def fn_prepare(self, gemm, state, paired):
        """Broadcast finalized register statistics over their source groups."""
        stats = state[0]
        coords, rAcc, row_base, visit_n, geom = (
            stats[1],
            stats[5],
            stats[6],
            stats[7],
            stats[8],
        )
        out = cute.make_rmem_tensor(coords.layout.shape, Float32)
        num_rows, group_slots, groups_per_run = geom[0], geom[4], geom[5]
        for r in cutlass.range_constexpr(num_rows):
            for g in cutlass.range_constexpr(groups_per_run):
                value = rAcc[row_base + r, visit_n, g]
                slots = const_expr(group_slots[r][g])
                for j in cutlass.range_constexpr(len(slots)):
                    out[slots[j]] = value
        return out


class GroupedColStatsOut(EpiOp):
    """Write a sibling GroupedColStatsBase statistic to global memory."""

    def __init__(self, name, stats_op):
        super().__init__(name)
        assert isinstance(stats_op, GroupedColStatsBase), "stats_op must be a grouped-stats op"
        self.stats_op = stats_op

    def config_key(self):
        return (self.stats_op.cache_key(),)

    def host_fake_arg(self, key, fctx):
        dtype, ndim = key
        groups = cute.sym_int()
        shape = (fctx.l, fctx.m, groups) if ndim == 3 else (fctx.m, groups)
        return make_fake_tensor(dtype, shape, leading_dim=ndim - 1, divisibility=1)

    def host_validate(self, value, *, m, n, tile_M, tile_N, batch, varlen_m, epi_args, **_):
        descriptor = epi_args[self.stats_op.name]
        self.stats_op.host_validate(descriptor, n=n, tile_N=tile_N)
        width = self.stats_op._group_cols(descriptor)
        inner = (m, n // width)
        expected = inner if varlen_m or batch is None else (batch, *inner)
        if tuple(value.shape) != expected:
            raise ValueError(f"'{self.name}': expected shape {expected}, got {tuple(value.shape)}")

    def param_fields(self):
        return [(self.name, object, None)]

    def to_params(self, gemm, args):
        return {self.name: assume_stride_divisibility(getattr(args, self.name))}

    def get_smem_tensor(self, gemm, params, storage_epi):
        return getattr(storage_epi, f"s_{self.stats_op.name}").get_tensor(
            cute.make_layout(getattr(gemm, self.stats_op._stats_shape_attr()))
        )

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        return (smem_tensor, ctx.num_epi_threads)

    @cute.jit
    def end_loop_stage(
        self,
        gemm,
        param,
        state,
        epi_coord,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tidx,
    ):
        """First main-phase subtile: stage this op for the finish phase. No
        barrier request — the sibling's prepass barrier already ordered the
        raw warp_n stats planes this op reads."""
        if const_expr(epi_coord[0] == 0 and epi_coord[1] == 0):
            return (False, (state, tidx))
        return None

    @cute.jit
    def end_loop_finish(self, gemm, param, staged, tile_coord_mnkl, varlen_manager):
        """Elect one writer per (row, group), fold the sibling's raw warp_n
        planes, finalize, and write directly to gmem. The value port uses
        rStats and never reads stats smem in the main pass."""
        state, tidx = staged
        sStats, num_epi_threads = state
        rows, groups, warps_n = getattr(gemm, self.stats_op._stats_shape_attr())
        group_cols = const_expr(gemm.cta_tile_shape_mnk[1] // groups)
        combine = self.stats_op._combine_op()
        batch_idx = tile_coord_mnkl[3]
        limit_m = min(varlen_manager.len_m(batch_idx) - tile_coord_mnkl[0] * rows, rows)
        # (l, m, G) batched; (m, G) dense-2D (batch_idx == 0) or varlen_m
        # (total_m rows, segment offset via cu_seqlens).
        if const_expr(cute.rank(param) == 3):
            mOut = param[batch_idx, None, None]
        elif const_expr(varlen_manager.varlen_m):
            mOut = cute.domain_offset((varlen_manager.params.cu_seqlens_m[batch_idx],), param)
        else:
            mOut = param
        limit_g = mOut.shape[1]
        row0 = tile_coord_mnkl[0] * rows
        g0 = tile_coord_mnkl[1] * groups
        total_slots = const_expr(rows * groups)
        for i0 in cutlass.range(0, total_slots, num_epi_threads, unroll_full=True):
            i = i0 + tidx
            if i < total_slots:
                r = i // groups
                g = i % groups
                if r < limit_m and g0 + g < limit_g:
                    stat = sStats[r, g, 0]
                    for w in cutlass.range_constexpr(1, warps_n):
                        stat = combine(stat, sStats[r, g, w])
                    mOut[row0 + r, g0 + g] = self.stats_op.stat_value(stat, group_cols)


class OnlineLSEReduce(ColVecReduce):
    """Online log-sum-exp column reduction: out[m, n_tile] = log sum_n exp(v).

    Host finalize is the logsumexp fold over the per-N-tile partials (see
    ``VecReduce.host_finalize``).

    The coupled (running max, running sum) accumulator is what a plain
    ``combine=`` cannot express: every new value may rescale the sum. The fn
    just returns the logit under this op's name (sink port); numerical
    stability is owned here. Output is per-N-tile partials like ColVecReduce
    ((l, m, n_tiles)); the host finalizes with a (tiny, stable) logsumexp over
    the n_tiles axis.

    Ragged last N tiles are handled like plain max VecReduce: the fold masks
    OOB elements to the fold identity via a per-element select on the N
    coordinate (OOB accumulator zeros are not an LSE identity), so N need not
    be divisible by tile_N. The select keeps the exp chain straight-line —
    measured ~1.4% on an epilogue-exposed shape (H100, 8k x 8k x 512) and
    <0.7% elsewhere, vs ~38% for a per-element branch. Pass
    ``check_oob=False`` (CUTLASS's ``VisitCheckOOB``) to compile it out when
    N is known tile_N-divisible; the frontend host rejects ragged N then.
    """

    def __init__(self, name, check_oob=True):
        super().__init__(name)
        self.check_oob = check_oob

    def host_finalize(self, partials):
        return torch.logsumexp(partials, dim=-1)

    def config_key(self):
        return (self.combine, self.check_oob)

    # The fold identity is a true -inf (exact for genuinely -inf logits,
    # e.g. attention masks — a finite sentinel like -1e30 is not). The one
    # hazard is an all-identity ("empty") slot: exp(x - m_new) with
    # m_new == -inf is (-inf) - (-inf) = NaN, which poisons the sum through
    # every later rescale. Guard: subtract 0 instead when m_new == -inf
    # (one select per SLOT/merge step, not per element) — exp(-inf - 0) = 0,
    # leaving the exact empty state (m = -inf, s = 0).
    _NEG_INF = -math.inf

    # The inter-warp exchange carries the coupled (max, sum) pair per row per
    # non-leader warp; smem sizing and the whole end_loop protocol come from
    # ColVecReduce's tuple-valued engine.
    reduce_planes = 2

    @cute.jit
    def _end_loop_values(self, state, epi_coord):
        return (
            state[0][None, None, None, epi_coord[0], epi_coord[1]],
            state[1][None, None, None, epi_coord[0], epi_coord[1]],
        )

    @cute.jit
    def _end_loop_smem(self, state):
        return state[2]

    @cute.jit
    def _merge(self, vals, others):
        m, s = vals
        om, os = others
        m_new = cute.arch.fmax(m, om)
        m_sub = self._guard_neg_inf(m_new)
        s_new = s * cute.math.exp(m - m_sub, fastmath=True) + os * cute.math.exp(
            om - m_sub, fastmath=True
        )
        return (m_new, s_new)

    @cute.jit
    def _guard_neg_inf(self, m_new):
        """Subtrahend for the exp args: 0 when the running max is still the
        -inf identity (empty slot), so exp(-inf - 0) = 0 instead of
        exp(NaN). One select, amortized over the slot's elements."""
        return Float32(cutlass.select_(m_new == Float32(self._NEG_INF), Float32(0.0), m_new))

    @cute.jit
    def _finalize(self, vals):
        return cute.math.log(vals[1], fastmath=True) + vals[0]

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        vec_mma_layout = cute.make_layout((ctx.tile_M, ctx.tile_N), stride=self._broadcast_stride())
        acc_layout = ctx.partition_for_epilogue_fn(
            cute.make_rmem_tensor(vec_mma_layout, Float32)
        ).layout
        tDrMax = cute.make_rmem_tensor(acc_layout, Float32)
        tDrSum = cute.make_rmem_tensor(acc_layout, Float32)
        state = (tDrMax, tDrSum, smem_tensor)
        if const_expr(self.check_oob):
            # OOB accumulator zeros are NOT an LSE identity (unlike add), so the
            # fold predicates on the per-element N coordinate against the ragged
            # tile boundary. Same partitioning as the accumulators, so linear
            # indices line up in fn_sink_flush.
            tDcD = ctx.partition_for_epilogue_fn(
                cute.make_identity_tensor((ctx.tile_M, ctx.tile_N))
            )
            limit_n = ctx.varlen_manager.len_n() - ctx.tile_coord_mnkl[1] * ctx.tile_N
            state = (tDrMax, tDrSum, smem_tensor, tDcD, limit_n)
        return state

    @cute.jit
    def begin_loop(self, gemm, state, epi_coord):
        m_cur = state[0][None, None, None, epi_coord[0], epi_coord[1]]
        s_cur = state[1][None, None, None, epi_coord[0], epi_coord[1]]
        if const_expr(epi_coord[self._reduce_dim()] == 0):
            cute.filter_zeros(m_cur).fill(self._NEG_INF)
            cute.filter_zeros(s_cur).fill(0.0)
        loop_state = (m_cur, s_cur)
        if const_expr(self.check_oob):
            c_cur = state[3][None, None, None, epi_coord[0], epi_coord[1]]
            loop_state = (m_cur, s_cur, c_cur, state[4])
        return loop_state

    @cute.jit
    def _fold(self, m_acc, s_acc, frag, coords=None, limit_n=None, tile_shape_mn=None):
        # Two-pass block fold per accumulator slot (same-row elements alias
        # through the zero-stride slice; group them and fold the block):
        # THREAD-LOCAL fragment max first (FMNMX tree, no exp), then ONE
        # rescale of the running sum and ONE exp per element; the coupled
        # (max, sum) cross-lane exchange happens once per M stripe in
        # end_loop. The naive per-element online recurrence pays two exps per
        # element, and MUFU.EX2 (quarter-rate pipe) is the fold's wall — this
        # halves it. (Broadcasting a common row max across N lanes per
        # subtile instead would cost log2(lanes_in_N) shuffle+fmax per slot
        # PER SUBTILE to save one exp-ful merge PER STRIPE — strictly more
        # ops at 8 subtiles/stripe.) The OOB select and compare are ALU ops
        # that mostly hide under the MUFU wall.
        if const_expr(coords is not None):
            # Mask OOB lanes to the fold identity (-inf): fmax keeps m_old
            # and exp(-inf - m_sub) = 0, so masked elements contribute
            # nothing; an all-masked slot stays at the exact empty state
            # (m = -inf, s = 0) via the _guard_neg_inf subtrahend. The frag
            # is this sink's scratch (one flush per fragment), so masking in
            # place is safe.
            #
            # Rebased compare: coords[i][1] = n_base(thread) + off_n(i), with
            # off_n static from the partition layout (projected onto N by
            # composing with (tile_M, tile_N):(0, 1)). Comparing the static
            # offset — an ISETP immediate — against limit_n - n_base deletes
            # the per-element coordinate materialization; ptxas can't rebase
            # itself because it sees `base | off` and can't prove the OR adds.
            lay_n = cute.composition(cute.make_layout(tile_shape_mn, stride=(0, 1)), coords.layout)
            limit_rel = limit_n - coords[0][1]
            for i in cutlass.range(cute.size(frag), unroll_full=True):
                off_n = cute.crd2idx(i, lay_n)
                frag[i] = frag[i] if off_n < limit_rel else self._NEG_INF
        ref = m_acc.layout
        frag_g = layout_utils.convert_layout_zero_stride(frag, ref)
        m_g = layout_utils.convert_layout_zero_stride(m_acc, ref)[None, 0]
        s_g = layout_utils.convert_layout_zero_stride(s_acc, ref)[None, 0]
        n_aliased = const_expr(cute.size(frag_g, mode=[1]))
        for si in cutlass.range(cute.size(m_g), unroll_full=True):
            m_old = m_g[si]
            vmax = frag_g[si, 0]
            for j in cutlass.range_constexpr(1, n_aliased):
                vmax = cute.arch.fmax(vmax, frag_g[si, j])
            m_new = cute.arch.fmax(m_old, vmax)
            m_sub = self._guard_neg_inf(m_new)
            s_new = s_g[si] * cute.math.exp(m_old - m_sub, fastmath=True)
            for j in cutlass.range_constexpr(n_aliased):
                s_new = s_new + cute.math.exp(frag_g[si, j] - m_sub, fastmath=True)
            s_g[si] = s_new
            m_g[si] = m_new

    @cute.jit
    def fn_sink_flush(self, gemm, state, frag):
        # With check_oob, OOB columns (ragged last N tile) are masked to the
        # fold identity: the accumulator zeros there are not an LSE identity.
        m_acc, s_acc = state[0], state[1]
        if const_expr(self.check_oob):
            self._fold(
                m_acc, s_acc, frag, state[2], state[3], tile_shape_mn=gemm.cta_tile_shape_mnk[:2]
            )
        else:
            self._fold(m_acc, s_acc, frag)


@dsl_user_op
def _dup_s16sat_from_s32(x: Int32, *, loc=None, ip=None) -> Uint32:
    """sat16(x) packed into both halves of a b32 (one ALU instruction).

    Feeds the f16x2 EQUALITY compares below as raw bit patterns: f16
    equality is bit equality away from NaN/±0, and this scheme is exact for
    ALL int32 inputs when the comparands are small POSITIVE ints — in-range
    values map injectively, high saturation (0x7FFF) and the in-range slice
    [31745, 32767] are f16 NaNs (match nothing), low saturation 0x8000 is
    -0.0 (aliases only a ±0 comparand — callers bias by +1 so no comparand
    is zero), and f16 compares never flush denormals. This replaces a
    cvt.rn.f16.s32 (quarter-rate XU pipe) whose exactness needed a rounding
    argument."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int32(x).ir_value(loc=loc, ip=ip)],
            "cvt.pack.sat.s16.s32 $0, $1, $1;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _selp_pair_f16x2(
    a2: Uint32, b2: Uint32, f0: Float32, f1: Float32, vin: Float32, *, loc=None, ip=None
) -> Float32:
    """One packed f16x2 equality (HSETP2: TWO predicates per instruction) plus
    the two dependent f32 selects — 3 SASS for 2 elements vs 4 for the scalar
    ISETP+FSEL pair. Inline PTX: the DSL has no two-predicate compare."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Uint32(a2).ir_value(loc=loc, ip=ip),
                Uint32(b2).ir_value(loc=loc, ip=ip),
                Float32(f0).ir_value(loc=loc, ip=ip),
                Float32(f1).ir_value(loc=loc, ip=ip),
                Float32(vin).ir_value(loc=loc, ip=ip),
            ],
            "{.reg .pred p, q; .reg .f32 t;\n"
            "setp.eq.f16x2 p|q, $1, $2;\n"
            "selp.f32 t, $3, $5, p;\n"
            "selp.f32 $0, $4, t, q;}",
            "=f,r,r,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


class ColVecSelect(EpiOp):
    """Per-row column selection ("gather along N"): out[m] = the fn value at
    column idx[m], written directly to an (l, m) / (m,) f32 colvec.

    The fn returns the value under this op's name (a plain sink plane). The
    per-row column index arrives through a companion integer ColVecLoad
    (int32/int64) declared in ``extra_ops`` — the fn never sees it. This op
    reads idx[row] straight from the companion's staged smem (one broadcast
    LDS per aliased row-slot; smem is runtime-addressable where register
    fragments are not) and compares in INTEGER against the static
    per-element N offsets (rebased-compare idiom of OnlineLSEReduce._fold:
    one ISETP against an immediate per element). At most one element in the
    whole (M, N) grid satisfies ``col == idx[row]``, so unlike a reduce
    there is no fold, no lane/warp exchange, and no barrier: whichever
    thread holds the matching element predicated-stores it straight to
    gmem. Rows whose index falls outside [0, N) (e.g. ignore_index -100)
    are never written — pre-fill the output when that matters.

    (Routing the index through the fn as a value-port operand instead costs
    measurably more on an epilogue-exposed shape: the broadcast index frags
    get materialized per element, converted to f32, and collected into a
    second sink plane — all deleted by the companion-smem read.)

    Extraction structure (see fn_sink_flush): per M-stripe, precompute WHICH
    epi_n subtile each row-slot's target falls in and OR one-hot bits into a
    Uint32 register mask; every subtile flush then tests one trace-time-
    constant bit (`mask & (1 << epi_n)` = a LOP3 + branch — no smem load, no
    compares on the hot path). The rarely-taken block extracts via packed
    f16x2 equality (setp.eq.f16x2: TWO predicates per instruction, inline
    PTX) — static footprint matters because the block is instantiated per
    subtile and its I-cache pressure is the measurable residual on short-K
    D-less shapes. Free vs the plain online-LSE epilogue at K >= 1024.

    TOMBSTONES — measured dead ends, do not retry (H100, M=4096 V=128256,
    pp 128x192, same-process interleaved medians):
      * idx through the fn value port (scaled two-plane sink): broadcast
        index frags materialized per element + f32 cvt + a second collection
        plane — several times the select tax of the companion-smem read.
      * per-element `if` around the store: ptxas rebuilds the entire gmem
        address chain inside every element's BSSY/BRA block (+25% kernel).
      * separate `hit` boolean guarding the store: ptxas emits the compare
        chain TWICE (a predicate chain for the branch plus FSELs
        re-materialized inside it) — guard on the selected value instead.
      * FMNMX tree over independent selects (more ILP than a serial fold):
        LOSES on the throughput-bound cooperative schedule — the epilogue
        wall is the warpgroup-shared ALU pipe, not select latency.
      * UNFENCED one-hot R2P extraction (flash-attn mask.py digit-inversion
        of the j->offset map; R2P confirmed in SASS): loses to the fenced
        window everywhere (K=512 D-less 1.18x vs 0.98x of plain) — R2P pays
        when every element consumes a predicate every visit (attention
        masking), not needle-in-haystack equality.
      * one-hot R2P as the FENCED block body: correct, but static size is a
        wash vs the scalar chain; select tax at K=512 D-less: scalar
        ISETP+FSEL chain +0.166ms, one-hot R2P +0.137, packed f16x2 +0.124
        (1 I2F.F16 + 4 HSETP2 + 8 FSEL per 8-element slot; ptxas folds the
        offset pairs into HSETP2 half2 immediates and dual-issues the
        broadcast via the .H0_H0 operand) — f16x2 shipped.
    """

    fn_port = "sink"

    def __init__(self, name, idx_op):
        super().__init__(name)
        if not isinstance(idx_op, ColVecLoad):
            raise ValueError(
                f"ColVecSelect {name!r}: idx_op must be the companion ColVecLoad "
                "staging the per-row column indices"
            )
        self.idx_op = idx_op

    def config_key(self):
        return (self.idx_op.cache_key(),)

    def host_fake_arg(self, key, fctx):
        dtype, ndim = key
        shape = (fctx.l, fctx.m) if ndim == 2 else (fctx.m,)
        return make_fake_tensor(dtype, shape, leading_dim=ndim - 1, divisibility=1)

    def param_fields(self):
        return [(self.name, object, None)]

    def to_params(self, gemm, args):
        return {self.name: assume_stride_divisibility(getattr(args, self.name))}

    def sink_alloc_shape(self, lead, n, tile_m, tile_n, num_seqs=None):
        # Full colvec, not per-tile partials: config-independent (tiles
        # ignored), and there is no host_finalize — the buffer IS the result.
        return tuple(lead)

    def host_validate(self, value, *, m, n, tile_M, tile_N, batch, varlen_m, epi_args, **_):
        idx = epi_args.get(self.idx_op.name)
        if idx is None:
            raise ValueError(f"sink '{self.name}' requires the '{self.idx_op.name}' index operand")
        if idx.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"'{self.idx_op.name}' must be int32 or int64, got {idx.dtype}")
        expected = (m,) if varlen_m or batch is None else (batch, m)
        if tuple(value.shape) != expected:
            raise ValueError(
                f"sink '{self.name}': expected shape {expected}, got {tuple(value.shape)}"
            )
        if value.dtype != torch.float32:
            raise ValueError(f"sink '{self.name}' must be float32, got {value.dtype}")

    def get_smem_tensor(self, gemm, params, storage_epi):
        # The COMPANION's staged index vector (the smem field is declared by
        # the companion; this is the same view its get_smem_tensor returns).
        return getattr(storage_epi, f"s_{self.idx_op.name}").get_tensor(
            cute.make_layout(gemm.cta_tile_shape_mnk[0])
        )

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        # Reference colvec-broadcast partition: its zero-N-stride layout
        # groups aliased same-row elements in fn_sink_flush (layout only —
        # the tensor itself is never read or written, so it costs nothing).
        vec_mma_layout = cute.make_layout((ctx.tile_M, ctx.tile_N), stride=(1, 0))
        tDrRef = ctx.partition_for_epilogue_fn(cute.make_rmem_tensor(vec_mma_layout, Float32))
        tDcD = ctx.partition_for_epilogue_fn(cute.make_identity_tensor((ctx.tile_M, ctx.tile_N)))
        if const_expr(ctx.varlen_manager.varlen_m):
            mVec = cute.domain_offset(
                (ctx.varlen_manager.params.cu_seqlens_m[ctx.batch_idx],), param
            )
        elif const_expr(cute.rank(param) == 2):
            mVec = param[ctx.batch_idx, None]
        else:
            mVec = param  # dense rank-1 (m,)
        gVec = cute.local_tile(mVec, (ctx.tile_M,), (ctx.tile_coord_mnkl[0],))
        limit_m = min(
            ctx.varlen_manager.len_m(ctx.batch_idx) - ctx.tile_coord_mnkl[0] * ctx.tile_M,
            ctx.tile_M,
        )
        n_off = ctx.tile_coord_mnkl[1] * ctx.tile_N
        epi_shape = cute.zipped_divide(
            cute.make_layout((ctx.tile_M, ctx.tile_N)), ctx.epi_tile
        ).shape[1]
        n_epi = const_expr(cute.size(epi_shape[1]))
        etN = const_expr(ctx.tile_N // n_epi)
        # Per-stripe state for the subtile bitmask: the rebased target
        # per row-slot and the Uint32 hit mask over epi_n subtiles.
        assert n_epi <= 32, "ColVecSelect: > 32 N subtiles per tile needs mask chunking"
        ref0 = tDrRef[None, None, None, 0, 0]
        n_slots = const_expr(
            cute.size(layout_utils.convert_layout_zero_stride(ref0, ref0.layout), mode=[0])
        )
        tMask = cute.make_rmem_tensor(1, Uint32)
        tRel0 = cute.make_rmem_tensor(n_slots, Int32)
        return (smem_tensor, tDrRef, tDcD, gVec, limit_m, n_off, tMask, tRel0, etN)

    @cute.jit
    def begin_loop(self, gemm, state, epi_coord):
        ref_cur = state[1][None, None, None, epi_coord[0], epi_coord[1]]
        c_cur = state[2][None, None, None, epi_coord[0], epi_coord[1]]
        return (state[0], ref_cur, c_cur, *state[3:], epi_coord)

    @cute.jit
    def fn_sink_flush(self, gemm, state, frag):
        sIdx, ref_frag, coords, gVec, limit_m, n_off, tMask, tRel0, etN, epi_coord = state
        ref = ref_frag.layout
        frag_g = layout_utils.convert_layout_zero_stride(frag, ref)
        coords_g = layout_utils.convert_layout_zero_stride(coords, ref)
        # Static per-element N offset relative to the thread's base column
        # (rebased compare, same idiom as OnlineLSEReduce._fold): the compare
        # is one ISETP against an immediate, no per-element coordinate
        # materialization.
        lay_n = cute.composition(
            cute.make_layout(gemm.cta_tile_shape_mnk[:2], stride=(0, 1)), coords_g.layout
        )
        n_aliased = const_expr(cute.size(frag_g, mode=[1]))
        n_slots = const_expr(cute.size(frag_g, mode=[0]))
        # Stripe-mask window (mask.py philosophy applied to the guard):
        # the rows of a slot are fixed for a whole M stripe, so at the
        # stripe's first subtile compute, per slot, WHICH epi_n subtile
        # (if any) the target falls in, and OR one-hot bits into a
        # Uint32 mask over subtiles. Every flush then tests one
        # trace-time-constant bit — `mask & (1 << epi_n)` is a single
        # LOP3 + branch, with no smem load and no compares on the hot
        # path (vs an LDS -> IADD -> 2x ISETP -> branch chain per slot
        # per subtile). The rare taken block re-derives idx_rel from the
        # stashed rebased target with one immediate subtract per slot.
        # range_constexpr so si is a Python int: crd2idx then folds to
        # Python ints at trace time (static bounds / shift immediates).
        n_epi = const_expr(gemm.cta_tile_shape_mnk[1] // etN)
        epi_n = const_expr(epi_coord[1])
        if const_expr(epi_n == 0 or gemm.epi_m_major):
            # First subtile of this row-stripe (epi_m_major visits new
            # rows every subtile, so it recomputes every time and the
            # mask degenerates to a per-subtile test — still correct).
            n_base0 = coords_g[0, 0][1] + n_off - epi_n * etN
            mask = Uint32(0)
            for si in cutlass.range_constexpr(n_slots):
                offs = [cute.crd2idx((si, j), lay_n) for j in range(n_aliased)]
                row = coords_g[si, 0][0]
                # OOB rows hold garbage indices (predicated g2s copy):
                # bake the row bound in as a value nothing can hit.
                t_rel = Int32(cutlass.select_(row < limit_m, Int32(sIdx[row]) - n_base0, -1))
                tRel0[si] = t_rel
                e = t_rel // etN
                w = t_rel - e * etN
                # w/e >= 0 both checked: safe under floor OR trunc
                # division semantics for negative t_rel.
                ok = (w >= 0) & (w <= max(offs)) & (e >= 0) & (e < n_epi)
                e_safe = Int32(cutlass.select_(ok, e, 0))  # shift stays in [0, 31]
                mask = mask | Uint32(cutlass.select_(ok, Uint32(1) << e_safe, Uint32(0)))
            tMask[0] = mask
        if Boolean(tMask[0] & (Uint32(1) << epi_n)):
            # Rare block body = packed f16x2 equality (below): what
            # matters here is STATIC size — the block is instantiated
            # per subtile and its I-cache footprint is the dominant
            # residual on short-K D-less shapes (ladder in the class
            # tombstones). Fresh *_cur names: a name born inside a
            # dynamic-if body that is also assigned in a sibling
            # (possibly compiled-out const_expr) arm trips
            # TYPE_UNSTABLE_JOIN (None on the not-taken path).
            for si in cutlass.range_constexpr(n_slots):
                offs_cur = [cute.crd2idx((si, j), lay_n) for j in range(n_aliased)]
                row_cur = coords_g[si, 0][0]
                # +1 bias folded into the immediate (see _select_packed16).
                ir1_cur = Int32(tRel0[si] - (epi_n * etN - 1))
                self._select_packed16(frag_g, gVec, si, row_cur, ir1_cur, offs_cur, n_aliased)

    @cute.jit
    def _select_packed16(self, frag_g, gVec, si, row, idx1, offs, n_aliased):
        # Packed-16 compare chain: pack sat16(idx1 = idx_rel + 1, the bias
        # pre-folded into the caller's immediate subtract) into both halves
        # of a b32 once, then one setp.eq.f16x2 per OFFSET PAIR yields two
        # predicates per instruction feeding the two selects (inline PTX
        # helpers above). The compares run on raw bit patterns with
        # comparand halves = the +1-biased offsets — trace-time immediates
        # (the stripe rebase already made the offsets thread-independent
        # statics), exact for ALL idx_rel (see _dup_s16sat_from_s32); odd
        # tails pad with an f16 NaN half. The -inf miss sentinel guards the
        # store, so a genuinely -inf value at the target column is not
        # written (as in the legacy kernel).
        assert max(offs) + 1 < 31745, "packed16 select: biased offsets reach the f16 NaN range"
        a2 = _dup_s16sat_from_s32(idx1)
        val = Float32(-math.inf)
        for j in cutlass.range_constexpr(0, n_aliased, 2):
            lo = offs[j] + 1
            hi = offs[j + 1] + 1 if j + 1 < n_aliased else 0x7FFF  # NaN pad
            f1 = frag_g[si, j + 1] if j + 1 < n_aliased else Float32(0.0)
            val = _selp_pair_f16x2(a2, Uint32(lo | (hi << 16)), frag_g[si, j], f1, val)
        if val != Float32(-math.inf):
            gVec[row] = val

    # TOMBSTONE: the one-hot R2P extraction that used to live here (digit-
    # inversion of the j->offset map by pow2 strides, one-hot mask, R2P into
    # the predicate bank, valid-guarded store — flash-attn mask.py idiom) was
    # measured and removed: unfenced it loses to the stripe-mask fence by
    # ~15pp of plain-matmul time, and as the fenced block body it loses to
    # the packed f16x2 chain above (+0.137 vs +0.124ms select tax, K=512
    # D-less) while being no smaller statically. See the class docstring
    # tombstones for the full ladder before resurrecting any of it.
