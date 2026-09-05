# Copyright (c) 2026, Han Guo, Tri Dao.
"""Grouped local reductions inside a GEMM epilogue tile (FlexGEMM parity).

FlexGEMM (``torch/_inductor/kernel/flex_gemm``) recognizes epilogues that
reshape the GEMM output to expose contiguous groups along M or N and reduce
only that grouped dimension::

    out = acc.reshape(m, n // g, g).sum(-1)     # axis=1, group=g
    out = acc.reshape(m // g, g, n).sum(-2)     # axis=0, group=g

The reduced value either leaves the kernel *compressed* (one value per
``(row, group)`` — no per-tile partials, unlike
:class:`quack.epilogue.ops.VecReduce`)
or is broadcast back into the same epilogue pass ("feed main", e.g. a grouped
softmax denominator). This module provides three EpiOps for that contract:

* :class:`GroupedLocalReduce` — sink port. The fn returns the value under the
  op name; the op reduces each group physically and stores exactly one element
  per ``(row, group)`` into a compressed aux tensor.
* :class:`GroupedLocalReduceFeed` — apply port for same-warp axis-0 groups. The
  fn calls the op (``r = gsum(acc)``) and gets the reduction broadcast to every
  row lane. Passing a compressed aux tensor additionally stores the value.
* :class:`GroupedLocalReducePrepass` — tensorless value port for axis-1 groups
  up to 32. An accumulator prepass reduces to shared memory through
  :class:`quack.epilogue.ops.GroupedColStatsBase`; the main pass receives the group
  value broadcast per element, without a follow-up kernel.
* :class:`GroupedLocalReduceWithFinalizeArg` — sink port for a grouped sum whose
  scalar finalizer also consumes a prepass value, used by stable grouped LSE.

Reduction geometry (all static, derived from the epilogue tiled_copy)
---------------------------------------------------------------------
``_fragment_geometry`` recovers the (m, n) offset of every register slot from
the partitioned layouts and *validates* each assumption the folds below make
(one contiguous N run of ``cols`` columns per row for N groups, exactly one row
per thread for M groups, warps tiling the CTA rows in lane-sized blocks for the
cross-warp stitch), so an unsupported tile/warp layout fails at compile time
instead of silently reducing the wrong elements.

* axis=1 (N groups), ``group <= cols``: fold inside the thread's fragment
  (left-to-right, broadcast within the group), then one leader lane stores.
* axis=1, ``group > cols``: fold each fragment, then combine the
  ``group // cols`` consecutive epi-N subtiles of a group in ascending order
  (the oracle's "temporal" combine) at the last subtile of the group.
* axis=0 (M groups), ``group <= lanes_m``: butterfly across the group's row
  lanes (halving offsets), so every lane holds the group value.
* axis=0, ``group > lanes_m``: butterfly within each warp, then stitch
  ``group // lanes_m`` warps through smem in ascending warp order behind the
  epilogue barrier.

Every fold order is fixed and data-independent, so results are bitwise
reproducible run to run. Two intentional deviations from the legacy FlexGEMM
reducer (the behavior oracle):

* The oracle receives axis-1 groups that fit one fragment already reduced by
  generated TensorSSA code and only compresses the store (``combine=None``
  here does that). With a ``combine``, this module folds the fragment itself in
  a fixed left-to-right order; TensorSSA's ``reduce`` may use a different
  association, so f32 sums can differ by rounding (tolerance-level, not a
  contract change).
* The oracle's 4-wide vectorized f32 store fast path for axis=0 is not ported;
  stores are scalar (one element per group leader). Pure throughput, no
  semantics.

Tails and dynamic group counts
------------------------------
The store is predicated on the *runtime* extents of the compressed tensor
(``limit_groups``) and of the GEMM output (``limit_m`` / ``limit_n``), so a
ragged last tile writes fewer groups without host-side padding. Groups must not
straddle the GEMM boundary: ``group`` has to divide both the CTA tile and the
grouped GEMM dimension (:func:`validate_grouped_reduce_out` checks the latter
host-side; the tile divisibility is asserted at compile time). OOB accumulator
lanes are zero, which is the identity for ``add`` only — with ``mul``/``max``/
``min`` a partially OOB group would be wrong, and the divisibility rule is what
makes that unrepresentable.

Integration (EpiMod / quack.epilogue.frontend)
------------------------------------------
The ops work through the existing hook APIs::

    @gemm_epilogue(outs={"gsum": GroupedLocalReduce("gsum", axis=1, group=32)})
    def grouped_sum(acc):
        return {"D": acc, "gsum": acc}

    @gemm_epilogue(ops={"gmax": GroupedLocalReduceFeed("gmax", axis=0, group=8,
                                                       combine="max")})
    def grouped_center(acc, gmax):
        return {"D": acc - gmax(acc)}   # epi_args["gmax"] = compressed buffer

    def grouped_prepass(acc):
        return {"gsum": acc}

    @gemm_epilogue(
        ops={"gsum": GroupedLocalReducePrepass("gsum", group=16)},
        prepass=grouped_prepass,
        prepass_outs=("gsum",),
    )
    def grouped_n_center(acc, gsum):
        return {"D": acc - gsum}        # epi_args["gsum"] = None

Tensorless feeds need a parent-side hook: ``ComposableEpiMixin`` normally
filters ops whose argument is ``None`` and omits their shared-memory budget.
:class:`GroupedFeedMainMixin` keeps both apply feeds and prepass/value feeds
active; ``EpiMod._mint`` includes it in every generated kernel class.

Numerics, geometries, tails, and the config contract are pinned by
tests/test_grouped_reduce.py.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr

from torch._vendor.quack.epilogue.ops import (
    EpiOp,
    EpiSmemBytes,
    GroupedColStatsBase,
    _callable_config_key,
    _get_lane_warp_layouts,
    assume_stride_divisibility,
)
from torch._vendor.quack.gemm_runtime.identity import semantic_value_key
from torch._vendor.quack import layout_utils
from torch._vendor.quack.sm90_utils import partition_for_epilogue

# FlexGEMM's host-side gate width (``constraints.LOCAL_REDUCE_FRAGMENT_WIDTH``):
# the M-lane count of the SM100 epilogue partition, and the largest group that
# a feed can reduce inside one warp. The device paths below derive the real
# per-thread geometry instead of assuming this value.
GROUPED_FRAGMENT_WIDTH = 32


def nan_propagating_max(a, b):
    """Combine maxima while preserving PyTorch's NaN propagation semantics."""
    return cute.arch.fmax(a, b, nan=True)


def nan_propagating_min(a, b):
    """Combine minima while preserving PyTorch's NaN propagation semantics."""
    return cute.arch.fmin(a, b, nan=True)


_COMBINE_FNS = {
    "add": operator.add,
    "mul": operator.mul,
    "max": nan_propagating_max,
    "min": nan_propagating_min,
}
_COMBINE_IDENTITIES = {
    "add": 0.0,
    "mul": 1.0,
    "max": -math.inf,
    "min": math.inf,
}


def feed_main_capable(axis: int, group: int) -> bool:
    """Whether a feed can be reduced in-pass (same-warp M groups only, mirroring
    ``constraints.validate_local_reduce_feed_main_capability``)."""
    return axis == 0 and group <= GROUPED_FRAGMENT_WIDTH


def grouped_reduce_supports_config(config, axis: int, group: int) -> bool:
    """Whether a GEMM config exposes a supported logical grouped-sink layout.

    ``axis`` is caller-oriented. Swap-at-trace transposes the accumulator, so
    legality is checked against the opposite physical CTA axis. Swapped grouped
    sinks are initially SM100-only, matching the layout families covered by
    forced-kernel tests.
    """
    if axis not in (0, 1) or group <= 0:
        return False
    if config.swap_ab:
        if axis == 0 or getattr(config, "device_capacity", None) != 10:
            return False
        axis = 1 - axis
    tile = config.tile_m if axis == 0 else config.tile_n
    if config.tile_m < 128 or config.tile_n % GROUPED_FRAGMENT_WIDTH or tile % group:
        return False

    fragment_width = GROUPED_FRAGMENT_WIDTH
    if (
        axis == 1
        and config.tile_m == 128
        and config.tile_n in (128, 160, 224)
        and config.cluster_m > 1
    ):
        fragment_width //= 2
    if group <= GROUPED_FRAGMENT_WIDTH:
        return fragment_width % group == 0 and group < tile

    if group % GROUPED_FRAGMENT_WIDTH or config.cluster_n != 1:
        return False
    if config.tile_m == 128 and config.cluster_m == 1:
        return True
    if config.tile_m == 256 and config.cluster_m == 2:
        return axis == 1 or group < tile
    return (
        axis == 1
        and config.tile_m == 128
        and config.tile_n == 256
        and config.cluster_m == 2
        and group < tile
    )


def max_grouped_reduce_group(configs, axis: int) -> int | None:
    """Largest logical grouped sink extent supported by a config collection."""
    groups = (
        group
        for config in configs
        for group in (
            2,
            4,
            8,
            16,
            32,
            *range(
                64,
                (config.tile_m if (1 - axis if config.swap_ab else axis) == 0 else config.tile_n)
                + 1,
                GROUPED_FRAGMENT_WIDTH,
            ),
        )
        if grouped_reduce_supports_config(config, axis, group)
    )
    return max(groups, default=None)


def grouped_reduce_out_shape(m: int, n: int, group: int, axis: int, batch: int | None = None):
    """Compressed aux shape for a grouped reduce: the grouped GEMM dim / group."""
    if group <= 0:
        raise ValueError("group must be positive")
    if axis not in (0, 1):
        raise ValueError("axis must be 0 (group along M) or 1 (group along N)")
    inner = (m, n // group) if axis == 1 else (m // group, n)
    return inner if batch is None else (batch, *inner)


@dataclass(frozen=True)
class GroupedLocalReduceOutputLayout:
    """Caller-owned physical carrier exposed as a logical grouped-output tensor.

    ``tensor_fn(carrier, transposed)`` runs while tracing a concrete GEMM
    candidate and returns the rank-3 caller-oriented compressed matrix, or its
    transposed kernel orientation. ``carrier_shape_fn(batch, rows, cols)``
    validates the runtime carrier; ``fake_shape_fn`` rebuilds its symbolic
    signature in async workers. QuACK fingerprints and invokes the callbacks
    but does not interpret the physical layout.
    """

    name: str
    tensor_fn: object
    carrier_shape_fn: object
    fake_shape_fn: object
    carrier_ndim: int
    supports_config_fn: object | None = None
    validate_carrier_fn: object | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("output layout name must be non-empty")
        if not all(
            callable(fn) for fn in (self.tensor_fn, self.carrier_shape_fn, self.fake_shape_fn)
        ):
            raise TypeError(
                "output layout tensor, carrier-shape, and fake-shape callbacks must be callable"
            )
        if self.supports_config_fn is not None and not callable(self.supports_config_fn):
            raise TypeError("output layout config predicate must be callable")
        if self.validate_carrier_fn is not None and not callable(self.validate_carrier_fn):
            raise TypeError("output layout carrier validator must be callable")
        if self.carrier_ndim <= 0:
            raise ValueError("output layout carrier rank must be positive")

    def cache_key(self):
        """Return source-derived callback identity for the persistent kernel key."""
        return (
            self.name,
            semantic_value_key(self.tensor_fn, set(), force_source=True),
            semantic_value_key(self.carrier_shape_fn, set(), force_source=True),
            semantic_value_key(self.fake_shape_fn, set(), force_source=True),
            self.carrier_ndim,
            semantic_value_key(self.supports_config_fn, set(), force_source=True),
            semantic_value_key(self.validate_carrier_fn, set(), force_source=True),
        )


def validate_grouped_reduce_out(
    name: str,
    out,
    m: int,
    n: int,
    group: int,
    axis: int,
    batch: int | None = None,
    tile_M: int | None = None,
    tile_N: int | None = None,
) -> None:
    """Fail-closed host check of a compressed aux buffer against the reduce plan.

    ``EpiMod`` only shape-checks sinks that look like :class:`VecReduce`
    partials, so callers (or a parent-side hook in ``EpiMod.gemm``'s sink loop)
    call this to reject group/shape combinations the kernel cannot express:
    a group that straddles the GEMM boundary would fold zero-padded elements,
    and a mismatched or non-contiguous buffer would corrupt memory.
    """
    tile = tile_M if axis == 0 else tile_N
    dim = m if axis == 0 else n
    if dim % group:
        raise ValueError(f"{name}: group {group} must divide the grouped dim {dim} (axis={axis})")
    if tile is not None and (tile % group or group > tile):
        raise ValueError(f"{name}: group {group} must divide the CTA tile extent {tile}")
    expected = grouped_reduce_out_shape(m, n, group, axis, batch)
    if tuple(out.shape) != expected:
        raise ValueError(f"{name}: expected compressed shape {expected}, got {tuple(out.shape)}")
    if out.stride(-1) != 1:
        raise ValueError(f"{name}: compressed aux buffer must be contiguous in its last dim")


class _GroupGeometry(NamedTuple):
    """Static per-thread epilogue fragment geometry a grouped reduce consumes.

    Only the fields the device paths read; every other layout property is
    checked in :func:`_fragment_geometry` and never travels. ``axis`` is the
    physical kernel axis after swap-at-trace. ``cols`` is the N run one thread
    owns per row of an epi subtile, ``chunks`` the per-(row, aligned run) flat
    fragment indices in ascending column order (the axis-1 fold unit),
    ``fragments_per_group`` > 1 means a group spans that many consecutive epi-N
    subtiles, and ``group_warps`` > 1 that an M group spans that many warps.
    """

    axis: int
    cols: int
    chunk: int
    chunks: tuple
    row_chunks: tuple
    fragments_per_group: int
    lanes_m: int
    group_warps: int
    lane_layout_MN: object
    warp_layout_MN: object


def _row_chunks(by_row, cols, group):
    """Group flat fragment indices into per-row, column-aligned fold chunks.

    Requires every row of the thread's fragment to own the same contiguous N
    run: that makes each chunk exactly a group (or a group-aligned piece of
    one), and makes the thread's column base a multiple of the run length
    because the partition tiles the epi subtile exactly.
    """
    for m_off, entries in by_row.items():
        if sorted(n for n, _ in entries) != list(range(cols)):
            raise NotImplementedError(
                "grouped N reduce needs each thread to own one contiguous N run per "
                f"row of an epi subtile; row {m_off} owns {sorted(n for n, _ in entries)}"
            )
    chunk = min(group, cols)
    if cols % chunk:
        raise NotImplementedError(
            f"grouped local reduce needs the thread N run ({cols}) to be a multiple of the "
            f"in-fragment group extent ({chunk})"
        )
    chunks = tuple(
        tuple(i for _, i in sorted(entries)[c0 : c0 + chunk])
        for entries in by_row.values()
        for c0 in range(0, cols, chunk)
    )
    return chunk, chunks


def _fragment_geometry(gemm, epi_tile, tiled_copy, tidx, reference_src, axis, group):
    """Derive and validate the static fragment geometry a grouped reduce needs.

    The (m, n) offset of every register slot comes from partitioning the two
    broadcast reference layouts, so the checks below are exact for the actual
    epilogue layout rather than assumed from the arch.
    """
    tile_M, tile_N = gemm.cta_tile_shape_mnk[:2]
    partition = partial(
        partition_for_epilogue,
        epi_tile=epi_tile,
        tiled_copy=tiled_copy,
        tidx=tidx,
        reference_src=reference_src,
    )
    frags = tuple(
        partition(cute.make_rmem_tensor(cute.make_layout((tile_M, tile_N), stride=s), Float32))
        for s in ((1, 0), (0, 1))
    )
    m_sub, n_sub = (f[None, None, None, 0, 0].layout for f in frags)
    slots = [(cute.crd2idx(i, m_sub), cute.crd2idx(i, n_sub)) for i in range(cute.size(m_sub))]
    by_row: dict[int, list] = {}
    by_col: dict[int, list] = {}
    for i, (m_off, n_off) in enumerate(slots):
        by_row.setdefault(m_off, []).append((n_off, i))
        by_col.setdefault(n_off, []).append((m_off, i))
    rows = len(by_row)
    cols = len(next(iter(by_row.values())))
    chunk, chunks = _row_chunks(by_row, cols, group) if axis == 1 else (1, ())
    epi_m = cute.size(frags[0].layout.shape[3])
    epi_n = cute.size(frags[0].layout.shape[4])
    lane_layout_MN, warp_layout_MN = _get_lane_warp_layouts(tiled_copy, reference_src)
    lanes_m = cute.size(lane_layout_MN, mode=[0])
    warps_m = cute.size(warp_layout_MN, mode=[0])
    warps_n = cute.size(warp_layout_MN, mode=[1])
    tile = tile_M if axis == 0 else tile_N
    if group > tile or tile % group:
        raise NotImplementedError(
            f"grouped local reduce: group {group} must divide the CTA tile extent {tile}"
        )
    fragments_per_group, group_warps = 1, 1
    row_chunks = ()
    if axis == 1:
        if group > cols:
            n_step = cute.crd2idx((0, 0, 0, 0, 1), frags[1].layout) if epi_n > 1 else 0
            if n_step != cols:
                raise NotImplementedError(
                    "grouped N reduce larger than one fragment needs consecutive epi-N "
                    f"subtiles to be adjacent column runs (step {n_step} != {cols})"
                )
            if group % cols or (cols * epi_n) % group:
                raise NotImplementedError(
                    f"grouped N reduce: group {group} must be a multiple of the thread N run "
                    f"({cols}) and divide its per-thread N coverage ({cols * epi_n})"
                )
            fragments_per_group = group // cols
    else:
        if lanes_m % min(group, lanes_m):
            raise NotImplementedError(
                f"grouped M reduce: group {group} must divide the M lane count {lanes_m}"
            )
        row_chunks = tuple(
            tuple(i for _, i in entries)
            for col_entries in by_col.values()
            for entries in (
                tuple(
                    sorted(
                        (entry for entry in col_entries if entry[0] // group == group_idx),
                    )
                )
                for group_idx in sorted({m_off // group for m_off, _ in col_entries})
            )
        )
        rows_per_warp = lanes_m * rows
        if group > rows_per_warp:
            group_warps = group // rows_per_warp
            if group % rows_per_warp or group_warps > warps_m:
                raise NotImplementedError(
                    f"grouped M reduce: group {group} must be a multiple of the per-warp "
                    f"M coverage ({rows_per_warp}) within the {warps_m} M warps"
                )
            if tile_M // rows_per_warp != warps_m:
                raise NotImplementedError(
                    f"grouped M reduce across warps needs the {warps_m} M warps to tile the "
                    f"{tile_M} CTA rows in {rows_per_warp}-row blocks"
                )
    return _GroupGeometry(
        axis=axis,
        cols=cols,
        chunk=chunk,
        chunks=chunks,
        row_chunks=row_chunks,
        fragments_per_group=fragments_per_group,
        lanes_m=lanes_m,
        group_warps=group_warps,
        lane_layout_MN=lane_layout_MN,
        warp_layout_MN=warp_layout_MN,
    )


class _GroupedState(NamedTuple):
    """Per-tile state: register accumulator, coordinate partition, smem, geometry."""

    frag: object
    coord: object
    smem: object
    geom: object
    warp_m_idx: object


class _GroupedSlice(NamedTuple):
    """Per-subtile state handed to the fn ports (sink fold / apply callable)."""

    frag: object
    geom: object


class _GroupedOutputLayoutParams(NamedTuple):
    """Logical bounds paired with a caller-owned physical output view."""

    tensor: object
    logical_extent: object
    logical_groups: object


class GroupedReduceBase(EpiOp):
    """Shared config, host schema, and compressed store for grouped reduces.

    ``axis``: 0 groups contiguous M rows, 1 groups contiguous N columns.
    ``group``: elements per group; must divide the CTA tile extent and the
    grouped GEMM dimension. ``combine``: ``"add" | "mul" | "max" | "min"``, a
    2-argument callable, or None — None means the values arrive already reduced
    and broadcast per group (FlexGEMM's generated-TensorSSA contract) and this
    op only compresses the store. ``finalize``: None, ``"mean"`` (divide by
    ``group``), or a 1-argument callable applied once to each group value.
    ``reduce_planes`` is the fixed number of Float32 state components. For more
    than one plane, the sink returns a tuple and callable combine/finalize
    receive the complete state tuple. ``fragment_reduced`` means each returned
    plane is already reduced and broadcast within its register fragment; the op
    skips only that local fold and still combines subtiles, lanes, and warps.
    """

    supports_swap_ab = False

    def __init__(
        self,
        name,
        *,
        axis,
        group,
        combine="add",
        finalize=None,
        reduce_planes=1,
        fragment_reduced=False,
        output_layout: GroupedLocalReduceOutputLayout | None = None,
    ):
        super().__init__(name)
        if axis not in (0, 1):
            raise ValueError("axis must be 0 (group along M) or 1 (group along N)")
        if group <= 1:
            raise ValueError("group must be greater than 1")
        if isinstance(combine, str) and combine not in _COMBINE_FNS:
            raise ValueError(f"unsupported combine {combine!r}; use {sorted(_COMBINE_FNS)}")
        if combine is not None and not (isinstance(combine, str) or callable(combine)):
            raise TypeError("combine must be a name, a 2-argument callable, or None")
        if not (finalize is None or finalize == "mean" or callable(finalize)):
            raise TypeError("finalize must be None, 'mean', or a 1-argument callable")
        if not isinstance(reduce_planes, int) or reduce_planes < 1:
            raise ValueError("reduce_planes must be a positive integer")
        if reduce_planes > 1 and (not callable(combine) or not callable(finalize)):
            raise TypeError("multi-plane grouped reductions require callable combine and finalize")
        if not isinstance(fragment_reduced, bool):
            raise TypeError("fragment_reduced must be bool")
        if fragment_reduced and combine is None:
            raise ValueError("fragment_reduced requires a cross-fragment combine")
        self.axis = axis
        self.group = group
        self.combine = combine
        self.finalize = finalize
        self.reduce_planes = reduce_planes
        self.fragment_reduced = fragment_reduced
        self.output_layout = output_layout

    def config_key(self):
        """Fail-closed identity for geometry, state algebra, and output layout.

        Callable identity includes source, defaults, closures, and referenced
        globals, so distinct generated state algebra cannot share a kernel.
        A subclass that adds configuration must extend this key.
        """
        extra = tuple(
            sorted(
                set(vars(self))
                - {
                    "name",
                    "axis",
                    "group",
                    "combine",
                    "finalize",
                    "reduce_planes",
                    "fragment_reduced",
                    "output_layout",
                }
            )
        )
        if extra:
            raise NotImplementedError(
                f"{type(self).__name__} has static configuration {extra}; extend config_key()"
            )
        return (
            self.axis,
            self.group,
            self.combine if isinstance(self.combine, str) else _callable_config_key(self.combine),
            self.finalize
            if self.finalize is None or isinstance(self.finalize, str)
            else _callable_config_key(self.finalize),
            self.reduce_planes,
            self.fragment_reduced,
            None if self.output_layout is None else self.output_layout.cache_key(),
        )

    def supports_config(self, config) -> bool:
        """Whether a native GEMM config preserves the reduction and output layout."""
        if not grouped_reduce_supports_config(config, self.axis, self.group):
            return False
        return (
            self.output_layout is None
            or self.output_layout.supports_config_fn is None
            or self.output_layout.supports_config_fn(config, self.axis, self.group)
        )

    def config_support_error(self, configs) -> str:
        """Describe an unsupported group or caller-owned output layout."""
        if (
            self.output_layout is not None
            and self.output_layout.supports_config_fn is not None
            and not any(
                self.output_layout.supports_config_fn(config, self.axis, self.group)
                for config in configs
                if grouped_reduce_supports_config(config, self.axis, self.group)
            )
        ):
            return f"output layout {self.output_layout.name!r} has no supported GemmConfig"
        max_group = max_grouped_reduce_group(configs, self.axis)
        return f"requested group={self.group}, max supported group={max_group} for axis={self.axis}"

    @property
    def sink_arity(self):
        """Number of state planes returned to this sink by the epilogue fn."""
        return self.reduce_planes

    @property
    def combine_fn(self):
        """Resolved 2-argument combine, or None for pre-reduced values."""
        return _COMBINE_FNS[self.combine] if isinstance(self.combine, str) else self.combine

    def _state_planes(self, value):
        """Return a tuple view of scalar or multi-plane reduction state."""
        return (value,) if self.reduce_planes == 1 else value

    @cute.jit
    def _combine_state(self, lhs, rhs):
        """Combine two tuples of grouped-reduction state components."""
        if const_expr(self.reduce_planes == 1):
            return (self.combine_fn(lhs[0], rhs[0]),)
        result = self.combine_fn(lhs, rhs)
        assert isinstance(result, tuple) and len(result) == self.reduce_planes, (
            f"combine must return {self.reduce_planes} state planes"
        )
        return result

    @cute.jit
    def _finalize_state(self, values):
        """Project grouped-reduction state to the scalar output value."""
        if const_expr(self.reduce_planes == 1):
            return self.finalize_value(values[0])
        return self.finalize(values)

    def _is_temporal(self, geom):
        """Whether one group spans several physical-N subtiles."""
        return geom.axis == 1 and self.combine_fn is not None and geom.fragments_per_group > 1

    @cute.jit
    def finalize_value(self, value):
        """Apply the group finalize: mean (a true divide, matching FlexGEMM's
        generated ``value / group.0``), a generated callable, or nothing."""
        if const_expr(self.finalize == "mean"):
            return value / Float32(self.group)
        if const_expr(self.finalize is not None):
            return self.finalize(value)
        return value

    # --- Host schema -------------------------------------------------------
    def host_arg_key(self, value):
        if value is None:
            return None
        from torch._vendor.quack.cute_dsl_utils import torch2cute_dtype_map

        if self.output_layout is not None:
            if value.ndim != self.output_layout.carrier_ndim:
                raise ValueError(
                    f"{self.name}: {self.output_layout.name} carrier must be rank "
                    f"{self.output_layout.carrier_ndim}"
                )
        else:
            if value.ndim not in (2, 3):
                raise ValueError(f"{self.name}: compressed aux buffer must be rank 2 or 3")
            if value.stride(-1) != 1:
                raise ValueError(
                    f"{self.name}: compressed aux buffer must be contiguous in its last dim"
                )
        return (torch2cute_dtype_map[value.dtype], value.ndim)

    def host_validate(
        self,
        value,
        *,
        m,
        n,
        tile_M,
        tile_N,
        batch,
        varlen_m,
        swap_ab=False,
        **_,
    ):
        """Validate caller-owned carrier geometry against the GEMM problem."""
        if varlen_m:
            raise ValueError(f"{self.name}: grouped reductions do not support varlen_m")
        if value is None:
            if getattr(self, "keep_tensorless", False):
                return
            kind = (
                "compressed aux buffer" if self.output_layout is None else "output-layout carrier"
            )
            raise ValueError(f"{self.name}: {kind} is required")
        self.host_arg_key(value)
        dim = m if self.axis == 0 else n
        physical_axis = 1 - self.axis if swap_ab else self.axis
        tile = tile_M if physical_axis == 0 else tile_N
        if dim % self.group:
            raise ValueError(
                f"{self.name}: group {self.group} must divide the grouped dim "
                f"{dim} (axis={self.axis})"
            )
        if tile % self.group or self.group > tile:
            raise ValueError(
                f"{self.name}: group {self.group} must divide the CTA tile extent {tile}"
            )
        if self.output_layout is None:
            expected = grouped_reduce_out_shape(m, n, self.group, self.axis, batch)
            actual = tuple(value.shape)
            if actual != expected:
                raise ValueError(f"{self.name}: expected compressed shape {expected}, got {actual}")
            return
        rows, cols = (m, n // self.group) if self.axis == 1 else (m // self.group, n)
        expected = tuple(
            self.output_layout.carrier_shape_fn(
                1 if batch is None else batch,
                rows,
                cols,
            )
        )
        actual = tuple(value.shape)
        if actual != expected:
            raise ValueError(
                f"{self.name}: {self.output_layout.name} carrier must have shape "
                f"{expected}, got {actual}"
            )
        if self.output_layout.validate_carrier_fn is not None:
            self.output_layout.validate_carrier_fn(value)

    def host_fake_arg(self, key, fctx):
        from torch._vendor.quack.compile_utils import make_fake_tensor

        dtype, ndim = key
        m, n = (fctx.n, fctx.m) if fctx.swapped else (fctx.m, fctx.n)
        if self.output_layout is not None:
            rows, cols = (m, n // self.group) if self.axis == 1 else (m // self.group, n)
            shape = self.output_layout.fake_shape_fn(fctx.l, rows, cols)
            return make_fake_tensor(
                dtype,
                shape,
                leading_dim=self.output_layout.carrier_ndim - 1,
                divisibility=1,
            )
        groups = cute.sym_int()
        inner = (m, groups) if self.axis == 1 else (groups, n)
        shape = (fctx.l, *inner) if ndim == 3 else inner
        return make_fake_tensor(dtype, shape, leading_dim=ndim - 1, divisibility=1)

    def param_fields(self):
        return [(self.name, object, None)]

    def physical_axis(self, gemm) -> int:
        """Return the kernel axis corresponding to the caller-oriented axis."""
        return 1 - self.axis if const_expr(gemm.a_transposed) else self.axis

    def to_params(self, gemm, args):
        tensor = getattr(args, self.name)
        if self.output_layout is not None:
            tensor = self.output_layout.tensor_fn(tensor, gemm.a_transposed)
            assert cute.rank(tensor) == 3
            logical_extent, logical_groups = (
                (gemm.caller_m, gemm.caller_n // self.group)
                if self.axis == 1
                else (gemm.caller_n, gemm.caller_m // self.group)
            )
            return {self.name: _GroupedOutputLayoutParams(tensor, logical_extent, logical_groups)}
        if const_expr(gemm.a_transposed):
            tensor = layout_utils.select(tensor, [0, 2, 1] if cute.rank(tensor) == 3 else [1, 0])
        return {self.name: assume_stride_divisibility(tensor)}

    def epi_m_major_score(self, arg_tensor, gemm):
        # Prefer N-major subtile order: the temporal combine then keeps only one
        # group's fragments live at a time.
        return -1

    # --- Cross-warp smem (axis-0 groups wider than one warp) ---------------
    def _uses_smem(self, gemm=None):
        """Whether physical-M groups may need cross-warp shared memory.

        Stage computation runs before the GEMM instance receives its swap flag,
        so it conservatively reserves the buffer for any wide group. The traced
        instance keeps the buffer only when its physical axis is M.
        """
        if self.group <= GROUPED_FRAGMENT_WIDTH:
            return False
        return gemm is None or self.physical_axis(gemm) == 0

    def _smem_warps(self, warps_m):
        return max(warps_m - 1, 0)

    def smem_bytes(self, arg_tensor, cta_tile_shape_mnk, epi_tile, warp_shape_mnk=None):
        if not self._uses_smem():
            return EpiSmemBytes()
        warps_m = warp_shape_mnk[0] if warp_shape_mnk is not None else 1
        planes = self._smem_warps(warps_m)
        return EpiSmemBytes(
            unstaged=cta_tile_shape_mnk[1] * planes * self.reduce_planes * (Float32.width // 8)
        )

    def _smem_shape(self, gemm):
        planes = self._smem_warps(gemm.epi_smem_warp_shape_mnk()[0])
        if not planes:
            return None
        shape = (gemm.cta_tile_shape_mnk[1], planes)
        return shape if self.reduce_planes == 1 else (*shape, self.reduce_planes)

    def smem_struct_field(self, gemm, params):
        shape = self._smem_shape(gemm) if self._uses_smem(gemm) else None
        if shape is None:
            return None
        size = math.prod(shape)
        return (f"s_{self.name}", cute.struct.Align[cute.struct.MemRange[Float32, size], 16])

    def get_smem_tensor(self, gemm, params, storage_epi):
        shape = self._smem_shape(gemm) if self._uses_smem(gemm) else None
        if shape is None:
            return None
        return getattr(storage_epi, f"s_{self.name}").get_tensor(cute.make_layout(shape))

    # --- Device: shared setup ---------------------------------------------
    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        """Register accumulator, coordinate partition, smem, validated geometry."""
        tiled_copy = ctx.tiled_copy_t2r if ctx.tiled_copy_t2r is not None else ctx.tiled_copy_r2s
        geom = _fragment_geometry(
            gemm,
            ctx.epi_tile,
            tiled_copy,
            ctx.tidx,
            ctx.tiled_copy_t2r is None,
            self.physical_axis(gemm),
            self.group,
        )
        frag = None
        coord = None
        if const_expr(param is not None):
            # Only the partitioned SHAPE is used (a compact register fragment,
            # congruent with the accumulator by flat index), so the reference
            # layout is a broadcast one: its cosize, not the whole MN tile,
            # bounds the trace-time tensor.
            shape = ctx.partition_for_epilogue_fn(
                cute.make_rmem_tensor(
                    cute.make_layout((ctx.tile_M, ctx.tile_N), stride=(1, 0)), Float32
                )
            ).shape
            keep_subtiles = const_expr(geom.fragments_per_group > 1)
            frag_shape = shape if keep_subtiles else shape[:3]
            frag = (
                cute.make_rmem_tensor(frag_shape, Float32)
                if const_expr(self.reduce_planes == 1)
                else tuple(
                    cute.make_rmem_tensor(frag_shape, Float32) for _ in range(self.reduce_planes)
                )
            )
            coord = ctx.partition_for_epilogue_fn(
                cute.make_identity_tensor((ctx.tile_M, ctx.tile_N))
            )
        warp_m_idx = geom.warp_layout_MN.get_hier_coord(
            cute.arch.make_warp_uniform(ctx.tidx // cute.arch.WARP_SIZE)
        )[0]
        return _GroupedState(frag, coord, smem_tensor, geom, warp_m_idx)

    @cute.jit
    def _frag_slice(self, state, epi_coord):
        """This subtile's slice of each register-state plane."""
        if const_expr(state.frag is None):
            return None
        if const_expr(self.reduce_planes == 1):
            if const_expr(cute.rank(state.frag) == 3):
                return state.frag
            return state.frag[None, None, None, epi_coord[0], epi_coord[1]]
        if const_expr(cute.rank(state.frag[0]) == 3):
            return state.frag
        return tuple(plane[None, None, None, epi_coord[0], epi_coord[1]] for plane in state.frag)

    def begin_loop(self, gemm, state, epi_coord):
        return _GroupedSlice(self._frag_slice(state, epi_coord), state.geom)

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
        """Stage this self-synchronized grouped flush for the finish phase."""
        return (False, (state, epi_coord, epi_tile, tiled_copy_t2r, tiled_copy_r2s, tidx))

    @cute.jit
    def end_loop_finish(self, gemm, param, staged, tile_coord_mnkl, varlen_manager):
        """Run the grouped flush through the current two-phase EpiOp protocol."""
        state, epi_coord, epi_tile, tiled_copy_t2r, tiled_copy_r2s, tidx = staged
        self.end_loop(
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
        )

    # --- Device: compressed store -----------------------------------------
    @cute.jit
    def _store_groups(
        self,
        gemm,
        param,
        state,
        epi_coord,
        tile_coord_mnkl,
        varlen_manager,
        values,
        finalize=True,
    ):
        """Store one element per (row, group) from the group leaders.

        ``values`` is indexable by the same flat index as the coordinate
        fragment; the leader predicate picks the slot whose group offset is the
        group's first column (axis 1) / first row (axis 0), and the runtime
        extents of the compressed tensor bound the group index so ragged tiles
        write fewer groups. ``finalize=False`` for values a caller already
        finalized (the feed port).
        """
        if const_expr(self.output_layout is not None):
            logical_extent = param.logical_extent
            logical_groups = param.logical_groups
            param = param.tensor
        else:
            logical_extent = None
            logical_groups = None
        tile_M, tile_N = gemm.cta_tile_shape_mnk[:2]
        axis = state.geom.axis
        tile = tile_M if const_expr(axis == 0) else tile_N
        groups_per_cta = const_expr(tile // self.group)
        batch_idx = tile_coord_mnkl[3]
        if const_expr(cute.rank(param) == 3):
            mReduce = param[batch_idx, None, None]
        else:
            mReduce = param[None, None]
        if const_expr(axis == 1):
            tile_shape = (tile_M, groups_per_cta)
            if const_expr(self.output_layout is not None):
                assert not varlen_manager.varlen_m
                limit_groups = logical_groups
            else:
                logical_extent = varlen_manager.len_m(batch_idx)
                limit_groups = cute.size(mReduce, mode=[1])
            limit = min(logical_extent - tile_coord_mnkl[0] * tile_M, tile_M)
        else:
            tile_shape = (groups_per_cta, tile_N)
            if const_expr(self.output_layout is not None):
                assert not varlen_manager.varlen_m
                limit_groups = logical_groups
            else:
                logical_extent = cute.size(mReduce, mode=[1])
                limit_groups = cute.size(mReduce, mode=[0])
            limit = min(logical_extent - tile_coord_mnkl[1] * tile_N, tile_N)
        gReduce = cute.local_tile(mReduce, tile_shape, (tile_coord_mnkl[0], tile_coord_mnkl[1]))
        coord = cute.filter_zeros(state.coord[None, None, None, epi_coord[0], epi_coord[1]])
        # Group leaders: the group's first column (N groups) / first row (M
        # groups) — except after a temporal combine, which lands on the LAST
        # fragment of the group.
        leader_off = const_expr(
            self.group - state.geom.cols if self._is_temporal(state.geom) else 0
        )
        tile_idx = tile_coord_mnkl[1] if const_expr(axis == 1) else tile_coord_mnkl[0]
        value_planes = self._state_planes(values)
        for i in cutlass.range(cute.size(coord), unroll_full=True):
            row_idx, n_idx = coord[i][0], coord[i][1]
            pos = n_idx if const_expr(axis == 1) else row_idx
            group_idx = pos // self.group
            in_bounds = (row_idx if const_expr(axis == 1) else n_idx) < limit
            if (
                pos % self.group == leader_off
                and in_bounds
                and tile_idx * groups_per_cta + group_idx < limit_groups
            ):
                state_value = tuple(plane[i] for plane in value_planes)
                value = (
                    self._finalize_state(state_value) if const_expr(finalize) else state_value[0]
                )
                if const_expr(param.element_type != Float32):
                    value = value.to(param.element_type)
                if const_expr(axis == 1):
                    gReduce[row_idx, group_idx] = value
                else:
                    gReduce[group_idx, n_idx] = value


class GroupedLocalReduce(GroupedReduceBase):
    """Compressed grouped local reduction (sink port).

    The fn returns the per-element value under this op's name; the op folds each
    group physically (see the module docstring for the four geometries) and
    stores one element per ``(row, group)`` into the compressed aux tensor.
    ``combine=None`` skips every fold because values are fully group-reduced.
    ``fragment_reduced=True`` skips only the in-fragment fold because TensorSSA
    or another producer already reduced and broadcast each fragment partial.
    """

    fn_port = "sink"
    supports_swap_ab = True

    @cute.jit
    def fn_sink_flush(self, gemm, state, *fragments):
        """Collect each state plane; the physical fold runs in end_loop."""
        assert len(fragments) == self.reduce_planes
        destinations = self._state_planes(state.frag)
        for source, destination in zip(fragments, destinations):
            cute.autovec_copy(source, destination)

    @cute.jit
    def _fold_chunks(self, frag, chunks):
        """Fold each static state chunk in place and broadcast its result."""
        planes = self._state_planes(frag)
        for chunk in cutlass.range_constexpr(len(chunks)):
            slots = const_expr(chunks[chunk])
            values = tuple(plane[slots[0]] for plane in planes)
            for j in cutlass.range_constexpr(1, len(slots)):
                values = self._combine_state(values, tuple(plane[slots[j]] for plane in planes))
            for j in cutlass.range_constexpr(len(slots)):
                for plane, value in zip(planes, values):
                    plane[slots[j]] = value

    @cute.jit
    def _butterfly_rows(self, frag, geom):
        """Reduce across the group's row lanes and broadcast each state plane."""
        planes = self._state_planes(frag)
        reduce_lanes = const_expr(min(self.group, geom.lanes_m))
        for i in cutlass.range(cute.size(planes[0]), unroll_full=True):
            values = tuple(plane[i] for plane in planes)
            rows = reduce_lanes // 2
            while rows > 0:
                offset = cute.crd2idx((rows, 0), geom.lane_layout_MN)
                values = self._combine_state(
                    values,
                    tuple(cute.arch.shuffle_sync_bfly(value, offset=offset) for value in values),
                )
                rows = rows // 2
            for plane, value in zip(planes, values):
                plane[i] = value

    @cute.jit
    def _combine_subtiles(self, state, epi_coord, geom):
        """Combine consecutive epi-N subtiles into one grouped state."""
        first = const_expr(epi_coord[1] + 1 - geom.fragments_per_group)
        state_planes = self._state_planes(state.frag)
        merged_planes = tuple(
            cute.make_rmem_tensor_like(plane[None, None, None, epi_coord[0], first], Float32)
            for plane in state_planes
        )
        for source, merged in zip(state_planes, merged_planes):
            cute.autovec_copy(source[None, None, None, epi_coord[0], first], merged)
        merged_planes = tuple(cute.filter_zeros(plane) for plane in merged_planes)
        for offset in cutlass.range_constexpr(1, geom.fragments_per_group):
            other_planes = tuple(
                cute.filter_zeros(plane[None, None, None, epi_coord[0], first + offset])
                for plane in state_planes
            )
            for i in cutlass.range(cute.size(merged_planes[0]), unroll_full=True):
                values = self._combine_state(
                    tuple(plane[i] for plane in merged_planes),
                    tuple(plane[i] for plane in other_planes),
                )
                for plane, value in zip(merged_planes, values):
                    plane[i] = value
        return merged_planes[0] if const_expr(self.reduce_planes == 1) else merged_planes

    @cute.jit
    def _stitch_warps(self, gemm, state, frag, epi_coord, geom):
        """Stitch the ``group_warps`` warps of an M group through smem: each
        non-leader warp publishes its butterfly result once per column, the
        leader warp folds the planes in ascending warp order."""
        sReduce = state.smem
        assert sReduce is not None, "grouped M reduce across warps needs its smem buffer"
        planes = self._state_planes(frag)
        coord = cute.filter_zeros(state.coord[None, None, None, epi_coord[0], epi_coord[1]])
        for i in cutlass.range(cute.size(planes[0]), unroll_full=True):
            row_idx, n_idx = coord[i][0], coord[i][1]
            group_idx = row_idx // self.group
            warp_in_group = state.warp_m_idx - group_idx * geom.group_warps
            if warp_in_group > 0 and warp_in_group < geom.group_warps:
                smem_warp = state.warp_m_idx - group_idx - 1
                for plane_idx, plane in enumerate(planes):
                    if const_expr(self.reduce_planes == 1):
                        sReduce[n_idx, smem_warp] = plane[i]
                    else:
                        sReduce[n_idx, smem_warp, plane_idx] = plane[i]
        gemm.epilogue_barrier.arrive_and_wait()
        for i in cutlass.range(cute.size(planes[0]), unroll_full=True):
            row_idx, n_idx = coord[i][0], coord[i][1]
            group_idx = row_idx // self.group
            if state.warp_m_idx == group_idx * geom.group_warps:
                values = tuple(plane[i] for plane in planes)
                for offset in cutlass.range_constexpr(1, geom.group_warps):
                    smem_warp = group_idx * geom.group_warps + offset - group_idx - 1
                    others = tuple(
                        sReduce[n_idx, smem_warp]
                        if const_expr(self.reduce_planes == 1)
                        else sReduce[n_idx, smem_warp, plane_idx]
                        for plane_idx in range(self.reduce_planes)
                    )
                    values = self._combine_state(values, others)
                for plane, value in zip(planes, values):
                    plane[i] = value

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
        """Fold this subtile's groups and store the completed ones."""
        if const_expr(param is None):
            return
        geom = state.geom
        frag = self._frag_slice(state, epi_coord)
        frag = (
            cute.filter_zeros(frag)
            if const_expr(self.reduce_planes == 1)
            else tuple(cute.filter_zeros(plane) for plane in frag)
        )
        if const_expr(self.combine_fn is not None):
            if const_expr(geom.axis == 1):
                if const_expr(not self.fragment_reduced and geom.chunk > 1):
                    self._fold_chunks(frag, geom.chunks)
                if const_expr(self._is_temporal(geom)):
                    # Only the group's last subtile completes a group value.
                    if const_expr((epi_coord[1] + 1) % geom.fragments_per_group != 0):
                        return
                    frag = self._combine_subtiles(state, epi_coord, geom)
            else:
                if const_expr(
                    not self.fragment_reduced and any(len(chunk) > 1 for chunk in geom.row_chunks)
                ):
                    self._fold_chunks(frag, geom.row_chunks)
                if const_expr(min(self.group, geom.lanes_m) > 1):
                    self._butterfly_rows(frag, geom)
                if const_expr(geom.group_warps > 1):
                    self._stitch_warps(gemm, state, frag, epi_coord, geom)
        self._store_groups(gemm, param, state, epi_coord, tile_coord_mnkl, varlen_manager, frag)
        if const_expr(geom.group_warps > 1):
            # Re-arm the smem planes for the next subtile / persistent tile.
            gemm.epilogue_barrier.arrive_and_wait()


class GroupedLocalReduceFeed(GroupedReduceBase):
    """Same-pass grouped M reduction fed back into the fn (apply port).

    The fn calls the op (``r = gsum(acc)``) and receives the group reduction
    broadcast to every row lane of the group — no second accumulator pass and no
    smem, which is why it is limited to groups inside one warp's row lanes
    (FlexGEMM's ``validate_local_reduce_feed_main_capability``). Passing a
    compressed aux tensor also stores the reduced values, and is what keeps the
    op active without the :class:`GroupedFeedMainMixin` hook.
    """

    fn_port = "apply"
    keep_tensorless = True

    def __init__(self, name, *, axis=0, group, combine="add", finalize=None):
        if not feed_main_capable(axis, group):
            raise NotImplementedError(
                "grouped feed-main supports same-warp M groups only "
                f"(axis=0, group <= {GROUPED_FRAGMENT_WIDTH}); got axis={axis}, group={group}"
            )
        if combine is None:
            raise ValueError("a grouped feed must reduce: combine=None has nothing to broadcast")
        super().__init__(name, axis=axis, group=group, combine=combine, finalize=finalize)

    @cute.jit
    def reduce_broadcast(self, value, geom):
        """Group reduction of one register value, broadcast to the group's row
        lanes. The primitive the apply port calls per element, and the entry
        point a hand-written mixin uses in place of the oracle's
        ``grouped_rowvec_reduce_value`` (loop it over the fragment)."""
        combine_fn = const_expr(self.combine_fn)
        rows = const_expr(self.group // 2)
        while rows > 0:
            value = combine_fn(
                value,
                cute.arch.shuffle_sync_bfly(
                    value, offset=cute.crd2idx((rows, 0), geom.lane_layout_MN)
                ),
            )
            rows = rows // 2
        return self.finalize_value(value)

    @cute.jit
    def fn_apply(self, gemm, pstate, i, value):
        """Broadcast group value for element ``i``; also records it for the store.

        Packed values (SM100 ``F2`` lanes, ``acc_pair`` ``Pair``s) hold the two
        adjacent fragment slots ``2i`` / ``2i + 1``, so the same write-through
        works in every fn loop shape.
        """
        geom = pstate.geom
        if const_expr(isinstance(value, tuple)):
            lanes = tuple(self.reduce_broadcast(lane, geom) for lane in value)
            if const_expr(pstate.frag is not None):
                pstate.frag[2 * i], pstate.frag[2 * i + 1] = lanes
            return type(value)(*lanes)
        reduced = self.reduce_broadcast(value, geom)
        if const_expr(pstate.frag is not None):
            pstate.frag[i] = reduced
        return reduced

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
        """Store the broadcast group values recorded by ``fn_apply`` (already
        finalized there, so the store must not apply the finalize twice)."""
        if const_expr(param is None):
            return
        frag = cute.filter_zeros(self._frag_slice(state, epi_coord))
        self._store_groups(
            gemm,
            param,
            state,
            epi_coord,
            tile_coord_mnkl,
            varlen_manager,
            frag,
            finalize=False,
        )


class _GroupedFinalizeState(NamedTuple):
    """Grouped-reduction state carrying one per-element finalizer argument."""

    frag: object
    finalize_arg: object
    coord: object
    smem: object
    geom: object
    warp_m_idx: object


class _GroupedFinalizeSlice(NamedTuple):
    """Per-subtile grouped values and their finalizer arguments."""

    frag: object
    finalize_arg: object
    geom: object


class GroupedLocalReduceWithFinalizeArg(GroupedLocalReduce):
    """Axis-1 sum sink whose scalar finalizer also receives a prepass value."""

    sink_arity = 2
    supports_swap_ab = False

    def __init__(self, name, *, axis, group, finalize, combine="add"):
        if (
            axis != 1
            or group > GROUPED_FRAGMENT_WIDTH
            or group & (group - 1)
            or combine != "add"
            or not callable(finalize)
        ):
            raise NotImplementedError(
                "two-argument grouped finalizers require a fragment-local axis-1 sum"
            )
        super().__init__(
            name,
            axis=axis,
            group=group,
            combine=combine,
            finalize=finalize,
        )

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        """Allocate a companion fragment beside the normal sink accumulator."""
        state = super().begin(gemm, param, smem_tensor, ctx)
        finalize_arg = (
            None
            if const_expr(state.frag is None)
            else cute.make_rmem_tensor_like(state.frag, Float32)
        )
        return _GroupedFinalizeState(
            state.frag,
            finalize_arg,
            state.coord,
            state.smem,
            state.geom,
            state.warp_m_idx,
        )

    def begin_loop(self, gemm, state, epi_coord):
        """Slice both sink fragments for the current epilogue subtile."""
        finalize_arg = state.finalize_arg
        if const_expr(finalize_arg is not None and cute.rank(finalize_arg) != 3):
            finalize_arg = finalize_arg[None, None, None, epi_coord[0], epi_coord[1]]
        return _GroupedFinalizeSlice(self._frag_slice(state, epi_coord), finalize_arg, state.geom)

    @cute.jit
    def fn_sink_flush(self, gemm, state, *fragments):
        """Capture one reduction plane and one independent finalizer argument."""
        assert len(fragments) == 2
        cute.autovec_copy(fragments[0], state.frag)
        cute.autovec_copy(fragments[1], state.finalize_arg)

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
        """Fold the outer sum, apply the binary finalizer, and store in-kernel."""
        if const_expr(param is None):
            return
        geom = state.geom
        assert not self._is_temporal(geom), (
            "two-argument grouped finalizers require one fragment per group"
        )
        frag = cute.filter_zeros(self._frag_slice(state, epi_coord))
        finalize_arg = cute.filter_zeros(
            state.finalize_arg
            if const_expr(cute.rank(state.finalize_arg) == 3)
            else state.finalize_arg[None, None, None, epi_coord[0], epi_coord[1]]
        )
        if const_expr(geom.chunk > 1):
            self._fold_chunks(frag, geom.chunks)
        for i in cutlass.range(cute.size(frag), unroll_full=True):
            frag[i] = self.finalize(frag[i], finalize_arg[i])
        self._store_groups(
            gemm,
            param,
            state,
            epi_coord,
            tile_coord_mnkl,
            varlen_manager,
            frag,
            finalize=False,
        )


class GroupedLocalReducePrepass(GroupedColStatsBase):
    """Axis-1 grouped reduction broadcast from an accumulator prepass.

    This is the feed-main path for groups that fit FlexGEMM's 32-column
    fragment contract. The prepass returns the source value under this op's
    name; :class:`GroupedColStatsBase` folds it into deterministic per-(row,
    group) shared-memory statistics, and the main-pass value port broadcasts
    the finalized group value to every element. The op is tensorless: callers
    pass ``None`` for its epilogue argument, and no reduction leaves the
    kernel.
    """

    keep_tensorless = True

    def __init__(self, name, *, axis=1, group, combine="add", finalize=None):
        super().__init__(name)
        if axis != 1:
            raise NotImplementedError(
                f"grouped prepass feed-main supports axis=1 only; got axis={axis}"
            )
        if group <= 1 or group > GROUPED_FRAGMENT_WIDTH or group & (group - 1):
            raise ValueError(f"group must be a power of two in [2, {GROUPED_FRAGMENT_WIDTH}]")
        if combine not in _COMBINE_FNS:
            raise ValueError(f"unsupported combine {combine!r}; use {sorted(_COMBINE_FNS)}")
        if not (finalize is None or finalize == "mean" or callable(finalize)):
            raise TypeError("finalize must be None, 'mean', or a 1-argument callable")
        self.axis = axis
        self.group = group
        self.combine = combine
        self.finalize = finalize

    def supports_config(self, config) -> bool:
        """Whether a native GEMM config can preserve this grouped geometry."""
        return grouped_reduce_supports_config(config, self.axis, self.group)

    def config_key(self):
        """Static reduction semantics used by the persistent kernel cache."""
        extra = tuple(sorted(set(vars(self)) - {"name", "axis", "group", "combine", "finalize"}))
        if extra:
            raise NotImplementedError(
                f"{type(self).__name__} has static configuration {extra}; extend config_key()"
            )
        return (
            self.axis,
            self.group,
            self.combine,
            self.finalize
            if self.finalize is None or isinstance(self.finalize, str)
            else _callable_config_key(self.finalize),
        )

    def host_arg_key(self, value):
        """Validate the tensorless host contract; no runtime argument is emitted."""
        if value is not None:
            raise ValueError(f"{self.name}: grouped prepass value port is tensorless")
        return None

    def host_validate(self, value, *, n, tile_N, **_):
        """Validate the tensorless argument and fixed grouped-N geometry."""
        self.host_arg_key(value)
        super().host_validate(self.group, n=n, tile_N=tile_N)

    def param_fields(self):
        return [(self.name, object, None)]

    def _group_cols(self, arg_tensor):
        return self.group

    def stats_identity(self):
        """Reduction identity used for register and shared-memory slots."""
        return _COMBINE_IDENTITIES[self.combine]

    def stats_combine_fn(self):
        """Resolved physical combine used by the prepass reduction tree."""
        return _COMBINE_FNS[self.combine]

    def stat_value(self, total, group_cols):
        """Leave finalization to the value-port broadcast."""
        return total

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        assert gemm.arch in (90, 100, 120), (
            "grouped prepass feed-main needs a re-readable accumulator"
        )
        assert ctx.tile_N % self.group == 0, (
            "grouped prepass feed-main needs whole groups in each CTA N tile"
        )
        return [self.stats_begin(gemm, smem_tensor, ctx, self.group), ctx.tRS_rD_layout]

    @cute.jit
    def begin_loop(self, gemm, state, epi_coord):
        return [self.stats_slice(state[0], epi_coord), state[1]]

    @cute.jit
    def finalize_value(self, value):
        """Apply the optional scalar finalizer once to a completed group."""
        if const_expr(self.finalize == "mean"):
            return value / Float32(self.group)
        if const_expr(self.finalize is not None):
            return self.finalize(value)
        return value

    @cute.jit
    def fn_prepare(self, gemm, state, paired):
        """Broadcast each finalized (row, group) statistic over its elements."""
        stats, out_layout = state[0], state[1]
        coords, geom = stats[1], stats[8]
        out = cute.make_rmem_tensor(out_layout, Float32)
        num_rows, group_slots, groups_per_run = geom[0], geom[4], geom[5]
        for r in cutlass.range_constexpr(num_rows):
            for g in cutlass.range_constexpr(groups_per_run):
                slots = const_expr(group_slots[r][g])
                coord = coords[slots[0]]
                value = self.finalize_value(
                    self.stat_total(stats, coord[0], coord[1] // self.group)
                )
                for j in cutlass.range_constexpr(len(slots)):
                    out[slots[j]] = value
        return out


class GroupedFeedMainMixin:
    """Parent-side hook that keeps tensorless grouped value producers active.

    ``ComposableEpiMixin`` normally drops ops whose argument is ``None``.
    Grouped apply feeds and prepass/value feeds still participate when no
    compressed tensor exists, so this mixin retains ops marked
    ``keep_tensorless`` and includes their shared-memory budget.
    """

    def _filter_epi_ops(self, args):
        super()._filter_epi_ops(args)
        active = {op.name for op in self._epi_ops}
        self._epi_ops = tuple(
            op
            for op in type(self)._epi_ops
            if op.name in active or getattr(op, "keep_tensorless", False)
        )

    @classmethod
    def epi_smem_bytes(cls, args, cta_tile_shape_mnk, epi_tile, warp_shape_mnk=None):
        """Include tensorless prepass resources omitted by the base filter."""
        result = super().epi_smem_bytes(args, cta_tile_shape_mnk, epi_tile, warp_shape_mnk)
        for op in cls._epi_ops:
            if getattr(args, op.name, None) is None and getattr(op, "keep_tensorless", False):
                result += op.smem_bytes(None, cta_tile_shape_mnk, epi_tile, warp_shape_mnk)
        return result
