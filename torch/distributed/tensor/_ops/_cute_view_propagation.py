# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
"""
CuTe layout composition for DTensor view ops sharding propagation.

CuTe layouts represent tensor distribution as hierarchical sub-modes within
each dimension.  A GPU sub-mode at some position in a dim's layout encodes
what DTensor calls Shard (outermost) or _StridedShard (inner).  View
composition preserves GPU modes through shape changes by routing sub-modes
across output dims via the DimMap rule tree.

Replaces Phase 2 (``rewrite_output_placements``) of
``_ViewShardingPropagator`` for the common case where each tensor dim is
sharded by at most one mesh dim.  Multi-mesh-same-dim cases return ``None``
to signal the caller to fall back to the existing Phase 2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence

from torch.distributed._pycute import (
    flatten as cute_flatten,
    is_tuple,
    Layout,
    make_layout,
    suffix_product,
)
from torch.distributed.tensor._ops._view_ops import (
    Broadcast,
    DimMap,
    Flatten,
    InputDim,
    NewDim,
    Repeat,
    Singleton,
    Split,
)
from torch.distributed.tensor.placement_types import (
    _StridedShard,
    Partial,
    Placement,
    Replicate,
    Shard,
)


class _UnsupportedCase(Exception):
    pass


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GpuMode:
    """Identifies a GPU sub-mode within a flattened layout."""

    mesh_dim: int
    flat_index: int


@dataclass
class DistLayout:
    """Tensor distribution as a CuTe layout with GPU mode tags.

    The layout maps tensor coordinates to flat element indices.
    Sharded dims have hierarchical sub-modes; ``gpu_modes`` tracks which
    sub-modes correspond to mesh dimensions.
    """

    layout: Layout
    num_dims: int
    gpu_modes: list[GpuMode] = field(default_factory=list)

    def gpu_dim(self, mesh_dim: int) -> int | None:
        """Which tensor dim contains the GPU mode for *mesh_dim*, or ``None``."""
        for gm in self.gpu_modes:
            if gm.mesh_dim == mesh_dim:
                dim_idx, _, _ = _locate_submode(
                    self.layout, self.num_dims, gm.flat_index
                )
                return dim_idx
        return None

    def gpu_mode_shape(self, mesh_dim: int) -> int | None:
        """Size of the GPU sub-mode for *mesh_dim* (should equal mesh size)."""
        for gm in self.gpu_modes:
            if gm.mesh_dim == mesh_dim:
                dim_idx, sub_idx, _ = _locate_submode(
                    self.layout, self.num_dims, gm.flat_index
                )
                mode = self.layout[dim_idx]
                shapes = (
                    cute_flatten(mode.shape) if is_tuple(mode.shape) else (mode.shape,)
                )
                return shapes[sub_idx]
        return None

    def local_contiguous(self, mesh_dim: int) -> bool:
        """True if GPU mode is outermost in its dim (Shard vs _StridedShard).

        "Outermost" means no *local* (non-GPU) sub-mode has a larger stride.
        """
        for gm in self.gpu_modes:
            if gm.mesh_dim == mesh_dim:
                dim_idx, sub_idx, _ = _locate_submode(
                    self.layout, self.num_dims, gm.flat_index
                )
                mode = self.layout[dim_idx]
                strides = (
                    cute_flatten(mode.stride)
                    if is_tuple(mode.stride)
                    else (mode.stride,)
                )
                gpu_stride = strides[sub_idx]
                gpu_sub_indices: set[int] = set()
                for other_gm in self.gpu_modes:
                    other_dim, other_sub, _ = _locate_submode(
                        self.layout, self.num_dims, other_gm.flat_index
                    )
                    if other_dim == dim_idx:
                        gpu_sub_indices.add(other_sub)
                return all(
                    st <= gpu_stride or j in gpu_sub_indices
                    for j, st in enumerate(strides)
                )
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _locate_submode(
    layout: Layout, num_dims: int, flat_index: int
) -> tuple[int, int, int]:
    """``(dim_idx, sub_idx_within_dim, dim_num_submodes)`` for a flat index."""
    running = 0
    for d in range(num_dims):
        mode = layout[d]
        n = len(cute_flatten(mode.shape)) if is_tuple(mode.shape) else 1
        if flat_index < running + n:
            return d, flat_index - running, n
        running += n
    raise ValueError(f"flat_index {flat_index} out of range")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def _add_shard_to_dim(
    dim_modes: dict[int, tuple[list[int], list[int], list[tuple[int, int]]]],
    dim: int,
    mesh_dim: int,
    M: int,
) -> None:
    """Add a Shard for *mesh_dim* to a dim that already has sub-modes.

    DTensor processes placements left-to-right: this Shard operates on the
    local data after all earlier mesh dims.  The local data may include
    group structure from earlier _StridedShards — Shard splits the groups
    evenly (outermost local sub-mode).
    """
    shapes, strides, gpu_info = dim_modes[dim]
    # Find the outermost non-GPU sub-mode (largest stride, not a GPU sub)
    gpu_sub_set = {si for _, si in gpu_info}
    outer_idx = -1
    outer_stride = -1
    for j, st in enumerate(strides):
        if j not in gpu_sub_set and st > outer_stride:
            outer_idx = j
            outer_stride = st
    if outer_idx == -1:
        raise _UnsupportedCase("no local sub-mode to divide for multi-mesh Shard")
    outer_shape = shapes[outer_idx]
    if outer_shape % M != 0:
        raise _UnsupportedCase(
            f"local sub-mode shape {outer_shape} not divisible by mesh {M}"
        )
    inner = outer_shape // M
    if inner == 1:
        # Degenerate: GPU encompasses this sub-mode entirely — replace in place
        shapes[outer_idx] = M
        gpu_info.append((mesh_dim, outer_idx))
    else:
        # Split: (inner, M) replaces the outer sub-mode
        shapes[outer_idx] = inner
        shapes.insert(outer_idx + 1, M)
        strides.insert(outer_idx + 1, outer_stride * inner)
        # Adjust gpu_info sub-indices for sub-modes shifted by the insertion
        for k, (md, si) in enumerate(gpu_info):
            if si > outer_idx:
                gpu_info[k] = (md, si + 1)
        gpu_info.append((mesh_dim, outer_idx + 1))


def _add_strided_shard_to_dim(
    dim_modes: dict[int, tuple[list[int], list[int], list[tuple[int, int]]]],
    dim: int,
    mesh_dim: int,
    M: int,
    sf: int,
) -> None:
    """Add a _StridedShard for *mesh_dim* to a dim with existing sub-modes."""
    shapes, strides, gpu_info = dim_modes[dim]
    gpu_sub_set = {si for _, si in gpu_info}
    # Find the outermost non-GPU sub-mode
    outer_idx = -1
    outer_stride = -1
    for j, st in enumerate(strides):
        if j not in gpu_sub_set and st > outer_stride:
            outer_idx = j
            outer_stride = st
    if outer_idx == -1:
        raise _UnsupportedCase("no local sub-mode for multi-mesh _StridedShard")
    outer_shape = shapes[outer_idx]
    if sf > outer_shape or outer_shape % sf != 0:
        raise _UnsupportedCase(
            f"sf={sf} doesn't divide outer sub-mode shape {outer_shape}"
        )
    group_size = outer_shape // sf
    if group_size % M != 0:
        raise _UnsupportedCase(
            f"local {outer_shape}: group_size {group_size} not divisible by M={M}"
        )
    lpg = group_size // M
    if lpg == 1:
        # (M, sf) replaces the outer sub-mode
        shapes[outer_idx] = M
        shapes.insert(outer_idx + 1, sf)
        strides.insert(outer_idx + 1, outer_stride * group_size)
        for k, (md, si) in enumerate(gpu_info):
            if si > outer_idx:
                gpu_info[k] = (md, si + 1)
        gpu_info.append((mesh_dim, outer_idx))
    else:
        # (lpg, M, sf) replaces the outer sub-mode
        shapes[outer_idx] = lpg
        shapes.insert(outer_idx + 1, M)
        shapes.insert(outer_idx + 2, sf)
        strides.insert(outer_idx + 1, outer_stride * lpg)
        strides.insert(outer_idx + 2, outer_stride * group_size)
        for k, (md, si) in enumerate(gpu_info):
            if si > outer_idx:
                gpu_info[k] = (md, si + 2)
        gpu_info.append((mesh_dim, outer_idx + 1))


def from_placements(
    shape: tuple[int, ...],
    placements: Sequence[Placement],
    mesh_sizes: tuple[int, ...],
) -> DistLayout:
    """Build a ``DistLayout`` from placements.

    Handles single-mesh-per-dim and multi-mesh-same-dim cases.  Placements
    are processed left-to-right (matching DTensor semantics): each mesh dim's
    placement operates on the local result of all earlier mesh dims.
    """
    bstrides: tuple[int, ...] = suffix_product(shape)  # type: ignore[assignment]

    # Per-dim sub-mode overrides: dim -> (shapes, strides, [(mesh_dim, sub_idx)])
    dim_modes: dict[int, tuple[list[int], list[int], list[tuple[int, int]]]] = {}

    for mesh_dim, p in enumerate(placements):
        if isinstance(p, Shard):
            dim = p.dim
            M = mesh_sizes[mesh_dim]
            if dim in dim_modes:
                _add_shard_to_dim(dim_modes, dim, mesh_dim, M)
            else:
                S, b = shape[dim], bstrides[dim]
                local = S // M
                if S % M != 0:
                    raise _UnsupportedCase(
                        f"dim {dim} size {S} not divisible by mesh {M}"
                    )
                if local == 1:
                    dim_modes[dim] = ([M], [b], [(mesh_dim, 0)])
                else:
                    dim_modes[dim] = ([local, M], [b, b * local], [(mesh_dim, 1)])

        elif isinstance(p, _StridedShard):
            dim = p.dim
            if dim in dim_modes:
                _add_strided_shard_to_dim(
                    dim_modes, dim, mesh_dim, mesh_sizes[mesh_dim], p.split_factor
                )
            else:
                S, b, M, sf = (
                    shape[dim],
                    bstrides[dim],
                    mesh_sizes[mesh_dim],
                    p.split_factor,
                )
                group_size = S // sf
                if group_size % M != 0:
                    raise _UnsupportedCase(
                        f"dim {dim}: group_size {group_size} not divisible by M={M}"
                    )
                lpg = group_size // M
                if lpg == 1:
                    dim_modes[dim] = (
                        [M, sf],
                        [b, b * group_size],
                        [(mesh_dim, 0)],
                    )
                else:
                    dim_modes[dim] = (
                        [lpg, M, sf],
                        [b, b * lpg, b * group_size],
                        [(mesh_dim, 1)],
                    )

    # Assemble layout
    modes: list[Layout] = []
    gpu_modes: list[GpuMode] = []
    flat_idx = 0

    for d in range(len(shape)):
        if d in dim_modes:
            shapes, strides, gpu_info = dim_modes[d]
            if len(shapes) == 1:
                modes.append(Layout(shapes[0], strides[0]))
            else:
                modes.append(Layout(tuple(shapes), tuple(strides)))
            for mesh_dim, sub_idx in gpu_info:
                gpu_modes.append(
                    GpuMode(mesh_dim=mesh_dim, flat_index=flat_idx + sub_idx)
                )
            flat_idx += len(shapes)
        else:
            modes.append(Layout(shape[d], bstrides[d]))
            flat_idx += 1

    if modes:
        L = make_layout(*modes)
    else:
        # Scalar (0-d) tensor
        L = Layout(1, 0)
    return DistLayout(layout=L, num_dims=len(shape), gpu_modes=gpu_modes)


# ---------------------------------------------------------------------------
# View composition
# ---------------------------------------------------------------------------


def compose_view(dist: DistLayout, rule: DimMap) -> DistLayout:
    """Apply a view (reshape) *rule* to a distribution layout."""
    output_modes: list[Layout] = []
    gpu_modes_out: list[GpuMode] = []
    flat_idx_out = 0

    for cmd in rule:
        out_shapes, out_strides, out_gpu = _collect_submodes(cmd, dist)

        if len(out_shapes) == 1:
            output_modes.append(Layout(out_shapes[0], out_strides[0]))
        else:
            output_modes.append(Layout(tuple(out_shapes), tuple(out_strides)))

        for gm_mesh_dim, local_sub_idx in out_gpu:
            gpu_modes_out.append(
                GpuMode(mesh_dim=gm_mesh_dim, flat_index=flat_idx_out + local_sub_idx)
            )

        flat_idx_out += len(out_shapes)

    if output_modes:
        out_layout = make_layout(*output_modes)
    else:
        # Scalar output (e.g. squeeze of a 1-element tensor)
        out_layout = Layout(1, 0)
    return DistLayout(layout=out_layout, num_dims=len(rule), gpu_modes=gpu_modes_out)


def _collect_submodes(
    cmd, dist: DistLayout
) -> tuple[list[int], list[int], list[tuple[int, int]]]:
    """Collect sub-modes from *dist* for one output dim described by *cmd*.

    Returns ``(shapes, strides, [(mesh_dim, local_sub_idx), ...])``.
    """
    if isinstance(cmd, InputDim):
        d = cmd.input_dim
        mode = dist.layout[d]
        shapes = list(
            cute_flatten(mode.shape) if is_tuple(mode.shape) else (mode.shape,)
        )
        strides = list(
            cute_flatten(mode.stride) if is_tuple(mode.stride) else (mode.stride,)
        )
        gpu_info: list[tuple[int, int]] = []
        for gm in dist.gpu_modes:
            dim_idx, sub_idx, _ = _locate_submode(
                dist.layout, dist.num_dims, gm.flat_index
            )
            if dim_idx == d:
                gpu_info.append((gm.mesh_dim, sub_idx))
        return shapes, strides, gpu_info

    elif isinstance(cmd, Flatten):
        all_shapes: list[int] = []
        all_strides: list[int] = []
        gpu_info = []
        for inner_cmd in cmd.input_dims:
            s, st, gi = _collect_submodes(inner_cmd, dist)
            for mesh_dim, sub_idx in gi:
                gpu_info.append((mesh_dim, len(all_shapes) + sub_idx))
            all_shapes.extend(s)
            all_strides.extend(st)
        return all_shapes, all_strides, gpu_info

    elif isinstance(cmd, Split):
        inner_shapes, inner_strides, inner_gpu = _collect_submodes(cmd.input_dim, dist)
        base_stride = min(inner_strides) if inner_strides else 1
        piece_stride = base_stride * math.prod(cmd.group_shape[cmd.split_id + 1 :])
        piece_size = cmd.group_shape[cmd.split_id]
        piece_end_stride = piece_stride * piece_size

        # Divide sub-modes that straddle the piece boundary before filtering.
        # A sub-mode (s, st) with st < piece_stride but s * st > piece_stride
        # spans across multiple Split pieces and must be split into an inner
        # part (stays at lower strides) and an outer part (at piece_stride).
        divided_shapes: list[int] = []
        divided_strides: list[int] = []
        divided_gpu: list[tuple[int, int]] = []
        for i, (s, st) in enumerate(zip(inner_shapes, inner_strides)):
            is_gpu = any(orig == i for _, orig in inner_gpu)
            if st < piece_stride and s * st > piece_stride and not is_gpu:
                divisor = piece_stride // st
                if divisor > 0 and st * divisor == piece_stride and s % divisor == 0:
                    # Inner part: (divisor, st) — below piece_stride
                    divided_shapes.append(divisor)
                    divided_strides.append(st)
                    # Outer part: (s // divisor, piece_stride) — at piece_stride
                    divided_shapes.append(s // divisor)
                    divided_strides.append(piece_stride)
                else:
                    divided_shapes.append(s)
                    divided_strides.append(st)
            else:
                gpu_sub_idx = len(divided_shapes)
                divided_shapes.append(s)
                divided_strides.append(st)
                if is_gpu:
                    for mesh_dim, orig in inner_gpu:
                        if orig == i:
                            divided_gpu.append((mesh_dim, gpu_sub_idx))

        out_shapes: list[int] = []
        out_strides: list[int] = []
        out_gpu: list[tuple[int, int]] = []
        sub_idx_out = 0
        for i, (s, st) in enumerate(zip(divided_shapes, divided_strides)):
            if piece_stride <= st < piece_end_stride:
                out_shapes.append(s)
                out_strides.append(st)
                for mesh_dim, orig_sub_idx in divided_gpu:
                    if orig_sub_idx == i:
                        out_gpu.append((mesh_dim, sub_idx_out))
                sub_idx_out += 1

        if not out_shapes:
            out_shapes = [piece_size]
            out_strides = [piece_stride]
        elif math.prod(out_shapes) != piece_size:
            if out_gpu:
                # A GPU mode falls inside this piece's stride range but the
                # decomposition doesn't match the piece size — the sharding is
                # incompatible with this Split and can't be represented.
                raise _UnsupportedCase(
                    f"GPU mode in piece but prod {math.prod(out_shapes)} != "
                    f"piece_size {piece_size}"
                )
            # Sub-modes straddle the split boundary — the input sharding is
            # incompatible with this view.  Drop GPU modes so this mesh dim
            # becomes Replicate in the output.
            out_shapes = [piece_size]
            out_strides = [piece_stride]
        return out_shapes, out_strides, out_gpu

    elif isinstance(cmd, Singleton):
        return [1], [0], []

    elif isinstance(cmd, Broadcast):
        # Inner dim is singleton; broadcast to dim_size.  Phase 1 prevents
        # GPU modes on broadcast dims, so no GPU info to propagate.
        return [cmd.dim_size], [0], []

    elif isinstance(cmd, NewDim):
        return [cmd.size], [0], []

    elif isinstance(cmd, Repeat):
        # Phase 1 forces Replicate for repeated dims.
        inner_shapes, inner_strides, inner_gpu = _collect_submodes(cmd.input_dim, dist)
        if inner_gpu:
            raise _UnsupportedCase("GPU modes in Repeat dim")
        total = math.prod(inner_shapes) * cmd.times
        base = inner_strides[0] if inner_strides else 0
        return [total], [base], []

    else:
        raise _UnsupportedCase(f"Unsupported DimSpec: {type(cmd)}")


# ---------------------------------------------------------------------------
# Conversion back to placements
# ---------------------------------------------------------------------------


def to_placements(dist: DistLayout, mesh_sizes: tuple[int, ...]) -> list[Placement]:
    """Convert a ``DistLayout`` back to DTensor placements."""
    result: list[Placement] = []

    for mesh_dim in range(len(mesh_sizes)):
        dim_idx = dist.gpu_dim(mesh_dim)
        if dim_idx is None:
            result.append(Replicate())
            continue

        gm = next(g for g in dist.gpu_modes if g.mesh_dim == mesh_dim)
        _, sub_idx, _ = _locate_submode(dist.layout, dist.num_dims, gm.flat_index)
        mode = dist.layout[dim_idx]
        shapes = list(
            cute_flatten(mode.shape) if is_tuple(mode.shape) else (mode.shape,)
        )
        strides = list(
            cute_flatten(mode.stride) if is_tuple(mode.stride) else (mode.stride,)
        )
        gpu_stride = strides[sub_idx]

        # GPU sub-indices for earlier mesh dims (j < mesh_dim) on this dim.
        # Later mesh dims' GPU modes are treated as "local" from mesh_dim's
        # perspective because DTensor processes placements left-to-right:
        # mesh_dim sees the global tensor before later mesh dims partition it.
        earlier_gpu_subs: set[int] = set()
        for other_gm in dist.gpu_modes:
            if other_gm.mesh_dim < mesh_dim:
                other_dim, other_sub, _ = _locate_submode(
                    dist.layout, dist.num_dims, other_gm.flat_index
                )
                if other_dim == dim_idx:
                    earlier_gpu_subs.add(other_sub)

        # split_factor = product of shapes above GPU stride, excluding
        # earlier GPU modes and this mesh dim's own GPU sub-mode
        sf = math.prod(
            shapes[j]
            for j, st in enumerate(strides)
            if st > gpu_stride and j != sub_idx and j not in earlier_gpu_subs
        )

        if sf <= 1:
            result.append(Shard(dim_idx))
        else:
            result.append(_StridedShard(dim_idx, split_factor=sf))

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _find_input_dim(cmd) -> int | None:
    """Return the input_dim if *cmd* is or wraps a single InputDim, else None."""
    if isinstance(cmd, InputDim):
        return cmd.input_dim
    return None


def _symbolic_rewrite_output_placements(
    input_tgt_placements: Sequence[Placement],
    global_input_shape: tuple[int, ...],
    rule: DimMap,
    mesh_sizes: tuple[int, ...],
) -> list[Placement]:
    """Lightweight symbolic-shape path: trace DimMap rules without CuTe layouts.

    Handles Shard and _StridedShard through InputDim, Flatten, and Split rules.
    Falls back to Replicate for patterns that can't be resolved symbolically.
    """
    from torch.fx.experimental.symbolic_shapes import guard_or_false

    output: list[Placement] = [Replicate()] * len(mesh_sizes)
    for mesh_dim, p in enumerate(input_tgt_placements):
        if not isinstance(p, (Shard, _StridedShard)):
            if isinstance(p, Partial):
                output[mesh_dim] = p
            continue
        shard_dim = p.dim
        is_strided = isinstance(p, _StridedShard)
        sf = p.split_factor if is_strided else 1

        for out_idx, cmd in enumerate(rule):
            if isinstance(cmd, InputDim) and cmd.input_dim == shard_dim:
                if is_strided:
                    output[mesh_dim] = _StridedShard(out_idx, split_factor=sf)
                else:
                    output[mesh_dim] = Shard(out_idx)
                break

            if isinstance(cmd, Flatten):
                for i, inner in enumerate(cmd.input_dims):
                    if isinstance(inner, InputDim) and inner.input_dim == shard_dim:
                        if i == 0:
                            if is_strided:
                                output[mesh_dim] = _StridedShard(
                                    out_idx, split_factor=sf
                                )
                            else:
                                output[mesh_dim] = Shard(out_idx)
                        else:
                            preceding = [
                                d
                                for d in cmd.input_dims[:i]
                                if isinstance(d, InputDim)
                            ]
                            new_sf = sf  # 1 for Shard, existing sf for _StridedShard
                            for d in preceding:
                                new_sf *= global_input_shape[d.input_dim]
                            try:
                                new_sf = int(new_sf)
                                output[mesh_dim] = _StridedShard(
                                    out_idx, split_factor=new_sf
                                )
                            except (TypeError, RuntimeError):
                                pass  # stays Replicate
                        break

            if isinstance(cmd, Split):
                inner_dim = _find_input_dim(cmd.input_dim)
                if inner_dim is not None and inner_dim == shard_dim:
                    if is_strided:
                        expected_sf = math.prod(cmd.group_shape[: cmd.split_id])
                        for m in range(mesh_dim):
                            mp = input_tgt_placements[m]
                            if (
                                isinstance(mp, (Shard, _StridedShard))
                                and mp.dim == shard_dim
                                and guard_or_false(expected_sf % mesh_sizes[m] == 0)
                            ):
                                expected_sf //= mesh_sizes[m]
                        if expected_sf == sf:
                            output[mesh_dim] = Shard(out_idx)
                            break
                    elif cmd.split_id == 0:
                        if guard_or_false(
                            cmd.group_shape[0] % mesh_sizes[mesh_dim] == 0
                        ):
                            output[mesh_dim] = Shard(out_idx)
                            break
                if isinstance(cmd.input_dim, Flatten):
                    first = (
                        cmd.input_dim.input_dims[0]
                        if cmd.input_dim.input_dims
                        else None
                    )
                    if (
                        isinstance(first, InputDim)
                        and first.input_dim == shard_dim
                        and not is_strided
                        and cmd.split_id == 0
                    ):
                        if guard_or_false(
                            cmd.group_shape[0] % mesh_sizes[mesh_dim] == 0
                        ):
                            output[mesh_dim] = Shard(out_idx)
                            break

    return output


def _expected_split_factor(
    cmd: Split,
    sharded_dim: int,
    mesh_dim: int,
    placements: Sequence[Placement],
    mesh_sizes: tuple[int, ...],
) -> int | None:
    """Compute residual split factor for *cmd* after earlier mesh dims.

    Mirrors ``_ViewShardingPropagator._expected_split_factor``.
    """
    sf = math.prod(cmd.group_shape[: cmd.split_id])
    for m in range(mesh_dim):
        other_p = placements[m]
        if isinstance(other_p, (Shard, _StridedShard)) and other_p.dim == sharded_dim:
            if sf % mesh_sizes[m] != 0:
                return None
            sf //= mesh_sizes[m]
    return sf


def _trace_multi_mesh_placement(
    p: Placement,
    mesh_dim: int,
    placements: Sequence[Placement],
    mesh_sizes: tuple[int, ...],
    rule: DimMap,
    global_shape: tuple[int, ...],
) -> Placement:
    """Trace a single placement through the rule for multi-mesh-same-dim.

    For Split rules: match _StridedShard's split_factor against the expected
    split_factor at each split_id.  For identity/Flatten: keep the placement
    with adjusted dim index.
    """
    if not isinstance(p, (Shard, _StridedShard)):
        return p

    shard_dim = p.dim
    is_strided = isinstance(p, _StridedShard)
    sf = p.split_factor if is_strided else 1

    # Phase 1: resolve to Shard (exact match)
    for out_idx, cmd in enumerate(rule):
        if isinstance(cmd, InputDim) and cmd.input_dim == shard_dim:
            if is_strided:
                return _StridedShard(out_idx, split_factor=sf)
            return Shard(out_idx)

        if isinstance(cmd, Split):
            inner_dim = _find_input_dim(cmd.input_dim)
            if inner_dim == shard_dim:
                if is_strided:
                    expected_sf = _expected_split_factor(
                        cmd, shard_dim, mesh_dim, placements, mesh_sizes
                    )
                    if expected_sf is not None and expected_sf == sf:
                        return Shard(out_idx)
                elif cmd.split_id == 0:
                    return Shard(out_idx)
            # Split(Flatten(...)): check if shard_dim is in the Flatten
            if isinstance(cmd.input_dim, Flatten):
                flat_dims = [
                    d.input_dim
                    for d in cmd.input_dim.input_dims
                    if isinstance(d, InputDim)
                ]
                if shard_dim in flat_dims:
                    pos = flat_dims.index(shard_dim)
                    if pos == 0 and not is_strided:
                        # First dim in flatten: Shard maps to split_id=0
                        if cmd.split_id == 0:
                            return Shard(out_idx)
                    elif not is_strided:
                        # Non-first dim: compute split_factor from local sizes
                        # of preceding dims, then match against split structure
                        local_sizes = {d: global_shape[d] for d in flat_dims[:pos]}
                        for m in range(mesh_dim):
                            mp = placements[m]
                            if (
                                isinstance(mp, (Shard, _StridedShard))
                                and mp.dim in local_sizes
                            ):
                                local_sizes[mp.dim] //= mesh_sizes[m]
                        local_sf = math.prod(local_sizes.values()) if local_sizes else 1
                        # The shard maps to the split_id where the stride
                        # aligns: prod(group_shape[:k]) matches local_sf
                        # times the local shard dim size
                        expected_split_prefix = math.prod(
                            cmd.group_shape[: cmd.split_id]
                        )
                        # Divide by earlier mesh dims on same dim
                        for m in range(mesh_dim):
                            mp = placements[m]
                            if (
                                isinstance(mp, (Shard, _StridedShard))
                                and mp.dim == shard_dim
                            ):
                                if expected_split_prefix % mesh_sizes[m] == 0:
                                    expected_split_prefix //= mesh_sizes[m]
                        if expected_split_prefix == local_sf:
                            return Shard(out_idx)

        if isinstance(cmd, Flatten):
            for i, inner in enumerate(cmd.input_dims):
                if isinstance(inner, InputDim) and inner.input_dim == shard_dim:
                    if i == 0:
                        if is_strided:
                            return _StridedShard(out_idx, split_factor=sf)
                        return Shard(out_idx)
                    else:
                        preceding_dims = [
                            d.input_dim
                            for d in cmd.input_dims[:i]
                            if isinstance(d, InputDim)
                        ]
                        local_sizes = {d: global_shape[d] for d in preceding_dims}
                        for m in range(mesh_dim):
                            mp = placements[m]
                            if (
                                isinstance(mp, (Shard, _StridedShard))
                                and mp.dim in local_sizes
                            ):
                                local_sizes[mp.dim] //= mesh_sizes[m]
                        split_factor = (
                            math.prod(local_sizes.values()) if local_sizes else 1
                        )
                        return _StridedShard(out_idx, split_factor=split_factor)

    # Phase 2: keep _StridedShard on a compatible Split output dim
    if is_strided:
        total_shard = mesh_sizes[mesh_dim] * sf
        if global_shape[shard_dim] % total_shard == 0:
            shard_size = global_shape[shard_dim] // total_shard
            for out_idx, cmd in enumerate(rule):
                if isinstance(cmd, Split):
                    inner_dim = _find_input_dim(cmd.input_dim)
                    if inner_dim == shard_dim:
                        inner_size = math.prod(cmd.group_shape[cmd.split_id + 1 :])
                        if shard_size >= inner_size and shard_size % inner_size == 0:
                            return _StridedShard(out_idx, split_factor=sf)
                    elif isinstance(cmd.input_dim, Flatten):
                        flat_dims = [
                            d.input_dim
                            for d in cmd.input_dim.input_dims
                            if isinstance(d, InputDim)
                        ]
                        if shard_dim in flat_dims:
                            pos = flat_dims.index(shard_dim)
                            inner_size = math.prod(cmd.group_shape[cmd.split_id + 1 :])
                            trailing_size = math.prod(
                                global_shape[d] for d in flat_dims[pos + 1 :]
                            )
                            flattened_shard_size = shard_size * trailing_size
                            if (
                                flattened_shard_size >= inner_size
                                and flattened_shard_size % inner_size == 0
                            ):
                                return _StridedShard(out_idx, split_factor=sf)

    return Replicate()


def _rewrite_per_mesh_dim(
    input_tgt_placements: Sequence[Placement],
    global_input_shape: tuple[int, ...],
    rule: DimMap,
    mesh_sizes: tuple[int, ...],
) -> list[Placement]:
    """Process each mesh dim independently: try CuTe, fall back to rule-trace.

    For multi-mesh-same-dim cases (all shards on the same tensor dim),
    per-mesh-dim CuTe with a 1-element mesh produces correct placements.
    For cases where CuTe layout construction fails (uneven sharding),
    falls back to rule-tracing for that specific mesh dim.
    """
    output: list[Placement] = [Replicate()] * len(mesh_sizes)
    for mesh_dim, p in enumerate(input_tgt_placements):
        if isinstance(p, (Shard, _StridedShard)):
            try:
                dist = from_placements(global_input_shape, [p], (mesh_sizes[mesh_dim],))
                out_dist = compose_view(dist, rule)
                out_plc = to_placements(out_dist, (mesh_sizes[mesh_dim],))
                output[mesh_dim] = out_plc[0]
            except _UnsupportedCase:
                output[mesh_dim] = _trace_multi_mesh_placement(
                    p,
                    mesh_dim,
                    input_tgt_placements,
                    mesh_sizes,
                    rule,
                    global_input_shape,
                )
        elif isinstance(p, Partial):
            output[mesh_dim] = p
    return output


def _validate_flatten_uneven_sharding(
    input_tgt_placements: Sequence[Placement],
    global_input_shape: tuple[int, ...],
    rule: DimMap,
    mesh_sizes: tuple[int, ...],
) -> None:
    """Reject uneven sharding on non-last flatten dims.

    Uneven sharding makes local shapes vary across ranks, breaking the
    uniform-stride assumption that _StridedShard relies on.
    """
    flatten_ranges: list[tuple[int, int]] = []
    for cmd in rule:
        if isinstance(cmd, Flatten):
            dims = [d.input_dim for d in cmd.input_dims if isinstance(d, InputDim)]
            if len(dims) >= 2:
                flatten_ranges.append((min(dims), max(dims) + 1))
    for start, end in flatten_ranges:
        local_shapes = list(global_input_shape)
        for mesh_dim, p in enumerate(input_tgt_placements):
            if not isinstance(p, (Shard, _StridedShard)):
                continue
            if not (start <= p.dim < end):
                continue
            if local_shapes[p.dim] % mesh_sizes[mesh_dim] != 0:
                has_later = any(
                    isinstance(q, (Shard, _StridedShard))
                    and start <= q.dim < end
                    and q.dim >= p.dim
                    for q in input_tgt_placements[mesh_dim + 1 :]
                )
                if has_later:
                    raise RuntimeError(
                        f"Cannot flatten unevenly sharded tensor: "
                        f"dimension {p.dim} (size {local_shapes[p.dim]}) "
                        f"is not evenly divisible by mesh dimension "
                        f"{mesh_dim} (size {mesh_sizes[mesh_dim]}). "
                        f"Please redistribute the tensor before this operation."
                    )
            local_shapes[p.dim] //= mesh_sizes[mesh_dim]


def cute_rewrite_output_placements(
    input_tgt_placements: Sequence[Placement],
    global_input_shape: tuple[int, ...],
    rule: DimMap,
    mesh_sizes: tuple[int, ...],
) -> list[Placement]:
    """Compute output placements via CuTe layout composition.

    Architecture:
    - Symbolic shapes → lightweight rule-tracing (no CuTe layouts)
    - Multi-mesh same-dim → per-mesh-dim CuTe with 1-element mesh
    - Single-mesh-per-dim → full CuTe pipeline (preserves cross-dim interactions)
    - Any _UnsupportedCase → per-mesh-dim rule-tracing fallback
    """
    if any(not isinstance(s, int) for s in global_input_shape):
        return _symbolic_rewrite_output_placements(
            input_tgt_placements, global_input_shape, rule, mesh_sizes
        )

    _validate_flatten_uneven_sharding(
        input_tgt_placements, global_input_shape, rule, mesh_sizes
    )

    shard_dims: dict[int, list[int]] = {}
    for mesh_dim, p in enumerate(input_tgt_placements):
        if isinstance(p, (Shard, _StridedShard)):
            shard_dims.setdefault(p.dim, []).append(mesh_dim)
    multi_mesh_same_dim = any(len(v) > 1 for v in shard_dims.values())

    if multi_mesh_same_dim:
        output = _rewrite_per_mesh_dim(
            input_tgt_placements, global_input_shape, rule, mesh_sizes
        )
    else:
        try:
            dist = from_placements(global_input_shape, input_tgt_placements, mesh_sizes)
            output_dist = compose_view(dist, rule)
            output = to_placements(output_dist, mesh_sizes)
        except _UnsupportedCase:
            output = _rewrite_per_mesh_dim(
                input_tgt_placements, global_input_shape, rule, mesh_sizes
            )

    for i, p in enumerate(input_tgt_placements):
        if isinstance(p, Partial):
            output[i] = p

    return output
