"""
Prototype: CuTe layout composition for DTensor view ops sharding propagation.

CuTe layouts naturally represent what DTensor encodes as Shard and _StridedShard:
a GPU sub-mode at some position in a dim's hierarchical layout. There is no
"Shard vs _StridedShard" distinction — both are just GPU modes with different
strides. View composition preserves GPU modes through shape changes by
partitioning sub-modes across output dims.
"""

import math
import unittest
from dataclasses import dataclass, field

from torch.distributed._pycute import (
    flatten as cute_flatten,
    is_tuple,
    Layout,
    logical_divide,
    make_layout,
    suffix_product,
)
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.distributed.tensor._ops._view_ops import view_groups
from torch.testing._internal.common_utils import run_tests, TestCase


# ---------------------------------------------------------------------------
# Distribution Layout: CuTe layout + GPU mode metadata
# ---------------------------------------------------------------------------

@dataclass
class GpuMode:
    """Identifies a GPU sub-mode within a flattened layout."""
    mesh_dim: int
    flat_index: int


@dataclass
class DistLayout:
    """
    Tensor distribution as a CuTe layout with GPU mode tags.

    The layout maps tensor coordinates to flat element indices.
    Sharded dims have hierarchical sub-modes; gpu_modes tracks which
    sub-modes correspond to mesh dimensions.
    """
    layout: Layout
    num_dims: int
    gpu_modes: list[GpuMode] = field(default_factory=list)

    def gpu_dim(self, mesh_dim: int) -> int | None:
        """Which tensor dim contains the GPU mode for this mesh dim, or None if replicated."""
        for gm in self.gpu_modes:
            if gm.mesh_dim == mesh_dim:
                dim_idx, _, _ = _locate_submode(self.layout, self.num_dims, gm.flat_index)
                return dim_idx
        return None

    def gpu_mode_shape(self, mesh_dim: int) -> int | None:
        """Size of the GPU mode for this mesh dim (should equal mesh_size)."""
        for gm in self.gpu_modes:
            if gm.mesh_dim == mesh_dim:
                dim_idx, sub_idx, _ = _locate_submode(self.layout, self.num_dims, gm.flat_index)
                mode = self.layout[dim_idx]
                shapes = cute_flatten(mode.shape) if is_tuple(mode.shape) else (mode.shape,)
                return shapes[sub_idx]
        return None

    def local_contiguous(self, mesh_dim: int) -> bool:
        """
        True if local chunks are contiguous (GPU mode is outermost in its dim).

        This is what DTensor calls Shard (True) vs _StridedShard (False).
        CuTe doesn't need this distinction — it's just a stride property.
        """
        for gm in self.gpu_modes:
            if gm.mesh_dim == mesh_dim:
                dim_idx, sub_idx, _ = _locate_submode(self.layout, self.num_dims, gm.flat_index)
                mode = self.layout[dim_idx]
                strides = cute_flatten(mode.stride) if is_tuple(mode.stride) else (mode.stride,)
                gpu_stride = strides[sub_idx]
                # Collect other GPU sub-indices in this dim
                gpu_sub_indices = set()
                for other_gm in self.gpu_modes:
                    other_dim, other_sub, _ = _locate_submode(
                        self.layout, self.num_dims, other_gm.flat_index
                    )
                    if other_dim == dim_idx:
                        gpu_sub_indices.add(other_sub)
                # Contiguous iff no local sub-mode has larger stride
                return all(
                    st <= gpu_stride or j in gpu_sub_indices
                    for j, st in enumerate(strides)
                )
        return True  # replicated — vacuously contiguous

    def __repr__(self):
        return (
            f"DistLayout(layout={self.layout}, "
            f"num_dims={self.num_dims}, gpu_modes={self.gpu_modes})"
        )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def from_placements(
    shape: tuple[int, ...],
    placements: list,
    mesh_sizes: tuple[int, ...],
) -> DistLayout:
    """
    Build a DistLayout from Shard/Replicate placements.

    For Shard(d) on mesh dim j with mesh size M:
      logical_divide splits dim d into (local_size, M) sub-modes.
      The GPU mode is the complement (outer sub-mode, index 1).
    """
    L = Layout(shape, suffix_product(shape))
    gpu_modes: list[GpuMode] = []

    for mesh_dim, p in enumerate(placements):
        if isinstance(p, Shard):
            dim = p.dim
            M = mesh_sizes[mesh_dim]
            local_size = shape[dim] // M
            divider = tuple(
                local_size if i == dim else None
                for i in range(len(L))
            )
            L = logical_divide(L, divider)
            flat_idx = _flat_index_of_submode(L, dim, 1)
            gpu_modes.append(GpuMode(mesh_dim=mesh_dim, flat_index=flat_idx))
        elif not isinstance(p, Replicate):
            raise ValueError(f"Only Shard and Replicate supported, got {type(p)}")

    return DistLayout(layout=L, num_dims=len(shape), gpu_modes=gpu_modes)


def _flat_index_of_submode(layout: Layout, dim: int, sub_idx: int) -> int:
    """Flat index of a sub-mode: counts all sub-modes across dims 0..dim-1, then adds sub_idx."""
    flat_idx = 0
    for d in range(dim):
        mode = layout[d]
        flat_idx += len(cute_flatten(mode.shape)) if is_tuple(mode.shape) else 1
    return flat_idx + sub_idx


def _locate_submode(
    layout: Layout, num_dims: int, flat_index: int
) -> tuple[int, int, int]:
    """Given a flat sub-mode index, return (dim_idx, sub_idx_within_dim, dim_num_submodes)."""
    running = 0
    for d in range(num_dims):
        mode = layout[d]
        n = len(cute_flatten(mode.shape)) if is_tuple(mode.shape) else 1
        if flat_index < running + n:
            return d, flat_index - running, n
        running += n
    raise ValueError(f"flat_index {flat_index} out of range")


# ---------------------------------------------------------------------------
# View composition
# ---------------------------------------------------------------------------

def compose_view(
    dist: DistLayout,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> DistLayout:
    """
    Apply a view (reshape) to a distribution layout.

    A view doesn't change data — the distribution on the flat index space is
    invariant. We route sub-modes from input dims to output dims based on the
    DimMap rule from view_groups.
    """
    rule = view_groups(list(input_shape), list(output_shape))

    output_modes = []
    gpu_modes_out = []
    flat_idx_out = 0

    for out_dim, cmd in enumerate(rule):
        out_shapes, out_strides, out_gpu = _collect_submodes_for_output_dim(
            cmd, dist, input_shape, output_shape[out_dim]
        )

        if len(out_shapes) == 1:
            output_modes.append(Layout(out_shapes[0], out_strides[0]))
        else:
            output_modes.append(Layout(tuple(out_shapes), tuple(out_strides)))

        for gm_mesh_dim, local_sub_idx in out_gpu:
            gpu_modes_out.append(GpuMode(
                mesh_dim=gm_mesh_dim,
                flat_index=flat_idx_out + local_sub_idx,
            ))

        flat_idx_out += len(out_shapes)

    out_layout = make_layout(*output_modes)
    return DistLayout(
        layout=out_layout, num_dims=len(output_shape), gpu_modes=gpu_modes_out
    )


def _collect_submodes_for_output_dim(cmd, dist, input_shape, out_dim_size):
    """
    Collect sub-modes from the input distribution for one output dim.

    Returns (shapes, strides, [(mesh_dim, local_sub_idx), ...])
    """
    from torch.distributed.tensor._ops._view_ops import InputDim, Flatten, Split, Singleton

    if isinstance(cmd, InputDim):
        d = cmd.input_dim
        mode = dist.layout[d]
        shapes = list(cute_flatten(mode.shape) if is_tuple(mode.shape) else (mode.shape,))
        strides = list(cute_flatten(mode.stride) if is_tuple(mode.stride) else (mode.stride,))

        gpu_info = []
        for gm in dist.gpu_modes:
            dim_idx, sub_idx, _ = _locate_submode(dist.layout, dist.num_dims, gm.flat_index)
            if dim_idx == d:
                gpu_info.append((gm.mesh_dim, sub_idx))

        return shapes, strides, gpu_info

    elif isinstance(cmd, Flatten):
        all_shapes = []
        all_strides = []
        gpu_info = []

        for inner_cmd in cmd.input_dims:
            if isinstance(inner_cmd, InputDim):
                d = inner_cmd.input_dim
                mode = dist.layout[d]
                shapes = cute_flatten(mode.shape) if is_tuple(mode.shape) else (mode.shape,)
                strides = cute_flatten(mode.stride) if is_tuple(mode.stride) else (mode.stride,)

                for gm in dist.gpu_modes:
                    dim_idx, sub_idx, _ = _locate_submode(
                        dist.layout, dist.num_dims, gm.flat_index
                    )
                    if dim_idx == d:
                        gpu_info.append((gm.mesh_dim, len(all_shapes) + sub_idx))

                all_shapes.extend(shapes)
                all_strides.extend(strides)

        return list(all_shapes), list(all_strides), gpu_info

    elif isinstance(cmd, Split):
        # Partition inner sub-modes by stride range.
        # Piece at split_id i has logical stride base * prod(group_shape[i+1:]).
        # Sub-modes whose stride falls in [piece_stride, piece_stride * piece_size)
        # belong to that piece.
        inner_shapes, inner_strides, inner_gpu = _collect_submodes_for_output_dim(
            cmd.input_dim, dist, input_shape, math.prod(cmd.group_shape)
        )

        base_stride = min(inner_strides) if inner_strides else 1
        piece_size = cmd.group_shape[cmd.split_id]
        piece_stride = base_stride * math.prod(cmd.group_shape[cmd.split_id + 1:])
        piece_end_stride = piece_stride * piece_size

        out_shapes = []
        out_strides = []
        out_gpu = []
        sub_idx_out = 0

        for i, (s, st) in enumerate(zip(inner_shapes, inner_strides)):
            if st >= piece_stride and st < piece_end_stride:
                out_shapes.append(s)
                out_strides.append(st)
                for mesh_dim, orig_sub_idx in inner_gpu:
                    if orig_sub_idx == i:
                        out_gpu.append((mesh_dim, sub_idx_out))
                sub_idx_out += 1

        if not out_shapes:
            out_shapes = [piece_size]
            out_strides = [piece_stride]

        return out_shapes, out_strides, out_gpu

    elif isinstance(cmd, Singleton):
        return [1], [0], []

    else:
        raise NotImplementedError(f"Unsupported DimSpec: {type(cmd)}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDistLayout(TestCase):
    """Test distribution layout construction."""

    def test_shard_dim0(self):
        """Shard(0) on shape (8, 6), mesh 4: dim 0 splits into (local=2, gpu=4)."""
        dist = from_placements((8, 6), [Shard(0)], (4,))

        mode0 = dist.layout[0]
        self.assertEqual(cute_flatten(mode0.shape), (2, 4))

        self.assertEqual(dist.gpu_dim(0), 0)
        self.assertEqual(dist.gpu_mode_shape(0), 4)
        self.assertTrue(dist.local_contiguous(0))

    def test_shard_dim1(self):
        """Shard(1) on shape (4, 6), mesh 3: dim 1 splits into (local=2, gpu=3)."""
        dist = from_placements((4, 6), [Shard(1)], (3,))

        mode1 = dist.layout[1]
        self.assertEqual(cute_flatten(mode1.shape), (2, 3))

        self.assertEqual(dist.gpu_dim(0), 1)
        self.assertTrue(dist.local_contiguous(0))

    def test_replicate(self):
        """Replicate: no GPU modes."""
        dist = from_placements((4, 6), [Replicate()], (2,))
        self.assertIsNone(dist.gpu_dim(0))
        self.assertEqual(len(dist.gpu_modes), 0)

    def test_2d_mesh(self):
        """[Shard(0), Shard(1)] on shape (6, 4), mesh (3, 2)."""
        dist = from_placements((6, 4), [Shard(0), Shard(1)], (3, 2))
        self.assertEqual(dist.gpu_dim(0), 0)
        self.assertEqual(dist.gpu_dim(1), 1)
        self.assertTrue(dist.local_contiguous(0))
        self.assertTrue(dist.local_contiguous(1))

    def test_rejects_strided_shard(self):
        """from_placements only accepts Shard and Replicate."""
        from torch.distributed.tensor.placement_types import _StridedShard
        with self.assertRaises(ValueError):
            from_placements((24,), [_StridedShard(0, split_factor=4)], (3,))


class TestComposeView(TestCase):
    """Test CuTe composition for view ops."""

    # -- Flatten --

    def test_flatten_shard_first(self):
        """(4, 6) Shard(0) → (24,): GPU outermost → contiguous."""
        dist = from_placements((4, 6), [Shard(0)], (4,))
        out = compose_view(dist, (4, 6), (24,))
        self.assertEqual(out.gpu_dim(0), 0)
        self.assertTrue(out.local_contiguous(0))

    def test_flatten_shard_last(self):
        """(4, 6) Shard(1) → (24,): GPU becomes inner → strided."""
        dist = from_placements((4, 6), [Shard(1)], (3,))
        out = compose_view(dist, (4, 6), (24,))
        self.assertEqual(out.gpu_dim(0), 0)
        self.assertFalse(out.local_contiguous(0))

    def test_flatten_3d_shard_last(self):
        """(2, 3, 4) Shard(2) → (24,): innermost dim, strided."""
        dist = from_placements((2, 3, 4), [Shard(2)], (4,))
        out = compose_view(dist, (2, 3, 4), (24,))
        self.assertEqual(out.gpu_dim(0), 0)
        self.assertFalse(out.local_contiguous(0))

    def test_flatten_3d_shard_middle(self):
        """(2, 3, 4) Shard(1) → (24,): middle dim, strided."""
        dist = from_placements((2, 3, 4), [Shard(1)], (3,))
        out = compose_view(dist, (2, 3, 4), (24,))
        self.assertEqual(out.gpu_dim(0), 0)
        self.assertFalse(out.local_contiguous(0))

    def test_flatten_replicate(self):
        """(4, 6) Replicate → (24,): stays replicated."""
        dist = from_placements((4, 6), [Replicate()], (3,))
        out = compose_view(dist, (4, 6), (24,))
        self.assertIsNone(out.gpu_dim(0))

    # -- Unflatten (expressed as compose chains from Shard) --

    def test_unflatten_resolves_to_dim1(self):
        """(4, 6) Shard(1) → (24,) → (4, 6): GPU back in dim 1, contiguous."""
        dist = from_placements((4, 6), [Shard(1)], (3,))
        flat = compose_view(dist, (4, 6), (24,))
        out = compose_view(flat, (24,), (4, 6))
        self.assertEqual(out.gpu_dim(0), 1)
        self.assertTrue(out.local_contiguous(0))

    def test_unflatten_shard_first(self):
        """(24,) Shard(0) → (4, 6): GPU stays in dim 0, contiguous."""
        dist = from_placements((24,), [Shard(0)], (4,))
        out = compose_view(dist, (24,), (4, 6))
        self.assertEqual(out.gpu_dim(0), 0)
        self.assertTrue(out.local_contiguous(0))

    def test_unflatten_shard_to_first_dim(self):
        """(24,) Shard(0) → (6, 4): GPU stays in dim 0, contiguous."""
        dist = from_placements((24,), [Shard(0)], (6,))
        out = compose_view(dist, (24,), (6, 4))
        self.assertEqual(out.gpu_dim(0), 0)
        self.assertTrue(out.local_contiguous(0))

    def test_unflatten_to_3d(self):
        """(2, 12) Shard(1) → (24,) → (2, 3, 4): GPU lands in dim 1."""
        dist = from_placements((2, 12), [Shard(1)], (3,))
        flat = compose_view(dist, (2, 12), (24,))
        out = compose_view(flat, (24,), (2, 3, 4))
        self.assertEqual(out.gpu_dim(0), 1)
        self.assertTrue(out.local_contiguous(0))

    # -- Partial flatten/unflatten --

    def test_partial_flatten(self):
        """(2, 3, 4) Shard(0) → (2, 12): GPU stays in dim 0."""
        dist = from_placements((2, 3, 4), [Shard(0)], (2,))
        out = compose_view(dist, (2, 3, 4), (2, 12))
        self.assertEqual(out.gpu_dim(0), 0)
        self.assertTrue(out.local_contiguous(0))

    def test_partial_flatten_shard_inner(self):
        """(2, 3, 4) Shard(2) → (2, 12): GPU in dim 1, strided."""
        dist = from_placements((2, 3, 4), [Shard(2)], (4,))
        out = compose_view(dist, (2, 3, 4), (2, 12))
        self.assertEqual(out.gpu_dim(0), 1)
        self.assertFalse(out.local_contiguous(0))

    def test_partial_unflatten(self):
        """(2, 12) Shard(0) → (2, 3, 4): GPU stays in dim 0."""
        dist = from_placements((2, 12), [Shard(0)], (2,))
        out = compose_view(dist, (2, 12), (2, 3, 4))
        self.assertEqual(out.gpu_dim(0), 0)
        self.assertTrue(out.local_contiguous(0))

    # -- Reshape (split + merge) --

    def test_reshape(self):
        """(6, 4) Shard(0) → (3, 8): GPU in dim 0, contiguous."""
        dist = from_placements((6, 4), [Shard(0)], (3,))
        out = compose_view(dist, (6, 4), (3, 8))
        self.assertEqual(out.gpu_dim(0), 0)
        self.assertTrue(out.local_contiguous(0))

    def test_identity_view(self):
        """(4, 6) Shard(1) → (4, 6): GPU unchanged in dim 1."""
        dist = from_placements((4, 6), [Shard(1)], (3,))
        out = compose_view(dist, (4, 6), (4, 6))
        self.assertEqual(out.gpu_dim(0), 1)
        self.assertTrue(out.local_contiguous(0))

    # -- Multi-mesh --

    def test_2d_mesh_flatten(self):
        """(6, 4) [Shard(0), Shard(1)] → (24,): both GPU modes in dim 0."""
        dist = from_placements((6, 4), [Shard(0), Shard(1)], (3, 2))
        out = compose_view(dist, (6, 4), (24,))
        self.assertEqual(out.gpu_dim(0), 0)  # mesh dim 0
        self.assertEqual(out.gpu_dim(1), 0)  # mesh dim 1
        # Mesh dim 0 is outermost → contiguous; mesh dim 1 is inner → strided
        self.assertTrue(out.local_contiguous(0))
        self.assertFalse(out.local_contiguous(1))

    # -- Round-trips --

    def test_flatten_unflatten_roundtrip(self):
        """(4, 6) Shard(1) → (24,) → (4, 6): full round-trip preserves distribution."""
        original = from_placements((4, 6), [Shard(1)], (3,))
        flat = compose_view(original, (4, 6), (24,))
        restored = compose_view(flat, (24,), (4, 6))

        # GPU mode back in same dim, same contiguity
        self.assertEqual(restored.gpu_dim(0), original.gpu_dim(0))
        self.assertEqual(restored.local_contiguous(0), original.local_contiguous(0))
        self.assertEqual(restored.gpu_mode_shape(0), original.gpu_mode_shape(0))

    def test_3d_roundtrip(self):
        """(2, 3, 4) Shard(1) → (24,) → (2, 3, 4): preserves distribution."""
        original = from_placements((2, 3, 4), [Shard(1)], (3,))
        flat = compose_view(original, (2, 3, 4), (24,))
        restored = compose_view(flat, (24,), (2, 3, 4))
        self.assertEqual(restored.gpu_dim(0), 1)
        self.assertTrue(restored.local_contiguous(0))


if __name__ == "__main__":
    run_tests()
