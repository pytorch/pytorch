# CuTe Layout Composition for DTensor View Ops

## Overview

`_cute_view_propagation.py` replaces Phase 2 (`rewrite_output_placements`) of
`_ViewShardingPropagator` with CuTe layout composition. Where Phase 2 threads
mutable state (`local_tensor_shapes`, `strided_shard_claimed_dims`) sequentially
across mesh dims, CuTe encodes cross-dim dependencies in the layout's stride
structure and computes output placements in one pass via view composition.

The module is gated behind `USE_CUTE_VIEW_PROPAGATION=1` (env var, off by
default). When enabled, `propagate_shape_and_sharding` tries CuTe first and
falls back to Phase 2 for unsupported cases.

## Data model

A `DistLayout` wraps a CuTe `Layout` with metadata tracking which sub-modes
correspond to mesh dimensions (GPU modes):

```
DistLayout
├── layout: Layout       # Hierarchical CuTe layout over all tensor dims
├── num_dims: int         # Tensor dimensionality
└── gpu_modes: [GpuMode]  # Which sub-modes are mesh-dim partitions
        ├── mesh_dim: int
        └── flat_index: int  # Position in flattened sub-mode list
```

Each tensor dimension's mode decomposes into sub-modes. A `Shard(d)` on mesh
dim `i` with mesh size `M` makes dim `d`'s mode `(S/M, M)` with the `M`
sub-mode marked as GPU. A `_StridedShard(d, sf)` produces `(lpg, M, sf)` where
`lpg = S/(sf*M)`.

## Pipeline

```
from_placements ──► compose_view ──► to_placements
   (input DTensor      (apply DimMap      (convert back
    placements to       rule: Split,       to Shard /
    DistLayout)         Flatten, etc.)     _StridedShard /
                                           Replicate)
```

**`from_placements`**: Builds a `DistLayout` by decomposing each sharded dim
into sub-modes. Processes placements left-to-right (matching DTensor semantics).
When multiple mesh dims shard the same tensor dim, `_add_shard_to_dim` /
`_add_strided_shard_to_dim` insert additional GPU sub-modes into the existing
decomposition.

**`compose_view`**: Walks the DimMap rule tree. For each output dim, collects
the sub-modes that fall within its stride range from the input layout. GPU
sub-modes are carried through, so the output layout inherits the right mesh-dim
assignments.

**`to_placements`**: For each mesh dim, finds its GPU sub-mode in the output
layout. If it's the outermost sub-mode in its dim → `Shard`; if inner →
`_StridedShard` (with `split_factor` derived from the product of outer
sub-mode shapes); if absent → `Replicate`.

## Fallback conditions

CuTe returns `None` (falling back to Phase 2) when:

| Condition | Reason |
|---|---|
| Any shape element is `SymInt` | CuTe uses plain arithmetic |
| Uneven sharding in flatten range with later mesh dim | Local shapes vary across ranks, breaking stride uniformity |
| `_StridedShard` `group_size % M != 0` | Layout requires non-integer sub-mode division |
| Split factor exceeds outer sub-mode shape | Strides from different mesh dims are incompatible |
| GPU mode in Split piece with mismatched product | Sharding doesn't align with unflatten boundaries |

All fallbacks are via `_UnsupportedCase` caught in `cute_rewrite_output_placements`.

## Split sub-mode division

When unflatten splits a dim into pieces, each piece occupies a stride range.
A local sub-mode may straddle a piece boundary — e.g., `(24, stride=1)` when
the piece boundary is at stride 12. The Split handler pre-divides straddling
non-GPU sub-modes:

```
(24, stride=1) with piece_stride=12
→ inner: (12, stride=1)   # below boundary
  outer: (2, stride=12)   # at boundary
```

GPU sub-modes that straddle a boundary indicate incompatible sharding and
trigger `_UnsupportedCase`.

## Scalar tensors

0-d tensors (scalars) produce empty mode lists. Both `from_placements` and
`compose_view` handle this by returning `Layout(1, 0)` — a trivial layout with
no GPU modes, correctly yielding all-Replicate output.

## Testing

```bash
# With CuTe enabled (exercises CuTe + fallback for all view ops)
USE_CUTE_VIEW_PROPAGATION=1 python -m pytest test/distributed/tensor/test_view_ops.py -v

# Without CuTe (baseline, should be identical)
python -m pytest test/distributed/tensor/test_view_ops.py -v
```
