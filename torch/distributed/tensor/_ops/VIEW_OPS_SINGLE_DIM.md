# View Ops: Single-Dim Sharding Propagation

## Problem

`_ViewShardingPropagator` processes all mesh dims together with cross-mesh-dim
coupling:

- `local_tensor_shapes` progressive division — mesh dim j divides before j+1
  reads
- `_expected_split_factor()` — iterates earlier mesh dims' placements
- `strided_shard_claimed_dims` — cross-mesh-dim output dim claims
- `_is_last_shard_in_flatten_range()` — reads later mesh dims
- `matched_strided_mesh_dims` in `_analyze_split()` — cross-mesh-dim tracking

Single-dim strategy (`single_dim_strategy.py`) requires each mesh dim to be
processable independently: `(placement, mesh_dim_size) → output_placement`.

## Solution

Constructing the propagator with `mesh_sizes=(mesh_dim_size,)` makes all
cross-mesh-dim interactions naturally vacuous:

- Progressive division: only one mesh dim, so no prior division
- `_expected_split_factor()`: iterates zero earlier placements
- `strided_shard_claimed_dims`: only one mesh dim can claim
- `_is_last_shard_in_flatten_range()`: vacuously true with one mesh dim

This means **no rewrite methods need to change** — existing
`_rewrite_plain_shard`, `_rewrite_split_flatten_shard`, and
`_rewrite_strided_shard` work correctly when the propagator sees a 1D mesh.

## API

```python
from torch.distributed.tensor._ops._view_ops import propagate_single_dim

inp_tgt, out_plc = propagate_single_dim(
    placement=Shard(0),
    global_input_shape=(2, 3, 4),
    rule=view_groups((2, 3, 4), (6, 4)),
    mesh_dim_size=2,
)
```

## Migration status

All view ops now use `register_single_dim_view_strategy`. The old
`register_op_strategy_map` has been deleted.

## strict_view and reject_redistribution

`aten.view.default` and `aten._unsafe_view.default` use `strict_view=True`,
which forbids communication. At strategy enumeration time, `_smallest_factor`
mesh sizes guarantee divisibility, so the propagator doesn't see the real mesh.
At expansion time, a `reject_redistribution` callback re-runs
`propagate_shape_and_sharding` with the actual mesh sizes across all mesh dims
simultaneously, preserving cross-mesh-dim validation (e.g. SS double-shard,
progressive local_tensor_shapes division). On success it returns a zero-cost
`OpStrategy`; on failure it raises `RuntimeError`.

## Inplace view ops

View ops are metadata-only, so inplace variants (e.g. `squeeze_`) are
registered with `is_view_op=True`. This skips the inplace placement
constraint in `expand_to_full_mesh_op_strategy` that would otherwise reject
valid dim-index shifts (e.g. `Shard(1) → Shard(0)` after squeezing dim 0).

## Refactoring

`_build_input_to_output_map(rule)` was extracted from `analyze()` as a
mesh-free structural mapping `{input_dim: [output_dims]}`. This separates
structural dim tracking (which input dims map to which output dims) from
shardability analysis (which dims can be sharded on which mesh dims).

## Semantic difference: single-dim vs multi-dim

For 2D mesh where both mesh dims shard the same input dim, single-dim produces
different results from multi-dim. This is intentional — the caller composes
independent per-mesh-dim results.
