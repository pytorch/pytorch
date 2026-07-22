# Partial RNG Support Plan

## Current Conclusion

`CheckpointableTensor` is sufficient for RNG geometry on a size-one
`Partial`, but not for general `Partial` semantics. It does not encode the
reduction type or mesh cardinality, and its attributes do not survive
`to_empty()`. Keep the protocol unchanged.

## Plan

1. Define singleton semantics: `Partial` on a mesh dimension of size one is
   numerically identical to `Replicate`.

2. Add `Partial -> Partial` strategies for `normal_()` and `uniform_()`, guarded
   by a full-mesh filter requiring every `Partial` dimension to have size one.
   Multi-rank `Partial` must remain rejected before generator advancement.

3. In the DTensor RNG layout mapper, treat an eligible singleton `Partial` as
   `Replicate`: use the full logical shape, zero offset, and the same dense
   Philox draw. Preserve the outward `Partial` placement.

4. Test the actual meta workflow: meta DTensor parameter -> `to_empty()` ->
   `normal_()` or `reset_parameters()`. Verify dense-equivalent values,
   placement retention, default and explicit generator state, the next draw,
   invalid parameters, and zero collectives.

5. Add a negative multi-rank test proving active `Partial` remains unsupported.
   The base [random strategy](torch/distributed/tensor/_ops/_random_ops.py#L16)
   still defines random sampling on active partial tensors as undefined.

6. Handle multi-rank `Partial` separately. A defensible contract is:

   ```text
   initialize Partial == distribute_tensor(initialize dense, same placements)
   ```

   Replay the same dense RNG values, then apply
   [`Partial._partition_value()`](torch/distributed/tensor/placement_types.py#L1794)
   for each partial mesh dimension. For `sum`, this divides by mesh size;
   `avg`, `min`, and `max` are unchanged. Reject unsupported reductions before
   reserving RNG state.

## Multi-Rank Numerical Contract

For `Partial("sum")`, dense-then-partition matches `distribute_tensor`'s local
numerics but may not reconstruct the dense tensor bit-for-bit after reduction.
If bitwise equality is required, an owner-plus-zero contribution policy needs
an explicit design decision.

## Pre-Change Behavior

The existing meta path already preserves `Partial` through `to_empty()`, as
covered by the [meta initialization test](test/distributed/tensor/test_tensor_ops.py#L339).
Before the singleton strategy was added, `normal_()` failed even on a one-device
`Partial` mesh because no valid strategy preserved the placement.
