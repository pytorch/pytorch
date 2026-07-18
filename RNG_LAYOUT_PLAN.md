# Logical RNG Layout Plan

## Status

Draft implementation plan for sharing one logical-index layout representation
between stateless Philox generation, legacy stateful CUDA generation, and
distributed tensor initialization.

## Goal

Represent a local output as an explicit selection of positions from one logical
random draw. Use that representation as the common substrate for:

- Stateless generation from an explicit Philox key.
- Existing stateful `Generator` APIs without changing their CUDA bitstream.
- DTensor and other distributed systems that initialize local shards while
  matching a dense logical initialization.
- An optional `RNGView` convenience API that does not masquerade as an
  independent PRNG key.

This project does not replace the default stateful RNG APIs.

## Stack Strategy

This plan stacks new PRs on top of HEAD rather than redesigning HEAD in place.

1. Keep HEAD (`#190289`) as the tested behavioral baseline for logical-index
   replay and dense `Generator` advancement.
2. Add the stateless indexed backend and shared layout machinery in follow-up
   PRs.
3. Refactor HEAD's generator-backed implementation through the shared core,
   preserving its values, final generator state, and next draw exactly.
4. Rebase and rework the current DTensor child (`#189546`) on top of the shared
   adapter. Do not retain its bypassing custom-handler architecture.
5. Add experimental frontend conveniences only after backend and DTensor
   semantics are stable.

An optional pre-landing cleanup may make HEAD's protocol and mode private, but
the plan does not require an in-place semantic rewrite of HEAD.

## Required Semantic Split

The stateless and legacy stateful streams are not interchangeable.

- `portable_linear_v1` uses an explicit tensor key, fixes Philox subsequence to
  zero, and maps logical indices directly to counter/lane positions. It is the
  policy for stateless APIs and should be independent of CUDA launch geometry.
- `cuda_generator_dense_v1` reproduces the current dense CUDA kernel. It uses
  dense launch geometry, thread-based subsequences, legacy generator offset
  units, and legacy uniform/normal transforms.

Both policies share layout validation and local-to-logical index decoding. The
Generator adapter must not simply convert `Generator` state into today's
stateless key and invoke `_philox_normal_` or `_philox_uniform_`; doing so would
change values and generator progression.

## Data Model

Start with private, immutable Python value objects registered as pytrees:

```python
@dataclass(frozen=True)
class RNGIndexBlock:
    start: int | torch.SymInt
    block_size: int | torch.SymInt
    block_stride: int | torch.SymInt
    block_count: int | torch.SymInt = 1


@dataclass(frozen=True)
class RNGLayout:
    logical_numel: int | torch.SymInt
    blocks: tuple[RNGIndexBlock, ...]


@dataclass(frozen=True)
class RNGView:
    key: torch.Tensor
    layout: RNGLayout
    event_shape: tuple[int | torch.SymInt, ...]
```

Expanding one block produces these half-open logical intervals:

```text
[start + i * block_stride,
 start + i * block_stride + block_size)
for i in range(block_count)
```

Intervals are concatenated in descriptor order. Local flat element `i` must
equal element `expanded_indices[i]` from dense generation over
`logical_numel` elements.

The layout is expressed in logical element units, never Philox counter units.
It contains no key, distribution, dtype, device mesh, generator, tracker, or
collective state.

## Layout Invariants

- `logical_numel` is nonnegative.
- Lists have equal lengths after lowering to native operator arguments.
- Block starts are nonnegative; block sizes and counts are positive.
- `block_stride >= block_size`.
- Endpoint and mapped-element calculations use checked 64-bit arithmetic.
- Every expanded interval ends at or before `logical_numel`.
- Expanded intervals within one layout are ordered and non-overlapping.
- The expanded element count equals the local output's `numel`.
- Empty local output is represented by `blocks=()`.
- Separate layouts may overlap. This is required for replicated values.
- Invalid stateful calls fail before reserving generator state.

Keep `RNGLayout` private and opaque initially. The four-value block encoding
may later need piecewise-affine tiles to avoid descriptor explosion for complex
multidimensional or ragged layouts.

## PR 1: Freeze Existing Semantics

- Add bitwise golden tests for existing stateless `uniform` and `normal`.
- Add golden tests for HEAD's values, final generator state, and next draw.
- Cover all floating dtypes, nonzero generator offsets, empty local shards,
  multiple blocks, uneven slices, and the dense grid-stride increment boundary.
- Lock down DTensor tracker-enabled and tracker-disabled behavior.
- Add a forced-rejection `trunc_normal_` test before claiming generic composite
  initializer support.

## PR 2: Stateless Indexed Backend

Add private ATen operations conceptually equivalent to:

```text
_philox_uniform_indexed_(out, key, logical_numel,
                         starts, sizes, strides, counts, low, high)
_philox_normal_indexed_(out, key, logical_numel,
                        starts, sizes, strides, counts, mean, std)
```

- Use `SymInt logical_numel` and `SymInt[]` descriptors at the schema boundary.
- Initially require one unbatched CUDA `uint64[2]` key on the output device.
- Map each logical index to the portable Philox counter and output lane.
- Share native validation and block decoding between distributions.
- Add identity-layout and single-block fast paths.
- Avoid allocating a per-element index tensor.
- Use checked 64-bit indexing and remove the current `INT_MAX` limitation.
- Add Meta/FakeTensor registration, functional/out variants, decomposition
  expectations, fullgraph compilation, export, and dynamic-shape coverage.
- Refactor current stateless dense generation through the identity fast path
  without changing its output.
- Land native schemas before frontend callers and observe the required
  forward-compatibility window.

## PR 3: Stateful Generator Adapter

Add separate private Generator-backed entry points using the same validator,
layout decoder, and launcher with the legacy dense sampling policy.

- Compute the full dense execution policy from `logical_numel`.
- Lock the CUDA generator and reserve its full logical increment exactly once.
- Preserve `PhiloxCudaState` capture pointers and intragraph offsets.
- Do not use Python `get_state()`/`set_state()` in the production path.
- Empty local output with nonempty logical output still reserves the full
  increment; zero logical output does not advance.
- Perform capability and argument validation before reservation or fallback.
- Keep generator reservation outside any loop over layout blocks.
- Route HEAD's scoped stateful mode through this adapter and prove bitwise
  equivalence with its golden tests.

Sharing happens at the indexed-kernel layer, not by forcing legacy generator
state through the portable stateless transform.

## PR 4: Experimental Frontend

Extend private `torch.func._random` APIs with an explicit layout keyword:

```python
random.uniform(key, local_shape, layout=layout)
random.uniform_(key, out, layout=layout)
random.normal(key, local_shape, layout=layout)
random.normal_(key, out, layout=layout)
```

- `layout=None` retains current dense behavior.
- Keep the key and layout as separate concepts.
- Lower frozen pytree layouts immediately to primitive operator arguments.
- Reject unsupported layouts rather than silently invoking mutable state.
- Initially support vmap over keys sharing one static layout.
- Defer batched or graph-input layouts until there is a tensorized descriptor
  representation and explicit batching rules.

## PR 5: DTensor Integration

- Construct one local `RNGLayout` from `DTensorSpec` in the common RNG dispatch
  path.
- Initially support contiguous CUDA local tensors, `normal_`, `uniform_`, a
  one-dimensional mesh, `Replicate`, and one `Shard(dim)` placement.
- Cover uneven and empty shards and ranks outside a submesh.
- Preserve `distribute_region_enabled` and existing tracker fallback policy.
- Capability-check before generator reservation so fallback cannot advance
  twice.
- Avoid direct custom handlers that bypass tracker policy.
- Avoid a hidden state broadcast on every RNG operation; synchronization policy
  belongs above the pure backend.
- Keep `Partial`, noncontiguous local tensors, unsupported devices, and
  unsupported placements on the existing fallback initially.
- Expand later to multidimensional rectangular shards, `_StridedShard`, and
  piecewise-affine layouts.

Use `#189546` as a bitwise semantic oracle, not as the final dispatch design.

## PR 6: RNGView and Partition Convenience

After explicit layouts are stable, add an optional convenience object:

```python
draw_key = random.fold_in(root_key, layer_id)
view = RNGView(draw_key, layout, event_shape)
local = random.uniform(view)
```

- `RNGView` is a registered pytree, not a Tensor or Tensor subclass.
- A view selects positions within one logical draw; it is not an independent
  random stream.
- `split(view)` and `fold_in(view)` are errors.
- Applying overlapping views to one key intentionally reproduces equal values.
- Production distributed code constructs only its local view.
- Consider `partition()` or `unbind()` only as sugar returning `RNGView`
  objects. Do not return offset-shifted ordinary keys.
- Settle naming only after tensorized layouts and batching rules are available.

## Acceptance Gates

### Correctness

- Reassembling stateless local views is bitwise equal to dense stateless output.
- Stateful shards match dense values, final generator state, and next draw.
- Normal generation is correct across odd block starts and pair boundaries.
- Tests cover every floating dtype, full/empty/uneven/replicated layouts,
  multiple blocks, overflow, offset wrap, invalid layouts, and repeated calls.
- Stateless portable output is invariant to physical kernel launch geometry.

### Composability

- Eager, AOT eager, Inductor fullgraph, FakeTensor, export, functionalization,
  dynamic shapes, and CUDA graph capture/replay agree.
- vmap over keys with a shared layout works before batched layouts are exposed.
- No tensor attributes are required to preserve RNG metadata through graph
  transforms.

### Distributed Behavior

- Default and explicit generators, LocalTensor simulation, submeshes, TP, FSDP,
  HSDP, and TorchTitan initialization retain their intended behavior.
- Unsupported cases demonstrate exactly one fallback and no double advancement.
- No backend operation performs a collective.

### Performance

- Dense identity layouts retain the current stateless fast path.
- Benchmark dense, one-block shard, repeated-block shard, and many-block cases.
- Track kernel launches, generator-lock duration, temporary allocations, and
  layout-construction overhead.
- Do not regress existing dense APIs to a per-element Philox initialization
  path.

## Non-Goals

- Replacing default mutation-based RNG APIs.
- Changing the existing CUDA Generator bitstream.
- Changing CPU's default Mersenne Twister behavior.
- Supporting every random distribution in the first stack.
- Enforcing linear key consumption at runtime.
- Promoting experimental APIs to public `torch.random` before CPU/CUDA,
  compile, and distributed semantics are stable.

## Primary Risks

- Confusing stateless counter units with legacy generator offset units.
- Accidentally changing uniform conversion or Box-Muller output ordering.
- Breaking CUDA graph capture while extracting generator state.
- Descriptor explosion for complex placements.
- Hidden collectives or divergent rank control flow in DTensor integration.
- Silent semantic discontinuity when falling back to the existing tracker.
- Metadata loss in composite initializers that allocate temporary tensors.
