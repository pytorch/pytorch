# Design Document: Reduced Placement for PyTorch DTensor

## Overview

This document proposes adding a `Reduced` placement type to PyTorch's DTensor system, inspired by JAX's sharding state model. The `Reduced` placement enables efficient FSDP-style gradient computation by tracking tensors whose gradients require reduction.

## Background

### Current DTensor Placements

| Placement | Data Layout | JAX Equivalent |
|-----------|-------------|----------------|
| `Shard(dim)` | Data split across devices on dimension `dim` | Varying |
| `Replicate()` | Same data on all devices | Invariant |
| `Partial(op)` | Partial results needing reduction | Unreduced |

### The Missing State: Reduced

JAX has a fourth state, **Reduced**, which represents:
- **Runtime**: Identical to Replicate (same data on all devices)
- **Semantics**: "This replicated data's gradient needs reduction"

The key insight is the **primal-cotangent duality**:

```
Primal Placement  →  Gradient Placement
───────────────────────────────────────
Shard(dim)        →  Shard(dim)
Replicate()       →  Replicate()
Partial()         →  Reduced()
Reduced()         →  Partial()         ← KEY!
```

## Motivation: FSDP Efficiency

### Without Reduced (Current Approach)

```python
# Forward: all-gather sharded weights
w_full = all_gather(w_shard)  # Shard → Replicate

# Backward:
# 1. grad is Replicate (same on all devices)
# 2. Need to redistribute to Shard
# This requires: all-reduce (2N bytes) + scatter
```

### With Reduced

```python
# Forward: all-gather to Reduced
w_full = all_gather(w_shard, out_placement=Reduced())  # Shard → Reduced

# Backward:
# 1. grad is Partial (cotangent of Reduced) - partial sums per device
# 2. reduce-scatter directly to Shard
# This requires: reduce-scatter (N bytes) - 50% less communication!
```

## Proposed Design

### 1. Reduced Placement Class

`Reduced` inherits from `Replicate` so that existing sharding propagation logic (which uses `isinstance(p, Replicate)`) automatically treats it as replicated data. This follows the same pattern as `_MaskPartial` inheriting from `Partial`.

```python
class Reduced(Replicate):
    """
    Replicated data where gradient should be Partial (unreduced).

    Subclasses Replicate so existing sharding propagation rules
    treat it as replicated data (which it is, at runtime).

    Always uses sum for gradient reduction (the only practical case
    for distributed training).
    """

    def is_reduced(self) -> bool:
        return True

    def __repr__(self) -> str:
        return "Reduced()"

    def __eq__(self, other) -> bool:
        return isinstance(other, Reduced)

    def __hash__(self) -> int:
        return hash("Reduced")
```

**Important**: Code that needs to distinguish `Reduced` from `Replicate` must check `isinstance(p, Reduced)` **before** checking `isinstance(p, Replicate)`, since `Reduced` is a subclass of `Replicate`.

### 2. Cotangent Mapping (Separate Module)

The cotangent mapping is kept separate from the placement classes in `torch/distributed/tensor/_cotangent.py`. This keeps the placement classes clean and centralizes the primal-cotangent duality logic in one place. Currently only `__torch_function__` uses this mapping for gradient redistribution.

```python
# torch/distributed/tensor/_cotangent.py

from torch.distributed.tensor import Shard, Replicate, Partial
from torch.distributed.tensor.placement_types import Reduced

def placement_to_cotangent(placement: Placement) -> Placement:
    """
    Map a primal placement to its cotangent (gradient) placement.

    The primal-cotangent duality:
    - Shard(dim) → Shard(dim)      # Sharding preserved
    - Replicate() → Replicate()    # True replication preserved
    - Partial() → Reduced()        # Unreduced → "needs reduction in backward"
    - Reduced() → Partial()        # "needs reduction in backward" → unreduced
    """
    if isinstance(placement, Reduced):
        return Partial()
    elif isinstance(placement, Shard):
        return Shard(placement.dim)
    elif isinstance(placement, Partial):
        return Reduced()
    elif isinstance(placement, Replicate):
        return Replicate()
    else:
        # Unknown placement type, return as-is
        return placement


def spec_to_cotangent(spec: DTensorSpec) -> DTensorSpec:
    """Convert a primal DTensorSpec to its cotangent (gradient) spec."""
    return DTensorSpec(
        mesh=spec.mesh,
        placements=tuple(placement_to_cotangent(p) for p in spec.placements)
    )
```

**Note**: The order of `isinstance` checks matters - `Reduced` must be checked before `Replicate` since `Reduced` is a subclass of `Replicate`.

## Propagation Strategy

### Key Insight: Reduced == Replicate for Sharding Math

Since `Reduced` and `Replicate` have identical runtime data layouts, we can:

1. **Let existing sharding propagation treat Reduced as Replicate**
2. **Post-process to restore Reduced annotation**

This avoids modifying every op's sharding rules.

### Implementation

```python
def propagate_reduced_annotation(
    input_specs: List[DTensorSpec],
    output_spec: DTensorSpec
) -> DTensorSpec:
    """
    After op dispatch, propagate Reduced annotation from inputs to output.

    Rules:
    - If any input has Reduced on a mesh dim, output gets Reduced on that dim
    - Uniform rule for ALL ops (no special cases)
    """

    # Collect which mesh dimensions have Reduced placements
    reduced_dims: Set[int] = set()

    for spec in input_specs:
        for dim, placement in enumerate(spec.placements):
            if isinstance(placement, Reduced):
                reduced_dims.add(dim)

    if not reduced_dims:
        return output_spec  # No Reduced inputs, nothing to propagate

    # Upgrade Replicate to Reduced on dimensions where any input was Reduced
    new_placements = list(output_spec.placements)

    for dim in reduced_dims:
        if isinstance(new_placements[dim], Replicate) and \
           not isinstance(new_placements[dim], Reduced):
            new_placements[dim] = Reduced()

    return DTensorSpec(
        mesh=output_spec.mesh,
        placements=tuple(new_placements)
    )
```

### Why Uniform Propagation Works (Including Matmul)

The "propagate if inputs agree" rule naturally handles all ops, including matmul:

**Typical FSDP case:**
```python
w = all_gather(w_shard)  # Reduced
x = input                 # Replicate (not Reduced)
y = x @ w                 # Inputs disagree → y is NOT Reduced ✓
```

**Both inputs Reduced (rare but valid):**
```python
w1 = all_gather(w1_shard)  # Reduced on mesh dim 0
w2 = all_gather(w2_shard)  # Reduced on mesh dim 0
y = w1 @ w2                 # Both agree → y is Reduced

# Backward:
# dy is Partial (cotangent of Reduced)
# dw1 = dy @ w2.T → Partial @ Replicate = Partial ✓
# dw2 = w1.T @ dy → Replicate @ Partial = Partial ✓
```

The uniform rule works because:
1. Most matmuls have only one Reduced input (weight), so propagation naturally doesn't apply
2. When both inputs are Reduced, the Partial gradient flows correctly through the backward matmul

### Integration Point: `__torch_function__`

**Why `__torch_function__` and not `__torch_dispatch__`?**

The DTensor `__torch_dispatch__` is implemented in C++ for performance (see `python_variable.cpp:dispatchDTensorOp()`). The Python `__torch_dispatch__` method raises `NotImplementedError` and is never actually called. Therefore, we must integrate Reduced propagation in `__torch_function__`, which:

1. Is called before dispatch and has access to original DTensor args with their placements
2. Already handles gradient placement hooks (see `_DTensorGradPlacementHook`)
3. Can post-process outputs after the underlying dispatch completes

```python
@classmethod
def __torch_function__(cls, func, types, args, kwargs=None):
    # Capture input specs before dispatch (for both gradient hooks and Reduced propagation)
    input_specs = [a._spec for a in pytree.tree_leaves(args) if isinstance(a, DTensor)]

    # Dispatch (goes through C++ __torch_dispatch__)
    out = super().__torch_function__(func, types, args, kwargs or {})

    # Post-process: propagate Reduced annotations
    def _propagate_reduced(t):
        if isinstance(t, DTensor):
            new_spec = propagate_reduced_annotation(input_specs, t._spec)
            if new_spec != t._spec:
                # Update spec (implementation TBD - may need internal API)
                pass
        return t

    out = tree_map(_propagate_reduced, out)

    # Register gradient placement hooks (existing logic)
    def _register_grad_placement_hook(t):
        if isinstance(t, torch.Tensor) and t.grad_fn is not None:
            _ = _DTensorGradPlacementHook(t.grad_fn, args)
        return t

    tree_map(_register_grad_placement_hook, out)
    return out
```

## Autograd Integration

### Gradient Placement Hook

The existing `__torch_function__` hook registers `_DTensorGradPlacementHook` on every op's output. This hook uses `spec_to_cotangent()` from `_cotangent.py` to redistribute gradients to cotangent placements during backward:

```python
from torch.distributed.tensor._cotangent import spec_to_cotangent

class _DTensorGradPlacementHook:
    """Hook to redistribute gradients to cotangent placements during backward."""

    def __init__(self, grad_fn, forward_args):
        self.placements = tuple(
            arg._spec.placements if isinstance(arg, DTensor) else None
            for arg in forward_args
        )
        self.specs = tuple(
            arg._spec if isinstance(arg, DTensor) else None
            for arg in forward_args
        )
        self.grad_fn = grad_fn
        self.grad_fn.register_hook(self._post)

    def _post(self, grad_inputs, grad_outputs):
        """Redistribute grad_inputs to cotangent placements."""
        result = []
        for spec, grad_input in zip(self.specs, grad_inputs):
            if isinstance(grad_input, DTensor) and spec is not None:
                target_spec = spec_to_cotangent(spec)
                if target_spec.placements != grad_input._spec.placements:
                    grad_input = grad_input.redistribute(
                        device_mesh=grad_input.device_mesh,
                        placements=target_spec.placements,
                    )
            result.append(grad_input)
        return tuple(result)
```

### Collective Operations with Reduced Support

```python
class AllGatherReduced(torch.autograd.Function):
    """All-gather that produces Reduced output for efficient backward."""

    @staticmethod
    def forward(ctx, x: DTensor, gather_dim: int) -> DTensor:
        ctx.gather_dim = gather_dim
        ctx.input_spec = x._spec

        # Perform all-gather
        gathered = all_gather_impl(x, gather_dim)

        # Mark output as Reduced (not Replicate)
        output_placements = list(x._spec.placements)
        output_placements[gather_dim] = Reduced()

        return DTensor(gathered, DTensorSpec(x.device_mesh, tuple(output_placements)))

    @staticmethod
    def backward(ctx, grad: DTensor) -> Tuple[DTensor, None]:
        # grad has Partial placement (cotangent of Reduced)
        # reduce-scatter: Partial → Shard
        return reduce_scatter_impl(grad, ctx.gather_dim), None
```

## Placement Duality Summary

| Primal | Cotangent | Forward Collective | Backward Collective |
|--------|-----------|-------------------|---------------------|
| `Shard(d)` | `Shard(d)` | - | - |
| `Replicate()` | `Replicate()` | - | - |
| `Partial()` | `Reduced()` | reduce (all-reduce) | broadcast/no-op |
| `Reduced()` | `Partial()` | no-op (already replicated) | reduce |
| `Shard → Reduced` | `Partial → Shard` | all-gather | reduce-scatter |
| `Shard → Replicate` | `Replicate → Shard` | all-gather | split/slice |

## Op Support Matrix

All ops use uniform propagation: if any input has Reduced on a mesh dim, output gets Reduced on that dim.

| Op Category | Reduced Propagation | Notes |
|-------------|---------------------|-------|
| Unary (sin, cos, exp, neg, ...) | ✅ Propagates | Input Reduced → Output Reduced |
| Binary (add, mul, sub, ...) | ✅ Propagates | Any Reduced input → Output Reduced |
| Reduction (sum, mean, ...) | ✅ Propagates | Reduced dims preserved |
| Matmul (mm, bmm, linear) | ✅ Propagates | Same rule; typically only one input is Reduced so output stays non-Reduced |
| Reshape (view, transpose, ...) | ✅ Propagates | Reduced annotation preserved |
| Indexing (slice, gather, ...) | ✅ Propagates | Reduced annotation preserved |

## Implementation Phases

### Phase 1: Core Infrastructure
- [x] Add `Reduced` placement class (inherits from `Replicate`)
- [x] Add `is_reduced()` method to `Reduced`
- [ ] Add `_cotangent.py` module with `placement_to_cotangent()` and `spec_to_cotangent()`
- [ ] Add tests in `test_dtensor_grad_placements.py`

### Phase 2: Propagation
- [ ] Implement `propagate_reduced_annotation()`
- [ ] Integrate into `__torch_function__`
- [ ] Add tests for propagation through common ops

### Phase 3: Autograd
- [ ] Update gradient hook to use `spec_to_cotangent()` from `_cotangent.py`
- [ ] Implement `all_gather_reduced` collective
- [ ] Add tests for backward pass correctness

### Phase 4: FSDP Integration
- [ ] Modify FSDP to use `all_gather_reduced` for weight gathering
- [ ] Benchmark communication savings
- [ ] End-to-end training tests

---

## LTensor Reduced Tracking Design

This section describes the implementation of reduced tracking in LTensor, which mirrors the DTensor `Reduced` placement in the local tensor abstraction.

### 1. Overview

Add `_reduced_dims` to LTensor to track mesh dimensions where the tensor is "reduced" (data is replicated, but gradient needs reduction). This is mutually exclusive with `_variant_dims`.

### 2. Key Differences from Variance Tracking

| Aspect | Variance (`_variant_dims`) | Reduced (`_reduced_dims`) |
|--------|----------------------------|---------------------------|
| Propagation | Union (ANY input variant → output variant) | Intersection (ALL inputs reduced → output reduced) |
| Semantic | Different values across ranks | Same values, but grad needs reduction |
| Mutual exclusivity | Cannot be reduced on same dim | Cannot be variant on same dim |

**Rationale for intersection propagation**: Reduced only propagates when ALL inputs agree because:
- If one input is reduced and another isn't, the gradient flow is different
- Conservative approach: only maintain reduced when it's unambiguous

### 3. Implementation Design

#### 3.1 New Registration Maps

```python
_CUSTOM_REDUCED_STRATEGY_MAP: dict[Callable, Callable] = {}

def register_reduced_strategy(ops):
    """Register custom reduced strategy for calculating output reduced dims.

    The decorated function receives:
      - input_reduced_dims: set[str] - intersection of reduced dims from all LTensor inputs
      - mesh: DeviceMesh - the mesh from input LTensors
      - *args, **kwargs: the original function arguments

    Returns:
      - output_reduced_dims: set[str] - which dims the output is reduced on
    """
    def wrapper(func):
        for op in ops:
            _CUSTOM_REDUCED_STRATEGY_MAP[op] = func
        return func
    return wrapper
```

#### 3.2 Updated LTensor Class

```python
class LTensor(torch.Tensor):
    _local_tensor: torch.Tensor
    _variant_dims: set[str]
    _reduced_dims: set[str]  # NEW
    _mesh: DeviceMesh

    def __new__(
        cls,
        local_tensor: torch.Tensor,
        variant_dims: set[str],
        mesh: DeviceMesh,
        reduced_dims: set[str] | None = None,  # NEW (optional for backward compat)
    ):
        # Validate mutual exclusivity
        reduced_dims = reduced_dims or set()
        if variant_dims & reduced_dims:
            raise ValueError(
                f"Dims cannot be both variant and reduced: {variant_dims & reduced_dims}"
            )

        r = local_tensor.as_subclass(cls)
        r._variant_dims = variant_dims
        r._reduced_dims = reduced_dims
        r._mesh = mesh
        r._local_tensor = local_tensor
        return r

    @staticmethod
    def compute_metadata_from_dtensor(dtensor: "DTensor"):
        """Extract metadata from DTensor placements."""
        mesh = dtensor.device_mesh
        placements = dtensor.placements

        if mesh.mesh_dim_names is None:
            raise ValueError(
                "DTensor's mesh must have mesh_dim_names to convert to LTensor."
            )

        variant_dims = set()
        reduced_dims = set()

        for mesh_dim_idx, placement in enumerate(placements):
            mesh_dim_name = mesh.mesh_dim_names[mesh_dim_idx]
            if isinstance(placement, Reduced):  # Check Reduced before Replicate!
                reduced_dims.add(mesh_dim_name)
            elif not isinstance(placement, Replicate):
                variant_dims.add(mesh_dim_name)
            # else: Replicate → neither variant nor reduced

        return {
            "mesh": mesh,
            "variant_dims": variant_dims,
            "reduced_dims": reduced_dims,
        }
```

#### 3.3 Updated `__torch_function__`

```python
@classmethod
def __torch_function__(cls, func, types, args=(), kwargs=None):
    if kwargs is None:
        kwargs = {}

    # === Collect metadata from all LTensor inputs ===
    out_variant_dims: set[str] = set()
    all_reduced_dims: list[set[str]] = []  # For intersection
    meshes: set[DeviceMesh] = set()
    has_ltensor_input = False

    def _extract_metadata(t):
        nonlocal has_ltensor_input
        if isinstance(t, LTensor):
            has_ltensor_input = True
            out_variant_dims.update(t._variant_dims)
            all_reduced_dims.append(t._reduced_dims)
            meshes.add(t._mesh)

    tree_map(_extract_metadata, args)

    if len(meshes) > 1:
        raise RuntimeError("Cannot mix LTensors from different meshes!")
    mesh = meshes.pop()

    # === Compute output reduced dims (INTERSECTION of all inputs) ===
    if all_reduced_dims:
        out_reduced_dims = all_reduced_dims[0].copy()
        for rd in all_reduced_dims[1:]:
            out_reduced_dims &= rd  # Intersection
    else:
        out_reduced_dims = set()

    # === Apply custom strategies ===
    if func in _CUSTOM_VARIANCE_STRATEGY_MAP:
        out_variant_dims = _CUSTOM_VARIANCE_STRATEGY_MAP[func](
            out_variant_dims, mesh, *args, **kwargs
        )

    if func in _CUSTOM_REDUCED_STRATEGY_MAP:
        out_reduced_dims = _CUSTOM_REDUCED_STRATEGY_MAP[func](
            out_reduced_dims, mesh, *args, **kwargs
        )

    # === Validate mutual exclusivity ===
    assert not (out_variant_dims & out_reduced_dims), (
        f"Bug: output cannot be both variant and reduced on {out_variant_dims & out_reduced_dims}"
    )

    func = _CUSTOM_OPERATOR_HANDLER_MAP.get(func, func)

    # === Unwrap, execute, wrap ===
    def unwrap_and_insert_mark_varying(t):
        # ... existing logic for mark_varying ...

    unwrapped_args = tree_map(unwrap_and_insert_mark_varying, args)
    unwrapped_kwargs = tree_map(unwrap_and_insert_mark_varying, kwargs)

    result = func(*unwrapped_args, **unwrapped_kwargs)

    def wrap(t):
        if isinstance(t, torch.Tensor) and not isinstance(t, LTensor):
            return LTensor(t, out_variant_dims, mesh, out_reduced_dims)
        return t

    return tree_map(wrap, result)
```

### 4. Example Custom Strategies

```python
# all_gather: Shard → Reduced (output becomes reduced on that dim)
@register_reduced_strategy([torch.ops._c10d_functional.all_gather_into_tensor])
def _all_gather_reduced_strategy(input_reduced_dims, mesh, *args, **kwargs):
    """all_gather makes output Reduced on the gather dim."""
    input_tensor, _, group_name = args
    dim_name = _get_dim_name_from_group(mesh, group_name)
    return input_reduced_dims | {dim_name}  # Add the gathered dim


# reduce_scatter: Reduced → Shard (output loses reduced on that dim)
@register_reduced_strategy([torch.ops._c10d_functional.reduce_scatter_tensor])
def _reduce_scatter_reduced_strategy(input_reduced_dims, mesh, *args, **kwargs):
    """reduce_scatter removes Reduced on the scatter dim."""
    input_tensor, _, group_name = args
    dim_name = _get_dim_name_from_group(mesh, group_name)
    return input_reduced_dims - {dim_name}  # Remove the scattered dim
```

### 5. Integration with local_map

```python
# In local_map, when converting DTensor → LTensor:
def dtensor_to_ltensor(dtensor: DTensor) -> LTensor:
    metadata = LTensor.compute_metadata_from_dtensor(dtensor)
    return LTensor(
        dtensor.to_local(),
        variant_dims=metadata["variant_dims"],
        mesh=metadata["mesh"],
        reduced_dims=metadata["reduced_dims"],
    )

# When converting LTensor → DTensor:
def ltensor_to_dtensor(ltensor: LTensor, out_placements: tuple[Placement, ...]) -> DTensor:
    # Validate that reduced_dims match Reduced() placements
    # ... conversion logic ...
```

### 6. Summary of Propagation Rules

| State | Propagation Rule | Example |
|-------|------------------|---------|
| `_variant_dims` | Union (ANY) | {A} ∪ {B} = {A, B} |
| `_reduced_dims` | Intersection (ALL) | {A, B} ∩ {A} = {A} |

## Open Questions

1. **Reduced + Shard interaction**: What if a tensor is `(Shard(0), Reduced())` on a 2D mesh? Need to define semantics clearly.

2. **Multiple Reduced dims**: Can a tensor be Reduced on multiple mesh dimensions? JAX supports this via `frozenset` of axes.

3. **Partial + Reduced coexistence**: Can a tensor be both Partial (needs reduction) and Reduced (gradient needs reduction) on different dims?

4. **Explicit vs. implicit**: Should users be able to explicitly construct `Reduced` tensors, or only through specific collectives like `all_gather_reduced`?

## References

- [JAX Reduced Sharding Design](./reduced_design.md) - Source design document
- [Current WIP: Gradient Placement Hook](./grad_code.diff) - Existing hook-based approach
