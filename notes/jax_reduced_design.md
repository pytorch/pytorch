# Design Document: The "Reduced" Sharding State in JAX

## Overview

This document describes the **Reduced** sharding state in JAX's distributed tensor system, its relationship to other states, and how it enables efficient gradient computation for FSDP-style distributed training. This is intended to help implement similar functionality in other ML frameworks (e.g., PyTorch).

## The Four Sharding States

JAX's sharding system tracks four distinct states for tensors in distributed computation:

| State | Data Layout | Stored In | Meaning |
|-------|-------------|-----------|---------|
| **Invariant** | Same value on all devices | (default, no annotation) | Replicated data, no special handling |
| **Varying** | Different values per device | `aval.vma` (set of axis names) | Sharded/partitioned data |
| **Unreduced** | Partial sums per device | `spec.unreduced` (set of axis names) | Needs reduction to get correct value |
| **Reduced** | Same value on all devices | `spec.reduced` (set of axis names) | Replicated, but gradient needs special handling |

### Key Insight: Reduced vs Invariant

**Reduced and Invariant have identical data layouts** - both store the same value on every device. The difference is purely semantic:

- **Invariant**: "This is normal replicated data"
- **Reduced**: "This is replicated data, but I'm tracking that its *gradient* will need reduction"

## The Cotangent (Gradient) Duality

The critical design principle is the **primal-cotangent duality**:

```
Primal State    →  Cotangent State
─────────────────────────────────────
Invariant       →  Invariant
Varying         →  Varying
Reduced         →  Unreduced    ← KEY!
Unreduced       →  Reduced
```

This is implemented in JAX as:
```python
def primal_spec_to_cotangent_spec(spec):
    return P(*spec, unreduced=spec.reduced, reduced=spec.unreduced)
```

**The Reduced↔Unreduced swap is the core mechanism** that enables efficient FSDP gradients.

## Why Reduced Exists: The FSDP Use Case

### Problem: Efficient Weight Gradient Computation

In FSDP (Fully Sharded Data Parallel):
1. Weights are sharded across devices (each device holds 1/N of weights)
2. For forward pass, we all-gather to get full weights
3. For backward pass, we need to get gradients back to sharded form

### Naive Approach (Inefficient)

```python
# Forward
w_full = all_gather(w, 'x')  # Varying → Varying (or Invariant)

# Backward
w_full_grad = ...  # compute gradient for w_full
w_full_grad = all_reduce(w_full_grad, 'x')  # ensure all devices have same grad
w_grad = slice_for_device(w_full_grad)  # slice to get local shard
```

This requires: **all-reduce (2N bytes) + local slice**

### Efficient Approach with Reduced

```python
# Forward
w_full = all_gather(w, 'x', to='reduced')  # Varying → Reduced

# Backward (automatic via transpose rules)
# w_full_grad is Unreduced (cotangent of Reduced)
w_grad = reduce_scatter(w_full_grad, 'x')  # Unreduced → Varying
```

This requires: **reduce-scatter (N bytes)** - 50% less communication!

## State Transition Primitives

### No-Op Casts (Type Annotations Only)

These change the type annotation without moving data:

| Primitive | Transition | Purpose |
|-----------|------------|---------|
| `pvary` | Invariant → Varying | Mark replicated as varying |
| `preduced` | Invariant → Reduced | Mark for future reduction |
| `vary_unreduced_cast` | Varying → Unreduced | Mark as partial sums |
| `reduced_vary_cast` | Reduced → Varying | Remove reduction marking |

### Data-Moving Collectives

| Primitive | Transition | Communication |
|-----------|------------|---------------|
| `psum_invariant` | Varying → Invariant | All-reduce |
| `all_gather_invariant` | Varying → Invariant | All-gather |
| `all_gather_reduced` | Varying → Reduced | All-gather |
| `all_gather` | Varying → Varying | All-gather |
| `reduce_scatter` | Varying → Varying | Reduce-scatter |
| `unreduced_psum` | Unreduced → Invariant | All-reduce |
| `unreduced_psum_scatter` | Unreduced → Varying | Reduce-scatter |

### Transpose Pairs (Autodiff)

Each forward primitive has a transpose for backward:

| Forward | Forward Effect | Transpose | Transpose Effect |
|---------|----------------|-----------|------------------|
| `pvary` | no-op (I→V) | `psum_invariant` | all-reduce (V→I) |
| `preduced` | no-op (I→R) | `unreduced_psum` | all-reduce (U→I) |
| `vary_unreduced_cast` | no-op (V→U) | `reduced_vary_cast` | no-op (R→V) |
| `reduced_vary_cast` | no-op (R→V) | `vary_unreduced_cast` | no-op (V→U) |
| `all_gather_reduced` | all-gather (V→R) | `unreduced_psum_scatter` | reduce-scatter (U→V) |
| `unreduced_psum_scatter` | reduce-scatter (U→V) | `all_gather_reduced` | all-gather (V→R) |

## Propagation Through Operations

### Which Operations Support Reduced?

Operations must explicitly define a `reduced_rule` to accept Reduced inputs. Without one:
```python
def call_reduced_rule(prim, reduced_rule, out_s, num_out, *avals, **kwargs):
    if reduced_rule is not None:
        return reduced_rule(out_s, *avals, **kwargs)
    if any(a.sharding.spec.reduced for a in avals):
        raise NotImplementedError(f'reduced rule for {prim.name} not implemented')
    return out_s
```

### Propagation Rules

**Unary operations** (sin, cos, exp, neg, etc.):
```python
def unop_reduced_rule(out_s, aval, **kwargs):
    return out_s.update(spec=out_s.spec.update(reduced=aval.sharding.spec.reduced))
```
- Copies `reduced` from input to output
- `sin(x{R:a})` → `{R:a}`

**N-ary operations** (add, mul, sub, etc.):
```python
def nary_reduced_rule(out_s, *avals, **params):
    specs = [a.sharding.spec for a in avals]
    reduced_specs = {s.reduced for s in specs if s.reduced}
    if len(reduced_specs) > 1:
        raise ShardingTypeError("inputs must have same reduced axes")
    return out_s.update(spec=out_s.spec.update(
        reduced=reduced_specs.pop() if reduced_specs else frozenset()))
```
- All inputs must have same `reduced` annotation
- `x{R:a} + y{R:a}` → `{R:a}`
- `x{R:a} + y{R:b}` → ERROR

**Matrix multiplication** (`dot_general`):
```python
def _dot_general_reduced_rule(out_s, lhs, rhs, *, dimension_numbers, **kwargs):
    return out_s  # Does NOT propagate from inputs!
```
- Requires explicit `out_sharding=P(..., reduced={...})` to get Reduced output

### Operations with Reduced Rules

| Operation | Propagation Behavior |
|-----------|---------------------|
| Unary ops (sin, cos, exp, log, neg, abs, ...) | Propagates from input |
| Binary ops (add, mul, sub, div, ...) | All inputs must match, then propagates |
| `convert_element_type` | Propagates from input |
| `transpose` | Propagates from input |
| `squeeze` | Propagates from input |
| `reduce_sum` | Propagates from input |
| `concatenate` | All inputs must match, then propagates |
| `dynamic_slice` | Propagates from input |
| `dynamic_update_slice` | Both must match |
| `dot_general` | NO propagation (use explicit out_sharding) |

## Reduced is for "Leaf" Tensors

Conceptually, Reduced is a **boundary contract** at shard_map edges:

```python
@shard_map(
    in_specs=(P('x', None), P('x', None, reduced={'y'})),  # inputs
    out_specs=P(None, None, reduced={'x'})                   # outputs
)
def f(a, b):
    ...
```

While Reduced CAN propagate through intermediate operations (via the rules above), it's primarily used to:
1. Mark inputs that came from a gather operation
2. Mark outputs that will need reduction outside

## Implementation Guidance for PyTorch

### Core Data Structures

```python
class ShardingSpec:
    partitions: Tuple[Optional[str], ...]  # axis assignments per dimension
    reduced: FrozenSet[str]    # mesh axes marked for reduction
    unreduced: FrozenSet[str]  # mesh axes with partial sums

class TensorType:
    shape: Tuple[int, ...]
    dtype: DType
    sharding: ShardingSpec
    vma: FrozenSet[str]  # varying manual axes
```

### Key Functions to Implement

1. **Cotangent spec conversion**:
```python
def primal_to_cotangent_spec(spec):
    return ShardingSpec(
        partitions=spec.partitions,
        reduced=spec.unreduced,    # SWAP
        unreduced=spec.reduced     # SWAP
    )
```

2. **Reduced propagation for ops**:
```python
def propagate_reduced_unary(input_spec):
    return output_spec.with_reduced(input_spec.reduced)

def propagate_reduced_nary(*input_specs):
    reduced_sets = [s.reduced for s in input_specs if s.reduced]
    if len(set(map(frozenset, reduced_sets))) > 1:
        raise Error("mismatched reduced axes")
    return reduced_sets[0] if reduced_sets else frozenset()
```

3. **Transpose rules for collectives**:
```python
# Register transpose pairs
register_transpose(all_gather_reduced, unreduced_reduce_scatter)
register_transpose(preduced, unreduced_psum)
register_transpose(vary_unreduced_cast, reduced_vary_cast)
```

### Integration with Autograd

In PyTorch's autograd, you'd need to:

1. **Track sharding state** on tensors (similar to requires_grad)
2. **Define backward functions** that respect the Reduced↔Unreduced duality
3. **Implement collective primitives** with proper forward/backward behavior

Example backward for `all_gather_reduced`:
```python
class AllGatherReduced(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, axis):
        ctx.axis = axis
        return all_gather_impl(x, axis)  # Returns Reduced tensor

    @staticmethod
    def backward(ctx, grad):
        # grad is Unreduced (cotangent of Reduced)
        return reduce_scatter_impl(grad, ctx.axis), None  # Returns Varying
```

## Summary

The Reduced state is a **compile-time annotation** that:
1. Has identical runtime data layout to Invariant (replicated)
2. Tracks that the tensor's gradient should be Unreduced
3. Enables efficient FSDP-style training via reduce-scatter instead of all-reduce
4. Propagates through most operations (except matmul which needs explicit annotation)

The key insight is the **Reduced↔Unreduced duality** which, when combined with proper transpose rules, automatically generates efficient gradient communication patterns.
