# MemoryShardedDTensor (MST) Proposal

## Disclaimer

> **⚠️ Proof of Concept — Feedback Welcome**
>
> This implementation is a **prototype** to demonstrate the core idea. Key caveats:
>
> - **Implementation details may change** — APIs and internals are not finalized
> - **Generated with Claude assistance** — Needs significant review and hardening for production use
> - **DTensor subclass for simplicity** — Currently implemented as a DTensor subclass for easy demonstration, but could alternatively be implemented as a plain Tensor subclass
> - **Feedback requested** — Please share thoughts on API design, use cases, and implementation approach
>
> **This PR includes 3 major changes:**
> 1. **MST implementation** (`torch/distributed/tensor/_memory_sharded.py`, `torch/distributed/memory_saving.py`)
> 2. **Demo use cases** (`test/distributed/tensor/test_mst_fsdp_demo.py`) — 8 tests covering FSDP v2, BSDP, FSDP v1, Muon, veScale, FSDP+TP, local shard ops, and Shampoo patterns
> 3. **FSDP v2 integration** — Using MST to replace the underlying FSDP v2 storage sharding implementation

---

## Motivation

FSDP is fundamentally a **memory optimization**: replicated data across ranks is partitioned so each rank holds a unique portion, reducing per-device memory. The full data is gathered only when needed for computation.

**Key insight**: Unlike other parallelism strategies, FSDP doesn't change *what* computation happens—it changes *where* data is stored. This means:

1. **No tensor ops on MST** - Data is "incomplete" on each rank
2. **Flexible sharding** - Many valid ways to partition replicated data
3. **Explicit unshard** - All-gather before computation, then optionally re-shard

### Why Not a New Placement?

DTensor already has many placements. FSDP as a memory optimization is more restricted in compute but needs more flexibility in data layout. MST provides this flexibility without adding placement complexity.

### Sharding Dimensions

Two key questions determine the sharding strategy:

| Question | Options |
|----------|---------|
| **Unit of sharding** | Per-tensor (each tensor split) or Per-group (tensors share a buffer) |
| **Split proportions** | Even (default, with padding) or Weighted (custom per-rank ratios) |

---

## Key APIs

### 1. `scatter_tensor_storage` — Per-Tensor Sharding

Scatter a **single tensor's** storage across ranks. Each tensor is independently sharded.

```python
from torch.distributed.memory_saving import scatter_tensor_storage
```

#### 1D Sharding (FSDP v2 style)

Shard along one dimension. Each rank holds a slice.

```python
sharded = scatter_tensor_storage(dtensor, dim=0, mesh_dim="dp")
```

```
Mesh: (dp=4)
Original tensor (16, 8):

        dim 1 →
      ┌─────────────────┐
   d  │                 │
   i  │   Full Tensor   │
   m  │    (16, 8)      │
   0  │                 │
   ↓  └─────────────────┘

After scatter_tensor_storage(dim=0):

      ┌─────────────────┐
      │ rank 0: (4, 8)  │  rows 0-3
      ├─────────────────┤
      │ rank 1: (4, 8)  │  rows 4-7
      ├─────────────────┤
      │ rank 2: (4, 8)  │  rows 8-11
      ├─────────────────┤
      │ rank 3: (4, 8)  │  rows 12-15
      └─────────────────┘

Unshard: 1 all-gather on dim 0
```

#### 2D Block Sharding (BSDP style)

Shard along two dimensions. Each rank holds a rectangular block.

```python
sharded = scatter_tensor_storage(dtensor, dim=[0, 1], mesh_dim=["dp_row", "dp_col"])
```

```
Mesh: (dp_row=4, dp_col=2)
Original tensor (16, 8):

After scatter_tensor_storage(dim=[0, 1]):

         col 0-3      col 4-7
        ┌──────────┬──────────┐
row 0-3 │  rank 0  │  rank 1  │   Each block: (4, 4)
        │  (4, 4)  │  (4, 4)  │
        ├──────────┼──────────┤
row 4-7 │  rank 2  │  rank 3  │
        │  (4, 4)  │  (4, 4)  │
        ├──────────┼──────────┤
row 8-11│  rank 4  │  rank 5  │
        │  (4, 4)  │  (4, 4)  │
        ├──────────┼──────────┤
row12-15│  rank 6  │  rank 7  │
        │  (4, 4)  │  (4, 4)  │
        └──────────┴──────────┘

Unshard: 2 sequential all-gathers (dim 1 first, then dim 0)
```

#### Weighted Sharding

Custom per-rank proportions for heterogeneous memory.

```python
sharded = scatter_tensor_storage(dtensor, dim=0, mesh_dim="dp", weights=[1, 2, 1])
```

```
Mesh: (dp=3), weights=[1, 2, 1] (sum=4)
Original tensor (16, 8):

After scatter_tensor_storage with weights:

      ┌─────────────────┐
      │ rank 0: (4, 8)  │  1/4 = 4 rows
      ├─────────────────┤
      │ rank 1: (8, 8)  │  2/4 = 8 rows (2x more)
      ├─────────────────┤
      │ rank 2: (4, 8)  │  1/4 = 4 rows
      └─────────────────┘
```

---

### 2. `scatter_tensor_group` — Per-Group Sharding

Scatter a **group of tensors** across ranks. Tensors share a logical buffer.

```python
from torch.distributed.memory_saving import scatter_tensor_group, ShardingBoundary
```

#### Element Boundary (FSDP v1 style)

Flatten all tensors, concatenate, shard by elements. A single tensor may span multiple ranks.

```python
sharded = scatter_tensor_group(tensors, mesh, "dp", boundary=ShardingBoundary.ELEMENT)
```

```
Mesh: (dp=4)
4 tensors: (4, 4)=16, (2, 4)=8, (3, 2)=6, (2, 2)=4  →  Total: 34 elements

Step 1 - Flatten and concat:
┌────────────────┬────────┬──────┬────┐
│   T0: 16 elem  │ T1: 8  │ T2:6 │T3:4│
└────────────────┴────────┴──────┴────┘

Step 2 - Shard by elements (34 / 4 ≈ 9 per rank, with padding):
┌───────────┬───────────┬───────────┬───────────┐
│  rank 0   │  rank 1   │  rank 2   │  rank 3   │
│ T0[0:9]   │ T0[9:16]  │ T1[6:8]   │ T2[3:6]   │
│           │ T1[0:2]   │ T2[0:3]   │ T3[0:4]   │
│  9 elem   │  9 elem   │  9 elem   │  7 elem   │
└───────────┴───────────┴───────────┴───────────┘

T0 spans ranks 0-1. T1 spans ranks 1-2.
Unshard: Single all-gather for entire buffer, then slice back to original shapes.
```

#### Tensor Boundary (Muon style)

Each rank owns complete tensors. No tensor is split.

```python
sharded = scatter_tensor_group(tensors, mesh, "dp", boundary=ShardingBoundary.TENSOR)
```

```
Mesh: (dp=4)
4 tensors assigned round-robin:

┌───────────┬───────────┬───────────┬───────────┐
│  rank 0   │  rank 1   │  rank 2   │  rank 3   │
├───────────┼───────────┼───────────┼───────────┤
│  Tensor 0 │  Tensor 1 │  Tensor 2 │  Tensor 3 │
│  (4, 4)   │  (2, 4)   │  (3, 2)   │  (2, 2)   │
│  WHOLE    │  WHOLE    │  WHOLE    │  WHOLE    │
└───────────┴───────────┴───────────┴───────────┘

No tensor is split across ranks.
Unshard: Broadcast from owner rank.
```

---

### 3. MST Critical Methods

| Method | Description |
|--------|-------------|
| `.local()` | Get the local shard as a plain tensor |
| `.unshard()` | All-gather to reconstruct full DTensor |
| `.full_tensor()` | Unshard storage + handle remaining placements (e.g., TP) |
| `.shape` | Local shard shape |
| `.full_shape` | Original full tensor shape |
| `from_local_shard()` | Create MST from pre-sharded local tensor |

---

## Use Cases

All examples use 8 GPUs. See `test/distributed/tensor/test_mst_fsdp_demo.py` for runnable tests.

### Use Case 1: FSDP v2 — Per-Tensor Dim 0 Sharding

Each tensor independently sharded along dim 0. Different tensors have different shard sizes.

```
Mesh: (dp=8)

Original tensors:          After scatter_tensor_storage(dim=0):
┌─────────────┐            ┌─────────────┐
│  (16, 8)    │            │  (2, 8)     │  ← each rank
├─────────────┤            ├─────────────┤
│  (32, 4)    │     →      │  (4, 4)     │
├─────────────┤            ├─────────────┤
│  (8, 16)    │            │  (1, 16)    │
├─────────────┤            ├─────────────┤
│  (24, 12)   │            │  (3, 12)    │
└─────────────┘            └─────────────┘
```

**Test**: `test_fsdpv2`
**API**: `scatter_tensor_storage(dt, dim=0, mesh_dim="dp")`

---

### Use Case 2: BSDP — 2D Block Sharding

Each tensor sharded into 2D blocks. Useful for distributed optimizers like Shampoo.

```
Mesh: (dp_row=4, dp_col=2)

Original tensor (16, 8):

     col 0-3    col 4-7
    ┌────────┬────────┐
r0  │ rank0  │ rank1  │  rows 0-3
    ├────────┼────────┤
r1  │ rank2  │ rank3  │  rows 4-7
    ├────────┼────────┤
r2  │ rank4  │ rank5  │  rows 8-11
    ├────────┼────────┤
r3  │ rank6  │ rank7  │  rows 12-15
    └────────┴────────┘

Each rank holds a (4, 4) block.
```

**Test**: `test_bsdp`
**API**: `scatter_tensor_storage(dt, dim=[0, 1], mesh_dim=["dp_row", "dp_col"])`

---

### Use Case 3: FSDP v1 — Flatten + Concat Element Sharding

All tensors flattened to 1D, concatenated into a single buffer, then sharded by elements. Enables single all-gather for all parameters.

```
Mesh: (dp=8)

Original tensors:              Flattened buffer:
┌──────────┐                   ┌────────────────────────────────────────┐
│ (16, 8)  │ → 128 elements    │ t0: 128 │ t1: 128 │ t2: 128 │ t3: 288 │
├──────────┤                   └────────────────────────────────────────┘
│ (32, 4)  │ → 128 elements              Total: 672 elements
├──────────┤
│ (8, 16)  │ → 128 elements    Sharded (672 / 8 = 84 per rank):
├──────────┤                   ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ (24, 12) │ → 288 elements    │ r0  │ r1  │ r2  │ r3  │ r4  │ r5  │ r6  │ r7  │
└──────────┘                   │ 84  │ 84  │ 84  │ 84  │ 84  │ 84  │ 84  │ 84  │
                               └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
                               A single tensor may span multiple ranks.
```

**Test**: `test_fsdpv1`
**API**: `scatter_tensor_group(tensors, mesh, "dp", boundary=ShardingBoundary.ELEMENT)`

---

### Use Case 4: Muon — Tensor-Level Distribution

Each rank owns complete tensors. No tensor is split across ranks.

```
Mesh: (dp=8), 4 tensors

Distribution:
┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
│ rank0  │ rank1  │ rank2  │ rank3  │ rank4  │ rank5  │ rank6  │ rank7  │
├────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤
│ tensor0│ tensor1│ tensor2│ tensor3│ (empty)│ (empty)│ (empty)│ (empty)│
│ (16,8) │ (32,4) │ (8,16) │(24,12) │        │        │        │        │
└────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘

Each tensor is whole on exactly one rank.
Unshard uses broadcast from owner.
```

**Test**: `test_muon`
**API**: `scatter_tensor_group(tensors, mesh, "dp", boundary=ShardingBoundary.TENSOR)`

---

### Use Case 5: veScale — Weighted Element Sharding

FSDP v1 with custom weights for heterogeneous memory scenarios.

```
Mesh: (dp=8), weights=[1, 2, 1, 1, 1, 1, 1, 1]  (sum=9)

Total: 810 elements, 90 per weight unit

┌─────┬──────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ r0  │  r1  │ r2  │ r3  │ r4  │ r5  │ r6  │ r7  │
│ 90  │ 180  │ 90  │ 90  │ 90  │ 90  │ 90  │ 90  │
│ w=1 │ w=2  │ w=1 │ w=1 │ w=1 │ w=1 │ w=1 │ w=1 │
└─────┴──────┴─────┴─────┴─────┴─────┴─────┴─────┘

Rank 1 holds 2x the elements (useful when rank 1 has more memory).
```

**Test**: `test_vescale`
**API**: `scatter_tensor_group(tensors, mesh, "dp", boundary=ShardingBoundary.ELEMENT, weights=[...])`

---

### Use Case 6: FSDP + TP Composition

Tensors already TP-sharded, then FSDP storage sharding applied on top.

```
Mesh: (dp=4, tp=2)

Original (16, 8):

Step 1 - TP sharding (mixed Shard(0) and Shard(1)):
┌──────────────────────┐
│ Tensor 0: Shard(1)   │  (16, 8) → (16, 4) per TP rank
│ Tensor 1: Shard(0)   │  (32, 4) → (16, 4) per TP rank
│ Tensor 2: Shard(1)   │  (8, 16) → (8, 8)  per TP rank
│ Tensor 3: Shard(0)   │  (24,12) → (12,12) per TP rank
└──────────────────────┘

Step 2 - FSDP storage sharding on TP shards (dim=0, dp=4):
┌──────────────────────┐
│ Tensor 0: (16,4)→(4,4)   │  dim 0 / 4
│ Tensor 1: (16,4)→(4,4)   │
│ Tensor 2: (8,8) →(2,8)   │
│ Tensor 3: (12,12)→(3,12) │
└──────────────────────┘
```

**Test**: `test_fsdp_plus_tp`
**API**: `scatter_tensor_storage(tp_sharded_dt, dim=0, mesh_dim="dp")`

---

### Use Case 7: Local Shard Operations

MST doesn't support direct tensor ops. Pattern for local updates:

```
Mesh: (dp=4)
Original tensor (16, 8), sharded on dim 0:

Step 1: MST holds local shards
┌─────────────┬─────────────┬─────────────┬─────────────┐
│   rank 0    │   rank 1    │   rank 2    │   rank 3    │
│   (4, 8)    │   (4, 8)    │   (4, 8)    │   (4, 8)    │
│    MST      │    MST      │    MST      │    MST      │
└─────────────┴─────────────┴─────────────┴─────────────┘
                      │
                      ▼ .local()
Step 2: Extract local tensor, apply element-wise ops
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ local * 2+1 │ local * 2+1 │ local * 2+1 │ local * 2+1 │
│   (4, 8)    │   (4, 8)    │   (4, 8)    │   (4, 8)    │
│   Tensor    │   Tensor    │   Tensor    │   Tensor    │
└─────────────┴─────────────┴─────────────┴─────────────┘
                      │
                      ▼ from_local_shard()
Step 3: Create new MST from modified locals
┌─────────────┬─────────────┬─────────────┬─────────────┐
│   rank 0    │   rank 1    │   rank 2    │   rank 3    │
│   (4, 8)    │   (4, 8)    │   (4, 8)    │   (4, 8)    │
│  new MST    │  new MST    │  new MST    │  new MST    │
└─────────────┴─────────────┴─────────────┴─────────────┘

No communication needed for element-wise operations!
```

```python
# Step 1: Extract local tensor
local = sharded.local()

# Step 2: Perform operations (element-wise, matmul, etc.)
updated = local * 2 + 1

# Step 3: Create new MST from modified local
new_sharded = MemoryShardedDTensor.from_local_shard(
    local_shard=updated,
    full_shape=sharded.full_shape,
    shard_dim=0,
    device_mesh=mesh,
    mesh_dim="dp",
)
```

**Test**: `test_local_shard_element_wise_update`

---

### Use Case 8: Shampoo Optimizer — BSDP with Local Matmul

Shampoo computes preconditioners from local gradient blocks without communication.

```
Mesh: (dp_row=4, dp_col=2)
Weight (16, 8) and Gradient (16, 8) both sharded as 2D blocks:

         col 0-3      col 4-7
        ┌──────────┬──────────┐
row 0-3 │  rank 0  │  rank 1  │   Weight block: (4, 4)
        │  W, G    │  W, G    │   Gradient block: (4, 4)
        ├──────────┼──────────┤
row 4-7 │  rank 2  │  rank 3  │
        │  W, G    │  W, G    │
        ├──────────┼──────────┤
   ...  │   ...    │   ...    │
        └──────────┴──────────┘

Per-rank computation (NO communication):
┌────────────────────────────────────────────────────────────┐
│  local_weight, local_grad = weight.local(), grad.local()   │
│                                                            │
│  # Compute preconditioners from local gradient block       │
│  L = local_grad @ local_grad.T        # (4,4) → (4,4)     │
│  R = local_grad.T @ local_grad        # (4,4) → (4,4)     │
│                                                            │
│  # Apply preconditioned update                             │
│  precond_grad = L @ local_grad @ R    # (4,4)             │
│  updated = local_weight - lr * precond_grad                │
└────────────────────────────────────────────────────────────┘

Each rank independently computes its block update.
Block structure enables local matmul without needing full tensor.
```

**Approximations in demo**:
- Real Shampoo accumulates L, R over iterations with momentum
- Real Shampoo uses L^{-1/4} @ grad @ R^{-1/4}

**Test**: `test_shampoo_local_matmul`
**API**: `shard_bsdp()` + `.local()` for block extraction

---

## Use Case 9: DCP Resharding

Block sharding enables efficient resharding for checkpoint loading with different world sizes.

**Scenario**: Save with 8 GPUs, load with 16 GPUs.

```
Saved (8 ranks, 2x4 blocks):     Load (16 ranks, 4x4 blocks):
┌───┬───┬───┬───┐                ┌─┬─┬─┬─┬─┬─┬─┬─┐
│ 0 │ 1 │ 2 │ 3 │                │0│1│2│3│4│5│6│7│
├───┼───┼───┼───┤       →        ├─┼─┼─┼─┼─┼─┼─┼─┤
│ 4 │ 5 │ 6 │ 7 │                │8│9│...       │
└───┴───┴───┴───┘                └─┴─┴─┴─┴─┴─┴─┴─┘
```

**Key optimization**: If new block size divides evenly into old (or vice versa):
- **Splitting** (old → smaller): Local operation, no communication
- **Merging** (old → larger): Partial all-gather on affected dimensions only
- **Incompatible**: Fall back to full unshard + re-shard

**Pattern** (current, without optimization):
```python
# Unshard MST storage
tp_sharded = bsdp_sharded.unshard()

# Redistribute DTensor placements
redistributed = tp_sharded.redistribute(placements=[...])

# Re-apply storage sharding
resharded = scatter_tensor_storage(redistributed, dim=0, mesh_dim="dp_row")
```

**TODO**: Implement `reshard_storage()` helper with efficient block-aligned resharding.

---

## FSDP + Autograd + Activation Checkpointing

MST is a **storage optimization**, not a compute type. To achieve memory savings during training, MST must integrate with autograd and activation checkpointing correctly.

**The challenge**: PyTorch autograd saves tensors needed for backward. Without management, calling `unshard()` saves the huge unsharded tensor, eliminating memory savings.

**The solution**: Use activation checkpointing. Checkpoint saves the MST (small) and recomputes `unshard()` during backward:

```python
def forward_with_params(x, param_mst):
    param_full = param_mst.unshard()  # Inside checkpoint - NOT saved
    return F.linear(x, param_full)

# Checkpoint saves param_mst (small), recomputes unshard on backward
output = checkpoint(forward_with_params, x, param_mst)
```

**Flow**:
```
Forward:
  checkpoint saves: [input, param_mst]  ← MST is small!
  discards: param_full (unsharded)

Backward:
  checkpoint recomputes forward_with_params()
  → param_mst.unshard() called again (all-gather)
  → gradients computed with recomputed param_full
```

**Memory savings**:
- Without checkpoint: peak = MST + unsharded + activations (no savings)
- With checkpoint: peak = MST + activations (huge savings!)

The key insight: `unshard()` is deterministic (all-gather same data), so recomputing it on backward is cheap compared to storing the huge unsharded tensor.

---

## Summary

| Use Case | API | Sharding Pattern |
|----------|-----|------------------|
| FSDP v2 | `scatter_tensor_storage(dim=0)` | Per-tensor, 1D |
| BSDP | `scatter_tensor_storage(dim=[0,1])` | Per-tensor, 2D blocks |
| FSDP v1 | `scatter_tensor_group(boundary=ELEMENT)` | Flatten+concat, elements |
| Muon | `scatter_tensor_group(boundary=TENSOR)` | Whole tensors per rank |
| veScale | `scatter_tensor_group(..., weights=[...])` | Weighted elements |
| FSDP+TP | `scatter_tensor_storage` on TP-sharded DTensor | Layered sharding |
| Shampoo | BSDP + `.local()` for matmul | Block ops |
| DCP | Block sharding + reshard helper (future) | Checkpoint resharding |
