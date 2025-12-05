# Overlap Scheduling Memory Regression Investigation

## Problem Statement

Overlap scheduling in PyTorch inductor causes a significant memory regression. The test case (`llama1d_bwd.py`) shows:

- **Original peak memory**: 65.73 GB
- **Rescheduled peak memory**: 105.75 GB (before fix)
- **Regression**: 40.02 GB (60.9% increase)

The overlap scheduler reorders FX graph nodes to overlap collective communication (all_gather, reduce_scatter) with compute operations. While this improves performance by hiding communication latency, it was causing massive memory increases.

## Key Files

- `/tmp/pytorch-work/torch/_inductor/fx_passes/overlap_scheduling.py` - Main scheduler
- `/tmp/pytorch-work/llama1d_bwd.py` - Test case (Llama backward pass, 64 ranks, 581 collectives)

## Investigation Findings

### 1. Prefetching is NOT the Problem

Initial hypothesis was that prefetching collectives early increases memory. Testing showed the opposite:

| Configuration | Peak Memory |
|--------------|-------------|
| With prefetching | 105.75 GB |
| Without prefetching | 173.54 GB |

**Prefetching HELPS by 68 GB** because it allows reduce_scatter operations (which free memory) to run earlier.

### 2. Off-path Collectives are Beneficial

- **On-path**: 290 all_gather (memory-increasing: small input → large output)
- **Off-path**: 291 reduce_scatter (memory-beneficial: large input → small output)

Blocking off-path prefetches made memory WORSE. The fix was to only block non-reduce_scatter off-path prefetches.

### 3. Root Cause: Wait Deferral

The actual cause of the 40 GB regression is **wait node deferral** in the main scheduling loop.

The scoring function in `_compute_score()` assigns:
- `compute_local_priority = 1` for exposed waits (defer them)
- `compute_local_priority = 0` for other nodes

This causes waits to be scheduled later than in the original order.

### 4. Why Deferring Waits Increases Memory

**Waits themselves don't free memory** - they just synchronize streams. The memory is freed by **downstream nodes that depend on waits**.

When a wait is deferred:
1. All nodes that depend on that wait are also deferred
2. Those downstream nodes are often the **last users** of large tensors
3. Deferring them delays when those tensors can be freed
4. Meanwhile, other nodes allocate new memory
5. Result: Memory builds up

Evidence:
- Original peak: index 557/8052 (7% through graph)
- Rescheduled peak: index 4511/8052 (56% through graph)
- 65 nodes after waits free 43.62 GB when scheduled

### 5. Why Forcing Waits Doesn't Directly Help

Attempted fix: Force-schedule waits when memory is high.

Problem: The wait's direct users have OTHER dependencies besides the wait:
- 204 out of 207 users were `not_ready` after scheduling the wait
- They couldn't be scheduled because they depend on other unscheduled nodes

This means we can't force the memory-freeing path - it has complex dependencies.

### 6. Heap Scoring is Stale

The scheduler uses a heap with pre-computed scores:
- Scores are computed at **push time** (when node becomes ready)
- Memory state changes between push and pop
- By the time a wait is popped, memory might be much higher than when it was scored

## Solution

The working fix: **When memory exceeds original profile + headroom, fall back to original node ordering**.

```python
# In _compute_score()
memory_tight = current_mem > original_target + gb_to_bytes(1.0)

if memory_tight:
    # When memory is tight, use original order as primary sort key
    # This prevents further reordering that could increase memory
    return (
        0,  # Don't deprioritize based on domination
        0,  # Don't deprioritize based on wait/collective
        self.node_idx[node],  # Stick to original order
    )
```

This approach:
1. Allows overlap optimization when memory is fine
2. Falls back to original order when memory is tight
3. Naturally keeps memory-freeing nodes scheduled earlier

### Results After Fix

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| Rescheduled peak | 105.75 GB | 72.15 GB |
| Regression | 40.02 GB (60.9%) | 6.42 GB (9.8%) |

## Memory Tracking Infrastructure Added

Several tracking mechanisms were added during investigation:

1. **`wait_freeing_potential`**: Pre-computed estimate of memory freed by scheduling each wait (via downstream nodes)
2. **`original_mem_before_compute_index`**: Original memory profile at each compute index
3. **`_should_force_wait_proactively()`**: Check if we should force a wait before memory builds up
4. **`_should_schedule_wait_instead()`**: Dynamic check after popping from heap

## Remaining Issues

1. **6.42 GB residual regression**: Still some memory increase vs original
2. **Debug code**: Many print statements need cleanup
3. **Performance impact**: The memory-tight check adds overhead to scoring

## Key Insights for Future Work

1. **Memory freeing is transitive**: Scheduling node A doesn't free memory; scheduling the nodes that USE A's inputs (as their last use) frees memory.

2. **Heap-based scheduling with pre-computed scores is fundamentally at odds with memory-aware scheduling**: Scores become stale as memory state changes.

3. **reduce_scatter vs all_gather**: These have opposite memory characteristics. Scheduling policy should treat them differently.

4. **Original order is a good baseline**: The original topological order was memory-efficient. Deviation from it should be limited when memory is constrained.

5. **Path scheduling matters**: When prefetching, we schedule an entire path from current position to the collective. Paths that include waits can trigger cascading prefetches.

## Test Commands

```bash
# Run the test
python llama1d_bwd.py

# Key output to look for:
#   original_peak_memory: X GB
#   rescheduled_peak_memory: Y GB
#   memory_increase (rescheduled): Z GB (W%)
```
