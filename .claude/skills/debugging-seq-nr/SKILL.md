---
name: debugging-seq-nr
description: Debug node.meta["seq_nr"] issues in PyTorch FX graphs. Use when investigating forward-backward node mapping issues, seq_nr mismatches, and unexpected sequence number increments. Also use when the user mentions seq_nr, sequence number, or forward-backward correspondence in compiled graphs.
---

# Debugging `seq_nr` in PyTorch FX Graphs

`seq_nr` maps nodes between forward and backward graphs. With the annotation and regional inductor APIs, `seq_nr` is now load-bearing and needs to be robust.

## When to Use This Skill

- Investigating forward-backward node mapping issues
- Investigating node metadata missing in backward graph issues
- Understanding unexpected `seq_nr` values
- Fixing `seq_nr` mismatches after graph transformations

## Quick Reference

```python
# Check seq_nr on a node
print(node.meta.get("seq_nr"))

# Check global sequence number
print(torch.autograd._get_sequence_nr())

# Check if node is gradient accumulation
print(node.meta.get("is_gradient_acc", False))
```

## How seq_nr Works

For detailed background, see [understanding.md](understanding.md).

### Key Points

1. **Forward nodes**: `seq_nr = torch.autograd._get_sequence_nr() - 1`
   - Assigned when creating proxy nodes
   - Should match the sequence number on the output tensor's `autograd_fn`

2. **Backward nodes**: `seq_nr` comes from prehooks on autograd nodes
   - Same `seq_nr` as corresponding forward nodes
   - Enables forward-backward mapping

3. **Grad accumulation nodes**: Marked with `is_gradient_acc = True`
   - Not part of normal forward-backward mapping
   - Identified by special stack marker

## Common Issues and Solutions

### Issue 1: seq_nr = -1 on Dynamo Graphs

**Symptom**: Forward nodes have `seq_nr = -1`

**Cause**: In Dynamo, proxy nodes are created before autograd nodes:
```python
# Proxy created first, then autograd node later
# seq_nr = 0 - 1 = -1
```

**Solution**: This is expected behavior in Dynamo.

### Issue 2: seq_nr Lost After Re-tracing

**Symptom**: After re-tracing a joint graph (e.g., in `invoke_subgraph`), all nodes have wrong `seq_nr`

**Cause**: Re-tracing assigns new sequence numbers, losing original forward-backward correspondence

**Solution**: Use `_preserve_node_seq_nr()` context manager:
```python
from torch.fx.traceback import _preserve_node_seq_nr

with _preserve_node_seq_nr():
    graph = reenter_make_fx(subgraph, ...)(*operands)
```

### Issue 3: Unexpected Sequence Number Increments

**Symptom**: `seq_nr` is off by some amount (e.g., always +1 or +2 from expected)

**Cause**: Something created an autograd node between the expected creation point and proxy node creation

**Debugging Steps**:
1. Add logging to track sequence number increments (see below)
2. Find what code is incrementing the global sequence number
3. Fix by either:
   - Preventing the unexpected increment
   - Explicitly passing sequence number to avoid `get_and_increment()`

### Issue 4: seq_nr Mismatch After Functionalization

**Symptom**: Replayed nodes have wrong `seq_nr`

**Cause**: Functionalization replays nodes without preserving original `seq_nr`

**Solution**: Use `set_current_replay_node()`:
```python
from torch.fx.traceback import set_current_replay_node, get_current_replay_node

# When replaying a node, set it as current
with set_current_replay_node(original_node):
    # Replayed operations will inherit seq_nr from original_node
    replayed_result = replay_operation(...)
```

## Debugging Techniques

### Add Logging for Sequence Number Increments

To find which Python line increments the global sequence number:

```python
import torch
import traceback

_original_get_and_increment = torch._C._autograd._get_sequence_nr

def _traced_get_sequence_nr():
    nr = _original_get_and_increment()
    print(f"seq_nr now: {nr}")
    traceback.print_stack(limit=10)
    return nr

# Monkey-patch (for debugging only)
# Note: This is approximate - the actual increment happens in C++
```

For C++ level debugging, add logging in `aten/src/ATen/SequenceNumber.cpp`:
```cpp
uint64_t get_and_increment() {
  auto old = sequence_nr_;
  sequence_nr_++;
  // Add logging here
  return old;
}
```

### Inspect seq_nr on All Nodes

```python
def print_seq_nr_info(graph_module):
    for node in graph_module.graph.nodes:
        seq_nr = node.meta.get("seq_nr", "N/A")
        is_grad_acc = node.meta.get("is_gradient_acc", False)
        print(f"{node.name}: seq_nr={seq_nr}, is_gradient_acc={is_grad_acc}")
```

### Validate Forward-Backward Mapping

```python
def validate_seq_nr_mapping(fw_graph, bw_graph):
    fw_seq_nrs = {n.meta.get("seq_nr") for n in fw_graph.nodes if "seq_nr" in n.meta}
    bw_seq_nrs = {n.meta.get("seq_nr") for n in bw_graph.nodes if "seq_nr" in n.meta}

    # Backward seq_nrs should be subset of forward (excluding grad acc)
    bw_non_acc = {n.meta.get("seq_nr") for n in bw_graph.nodes
                  if "seq_nr" in n.meta and not n.meta.get("is_gradient_acc")}

    missing = bw_non_acc - fw_seq_nrs
    if missing:
        print(f"Backward nodes with seq_nr not in forward: {missing}")
```

### Using the Claude Prompt

When debugging seq_nr issues, you can ask:

> "Add logging in pytorch to find out which exact python line did `torch.autograd._get_sequence_nr()` increment (optional: from X to Y in [command])."

This will help identify unexpected sequence number increments causing mismatches.

## Key Code Locations

| Component | Location |
|-----------|----------|
| seq_nr assignment | `torch/fx/proxy.py` - `_get_seq_nr()` |
| Global sequence number | `aten/src/ATen/SequenceNumber.cpp` |
| Backward prehook setup | `torch/_functorch/_aot_autograd/logging_utils.py` |
| Forward-backward copy | `torch/_functorch/_aot_autograd/utils.py` - `copy_fwd_metadata_to_bw_nodes()` |
| Preserve seq_nr context | `torch/fx/traceback.py` - `_preserve_node_seq_nr()` |
| Replay node context | `torch/fx/traceback.py` - `set_current_replay_node()` |

## Special Cases

### Grad Accumulation Nodes

- Have `node.meta["is_gradient_acc"] = True`
- seq_nr is `UINT64_MAX` (very large number)
- Not mapped to forward nodes

### Error Nodes

- Also have seq_nr `UINT64_MAX`
- Should never be reached during normal backprop

### invoke_subgraph Nodes

- Need `_preserve_node_seq_nr()` when re-tracing
- Inner nodes should retain their original seq_nr
