# Understanding `seq_nr` in PyTorch FX Graphs

We use `seq_nr` to map nodes between forward and backward graphs. While this is not new, the mapping has not been load-bearing in the past. However, with the new annotation and regional inductor APIs, the usage of `seq_nr` node meta is now load-bearing. So, it needs to be robust and we need to understand how it works.

In this document, I'll explain how `seq_nr` works for most nodes, and some special cases (e.g. `invoke_subgraph` nodes, grad accumulation nodes).

## Basics

First, some basic facts about `seq_nr`:

- `seq_nr` is an integer that lives in `node.meta["seq_nr"]`
- `seq_nr` is assigned to nodes in [`torch/fx/proxy.py` in `_get_seq_nr()`](https://github.com/pytorch/pytorch/blob/58a92ff33219dc55cff7351c8d4d8e2aef1f3ceb/torch/fx/proxy.py#L457-L508)

## Background: Global Sequence Number in Autograd Engine

There is a thread-local sequence number in [`aten/src/ATen/SequenceNumber.cpp`](https://github.com/pytorch/pytorch/blob/58a92ff33219dc55cff7351c8d4d8e2aef1f3ceb/aten/src/ATen/SequenceNumber.cpp). I'm going to call this the "global sequence number". We can peek at this number in python with
`torch.autograd._get_sequence_nr()`.
We query this global sequence number when deciding the `seq_nr` for proxy nodes, but conceptually **`seq_nr` is NOT the same as this global sequence number**.


```cpp
namespace at::sequence_number {
namespace {
thread_local uint64_t sequence_nr_ = 0;
}

uint64_t peek() {
  return sequence_nr_;
}

uint64_t get_and_increment() {
  return sequence_nr_++;
}
}
```

The global sequence number increments when we construct an autograd node without explicitly passing in a sequence number. This happens in the `Node` constructor in [`torch/csrc/autograd/function.h`](https://github.com/pytorch/pytorch/blob/58a92ff33219dc55cff7351c8d4d8e2aef1f3ceb/torch/csrc/autograd/function.h#L137-L140):

```cpp
explicit Node(edge_list&& next_edges = edge_list())
    : Node(
          /*sequence_nr=*/at::sequence_number::get_and_increment(),
          std::move(next_edges)) {}
```

This number is stored in the autograd node.

In most cases, we create an autograd node without explicitly specifying a sequence number, except:

1. **The grad accumulation node** ([`torch/csrc/autograd/functions/accumulate_grad.cpp`](https://github.com/pytorch/pytorch/blob/58a92ff33219dc55cff7351c8d4d8e2aef1f3ceb/torch/csrc/autograd/functions/accumulate_grad.cpp)) - we give it `UINT64_MAX` because we want it to execute first in topological order.

2. **The Error node** ([`torch/csrc/autograd/functions/basic_ops.h`](https://github.com/pytorch/pytorch/blob/58a92ff33219dc55cff7351c8d4d8e2aef1f3ceb/torch/csrc/autograd/functions/basic_ops.h#L15-L24)) - we give it `UINT64_MAX` because it should never be reached during backprop, and if executed, should run immediately and stop execution.

This global sequence number has many uses:
- Determine the order of execution in the grad engine
- Used by the profiler for correlating operations
- Used to compute the `seq_nr` for proxy nodes


## Our assumptions


## How `seq_nr` is Assigned to Nodes (The basic logics)

The assignment logic is in [`_get_seq_nr()` in `torch/fx/proxy.py`](https://github.com/pytorch/pytorch/blob/58a92ff33219dc55cff7351c8d4d8e2aef1f3ceb/torch/fx/proxy.py#L457-L508).

### Forward Nodes

When creating forward proxy nodes, we assign `torch.autograd._get_sequence_nr() - 1` to the nodes. The assumption is that before creating each proxy node, we also created an autograd node, which increments the global sequence number.

```python
# Here we decrement to account for the sequence_nr having
# just been incremented while tracing this lowered aten op.
new_seq_nr = torch.autograd._get_sequence_nr() - 1
```
So, the `seq_nr` on each forward node should be the same as the sequence number on the `autograd_fn` node on the output tensor produced by the node.

### Backward Nodes

When running the autograd engine backward() and creating backward proxy nodes, we set up a prehook on autograd nodes so we know the sequence number of the current autograd node. The setup is in [`torch/_functorch/_aot_autograd/logging_utils.py`](https://github.com/pytorch/pytorch/blob/58a92ff33219dc55cff7351c8d4d8e2aef1f3ceb/torch/_functorch/_aot_autograd/logging_utils.py#L106-L129):

```python
def get_prehook(stack_, seq_nr):
    def prehook(grad_output):
        fx_traceback.set_grad_fn_seq_nr(seq_nr)  # stores seq_nr in global current_meta
    return prehook

for node in iter_graph(roots):  # roots is an autograd graph
    node.register_prehook(get_prehook(forward_node_stack, node._sequence_nr()))
```

Then when creating the proxy node, we look at the `seq_nr` in `current_meta` and assign it to the backward node:

```python
if current_meta.get("in_grad_fn", 0) > 0:
    # This branch is used to get seq_nr for backward nodes
    new_seq_nr = current_meta["grad_fn_seq_nr"][-1]
```

This ensures backward nodes get the same `seq_nr` as their corresponding forward nodes.
In `pytorch/torch/csrc/autograd/engine.cpp`, you can see that the prehook is run right before we run the actual backward function. So, the `seq_nr` is set right before we create the backward proxy node.


### Grad Accumulation Nodes

Grad accumulation nodes are special because
 - They are not part of the autograd graph, so we can explicitly set hooks on them in python
 - They're not forward nodes and also not backward nodes, so the goal is not to find a corresponding forward or backward node for them. The goal is to explicitly mark that these are grad accumulation nodes.

If you're interested in details, the grad accumulation happens here:

- [`torch/csrc/autograd/input_buffer.cpp:99`](https://github.com/pytorch/pytorch/blob/58a92ff33219dc55cff7351c8d4d8e2aef1f3ceb/torch/csrc/autograd/input_buffer.cpp#L99) - `add()` method
- [`torch/csrc/autograd/input_buffer.cpp:220`](https://github.com/pytorch/pytorch/blob/58a92ff33219dc55cff7351c8d4d8e2aef1f3ceb/torch/csrc/autograd/input_buffer.cpp#L220) - accumulation logic
- [`torch/csrc/autograd/engine.cpp:1219`](https://github.com/pytorch/pytorch/blob/58a92ff33219dc55cff7351c8d4d8e2aef1f3ceb/torch/csrc/autograd/engine.cpp#L1219) - engine execution

Because we cannot explicit set hook on them like we do for other autograd nodes, we use the following trick.
After `auto outputs = call_function(graph_task, func, inputs);`, the post_grad hooks are run. In post_grad hooks, we add a special "grad accumulation stack" marker to `current_meta`'s stack:

```python
GRADIENT_ACC_SPECIAL_STACK = (
    "Gradient addition node due to multiple use of tensor around:"
)
```

When we run the next node, the pre_grad hook will re-set the stack again. So this special stack should only exist after the current `call_function` and before the next `call_function`. Our assumption is that the only operations that can happen between those are the grad accumulation nodes.

For these nodes, we mark them with `is_gradient_acc`:

```python
if fx_traceback.GRADIENT_ACC_SPECIAL_STACK in stack_trace:
    node.meta["is_gradient_acc"] = True
```


## When This Can Go Wrong (Special Cases)


### 1. Re-creating the node lost original seq_nr

#### Re-tracing a Joint Graph

When re-tracing a joint graph (e.g., in higher-order ops like `invoke_subgraph`), the `seq_nr` can get lost. All nodes would get a new `seq_nr` during re-trace, and the original useful `seq_nr` that carry the fwd-bwd correspondance is lost.

**Solution**: Use `_preserve_node_seq_nr()` context manager ([`torch/fx/traceback.py`](https://github.com/pytorch/pytorch/blob/58a92ff33219dc55cff7351c8d4d8e2aef1f3ceb/torch/fx/traceback.py#L270-L279)):

```python
# From torch/_higher_order_ops/invoke_subgraph.py
with torch.fx.traceback._preserve_node_seq_nr():
    graph = reenter_make_fx(subgraph, ...)(*operands)
```


#### Functionalization View Replay

In functionalization, nodes can be replayed, and these nodes should preserve the original `seq_nr`. This is handled via `set_current_replay_node()` ([`torch/fx/traceback.py`](https://github.com/pytorch/pytorch/blob/58a92ff33219dc55cff7351c8d4d8e2aef1f3ceb/torch/fx/traceback.py#L458-L479)):

```python
replay_node = fx_traceback.get_current_replay_node()
if replay_node is not None:
    if "seq_nr" in replay_node.meta:
        new_seq_nr = replay_node.meta["seq_nr"]
```

### 2. Unexpected Sequence Number Increments

The global sequence number should NOT increment again between the autograd node creation and the proxy node creation for forward nodes. If something creates an autograd node in between, the `seq_nr` will be wrong, because `torch.autograd._get_sequence_nr() - 1` is not the same as the original autograd node that corresponds to the proxy node.

We fixed one such bug in https://github.com/pytorch/pytorch/pull/172687 where Error nodes are incrementing the global sequence number.

### 3. Autograd node is created after proxy node in Dynamo.

This is not exactly a bug, but it's a case where our assumption is not True.

In dynamo, we create the proxy node first, then autograd nodes are created when we run the node on FakeTensor with `requires_grad=True` in [`torch/_dynamo/utils.py`](https://github.com/pytorch/pytorch/blob/58a92ff33219dc55cff7351c8d4d8e2aef1f3ceb/torch/_dynamo/utils.py#L3599-L3601):

```python
if out.requires_grad:
    out.grad_fn  # This can trigger autograd node creation
```

This happens AFTER we have created the proxy node, so initially the `seq_nr` is calculated as `0 - 1 = -1`.

Sometimes you'll observe `seq_nr = -1` on dynamo graphs where proxy nodes get created before autograd nodes.

## Usage: Mapping Forward to Backward Nodes

The primary use of `seq_nr` is in [`copy_fwd_metadata_to_bw_nodes()`](https://github.com/pytorch/pytorch/blob/58a92ff33219dc55cff7351c8d4d8e2aef1f3ceb/torch/_functorch/_aot_autograd/utils.py#L607-L638):

```python
def copy_fwd_metadata_to_bw_nodes(fx_g: torch.fx.GraphModule) -> None:
    """
    This function walks the graph and copies over metadata from forward nodes
    to backward nodes, using the `seq_nr` field as a one-to-many mapping
    from forward node to backward node.
    """
```

This enables:
- **Debugging**: Understanding which backward operations correspond to which forward operations
- **Metadata propagation**: Copying `stack_trace`, `nn_module_stack`, and other metadata from forward to backward nodes

## Tips for Debugging `seq_nr`

To debug sequence number issues, you can add logging to find out which exact Python line caused `torch.autograd._get_sequence_nr()` to increment.

**Prompt for Claude**: "Add logging in pytorch to find out which exact python line did `torch.autograd._get_sequence_nr()` increment (optional: from X to Y in [command])."

This will help identify unexpected sequence number increments that may be causing `seq_nr` mismatches.
