# Implicit Invariants and Tips for Writing FX Graph Passes
## Fake Tensor metadata on node
Each FX node has metadata on it, and in particular, stores a faketensor representing the metadata of that node `node.meta['val']`. This FakeTensor has properties like 1. shape, 2. stride, and 3. aliasing information. However, various passes may change the faketensor values, and so we need to maintain consistency.

Passes may assume that FakeTensor metadata is consistent when they begin. If a pass changes node inputs or outputs in a way that makes downstream metadata stale, it must update the affected metadata itself or run `FakeTensorUpdater` from `_inductor/fx_utils.py` before returning. Passes do not need to run `FakeTensorUpdater` before each metadata read.

## Graph outputs
After AOTDispatch, joint and post-grad graph outputs are either a single FX node or a flat list or tuple of FX nodes.

## Mutations throughout the stack
The invariant about mutation we have is:

**After AOTDispatch tracing and before Inductor, we have no mutation in our graph, except for a copy_ epilogue at the end of the graph.**

For example, passes operating on the joint_graph and post_grad graph do not need to worry about mutation, except for the rare `aten.set_`. Its presence means the graph contains mutation, so passes that rely on the functional graph invariant may conservatively exit early when they encounter it.

However, we do still have aliasing in the graph. This does not matter most of the time, but it does mean that **our passes are not allowed to cause any additional inputs/outputs to alias if they did not alias in the original graph**.

For example
```python
def f(x: Tensor):
    return x.clone()
```
cannot be turned into a no-op, as this would change the semantics of the compiled graph.

In addition, AOTDispatch can introduce a copy_ epilogue into the graph. For example, we may have a graph like
```python
def f(x: Tensor):
    y = x.clone()
    x.copy_(y)
    return y
```
In this case, we are also not allowed to eliminate `x.clone()`. Luckily, the
condition for when this can cause problems is the same as with aliasing,
which is that **our passes are not allowed to cause the input and output to
alias if they did not alias in the original graph**. To check whether the
inputs and outputs have any aliasing, it suffices to check whether the
storages of the input and the storages of the output have any overlap. See
`remove_noop_ops` for an example of how to do this.

Additionally, we do have one pass that *does* introduce mutation - `reinplace_inplaceable_ops`. This pass must run *just before Inductor lowering*, as otherwise this breaks our invariant.
