# Implicit Invariants and Tips for Writing FX Graph Passes
## Fake Tensor metadata on node
Each FX node has metadata on it, and in particular, stores a faketensor representing the metadata of that node `node.meta['val']`. This FakeTensor has properties like 1. shape, 2. stride, and 3. aliasing information. However, various passes may change the faketensor values, and so we need to maintain consistency.

Passes may assume that FakeTensor metadata is consistent when they begin. If a pass changes node inputs or outputs in a way that makes downstream metadata stale, it must update the affected metadata itself or run `FakeTensorUpdater` from `_inductor/fx_utils.py` before returning. Passes do not need to run `FakeTensorUpdater` before each metadata read.

## Alias analysis
Passes should determine tensor aliasing from FakeTensor storage identity rather than operator schema alias annotations. Two tensor nodes with the same non-`None` storage ID alias. A `None` storage ID means aliasing is unknown and must not be used to establish an alias relationship.

## Aliasing
Although these graphs are mutation-free, they may still contain aliasing. Passes may change aliasing relationships among intermediate tensors because internal storage identity is not part of the functional operator contract. In particular, functional custom operators cannot rely on whether two intermediate inputs share storage; there is no schema or tag for declaring such a dependency.

Passes must preserve aliasing relationships that escape through user-visible inputs and outputs. They must neither introduce nor remove user-visible input-output or output-output aliases. Saved activations that appear as inputs to backward graphs are internal compiler values and are not subject to these boundary aliasing constraints.

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

## Graph outputs
After AOTDispatch, joint and post-grad graph outputs are either a single FX node or a flat list or tuple of FX nodes.

## Mutations throughout the stack
The invariant about mutation we have is:

**After AOTDispatch tracing and before Inductor, we have no mutation in our graph, except for a `copy_` epilogue at the end of the graph and the rare `aten.set_`.**

Passes operating on the joint_graph and post_grad graph do not need to account for other mutations. Pass authors that do not support the rare `aten.set_` may conservatively exit early if it exists in the graph:

```python
if graph.find_nodes(op="call_function", target=aten.set_.default):
    return
```

Additionally, we do have one pass that *does* introduce mutation - `reinplace_inplaceable_ops`. This pass must run *just before Inductor lowering*, as otherwise this breaks our invariant.

## Order of random operations
With the default `fallback_random=False`, Inductor uses functional RNG lowering whose state is represented explicitly. Passes may reorder independent random operations in this mode, provided they preserve their explicit operands and data dependencies and do not duplicate or remove them.

When `fallback_random=True`, random operations consume the global CUDA RNG state, so all passes must preserve their relative order. Before `_chain_random_ops_for_ordering` runs, passes preserve the relative graph-list order of random operations, even when the graph is temporarily not topologically sorted. After it adds explicit control dependencies between random operations, passes preserve their order by respecting those dependencies. Passes may move deterministic nodes around random operations, but they must not reorder the random operations relative to one another.

Before chaining, passes must also avoid introducing dependencies that require the opposite RNG order. For example, if the graph-list order is `rng_1` followed by `rng_2`, introducing a dependency `rng_2 -> rng_1` conflicts with the later ordering dependency `rng_1 -> rng_2` and creates a cycle during `_chain_random_ops_for_ordering` before `stable_topological_sort` runs.

## Tensor layouts
The following invariants apply to intermediate tensors when Inductor is the backend; they are not general FX graph invariants:

- The exact storage offset of an intermediate tensor is not part of its semantic contract. Passes do not need to preserve it, and operators must not depend on its value.
- The exact strides of an intermediate tensor are not part of its semantic contract. During lowering, Inductor determines the input strides required by each consumer and ensures that the consumer receives them, restriding or materializing the input as needed.

These invariants do not apply to user-visible graph inputs and outputs, whose externally visible metadata and aliasing relationships must be preserved.

Saved activations are internal compiler values that appear as inputs to backward graphs, so they follow the intermediate-tensor rules instead. A plain FX graph does not reliably identify saved activations; passes that need this distinction require information propagated from the AOTAutograd graph signature rather than graph-structure heuristics.
