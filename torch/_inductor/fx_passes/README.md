# Implicit Invariants and Tips for Writing FX Graph Passes
## Fake Tensor metadata on node
Each FX node has metadata on it, and in particular, stores a faketensor representing the metadata of that node `node.meta['val']`. This FakeTensor has properties like 1. shape, 2. stride, and 3. aliasing information. However, various passes may change the faketensor values, and so we need to maintain consistency.

Passes may assume that FakeTensor metadata is consistent when they begin. If a pass changes node inputs or outputs in a way that makes downstream metadata stale, it must update the affected metadata itself or run `FakeTensorUpdater` from `_inductor/fx_utils.py` before returning. Passes do not need to run `FakeTensorUpdater` before each metadata read.

## Operator arguments
For dispatcher-traced `OpOverload` nodes in AOT-produced graphs, supplied schema arguments are recorded in schema order in `node.args`, regardless of whether the original Python call used positional or keyword arguments. Schema arguments do not appear in `node.kwargs`, so passes may index `node.args` according to the operator schema.

Passes that manually construct `OpOverload` nodes with `Graph.call_function`, including custom passes, must preserve this convention. This invariant does not apply to arbitrary Python functions or higher-order operators used as `call_function` targets.

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

## When an operation can be deleted
Before deleting an FX node, a pass must verify that it has no users and that `Node.is_impure()` returns false. A node with no users may still have observable side effects, such as modifying storage or advancing RNG state.

## Topological order of nodes
FX passes must NOT assume that the current graph-list order is topological. Earlier passes may leave dependencies out of list order, and passes are not generally required to restore topological order after each rewrite.

A pass that requires a topological traversal must either establish that ordering itself or be registered at a pipeline point whose preceding passes have been carefully verified to guarantee it. This registration-order dependency must be maintained when the pipeline changes.

Rewrites must still keep the dataflow graph acyclic. The post-grad pipeline runs `stable_topological_sort` before lowering to restore topological node order.

## Finding nodes
When searching for nodes with a particular operation and target, use `Graph.find_nodes` instead of scanning `Graph.nodes`. It uses the graph's lookup table and avoids visiting unrelated nodes. By default, `find_nodes` sorts matches into current graph-list order, which is not necessarily topological order. Pass `sort=False` when match order does not matter to avoid that sorting cost.

## Ordering dependencies
Pass authors do not need to special-case ordering requirements that are not represented by ordinary dataflow. If correctness depends on an implicit execution order, the component responsible for that invariant must encode the order as explicit graph dependencies. Other passes then treat those dependencies like any other graph edges. Graph insertion APIs only choose node position; they do not infer hidden ordering requirements or add the required dependencies automatically.

For example, when `fallback_random=True`, post-grad adds control dependencies between random operations before topological sorting and scheduling. Individual passes do not need to inspect `fallback_random` or reason about random-operation ordering.

## Tensor layouts
The following invariants apply to intermediate tensors when Inductor is the backend; they are not general FX graph invariants:

- The exact storage offset of an intermediate tensor is not part of its semantic contract. Passes do not need to preserve it, and operators must not depend on its value.
- The exact strides of an intermediate tensor are not part of its semantic contract. During lowering, Inductor determines the input strides required by each consumer and ensures that the consumer receives them, restriding or materializing the input as needed.

These invariants do not apply to user-visible graph inputs and outputs, whose externally visible metadata and aliasing relationships must be preserved.

Saved activations are internal compiler values that appear as inputs to backward graphs, so they follow the intermediate-tensor rules instead. A plain FX graph does not reliably identify saved activations; passes that need this distinction require information propagated from the AOTAutograd graph signature rather than graph-structure heuristics.
