# Implicit Invariants and Idioms For Writing FX Graph Passes
These are the invariants and idioms that Inductor (GPU) post-grad FX passes may assume. They are not general FX graph invariants and may not hold for other backends.

## Fake Tensor metadata on node
Each FX node has metadata on it, and in particular, stores a faketensor representing the metadata of that node `node.meta['val']`. This FakeTensor records the node's shape, stride, and storage (which determines aliasing). However, various passes may change the faketensor values, and so consistency must be maintained.

Passes may assume that FakeTensor metadata is consistent when they begin. If a pass changes node inputs or outputs in a way that makes downstream metadata stale, it must update the affected metadata itself or run `FakeTensorUpdater` from `_inductor/fx_utils.py` before returning.

## Operator arguments
Passes cannot assume that an operator's arguments have been normalized into `Node.args`, because manually constructed nodes may place schema arguments in `Node.kwargs`. When inspecting a known operator, use `get_arg_value(node, index, kwarg_name)` to handle either representation:

```python
dim = get_arg_value(split_node, 2, "dim")
```


## Mutations
After AOTDispatch tracing, joint-graph and post-grad graphs contain no mutation except:

- a `copy_` epilogue at the end of the graph that writes back mutated inputs, and
- rare storage-mutating ops: `aten.set_.source_Tensor` and `aten.shallow_copy_data_.default`.

Passes can assume the graph is otherwise functional.

Passes do not have to support the storage-mutating ops; because they are rare, it is usually fine to exit early if either appears:

```python
storage_mutation_ops = (aten.set_.source_Tensor, aten.shallow_copy_data_.default)
if any(graph.find_nodes(op="call_function", target=op, sort=False) for op in storage_mutation_ops):
    return
```

The `copy_` epilogue exists because functionalization traces input mutations as functional ops, then writes the final value back with `input.copy_(new_value)`. It still constrains passes. In

```python
def f(x: Tensor):
    y = x.clone()
    x.copy_(y)
    return y
```

a pass must not eliminate `x.clone()`, by the same rule as in [Altering Aliasing](#altering-aliasing): passes must neither introduce nor remove aliasing between user-visible inputs and outputs.

The final post-grad passes `slice_scatter_chunking_pass` and `reinplace_inplaceable_ops` reintroduce mutation just before lowering.


## Altering Aliasing
For non-mutating graphs (those with no storage-mutating ops, as explained in [Mutations](#mutations)):
- Passes may change aliasing relationships among intermediate tensors because internal storage identity is not part of the functional operator contract. In particular, functional custom operators cannot rely on whether two intermediate inputs share storage; there is no schema or tag for declaring such a dependency.

- Passes must preserve aliasing relationships that escape through user-visible inputs and outputs. They must neither introduce nor remove user-visible input-output or output-output aliases.

For example
```python
a = b.clone()
c = b.clone()
# rewritten to
a = alias(b)
c = alias(b)
```
is fine when `a`, `b`, and `c` are all intermediates, but breaks when `a` and `c` are both outputs (adds an output-output alias) or when `b` is an input and `a` or `c` is an output (adds an input-output alias).

Similarly,
```python
def f(x: Tensor):
    return x.clone()
```
cannot be turned into a no-op, as this would change the semantics of the compiled graph.

- Saved activations that appear as inputs to backward graphs are internal compiler values and are not subject to these boundary aliasing constraints.

## Alias analysis
To determine whether two tensors alias (e.g. whether `c = op1(a)` aliases `a`),
compare the FakeTensor storage identities in `a.meta["val"]` and `c.meta["val"]`
rather than inferring aliasing from the operator schema's alias annotations.

- Equal non-`None` storage IDs mean the tensors alias.
- Different storage IDs mean they do not alias.
- A `None` storage ID means aliasing is unknown and must not be used to establish
  an alias relationship.

The same check extends to graph inputs and outputs: they alias iff their sets of
storages overlap. See `remove_noop_ops` for an example.

Only use another approach when there is a strong reason to; using storage
identity everywhere keeps passes simple and consistent with each other.


## Tensor layouts
The following invariants apply to intermediate tensors:

- The exact storage offset of an intermediate tensor is not part of its semantic contract. Passes do not need to preserve it, and operators must not depend on its value.
- The exact strides of an intermediate tensor are not part of its semantic contract. During lowering, Inductor determines the input strides required by each consumer and ensures that the consumer receives them, restriding or materializing the input as needed.

These invariants do not apply to user-visible graph inputs and outputs, whose externally visible metadata must be preserved.

Saved activations are internal compiler values that appear as inputs to backward graphs, so they follow the intermediate-tensor rules instead.

## Topological order of nodes
FX passes must NOT assume that the current graph-list order is topological. Earlier passes may leave dependencies out of list order, and passes are not generally required to restore topological order after each rewrite.

A pass that requires a topological traversal must either establish that ordering itself or be registered at a pipeline point whose preceding passes have been carefully verified to guarantee it. This registration-order dependency must be maintained when the pipeline changes.

Rewrites must still keep the dataflow graph acyclic.

## When a node can be deleted
Before deleting an FX node, a pass must verify that it has no users and that `Node.is_impure()` returns false. A node with no users may still have observable side effects, such as modifying storage or advancing RNG state.

```python
if node.users or node.is_impure():
    continue
graph.erase_node(node)
```

## Finding nodes
When searching for nodes with a particular operation and target, use `Graph.find_nodes` instead of scanning `Graph.nodes`. It uses the graph's lookup table and avoids visiting unrelated nodes. By default, `find_nodes` sorts matches into current graph-list order, which is not necessarily topological order. Pass `sort=False` when match order does not matter to avoid that sorting cost.

## Passes that add dependencies
Passes that add new dependencies must run after all **mandatory** control-dependency
passes. A topological order computed before those dependencies are added is not
valid: adding edges based on it can create a cycle once the mandatory
dependencies are inserted (e.g. random-op ordering under `fallback_random=True`).
