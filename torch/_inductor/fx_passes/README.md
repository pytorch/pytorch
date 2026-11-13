# Implicit Invariants and Idioms For Writing FX Graph Passes
These are the invariants and idioms that Inductor (GPU) post-grad FX passes may assume. They are not general FX graph invariants and may not hold for other backends.

## Fake Tensor metadata on node
Each FX node has metadata on it, and in particular, stores a faketensor representing the metadata of that node `node.meta['val']`. This FakeTensor records the node's shape, stride, and storage (which determines aliasing). However, various passes may change the faketensor values, and so consistency must be maintained.

Passes may assume that FakeTensor metadata is consistent when they begin. If a rewrite makes downstream metadata stale, the pass must update the affected metadata itself or use `FakeTensorUpdater` from `torch._inductor.fx_utils`. Create `FakeTensorUpdater(gm)` before rewriting the graph, then call its `incremental_update()` method afterwards, before returning. The updater detects changes relative to its snapshot at construction; creating it after the rewrite will not detect those changes. The outermost graph's input/output semantics must remain unchanged after the updater is created.

Other `node.meta` entries, in particular `eager_input_vals` (used by lowering to enforce stride requirements), must also be preserved when a pass replaces or rebuilds a node.

## Operator arguments
Passes cannot assume that an operator's arguments have been normalized into `Node.args`, because manually constructed nodes may place schema arguments in `Node.kwargs`. When inspecting a known operator, use `get_arg_value(node, index, kwarg_name)` to handle either representation:

```python
dim = get_arg_value(split_node, 2, "dim")
```


## Mutations
Immediately after AOTDispatch tracing, joint-graph and post-grad graphs contain no mutation except:

- a `copy_` epilogue that writes back mutated inputs (administrative view ops may follow these mutations), and
- rare input mutation ops: `aten.set_.source_Tensor` and `aten.shallow_copy_data_.default`, plus `inductor.resize_storage_bytes_.default` and `fsdp.copy_.default` used by FSDP.

Passes can assume the graph is otherwise functional only until the first mutation-introducing pass runs, such as `bucket_all_gather` / `bucket_fsdp_all_gather` when enabled, or `slice_scatter_chunking_pass`. Passes registered after that point must handle the mutations introduced by preceding passes.

Passes do not have to support the rare input mutation ops; because they are rare, it is usually fine to exit early if any appears:

```python
rare_mutation_ops = [
    aten.set_.source_Tensor,
    aten.shallow_copy_data_.default,
    torch.ops.inductor.resize_storage_bytes_.default,
]
if hasattr(torch.ops.fsdp, "copy_"):
    rare_mutation_ops.append(torch.ops.fsdp.copy_.default)
if any(graph.find_nodes(op="call_function", target=op, sort=False) for op in rare_mutation_ops):
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

Mutation-introducing passes must run after passes that rely on the functional invariant. Near lowering, `slice_scatter_chunking_pass` must run before `reinplace_inplaceable_ops`; `decompose_auto_functionalized` runs later and also introduces mutation. Any pass running after mutation has been introduced must support the resulting non-functional graph. Preserve these ordering constraints when adding or reordering passes.


## Altering Aliasing
- Passes may change aliasing relationships among intermediate tensors only when doing so preserves the values observed across any mutations in the graph. Internal storage identity is not part of the functional operator contract. In particular, functional custom operators cannot rely on whether two intermediate inputs share storage; there is no schema or tag for declaring such a dependency.

Changes to intermediate aliasing must also preserve tensor values in the presence of the mutation exceptions described above.

- Passes must preserve aliasing relationships that escape through user-visible inputs and outputs. They must neither introduce nor remove user-visible input-output or output-output aliases.

  Note: [#198735](https://github.com/pytorch/pytorch/pull/198735) will relax the "introduce" half of this rule for forward and inference graphs. AOTAutograd records which user inputs and outputs share a storage in eager (`ViewAndMutationMeta.aliased_input_output_pairs` and `aliased_output_pairs`). When lowering the output node, Inductor compares these with the buffers the user-visible outputs actually use, and copies any output that has a new input-output or output-output alias, keeping its original strides. Once it lands, introducing such an alias costs a copy but is not a correctness bug. Two constraints remain: removing an eager alias is not repaired, and the repair copy is made at the output, after input mutations, so passes must still preserve the values observed across mutations (see [Mutations](#mutations)). This section will be updated then.

For example
```python
a = b.clone()
c = b.clone()
# rewritten to
a = alias(b)
c = alias(b)
```
is fine when `a`, `b`, and `c` are all intermediates and no mutation makes their users observe different values. It breaks when `a` and `c` are both outputs (adds an output-output alias) or when `b` is an input and `a` or `c` is an output (adds an input-output alias).

Similarly,
```python
def f(x: Tensor):
    return x.clone()
```
cannot be turned into a no-op, as this would change the semantics of the compiled graph.

- Saved activations that appear as inputs to backward graphs are internal compiler values, but rewrites must preserve the values saved for backward, including across input mutations performed by the graph or runtime epilogue. A clone that saves an input's value before mutation must not be replaced with an alias that exposes the mutated value to backward.

## Alias analysis
To determine whether two tensors alias (e.g. whether `c = op1(a)` aliases `a`),
compare the FakeTensor storage identities in `a.meta["val"]` and `c.meta["val"]`
rather than inferring aliasing from the operator schema's alias annotations.

- Equal non-`None` storage IDs mean the tensors alias.
- Different non-`None` storage IDs mean they do not alias.
- A `None` storage ID means aliasing is unknown and must not be used to establish
  either aliasing or non-aliasing.

The same check extends to graph inputs and outputs: overlapping sets of known,
non-`None` storage IDs establish aliasing. Disjoint sets establish non-aliasing
only when all relevant storage IDs are known. Otherwise, a pass must handle the
unknown cases conservatively or use another analysis before rewriting.

Only use another approach when there is a strong reason to; using storage
identity everywhere keeps passes simple and consistent with each other.


## Tensor layouts
- The exact storage offset of an intermediate tensor is not part of its semantic contract. Passes do not need to preserve it, and operators must not depend on its value.
- A pass does not need to preserve the strides a node produces; a rewrite may change them. Stride requirements are enforced at each use rather than at the definition: during lowering, Inductor determines the strides each consumer requires and restrides or materializes the input as needed. Uses with stride requirements include non-codegen'd kernels, such as custom and extern ops, and user-visible graph outputs: when `config.keep_output_stride` is enabled (the default), Inductor records each output's original strides before post-grad passes run and restrides the output to match during lowering. Changing strides is therefore a performance concern rather than a correctness one: when the strides a node produces do not match what a use requires, Inductor may insert a copy at that use.

Graph input layouts are set by the caller and cannot be changed by passes.

## Topological order of nodes
FX passes must NOT assume that the current graph-list order is topological. Earlier passes may leave dependencies out of list order, and passes are not generally required to restore topological order after each rewrite.

A pass that requires a topological traversal must either establish that ordering itself or be registered at a pipeline point whose preceding passes have been carefully verified to guarantee it. This registration-order dependency must be maintained when the pipeline changes. This implicit contract is fragile; we want to provide a better API for it, e.g. a way for a pass to declare that it requires (or preserves) topological order so the pipeline can enforce it.

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
Some passes add **mandatory** control dependencies. For example, when
`fallback_random=True`, post-grad chains random ops so they run in eager order.
Passes that add new dependencies must run after these. A topological order
computed before the mandatory dependencies exist may not be valid once they are
added, so edges based on it can create a cycle.
