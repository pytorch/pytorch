(torch.compiler_inductor_fx_passes)=

# FX Graph Passes

TorchInductor applies a series of FX graph passes at different stages of the
compilation pipeline. Each stage operates on a different form of the IR and has
different constraints. This page describes what the built-in passes do and when
they run.

For a walkthrough of where these passes fit in the overall pipeline, see the
[Architecture Overview](torch.compiler_inductor_overview.md).

## Pre-Grad Passes

**Source**: [torch/_inductor/fx_passes/pre_grad.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/fx_passes/pre_grad.py)

Pre-grad passes run on the high-level Torch IR before AOT Autograd. At this
stage, the graph contains the full set of 2000+ PyTorch operators (for example,
`torch.nn.functional.linear`). These passes transform the graph to improve
performance through operator fusion, elimination of redundant operations, and
graph simplification. The high-level IR can make it easier to perform pattern
matching and can expose fusion opportunities that would be harder to recognize
after decomposition.

However, the IR at this stage has **not** been normalized (canonicalized) or
functionalized (put in SSA form), making pre-grad passes more challenging to
write than post-grad passes. Specifically:

- Passes **must be safe with respect to aliasing and mutation**. A pass cannot
  assume that two tensor arguments point to different storage.
- Passes **must handle all possible argument schemas** for an operation, since
  the IR has not been canonicalized to a standard form.
- Manipulation at this stage is discouraged. TorchInductor uses limited-to-no
  manipulation on pre-grad IR.

## Joint Graph Passes

**Source**: [torch/_inductor/fx_passes/joint_graph.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/fx_passes/joint_graph.py)

Joint graph passes run on the combined forward and backward graph produced by
AOT Autograd. At this stage, the IR has been:

- **Functionalized** — put in SSA form with no mutations
- **Normalized** — canonicalized to a standard form
- **Decomposed** — reduced to fundamental ATen IR

This makes joint graph passes easier to write than pre-grad passes, since
the IR is functional and standardized. These passes focus on optimizations
that benefit from seeing both forward and backward operations together — for
example, patterns that need to match across the forward-backward boundary.

## Post-Grad Passes

**Source**: [torch/_inductor/fx_passes/post_grad.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/fx_passes/post_grad.py)

Post-grad passes run separately on the forward and backward graphs after the
joint graph has been split by the
[min-cut partitioner](https://dev-discuss.pytorch.org/t/min-cut-optimal-recomputation-i-e-activation-checkpointing-with-aotautograd/467).
These passes operate on normalized, functional ATen IR, making pattern matching
and transformations more straightforward than pre-grad passes. They focus on
low-level optimizations that benefit from seeing the fully decomposed and
functionalized representation, such as:

- **No-op elimination** — removing operations that have no effect
- **Dead code elimination** — removing unused computations
- **Pattern matching** — replacing recognized operation patterns with more
  efficient implementations

This is the **final stage** for high-level graph optimizations before the IR is
lowered to Inductor IR.

### Custom Post-Grad Pass Hooks

Users can add their own custom passes to the post-grad stage using
configuration hooks:

- `post_grad_custom_pre_pass` — runs **before** the built-in post-grad passes
- `post_grad_custom_post_pass` — runs **after** the built-in post-grad passes

These hooks allow you to insert custom graph transformations into the
compilation pipeline without modifying TorchInductor internals.

## Pattern Matching

TorchInductor includes a pattern matching framework used across all three pass
stages. The framework represents patterns as a DAG, searches for them
concurrently in a single graph traversal, and provides APIs for registering
custom patterns.

For details on how the pattern matcher works, the registration APIs
(`register_replacement`, `register_graph_pattern`, `register_lowering_pattern`),
and worked examples, see
[Pattern Matching](torch.compiler_inductor_pattern_matching.md).

## Key Invariants for FX Passes

When writing or reasoning about FX graph passes, there are important invariants
to keep in mind:

### FakeTensor Metadata

Each FX node stores a `FakeTensor` in `node.meta['val']` representing the
node's metadata (shape, stride, aliasing information). Passes that modify the
graph may need to update this metadata to maintain consistency. This is done
through `FakeTensorUpdater` in `torch/_inductor/fx_utils.py`.

### Mutation Constraints

After AOT Autograd tracing and before Inductor lowering, the graph has **no
mutation except for a `copy_` epilogue** at the end of the graph. Passes
operating on the joint graph and post-grad graph do not need to worry about
mutation.

However, there is still **aliasing** in the graph. Passes **must not cause
additional inputs or outputs to alias** if they did not alias in the original
graph. For example:

```python
def f(x: Tensor):
    return x.clone()
```

This cannot be turned into a no-op, because that would change the aliasing
semantics of the compiled graph.

:::{note}
The pass `reinplace_inplaceable_ops` is the one exception that introduces
mutation. It must run just before Inductor lowering to avoid breaking the
mutation invariant.
:::

For the full set of developer invariants, see
[torch/_inductor/fx_passes/README.md](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/fx_passes/README.md).

:::{seealso}
To learn how to write your own graph transformations on ATen IR, see
[Writing Graph Transformations](torch.compiler_transformations.md).
:::
