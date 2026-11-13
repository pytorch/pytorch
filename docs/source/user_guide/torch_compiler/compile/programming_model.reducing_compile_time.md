# Reducing Compile Time

`torch.compile` is a just-in-time compiler, so the first one or two calls to a
compiled function pay a compilation cost before you see any speedup. Cold-start
(uncached) compile time typically ranges from seconds to minutes for common
models, and can reach tens of minutes to a few hours for very large ones. This
page collects strategies for reducing that cost.

This is distinct from reducing the per-call *guard* overhead paid on every
invocation — for that, see
[Reducing Guard Overhead](programming_model.reducing_guard_overhead).

## Avoid unnecessary recompilations

Each recompilation pays the compile cost again, so the single most common cause
of excessive compile time is recompiling more than necessary. Diagnose
recompilations with tlparse or `TORCH_LOGS=recompiles`, then remove their
causes. See [Dealing with Recompilations](programming_model.recompilation) for
the full workflow.

## Use dynamic shapes to avoid shape-driven recompiles

By default, `torch.compile` specializes on the shapes it sees, so a program that
runs many different input shapes can recompile once per shape. Letting Dynamo
treat a dimension as dynamic compiles a single graph that covers a range of
sizes, avoiding a recompile per shape.

Whether dynamic shapes are attempted is controlled by the `dynamic` argument to
`torch.compile`; see
[Is Dynamic Shapes Enabled?](programming_model.recompilation) for how the
`dynamic=None`/`True`/`False` settings behave. To keep a specific dimension
dynamic from the first compile, annotate it explicitly with `mark_dynamic`:

```python
import torch

# Mark dim 0 of x as dynamic, optionally with a min/max range, BEFORE compiling.
torch._dynamo.mark_dynamic(x, 0, min=1, max=1024)

opt_fn = torch.compile(fn)
opt_fn(x)
```

`mark_dynamic` must be called on the input tensors *before* they are passed into
the compiled function (not inside the compiled region). For the full model of
what "dynamic" means, automatic dynamic, and other annotations, see
{ref}`dynamic_shapes`.

## Regional compilation

Instead of compiling an entire large model in one shot, you can compile a
smaller *region* that is repeated throughout the model — for example, a single
transformer block — and let that same compiled region run for every repetition.
Because the region is compiled once rather than once per repetition, cold-start
compile time drops substantially for models built from many identical blocks.

See the
[Regional compilation recipe](https://docs.pytorch.org/tutorials/recipes/regional_compilation.html)
for a worked example.

## Hierarchical compilation with `nested_compile_region`

`torch.compiler.nested_compile_region` marks a set of operations — again,
typically a repeated structural unit such as an LLM's transformer layer — as a
nested compile region. During a full-model `torch.compile`, the compiler emits
optimized code for the region the first time it is encountered and re-emits
("stamps out") that compiled code on every subsequent occurrence, rather than
re-compiling each one. This can substantially reduce overall compile time for
deeply-stacked, structurally-identical components.

```python
import torch

@torch.compiler.nested_compile_region
def transformer_layer(x):
    ...

@torch.compile
def model(x):
    for _ in range(num_layers):
        x = transformer_layer(x)
    return x
```

Unlike regional compilation, this works *inside* a single `torch.compile` call
and does not require restructuring how you apply `torch.compile`. It is also
safe: it does **not** guarantee the region is compiled exactly once — if new
input conditions (shape, dtype, device, stride, globals, etc.) would make the
cached region invalid, the compiler transparently re-compiles it, so correctness
is always preserved and you only pay the extra compile cost when required.
Outside a `torch.compile` context the call is a no-op.

## Measuring compile time

Before optimizing, measure where compile time is going:

- `torch._dynamo.utils.compile_times()` prints the time spent in each Dynamo
  compilation phase.
- `TORCH_COMPILE_DEBUG=1` surfaces Inductor phase timings and artifacts.
- tlparse / `TORCH_TRACE` gives a per-compilation report; see
  [tlparse / TORCH_TRACE](programming_model.observability).

Finally, remember that `torch.compile`'s caches persist compiled artifacts
across invocations and even across processes, so a warmed-up (cached) run
compiles much faster than a cold start.
