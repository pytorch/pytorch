(torch.compiler_inductor_pattern_matching)=

# Pattern Matching

The [pattern matcher](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/pattern_matcher.py)
performs peephole graph optimizations — replacing short sequences of known
operation patterns with fused or optimized sequences. It is the primary
mechanism that [FX graph passes](torch.compiler_inductor_fx_passes.md) use
to recognize and rewrite subgraphs.

For example, a well-known use case is identifying the multi-step sequence of
operations in scaled dot-product attention (matmul → scale → softmax → matmul)
and replacing it with a single fused `aten.scaled_dot_product_attention` call.
By doing so, TorchInductor can dispatch to an optimized flash-attention kernel
instead of executing the individual operations separately.

In general, the pattern matcher helps with:

- **Operation fusion** — combining sequences of ops into single optimized calls
- **Constant folding** — simplifying composite operations with known constants
- **Subgraph replacement** — swapping inefficient patterns with more efficient
  alternatives

**Source**: [torch/_inductor/pattern_matcher.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/pattern_matcher.py)

## How It Works

Internally, the pattern matcher represents patterns as a **DAG** (directed
acyclic graph). A single `PatternMatcherPass` may contain an arbitrary number
of patterns that the pass searches **concurrently** when iterating over the
graph. This is much more efficient than visiting the graph many times to apply
independent transformations — all registered patterns are matched in a single
traversal.

## Registration APIs

There are three APIs for registering patterns, each suited to different use
cases:

### `register_replacement()`

Create a replacement rule based on example functions that get traced to create
patterns. You provide a **search function** (the pattern to match) and a
**replacement function** (what to substitute), along with example input tensors
that are used to trace both functions into ATen IR.

This is the **recommended API** when applicable. It supports both training and
inference when run on a joint forward+backward graph, and it automatically
handles decomposition and canonicalization of the search pattern.

### `register_graph_pattern()`

Register a pattern that runs a custom function on the FX graph, allowing
arbitrary transformation code. This gives you full control over how the match
is processed and how the graph is modified. Use this when the replacement logic
is too dynamic to express as a simple function-to-function substitution.

### `register_lowering_pattern()`

Register an ATen-to-Inductor-IR replacement pattern. The decorated function
is saved and then called at lowering time, allowing direct conversion from a
matched pattern to Inductor IR.

:::{warning}
Inductor IR is not stable, so this path is **discouraged** for external use.
:::

## Example: Fusing `addmm + relu` with `register_graph_pattern`

The following example uses `register_graph_pattern` to recognize a
`mm → add → relu` sequence and replace it with CuBLAS's fused
`_addmm_activation` kernel:

```python
import torch
torch.set_default_device("cuda")

from torch._inductor.pattern_matcher import (
    CallFunction,
    KeywordArg,
    Arg,
    PatternMatcherPass,
    register_graph_pattern,
    Match,
)

aten = torch.ops.aten
from torch._inductor.virtualized import V

pattern_pass = PatternMatcherPass()


def is_valid_addmm_activation_fusion(match):
    # For brevity, we'll just check if tensors are CUDA, but for production we'd also
    # want to:
    # - Check that shapes are compatible with cuBLAS's API (1D bias, 2D matrices, etc.)
    # - Ensure we're not preventing more profitable fusions (e.g., max_autotune_gemm)
    # - Verify the relu doesn't have fusion opportunities with its users
    return match.output_node().meta["val"].is_cuda


# Pattern: mm -> add -> relu
@register_graph_pattern(
    CallFunction(aten.relu,
        CallFunction(aten.add,
            CallFunction(aten.mm, Arg(), Arg()),
            KeywordArg("input")
        )
    ),
    pass_dict=pattern_pass,
    extra_check=is_valid_addmm_activation_fusion,
)
def fuse_mm_add_relu(match: Match, mat1, mat2, *, input):
    def repl(input, mat1, mat2):
        return aten._addmm_activation(
            input, mat1, mat2, beta=1, alpha=1, use_gelu=False
        )

    # when tracing, theres no need to run the actual kernels, or alloc real memory
    # so we enable FakeTensorMode.
    with V.fake_mode:
        match.replace_by_example(repl, [input, mat1, mat2])


def my_pass(graph):
    print(f"FX Graph Before: {graph}")
    pattern_pass.apply(graph)
    print(f"FX Graph After: {graph}")


# hook in the custom pass to torch.compile. To compose with caching, we would need to
# subclass torch._inductor.custom_graph_pass.CustomGraphPassType
torch._inductor.config.post_grad_custom_pre_pass = my_pass


class LinearRelu(torch.nn.Linear):
    def forward(self, x):
        # This will create mm + add + relu pattern that can be fused
        return torch.nn.functional.relu(torch.add(x @ self.weight.T, self.bias))


with torch.no_grad():
    mod = LinearRelu(512, 512)
    inp = torch.rand([512, 512])

    compiled_mod = torch.compile(mod)
    result = compiled_mod(inp)

    print(f"Compilation successful, result shape: {result.shape}")

    # Verify correctness by comparing with eager mode
    eager_result = mod(inp)
    torch.testing.assert_close(result, eager_result, rtol=1e-4, atol=1e-4)
    print("Results match eager mode")
```

With `register_graph_pattern`, you manually construct the pattern using
`CallFunction`, `Arg`, and `KeywordArg` primitives, and you manually update
the graph inside the handler function.

## Example: Fusing `addmm + relu` with `register_replacement`

The same fusion can be expressed more concisely with `register_replacement`.
Instead of manually constructing the pattern DAG, you write the search and
replacement as ordinary Python functions:

```python
import torch
import torch._inductor.config
torch.set_default_device("cuda")

from torch._inductor.pattern_matcher import (
    register_replacement,
    fwd_only,
    PatternMatcherPass,
)

pattern_pass = PatternMatcherPass()


def addmm_relu_pattern(input, mat1, mat2):
    return torch.addmm(input, mat1, mat2).relu()


def addmm_activation_replacement(input, mat1, mat2):
    return torch.ops.aten._addmm_activation(
        input, mat1, mat2, beta=1, alpha=1, use_gelu=False
    )


def is_valid_fusion(match):
    # As above, we will just check if Tensors are cuda, but production may require
    # additional checks.
    return match.output_node().meta["val"].is_cuda


def get_example_args():
    return [
        torch.empty([10], device="cuda"),      # input (bias)
        torch.empty([10, 10], device="cuda"),   # mat1
        torch.empty([10, 10], device="cuda"),   # mat2
    ]


def register():
    # avoid using real memory in registrations
    with torch._subclasses.fake_tensor.FakeTensorMode():
        register_replacement(
            addmm_relu_pattern,
            addmm_activation_replacement,
            get_example_args(),
            fwd_only,
            [pattern_pass],
            extra_check=is_valid_fusion,
        )


register()


def my_pass(graph):
    print(f"FX Graph Before: {graph}")
    pattern_pass.apply(graph)
    print(f"FX Graph After: {graph}")


torch._inductor.config.post_grad_custom_pre_pass = my_pass


class LinearRelu(torch.nn.Linear):
    def forward(self, x):
        return torch.addmm(self.bias, x, self.weight.T).relu()


with torch.no_grad():
    mod = LinearRelu(512, 512)
    inp = torch.rand([512, 512])

    compiled_mod = torch.compile(mod)
    result = compiled_mod(inp)

    print(f"Compilation successful, result shape: {result.shape}")

    # Verify correctness by comparing with eager mode
    eager_result = mod(inp)
    torch.testing.assert_close(result, eager_result, rtol=1e-4, atol=1e-4)
    print("Results match eager mode")
```

With `register_replacement`, you don't need to manually construct a pattern
or update the graph. You can write the search pattern as a user would, and
`register_replacement` will decompose and canonicalize it into ATen IR. If any
part of that process changes, it will also update the decomposition of the
search pattern, making it more robust to future compiler changes and less
error-prone to write. However, for other use cases where the replacement
pattern is more dynamic, `register_graph_pattern` can be useful.

:::{tip}
The example argument tensors are used to trace the pattern. If the sequence
of ATen ops that get traced is affected by either the dtype or device, the
inputs will need to be constructed appropriately. For instance, the
decomposition of softmax includes an upcast if the input is in low precision;
when generating patterns,
[we trace with both fp16 and fp32](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/fx_passes/fuse_attention.py).
In this example, the pattern will get traced as `aten.addmm` and `aten.relu`
regardless of device or dtype, so we just need a single registration.
:::

## Joint Graph Patterns

Post-grad patterns do not need to maintain autograd compatibility. You can
match a pattern in the forward graph and replace it with an operator that has
no backward defined. However, sometimes you want to replace an operator's
forward **and** backward together. For this, `register_replacement` has an API
variant that traces both the forward and backward, then performs search and
replace on the **joint graph**.

An example of this usage is TorchInductor's
[fused attention pattern matching](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/fx_passes/fuse_attention.py),
which replaces attention subgraphs in both the forward and backward passes.

## Pre-Grad Graph Patterns

Pre-grad patterns operate on the FX graph **before** AOT Autograd, so
operators have not been decomposed or functionalized. Pattern matching at
this stage can be easier in some cases (matching high-level ops directly),
but tends to match less reliably because:

- Equivalent operations may have different surface forms (e.g.,
  `torch.nn.functional.relu(x)` vs. `x.relu()`).
- You may need to reason about mutation and aliasing.

TorchInductor uses limited-to-no manipulation at the pre-grad stage. Pre-grad
patterns are discouraged unless the pattern is significantly easier to express
on un-decomposed IR.

## Precompiled Patterns

New patterns added using `register_replacement()` incur compile-time overhead
because they need to be traced before use. To avoid that overhead, patterns
can be **precompiled** using `gen_register_replacement()`.

The arguments are the same as `register_replacement()` except for an additional
unique name used as a lookup key. `gen_register_replacement()` is more
appropriate for patterns that are heavily parameterized or use joint graph
forward + backward training matching.

:::{seealso}
- [FX Graph Passes](torch.compiler_inductor_fx_passes.md) for an overview of
  where pattern matching fits in the compilation pipeline.
- [Writing Graph Transformations](torch.compiler_transformations.md) for the
  general ATen IR transformation framework.
:::
