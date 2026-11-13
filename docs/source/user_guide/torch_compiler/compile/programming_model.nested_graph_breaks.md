# Nested Graph Breaks

Summary:

- Nested graph breaks are enabled by default in open-source PyTorch.
- When possible, Dynamo resumes every inlined frame after a nested graph break instead of moving the break to each caller.
- Use `torch._dynamo.disable_nested_graph_breaks` to opt out for code that depends on the legacy behavior.

Recall that when `torch.compile` is applied to a function, any nested function calls are also traced.
A **nested graph break** refers to any graph break that happens in a nested function call.

```python
def inner(x):
    ...
    torch._dynamo.graph_break()  # nested graph break
    ...

@torch.compile
def outer(x):
    ...
    y = inner(x)
    ...
```

Recall that in `fullgraph=False`, [graph breaks are handled](programming_model.dynamo_core_concepts.graph_breaks) by compiling the FX graph that has been determined so far,
running the unsupported code in regular Python, then resuming tracing after the unsupported code with a new FX graph.

## Default behavior

When a graph break occurs in an eligible nested frame, Dynamo creates resume functions for that frame and its eligible callers.
Consider this example:

```python
def inner1(x):
    x = x + 1
    torch._dynamo.graph_break()
    return x + 2

def inner2(x):
    x = x + 4
    x = inner1(x)
    return x + 8

@torch.compile
def f(x):
    x = x + 16
    x = inner2(x)
    return x + 32

f(torch.randn(3))
```

Dynamo traces from `f` through `inner2` and into `inner1`.
It compiles the operations through `x + 1`, executes the graph-break site in Python, and then resumes `inner1` after the break.
Tracing continues through the return from `inner1`, the remainder of `inner2`, and the remainder of `f`.
The result is roughly two compiled regions:

```python
def compiled_before_break(x):
    return x + 16 + 4 + 1

def compiled_after_break(x):
    return x + 2 + 8 + 32

def compiled_f_semantics(x):
    x = compiled_before_break(x)
    torch._dynamo.graph_break()  # runs between the compiled regions
    return compiled_after_break(x)
```

Nested resumption requires every frame in the caller chain to be eligible for partial graph compilation.
It is unavailable for `fullgraph=True`, generators, and functions that Dynamo explicitly suppresses.
Depending on why resumption is unavailable, Dynamo propagates the break outward, skips the frame, or reports an error.

## Disabling nested graph breaks

Use `torch._dynamo.disable_nested_graph_breaks` as a decorator or context manager to restore the legacy behavior for a region:

```python
@torch._dynamo.disable_nested_graph_breaks
def inner2(x):
    return inner1(x)

with torch._dynamo.disable_nested_graph_breaks():
    compiled_fn(x)
```

When nested graph breaks are disabled, Dynamo uses the legacy behavior: a break in an inlined function bubbles to its caller.
The nested function may then be compiled separately, so the same break can be encountered again at successive nesting levels.
