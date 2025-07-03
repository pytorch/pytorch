(programming_model.nested_graph_breaks)=

# Nested Graph Breaks

**Summary:**

- Graph breaks in nested functions can result in hard-to-understand compiler behavior, which we document below.
- In older versions of PyTorch, a nested graph break results in $O(N)$ duplicate graph break behavior.

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

The semantics around nested graph breaks can be confusing, so we describe the behavior below.

## Old Nested Graph Break Behavior

Recall that in `fullgraph=False`, graph breaks are handled by compiling the FX graph that has been determined so far,
running the unsupported code in regular Python, then resuming tracing after the unsupported code with a new FX graph.
Resuming a function is actually a fairly complicated technical feat, so in older versions of PyTorch,
resuming tracing is only supported on top-level (i.e non-nested) functions.

Nested graph breaks are thus handled in this way:

First, `torch.compile` traces from `f` and traces all the way until the graph break in `inner1` is encountered.

```python
def inner1(x):
    x = x + 1
    torch._dynamo.graph_break()  # stop tracing due to graph break
    return x + 2

def inner2(x):
    x = x + 4
    x = inner1(x)
    x = x + 8

@torch.compile
def f(x):
    # start tracing from here
    x = x + 16
    x = inner2(x)
    x = x + 32

f(torch.randn(3))
```

Since we can only resume from top-level functions, we graph break on the `inner2` call in `f`.

```python
# The semantics of torch.compile(f)(x) is roughly this:
def compiled_f_semantics(x):
    y = x + 16
    z = inner2(y)
    return torch.compile(resume_f_semantics)(z)

def resume_f_semantics(x):
    return x + 32

compiled_f_semantics(torch.randn(3))
```

`inner2` is then automatically compiled as a top-level function.
We trace all the way until the graph break in `inner1` is encountered again.

```python
def inner1(x):
    x = x + 1
    torch._dynamo.graph_break()  # stop tracing due to graph break
    return x + 2

# this torch.compile is automatically applied
@torch.compile
def inner2(x):
    # start tracing from here
    x = x + 4
    x = inner1(x)
    x = x + 8

def compiled_f_semantics(x):
    y = x + 16
    z = inner2(y)
    return torch.compile(resume_f_semantics)(z)

def resume_f_semantics(x):
    return x + 32

compiled_f_semantics(torch.randn(3))
```

Then we graph break on the `inner1` call in `inner2`.

```python
def compiled_inner2_semantics(x):
    y = x + 4
    z = inner1(y)
    return torch.compile(resume_inner2_semantics)(z)

def resume_inner2_semantics(x):
    return x + 8
```

`inner1` is then automatically compiled as a top-level function.
The graph break is from `inner1`, so we handle the graph break normally.

```python
# this torch.compile is automatically applied
@torch.compile
def inner1(x):
    # start tracing from here
    x = x + 1
    torch._dynamo.graph_break()  # stop tracing due to graph break
    return x + 2

def compiled_f_semantics(x):
    y = x + 16
    z = compiled_inner2_semantics(y)
    return torch.compile(resume_f_semantics)(z)

def resume_f_semantics(x):
    return x + 32

def compiled_inner2_semantics(x):
    y = x + 4
    z = inner1(y)
    return torch.compile(resume_inner2_semantics)(z)

def resume_inner2_semantics(x):
    return x + 8

compiled_f_semantics(torch.randn(3))
```

`inner1` is handled normally:

```python
def compiled_inner1_semantics(x):
    y = x + 1
    torch._dynamo.graph_break()
    return torch.compile(resume_inner1_semantics)(y)

def resume_inner1_semantics(x):
    return x + 2
```

So the initial code is semantically equivalent to

```python
def compiled_f_semantics(x):
    y = x + 16
    z = compiled_inner2_semantics(y)
    return torch.compile(resume_f_semantics)(z)

def resume_f_semantics(x):
    return x + 32

def compiled_inner2_semantics(x):
    y = x + 4
    z = compiled_inner1_semantics(y)
    return torch.compile(resume_inner2_semantics)(z)

def resume_inner2_semantics(x):
    return x + 8

def compiled_inner1_semantics(x):
    y = x + 1
    torch._dynamo.graph_break()
    return torch.compile(resume_inner1_semantics)(y)

def resume_inner1_semantics(x):
    return x + 2

compiled_f_semantics(torch.randn(3))
```

Note in particular that we traced 3 top-level functions, and that we traced the same graph break 3 times.
This explains why you may encounter duplicate graph breaks when using `torch.compile`.

In summary, nested graph breaks are handled by:
- Tracing from the top-level function all the way to the nested graph break
- Graph breaking on the top-level function at the call to the second-level function
- Compiling the PyTorch ops tracked so far and running the compiled graph.
- Calling the second-level function, which gets automatically compiled as a top-level function
- Resuming tracing the top-level function after the second-level function call is complete

Note that the runtime of handling this graph break is $O(NK)$, where $N$ is the nesting depth,
and $K$ is the number of instructions from the top-level function to the graph break.

## New Nested Graph Break Behavior

NOTE: this implementation is WIP

In later versions of PyTorch, we fixed the above graph break behavior such that the repeated tracing behavior no longer happens.
This was accomplished by implementing nested resume functions,
so that we can immediately resume tracing from the point of the graph break.

```python
def compiled_f_inner2_inner1_semantics(x):
    # the next 3 instructions are compiled
    x1 = x + 16
    x2 = x1 + 4
    x3 = x2 + 1
    torch._dynamo.graph_break()
    return torch.compile(resume_f_inner2_inner1_semantics)(x3)

def resume_f_inner2_inner1_semantics(x):
    y = resume_inner2_inner1_semantics(x)
    return y + 32

def resume_inner2_inner1_semantics(x)
    y = resume_inner1_semantics(x)
    return y + 8

def resume_inner1_semantics(x)
    return x + 2
```

In the end, we only trace the graph break once - each instruction is traced once.
The runtime of handling this graph break is $O(N + K)$, where $N$ is the nesting depth,
and $K$ is the number of instructions from the top-level function to the graph break.
