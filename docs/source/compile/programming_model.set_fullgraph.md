---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  execution_timeout: 60
---

(programming_model.set_fullgraph)=

```{code-cell}
:tags: [remove-cell]
import torch
torch._logging.set_logs(graph_breaks=True)
```

# Toggling `fullgraph`

**Summary:**

- `torch.compile`'s `fullgraph` setting can be toggled during tracing to provide more flexibility in
  dealing with graph breaks.

Let’s compare `torch.compile`’s different `fullgraph` modes:

| `fullgraph=False` | `fullgraph=True` |
| --- | --- |
| Will continue to compile after encountering graph breaks
| Will error on the first graph break |

| Doesn’t require many user code changes to work
| User code must be fully compatible with `torch.compile`; no graph breaks allowed |

| Performance may be negatively impacted due to graph breaks
| Guarantees no performance hits from graph breaks (because there are no graph breaks) |

| All graph breaks will be reported
| Only the first graph break will be reported |

| Ideal for out-of-the-box use cases, on “non-weird” code, or where squeezing maximal performance is not necessary
| Ideal for code sensitive to graph breaks: framework/library code or cases where getting maximum performance is required |


By using the `torch._dynamo.set_fullgraph` context manager/decorator, we can mark which regions of the code should be run with `fullgraph=False` or `fullgraph=True`.

More precisely, if a graph break or compiler error occurs in code while `fullgraph` is set to `False`, then `torch.compile` will attempt to continue compilation after the graph break/error.
If `fullgraph` is set to `True`, then `torch.compile` will abort compilation and propagate the error to user code.

## `set_fullgraph(False)`

```{code-cell}
@torch._dynamo.set_fullgraph(False)
def code_with_a_difficult_graph_break(x):
    x = x + 1
    torch._dynamo.graph_break()
    return x + 2

def inner(x):
    return code_with_a_difficult_graph_break(x)

@torch.compile(fullgraph=True)
def fn(x):
    return inner(x)

# No error, but there is a graph break
fn(torch.randn(3))
```

Using `set_fullgraph(False)` under `torch.compile(fullgraph=True)` is helpful for when we want to minimize graph breaks (i.e. follow the `fullgraph=True` programming model),
but there are some sections of code with non-performance-critical graph breaks that are difficult to work around.

## `set_fullgraph(True)`

```{code-cell}
@torch._dynamo.set_fullgraph(True)
def inner2(x):
    x = x + 1
    torch._dynamo.graph_break()  # error
    return x + 2

def inner(x):
    return inner2(x)

@torch.compile(fullgraph=False)
def fn(x):
    x = x + 4
    torch._dynamo.graph_break()  # no error
    return inner(x)

try:
    fn(torch.randn(3))
except Exception as e:
    print(e)
```

Using `set_fullgraph(True)` under `torch.compile(fullgraph=False)` is helpful for when we want to use `torch.compile` flexibly (i.e. follow the `fullgraph=False` programming model),
but there are some sections of the code that are performance-critical and we want to ensure that those sections do not contain graph breaks.

## Notes

`set_fullgraph` affects the `fullgraph` setting of nested calls as well:

```{code-cell}
def inner(x):
    x = x + 1
    torch._dynamo.graph_break()
    return x + 2

def inner2(x):
    with torch._dynamo.set_fullgraph(False):
        return inner(x)

@torch.compile(fullgraph=True)
def fn(x):
    return inner2(x)

# no error
fn(torch.randn(3))
```

`set_fullgraph` can be used under another `set_fullgraph` region:

```{code-cell}
def inner(x):
    x = x + 1
    with torch._dynamo.set_fullgraph(False):
        torch._dynamo.graph_break()
    return x + 2

def inner2(x):
    with torch._dynamo.set_fullgraph(True):
        return inner(x)

@torch.compile(fullgraph=False)
def fn(x):
    return inner2(x)

# no error
fn(torch.randn(3))
```
