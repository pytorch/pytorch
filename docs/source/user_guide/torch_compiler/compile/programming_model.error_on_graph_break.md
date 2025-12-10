---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  execution_timeout: 30
  execution_show_tb: True
  merge_streams: True
---

```{code-cell}
:tags: [remove-cell]
import torch

import header_code
torch._logging.set_logs(graph_breaks=True)
```

# Toggling `error_on_graph_break`

**Summary:**

- When `fullgraph=False`, we can use `torch._dynamo.error_on_graph_break()` for more flexibility in
  dealing with graph breaks.

So far, we have introduced two ways in dealing with graph breaks in `torch.compile`:
1. `fullgraph=True` errors on the first graph break and additionally guarantees that only one graph is traced from the code.
2. `fullgraph=False` continues tracing even when encountering graph breaks.

What if we want to disallow graph breaks for most of the code, but there are a few problematic functions where the graph breaks are hard to remove,
and we are okay with having those graph breaks? We can use `torch._dynamo.error_on_graph_break()` to achieve this.

`torch.compile` has an `error_on_graph_break` setting (initially set to `False`).
If a graph break or compiler error occurs in code while `error_on_graph_break` is set to `False`, then `torch.compile` will attempt to continue compilation after the graph break/error.
If `error_on_graph_break` is set to `True`, then `torch.compile` will abort compilation and propagate the error to user code.

A significant difference between `error_on_graph_break=True` and `fullgraph=True` is that the former **does not guarantee that a single graph will be captured**.
`error_on_graph_break` **can be arbitrarily toggled during compile time** by using the `torch._dynamo.error_on_graph_break()` context manager/decorator.
In comparison, once `fullgraph` is set to `True`, it cannot be set back to `False`.
Finally, `error_on_graph_break` has lower precedence than `fullgraph` - `error_on_graph_break` only takes effect when `fullgraph=False`.


## `error_on_graph_break(False)` example

```{code-cell}
@torch._dynamo.error_on_graph_break(False)
def code_with_a_difficult_graph_break(x):
    x = x + 1
    torch._dynamo.graph_break()
    return x + 2

def inner(x):
    return code_with_a_difficult_graph_break(x)

# NOTE: fullgraph=False
@torch._dynamo.error_on_graph_break(True)
@torch.compile
def fn(x):
    return inner(x)

# No error, but there is a graph break
fn(torch.randn(3))
```

Using `error_on_graph_break(False)` under `error_on_graph_break(True)` is helpful for when we want to minimize graph breaks (i.e. follow the `fullgraph=True` programming model),
but there are some sections of code with non-performance-critical graph breaks that are difficult to work around.

`error_on_graph_break()` can be used as a context manager as well:

```{code-cell}
# NOTE: fullgraph=False
@torch._dynamo.error_on_graph_break(True)
@torch.compile
def fn(x):
    x = x + 1
    with torch._dynamo.error_on_graph_break(False):
        torch._dynamo.graph_break()  # no error
    return x + 2

# No error, but there is a graph break
fn(torch.randn(3))
```

You can use monkey patching to toggle `error_on_graph_break` for code where you cannot edit the source (e.g. framework code):

```{code-cell}
class ThirdPartyModule(torch.nn.Module):
    def forward(self, x):
        x = x + 1
        torch._dynamo.graph_break()
        return x + 2

tp_mod = ThirdPartyModule()
tp_mod.forward = torch._dynamo.error_on_graph_break(False)(tp_mod.forward)

@torch._dynamo.error_on_graph_break(True)
@torch.compile
def fn(x):
    return tp_mod.forward(x)

# No error, but there is a graph break
fn(torch.randn(3))
```

## `error_on_graph_break(True)` example

```{code-cell}
@torch._dynamo.error_on_graph_break(True)
def inner2(x):
    x = x + 1
    torch._dynamo.graph_break()  # error
    return x + 2

def inner(x):
    return inner2(x)

# fullgraph=False, error_on_graph_break=False
@torch.compile
def fn(x):
    x = x + 4
    torch._dynamo.graph_break()  # no error
    return inner(x)

try:
    fn(torch.randn(3))
except Exception as e:
    print(e)
```

Using `error_on_graph_break(True)` under `error_on_graph_break(False)` is helpful for when we want to use `torch.compile` flexibly (i.e. follow the `fullgraph=False` programming model),
but there are some sections of the code that are performance-critical and we want to ensure that those sections do not contain graph breaks.

## `error_on_graph_break` nesting behavior

`torch._dynamo.error_on_graph_break()` affects the `error_on_graph_break` setting of nested calls as well:

```{code-cell}
def inner(x):
    x = x + 1
    torch._dynamo.graph_break()
    return x + 2

def inner2(x):
    with torch._dynamo.error_on_graph_break(False):
        return inner(x)

@torch._dynamo.error_on_graph_break(True)
@torch.compile
def fn(x):
    return inner2(x)

# no error
fn(torch.randn(3))
```

`torch._dynamo.error_on_graph_break()` can be used under another `torch._dynamo.error_on_graph_break()` region:

```{code-cell}
def inner(x):
    x = x + 1
    with torch._dynamo.error_on_graph_break(False):
        torch._dynamo.graph_break()
    return x + 2

def inner2(x):
    with torch._dynamo.error_on_graph_break(True):
        return inner(x)

@torch.compile
def fn(x):
    return inner2(x)

# no error
fn(torch.randn(3))
```

## Interaction with `fullgraph`

`fullgraph=True` takes higher precedence than `error_on_graph_break`:


```{code-cell}
@torch._dynamo.error_on_graph_break(False)
def inner(x):
    x = x + 1
    torch._dynamo.graph_break()
    return x + 2

@torch.compile(fullgraph=True)
def fn(x):
    return inner(x)

try:
    fn(torch.randn(3))
except Exception as e:
    print(e)
```

`fullgraph=True` cannot be toggled back to `fullgraph=False`:

```{code-cell}
@torch.compile(fullgraph=False)
def inner(x):
    x = x + 1
    torch._dynamo.graph_break()
    return x + 2

@torch.compile(fullgraph=True)
def fn(x):
    return inner(x)

try:
    fn(torch.randn(3))
except Exception as e:
    print(e)
```

```{code-cell}
@torch.compile(fullgraph=True)
def inner(x):
    x = x + 1
    torch._dynamo.graph_break()
    return x + 2

@torch.compile(fullgraph=False)
def fn(x):
    return inner(x)

try:
    fn(torch.randn(3))
except Exception as e:
    print(e)
```

## Summary of `fullgraph=True/False` vs `error_on_graph_break`

Here is a table summarizing the differences between `fullgraph=True/False` and `error_on_graph_break`:

|  | `error_on_graph_break=True` | `error_on_graph_break=False` (default) |
| --- | --- | --- |
| `fullgraph=True` | Graph breaks result in errors. Only the first graph break will be reported. **One graph guarantee.**<br><br>`fullgraph` cannot be toggled to `False`. `error_on_graph_break` has no effect.<br><br>User code must be fully compatible with `torch.compile`. Guarantees no performance hits from graph breaks (because there are no graph breaks).<br><br>Ideal for code sensitive to graph breaks: framework/library code or cases where getting maximum performance is required. Prevents downstream user code from inadvertently allowing graph breaks. | Same as `fullgraph=True` and `error_on_graph_break=True` as `error_on_graph_break` has no effect when `fullgraph=True`. |
| `fullgraph=False` (default) | Graph breaks result in errors. Only the first graph break will be reported. **No one graph guarantee.**<br><br>`error_on_graph_break` can be toggled to `False`.<br><br>User code must be fully compatible with `torch.compile`. Guarantees no performance hits from graph breaks (because there are no graph breaks).<br><br>Ideal for user code sensitive to graph breaks. `error_on_graph_break` can be toggled to `False` to deal with sections that have graph breaks that are difficult to work around. | Will continue to compile after encountering graph breaks. All graph breaks will be reported.<br><br>`error_on_graph_break` can be toggled to `True`.<br><br>Doesn’t require many user code changes to work. Performance may be negatively impacted due to graph breaks.<br><br>Ideal for out-of-the-box use cases, on “non-weird” code, or where squeezing maximal performance is not necessary |
