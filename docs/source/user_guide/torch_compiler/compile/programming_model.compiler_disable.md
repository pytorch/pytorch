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

torch._logging.set_logs(graph_breaks=True, graph_code=True)
```

# Disabling and Suppressing Errors
For some model architectures, there are portions of the model which are particularly difficult to compile -
either there are many graph breaks, or there are crashes.
You may want to explicitly disable these portions of the model which are problematic so that you can apply
`torch.compile` to the parts that work. You can do this by using the `@torch.compiler.disable` decorator.
When `torch.compile` attempts to call a disabled function, it breaks the graph and skips tracing the disabled function,
resuming tracing after the call. By default, all recursive calls made from a disabled function are also disabled.
Use the `recursive=False` option to allow compilation for recursive calls.

```{code-cell}
def inner1(x):
    torch._dynamo.graph_break()  # not traced
    return x + 1  # not traced

@torch.compiler.disable
def outer1(x):
    x = x + 2  # not traced
    torch._dynamo.graph_break()  # not traced
    return inner1(x)

@torch.compile
def f(x):
    x = outer1(x)
    return x + 4  # traced

print(f(torch.ones(3)))
```

```{code-cell}
def inner2(x):
    torch._dynamo.graph_break()  # traced
    return x + 1  # traced

@torch.compiler.disable(recursive=False)
def outer2(x):
    x = x + 2  # not traced
    torch._dynamo.graph_break()  # not traced
    return inner2(x)

@torch.compile
def g(x):
    x = outer2(x)
    return x + 4  # traced

print(g(torch.ones(3)))
```

For example, one can use `torch.compiler.disable` to disable `torch.compile` on sparse architecture in
recommendation models, as the sparse arch is difficult to compile.
Preprocessing and logging functions are other examples of functions that typically cause
a lot of graph breaks and do not get value from being compiled.

If you are experiencing compiler crashes and you want to continue regardless,
you can set `torch._dynamo.config.suppress_errors = True`.
When the compiler crashes, we will just skip tracing the function and try again later.
**This is not best practice** - it is better to eventually manually add `disable` annotations as necessary.
