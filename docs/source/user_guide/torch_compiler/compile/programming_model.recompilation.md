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

torch._logging.set_logs(recompiles=True)
```

# Dealing with Recompilations

Recompilations are necessary for `torch.compile` soundness, but can result in significantly increased compile time.
Thus, minimizing recompilations while preserving soundness is essential for reducing compile time.

You can view recompilations and their reasons using tlparse or `TORCH_LOGS=recompiles`.

## Is Dynamic Shapes Enabled?

In the below example, we recompile due to mismatched shapes:

```{code-cell}
@torch.compile
def fn(x):
    return x + 1
fn(torch.ones(3))
fn(torch.ones(4))
```

Make sure that the dynamic option of `torch.compile` is not set to `False`.
The default option, `dynamic=None`, will only attempt dynamic shapes after the first compilation.
You can set `dynamic=True` to upfront compile as dynamic as possible:

```{code-cell}
@torch.compile(dynamic=True)
def gn(x):
    return x + 1
gn(torch.ones(3))
gn(torch.ones(4))
```

For more information on dynamic shapes, including dealing with errors/recompilations due to
dynamic shapes, see [the dynamic shapes manual](https://docs.google.com/document/d/1GgvOe7C8_NVOMLOCwDaYV1mXXyHMXY7ExoewHqooxrs/edit?tab=t.0#heading=h.fh8zzonyw8ng).

## Wrapping Constants with Tensors
By default, `int` / `float` variables are treated as constants and are guarded on their exact value.
In the below example, we have a recompilation for each function call.

```{code-cell}
@torch.compile
def fn(x, c):
    return x + c
for i in range(5):
    fn(torch.ones(i), 0.5 + i)
```

In particular, for LR schedulers, initializing with a constant can lead to recompilations:

```{code-cell}
mod = torch.nn.Linear(3, 3)
opt = torch.optim.Adam(mod.parameters(), lr=0.01)
sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.9)
@torch.compile
def gn(inp):
    opt.zero_grad(True)
    out = mod(inp).sum()
    out.backward()
    opt.step()
    sched.step()
for i in range(5):
    gn(torch.ones(3, 3))
```

In both examples, we can wrap `float` variables in tensors in order to prevent recompilations.

```{code-cell}
:tags: [remove-cell]
torch._dynamo.reset()
```

```{code-cell}
# first example
for i in range(5):
    fn(torch.ones(i), torch.tensor(0.5 + i))
# second example
opt = torch.optim.Adam(mod.parameters(), lr=torch.tensor(0.01))
sched = torch.optim.lr_scheduler.ExponentialLR(opt, torch.tensor(0.9))
for i in range(5):
    gn(torch.ones(3, 3))
```

(programming_model.recompilation.changing_cache_size_limit)=
## Changing the Cache Size Limit

There is a limit to how many times a function can be recompiled,
determined by `torch._dynamo.config.cache_size_limit` and `torch._dynamo.config.accumulated_cache_size_limit`
(The exact difference between these 2 values is detailed in [`torch/_dynamo/cache_size.py`](https://github.com/pytorch/pytorch/blob/4ce6e6ec8890a3f6ee604c9efb3ff153825ce575/torch/_dynamo/cache_size.py#L14)).
If the Dynamo cache limit is hit, then all future compilation attempts **will result in the function being skipped (run eagerly)**.
Dynamo will still attempt to use previously compiled bytecode for future function calls, if the guards pass.
Note that in the case of a recompilation limit hit, **all nested function calls WILL be skipped**
(Dynamo will try to use previously compiled bytecode for the nested functions).
Dynamo will also issue a warning containing the affected function and which limit was hit.
In the example below, each function call results in a recompile attempt.
When we hit the cache size limit (by default, 8), we stop attempting to recompile.
(Note that we set `dynamic=False` for demonstration purposes to force recompilation every time).

```{code-cell}
@torch.compile(dynamic=False)
def fn(x):
    return x + 1
for i in range(1, 10):
    # recompile every time due to dynamic=False
    fn(torch.ones(i))
```

If you know that the number of recompilations has a reasonable constant upper bound, you can raise the cache size limit.
If the cost of recompilation outweighs the benefit of compilation, then you can consider lowering the cache size limit.

```{code-cell}
torch._dynamo.config.cache_size_limit = 16
@torch.compile(dynamic=False)
def gn(x):
    return x + 1
for i in range(1, 10):
    gn(torch.ones(i))
```

## Graph Breaking to Reduce Recompilation Costs
If a large graph is recompiling and causing high compile time, you can intentionally introduce
a graph break in order to reduce recompilation costs, at the expense of introducing a performance hit.

```{code-cell}
def very_large_function(x):
    return x + 1

@torch.compile(dynamic=False)
def fn(x, c):
    y = very_large_function(x)  # recompiled every time
    return y + c

for i in range(1, 5):
    fn(torch.ones(3), i)

@torch.compile(dynamic=False)
def gn(x, c):
    y = very_large_function(x)  # compiled only once
    torch._dynamo.graph_break()
    return y + c  # recompiled every time

for i in range(1, 5):
    gn(torch.ones(3), i)
```
