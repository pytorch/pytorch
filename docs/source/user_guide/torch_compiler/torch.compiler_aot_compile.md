(torch.compiler_aot_compile)=

# Ahead-of-Time Compilation with torch.compile

:::{warning}
This feature is experimental and subject to change.
:::

`torch.compile().aot_compile()` performs ahead-of-time (AOT) compilation on a
`torch.compile`-d function or module. Unlike the standard `torch.compile` flow
-- which compiles lazily on first invocation -- `aot_compile()` eagerly traces,
compiles, and packages the result into a serializable artifact.

The entire compilation pipeline runs ahead of time, including graph tracing,
Inductor code generation, Triton kernel compilation, and autotuning. The
compiled artifact can be saved to disk and loaded later, skipping all of these
steps at runtime.

This is useful when you want to:

- **Eliminate cold-start compilation latency** in production.
- **Serialize compiled artifacts** for deployment to other processes or machines.
- **Cross-compile** on a host machine for a different target device by tracing
  with fake tensors.

## How it differs from AOTInductor

{ref}`AOTInductor <torch.compiler_aot_inductor>` operates on
{func}`torch.export`-ed models and produces shared libraries for deployment in
non-Python environments. `aot_compile()` operates directly on `torch.compile`
and stays within the Python runtime -- the saved artifact is loaded back as a
Python callable. Choose AOTInductor when you need C++ deployment; choose
`aot_compile()` when you want to stay in Python and precompute compilation.

## Quick start

### Compiling a free function

```python
import torch

def fn(x, y):
    return x + y

# Step 1: AOT compile with example inputs.
#   fullgraph=True is required (graph breaks are not supported).
#   example_inputs is a tuple of (args_tuple, kwargs_dict).
compiled_fn = torch.compile(fn, fullgraph=True).aot_compile(
    ((torch.randn(3, 4), torch.randn(3, 4)), {})
)

# Step 2: Run the compiled function.
result = compiled_fn(torch.randn(3, 4), torch.randn(3, 4))

# Step 3: Save the compiled artifact to disk.
compiled_fn.save_compiled_function("compiled_add.pt")

# Step 4: Load and run in another process (no recompilation).
with open("compiled_add.pt", "rb") as f:
    loaded_fn = torch.compiler.load_compiled_function(f)

result = loaded_fn(torch.randn(3, 4), torch.randn(3, 4))
```

(compiling-a-module)=

### Compiling a module

When compiling an `nn.Module`, call `aot_compile()` on the `.forward` method
of the compiled module. The compiled function expects the module instance as
its first argument (the `self` parameter).

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)

model = MyModel()

# AOT compile the forward method.
compiled_fn = torch.compile(
    model, fullgraph=True
).forward.aot_compile(((torch.randn(3, 4),), {}))

# Call with the module instance as the first argument.
result = compiled_fn(model, torch.randn(3, 4))

# Backward works through the compiled function because model parameters
# require gradients.
loss = result.sum()
loss.backward()
print(model.linear.weight.grad)  # gradients flow to model parameters

# Save and load.
compiled_fn.save_compiled_function("compiled_model.pt")

with open("compiled_model.pt", "rb") as f:
    loaded_fn = torch.compiler.load_compiled_function(f)

# Backward also works after loading from disk.
model.zero_grad()
result = loaded_fn(model, torch.randn(3, 4))
result.sum().backward()
print(model.linear.weight.grad)
```

## API reference

### `torch.compile(...).aot_compile(example_inputs)`

Ahead-of-time compiles the `torch.compile()`-wrapped function.

**Args:**

- **example_inputs** (`tuple[tuple[Any, ...], dict[str, Any]]`) -- A tuple of
  `(args, kwargs)` providing example inputs for tracing. These determine the
  tensor shapes, dtypes, and devices that the compiled artifact is valid for.

**Returns:** An `AOTCompiledFunction` -- a callable that behaves like the
original function but runs the pre-compiled code. It also exposes:

- `save_compiled_function(path)` -- Serialize the compiled artifact to disk.
- `disable_guard_check()` -- Disable runtime guard validation (advanced use).
  On this function path the compiled function then runs whatever it is called
  with, without evaluating its guards; module dispatch over several compiled
  inputs still evaluates them, as described under
  {ref}`Developer notes <aot-compile-developer-notes>`.

**Requirements:**

- `fullgraph=True` must be passed to `torch.compile()`. Graph breaks are not
  supported with AOT compilation.
- The backend must be callable (string backends like `"inductor"`,
  `"eager"` and `"aot_eager"` are supported).

### `torch.compiler.load_compiled_function(file, *, f_globals=None, external_data=None)`

Load a previously saved AOT-compiled function from a file.

**Args:**

- **file** -- A file-like object (opened in binary read mode) containing the
  serialized compiled function.
- **f_globals** (`dict | None`) -- Optional global scope enclosing the
  compiled function, and the scope the kept guards resolve against: it must
  bind every global they read, with values that satisfy them. When a kept
  guard reads a global -- which takes a `guard_filter_fn` that keeps global
  guards, since the default drops them all -- that means `vars(my_module)` for
  the module that defined the original function (as in the example below)
  rather than a dict of a few extra names; under the default filter the only
  kept guard that reads a global is a symbolic-shape guard on a global with a
  dynamic dim, so otherwise the dict only widens what the bytecode merges over
  (below) with nothing checking it, and only the names the load cannot
  otherwise resolve belong in it. Guards
  read this dict by reference, so a global rebound after loading is seen on
  the next call, and a guarded global the dict lacks fails the guard until
  that name is bound in it -- there is no fallback to the values serialized
  with the artifact. That holds for a symbolic-shape guard too, whether it
  installs as a Python lambda (the default) or as a C++ guard under
  `enable_cpp_symbolic_shape_guards`: its global operands resolve here. Loading may
  insert names of its own, never overwriting an existing key: the
  Dynamo-generated globals a kept guard is rooted at, and
  `__builtins__` when it has to build the builtins dict one of those names
  holds. The bytecode does not read this dict: it reads a snapshot, taken at
  load time, of the globals serialized with the artifact with this dict merged
  over them, so a name the dict omits still resolves there. That the two can
  disagree is a known limitation rather than a contract to rely on: a rebind
  the guards accept leaves the call computing with the load-time value, so
  only a rebind they reject changes what the call does, by raising. When
  omitted, global guards are resolved against the scope rebuilt from the
  artifact instead, where a rebinding in this process is invisible. Passing
  `{}` is not that: it installs a live but empty guard scope, so every kept
  guard rooted at a global the load does not seed itself fails with
  `KeyError on G['NAME']` until that name is bound in the same dict, which the
  load holds by reference.
- **external_data** (`dict | None`) -- Optional data to be loaded into the
  runtime environment. Required when the original function captures objects
  that could not be serialized (e.g., `nn.Module` instances). The keys should
  match those passed to `save_compiled_function(external_data=...)`.

**Returns:** A callable with compilation preloaded from disk.

## Choosing a backend

`aot_compile()` works with any backend that implements the
`SerializableCallable` interface. The built-in `"inductor"`, `"eager"` and `"aot_eager"`
backends are supported out of the box:

```python
# With inductor (default, optimized code generation).
compiled_fn = torch.compile(fn, fullgraph=True, backend="inductor").aot_compile(
    ((torch.randn(3, 4),), {})
)

# With eager (useful for debugging, no codegen).
compiled_fn = torch.compile(fn, fullgraph=True, backend="eager").aot_compile(
    ((torch.randn(3, 4),), {})
)
```

## Handling closures and external references

Functions that capture free variables (closures) are supported. The closure
state is serialized along with the compiled artifact:

```python
scale = 2

def fn(x, y):
    return (x + y) * scale

compiled_fn = torch.compile(fn, fullgraph=True).aot_compile(
    ((torch.randn(3, 4), torch.randn(3, 4)), {})
)
compiled_fn.save_compiled_function("scaled_add.pt")

with open("scaled_add.pt", "rb") as f:
    loaded_fn = torch.compiler.load_compiled_function(f)
```

When the function references user-defined types that cannot be found by the
deserializer, pass `f_globals` to provide the necessary namespace. The same dict
is what the kept guards resolve against, so an artifact that keeps global guards
needs the defining module's namespace rather than a dict of the missing names
alone:

```python
with open("my_fn.pt", "rb") as f:
    loaded_fn = torch.compiler.load_compiled_function(
        f, f_globals=my_module.__dict__
    )
```

When the function captures non-serializable objects (like `nn.Module`
instances), use `external_data`:

```python
# Saving.
compiled_fn.save_compiled_function(
    "fn_with_model.pt",
    external_data={"model": model},
)

# Loading.
with open("fn_with_model.pt", "rb") as f:
    loaded_fn = torch.compiler.load_compiled_function(
        f, external_data={"model": model}
    )
```

## Training

`aot_compile()` supports training out of the box. When any parameter requires
gradients, the compilation automatically traces the joint forward+backward
graph, partitions it, and compiles both halves. The resulting function is
autograd-aware -- calling `.backward()` on its output works as expected.

Using the same `MyModel` from {ref}`Compiling a module <compiling-a-module>`
above:

```python
model = MyModel()

compiled_fn = torch.compile(
    model, fullgraph=True
).forward.aot_compile(((torch.randn(3, 4),), {}))

# Training loop using the AOT-compiled function.
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for _ in range(3):
    optimizer.zero_grad()
    output = compiled_fn(model, torch.randn(3, 4))
    loss = output.sum()
    loss.backward()
    optimizer.step()
```

Save and load preserve the autograd support -- backward works after loading
from disk:

```python
compiled_fn.save_compiled_function("train_model.pt")

with open("train_model.pt", "rb") as f:
    loaded_fn = torch.compiler.load_compiled_function(f)

output = loaded_fn(model, torch.randn(3, 4))
output.sum().backward()  # gradients flow correctly
```

### Distributed training with DTensor

When training with tensor-parallelized models using
{class}`~torch.distributed.tensor.DTensor`, `aot_compile()` can be combined
with `compile_on_one_rank` to produce rank-independent compiled graphs. Without
this flag, rank-specific values (mesh coordinates, shard offsets) get baked into
the compiled graph as constants, producing a different graph per rank. With the
flag enabled, these values become symbolic and are computed at runtime.

Set the flag via `torch.distributed.config.patch`:

```python
import torch.distributed.config as dist_config

with dist_config.patch(compile_on_one_rank=True):
    compiled_fn = torch.compile(
        model, fullgraph=True
    ).forward.aot_compile(((example_input,), {}))
```

Or via the environment variable `TORCH_DISTRIBUTED_COMPILE_ON_ONE_RANK=1`.

Here is a complete example of a tensor-parallel model compiled and trained
across multiple GPUs via `torchrun`:

```python
# train_tp.py -- run with: torchrun --nproc_per_node=8 train_tp.py
import torch
import torch.distributed as dist
import torch.distributed.config as dist_config
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))

    model = FeedForward(64, 128).cuda()
    parallelize_module(model, mesh, {
        "linear1": ColwiseParallel(),
        "linear2": RowwiseParallel(),
    })

    x = DTensor.from_local(
        torch.randn(4, 64, device=f"cuda:{rank}"),
        mesh, [Replicate()], run_check=False,
    )

    # Compile with compile_on_one_rank -- all ranks produce the same graph.
    with dist_config.patch(compile_on_one_rank=True):
        compiled_fn = torch.compile(
            model, fullgraph=True,
        ).forward.aot_compile(((x,), {}))

    # Training loop.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(5):
        optimizer.zero_grad()
        x = DTensor.from_local(
            torch.randn(4, 64, device=f"cuda:{rank}"),
            mesh, [Replicate()], run_check=False,
        )
        out = compiled_fn(model, x)
        loss = out.to_local().sum()
        loss.backward()
        optimizer.step()
        if rank == 0:
            print(f"step {step}: loss = {loss.item():.4f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

Saving and loading works the same way as the single-process case -- call
`save_compiled_function` on any rank and `load_compiled_function` on any rank.
Because `compile_on_one_rank=True` produces rank-independent graphs, the same
artifact can be loaded on every rank without per-rank compilation.

## Limitations

- **`fullgraph=True` is required.** If `torch.compile` encounters a graph
  break, `aot_compile()` raises an error.
- **Input shapes are specialized.** The compiled artifact is valid for the
  tensor shapes, dtypes, and devices provided as example inputs. Inputs with
  different shapes will trigger a guard failure at runtime unless guards are
  explicitly disabled.
- **Not all backends are supported.** Custom backends must implement the
  `SerializableCallable` interface to be compatible with save/load.

(aot-compile-developer-notes)=

## Developer notes

**Private, unstable -- may change or disappear without notice.** A module can
also be compiled for several calls at once:
`torch.compile(model, fullgraph=True)._aot_compile(inputs)` takes a list of
`torch._dynamo.aot_compile.ModelInput`, compiles one graph per input and
replaces the wrapper's `forward` with a dispatcher over their guards. It needs
`torch._dynamo.config.enable_aot_compile`, which is on by default, so only a
caller who turned it off has to restore it. The dispatcher serves the first
input whose guards match, and evaluates the guards of an input opted out
through `model.forward.compiled_results[i].disable_guard_check()` as well:
opting out here suppresses the failure, not the evaluation, so such an input is
served on a match like any other, and on the strength of its opt-out alone only
when nothing matched -- one opt-out replaces the
`No AOT compiled graph matched this call` error for the whole model.
