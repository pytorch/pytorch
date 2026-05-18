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
).forward.aot_compile(((torch.randn(3, 4, requires_grad=True),), {}))

# Call with the module instance as the first argument.
inputs = torch.randn(3, 4, requires_grad=True)
result = compiled_fn(model, inputs)

# Backward works through the compiled function.
loss = result.sum()
loss.backward()
print(inputs.grad)  # gradients flow back correctly

# Save and load.
compiled_fn.save_compiled_function("compiled_model.pt")

with open("compiled_model.pt", "rb") as f:
    loaded_fn = torch.compiler.load_compiled_function(f)

# Backward also works after loading from disk.
inputs = torch.randn(3, 4, requires_grad=True)
result = loaded_fn(model, inputs)
result.sum().backward()
print(inputs.grad)
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
- **f_globals** (`dict | None`) -- Optional global scope for the compiled
  function. Required when the original function references user-defined types
  or other non-standard globals.
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
deserializer, pass `f_globals` to provide the necessary namespace:

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

`aot_compile()` supports training out of the box. When any input or parameter
requires gradients, the compilation automatically traces the joint
forward+backward graph, partitions it, and compiles both halves. The resulting
function is autograd-aware -- calling `.backward()` on its output works as
expected.

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

# AOT compile -- backward is captured automatically when inputs require grad.
compiled_fn = torch.compile(
    model, fullgraph=True
).forward.aot_compile(((torch.randn(3, 4, requires_grad=True),), {}))

# Training loop using the AOT-compiled function.
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for _ in range(3):
    optimizer.zero_grad()
    inputs = torch.randn(3, 4, requires_grad=True)
    output = compiled_fn(model, inputs)
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

inputs = torch.randn(3, 4, requires_grad=True)
output = loaded_fn(model, inputs)
output.sum().backward()  # gradients flow correctly
```

(distributed-training-with-dtensor)=

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
        local_out = out.to_local() if isinstance(out, DTensor) else out
        loss = local_out.sum()
        loss.backward()
        optimizer.step()
        if rank == 0:
            print(f"step {step}: loss = {loss.item():.4f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

## Limitations

- **`fullgraph=True` is required.** If `torch.compile` encounters a graph
  break, `aot_compile()` raises an error.
- **Input shapes are specialized.** The compiled artifact is valid for the
  tensor shapes, dtypes, and devices provided as example inputs. Inputs with
  different shapes will trigger a guard failure at runtime unless guards are
  explicitly disabled.
- **Not all backends are supported.** Custom backends must implement the
  `SerializableCallable` interface to be compatible with save/load.
