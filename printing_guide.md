# Printing and Inspecting Tensors Inside `torch.compile`

## Motivation

When debugging numerical issues in compiled models, you often want to inspect
tensor values at various points in your model — shapes, norms, means, hashes, whether
NaNs are present, etc. In this doc, we describe some printing APIs that doesn't cause
graph breaks and works for both strict (`dynamo`) and non-strict (`make_fx`) tracing.

**TL;DR:**
- **`torch._higher_order_ops.print`** — format-string-based print, forward-only. Good for printing quick tensor/scalar/SymInt messages.
- **`torch._higher_order_ops.opaque_inspect_tensor.make_opaque_inspect_tensor_fn`** — arbitrary Python callbacks on full tensors, prints in both forward and backward (tensor gradient is printed in backward), opaque to the compiler. Good for when you want to inspect gradients
or do more complicated logging operations.

| | `torch._higher_order_ops.print` | `make_opaque_inspect_tensor_fn` |
|---|---|---|
| Log in forward | Yes | Yes |
| Log in backward | No | Yes |
| Support logging other than builtin.print | No | Yes |
| DTensor support | WIP | WIP |
| Filter by rank | No | Yes |
| Preserved in Inductor| Yes (as `print`) | No (inductor lowering will fail) |

**Rule of thumb:** Use `torch._higher_order_ops.print` when you want simple
formatted messages in the forward pass or want to preserve the print in inductor. Use `make_opaque_inspect_tensor_fn`
when you need custom logging other than `builtin.print`, or observe gradients in the backward pass.

---

## `torch._higher_order_ops.print` (the print HOP)

A HOP version of Python's `print()` that appears as a node in the FX
graph. It uses format strings with `str.format()` syntax.


### Signature

```python
torch._higher_order_ops.print(format_str: str, *args, **kwargs) -> None
```

- `format_str` — a Python format string (e.g. `"x={x}"` or `"x={}"`)
- `*args, **kwargs` — values substituted into the format string via `.format()`

Values can be tensors (printed as their repr), SymInts, or plain Python
scalars/strings.

### Examples

**Basic string messages:**

```python
@torch.compile(backend="aot_eager", fullgraph=True)
def fn(x):
    x = x + 1
    torch._higher_order_ops.print("after add: {}", x)
    x = x * 2
    torch._higher_order_ops.print("after mul: {}", x)
    return x

fn(torch.tensor(3.0))
# after add: 4.0
# after mul: 8.0
```

**Keyword arguments:**

```python
@torch.compile(backend="inductor", fullgraph=True)
def fn(x):
    torch._higher_order_ops.print("shape={s}, val={v}", s=x.shape[0], v=x.sum())
    return x * 2

fn(torch.randn(4))
# shape=4, val=-0.2345...
```

### Limitations

**Cannot print non-scalar values like shapes directly.** Shapes are tuples, and
the print HOP doesn't support tuple formatting yet:

```python
# This does NOT work:
torch._higher_order_ops.print("x.shape={}", x.shape)

# Workaround — print individual dimensions:
torch._higher_order_ops.print("x.shape=({}, {})", x.shape[0], x.shape[1])
```

**Computed values appear in the graph.** Any tensor operations you pass as
arguments are traced into the compiled graph:

```python
# x.norm() and x + 1 become nodes in the graph
torch._higher_order_ops.print("norm={}, plus1={}", x.norm(), x + 1)
```

This is usually fine, but if you want the computation to be opaque to the
compiler, use `make_opaque_inspect_tensor_fn` instead (see below).

**Forward-only.** The print HOP does not participate in autograd — it has no
backward pass. If you need to observe gradients, use
`make_opaque_inspect_tensor_fn`.

---

## `make_opaque_inspect_tensor_fn` (opaque tensor inspector)

A wrapper over the `@leaf_function` HOP that lets you run **arbitrary Python
callbacks** on tensors during both forward and backward passes. The callbacks
are completely opaque to the compiler — they execute at runtime without being
traced.

### Creating an inspector

```python
opaque_inspect = make_opaque_inspect_tensor_fn(
    fwd_f=callable,  # called as fwd_f(tensor) on each tensor in forward
    bwd_f=callable,  # called as bwd_f(grad) on each gradient in backward
)
```

### Calling the inspector

```python
opaque_inspect(*tensors, tag="", ranks=None, phase=None)
```

- `*tensors` — one or more tensors to inspect
- `tag` (str) — optional label; prints `[tag][fwd]` / `[tag][bwd]` prefixes
- `ranks` (int | set[int] | None) — filter by distributed rank (`None` = all
  ranks, `int` = single rank, `set[int]` = multiple ranks)
- `phase` (str | None) — override which callback runs:
  - `None` (default): calls `fwd_f`, registers backward hooks for `bwd_f`
  - `"fwd"`: calls `fwd_f` only, no backward hooks
  - `"bwd"`: calls `bwd_f` directly, no backward hooks

### Examples

**Basic usage — works in both eager and compiled:**

```python
opaque_inspect = make_opaque_inspect_tensor_fn(
    fwd_f=lambda t: print(f"  fwd: shape={t.shape}, norm={t.norm():.4f}"),
    bwd_f=lambda t: print(f"  bwd: shape={t.shape}, norm={t.norm():.4f}"),
)

def fn(x):
    opaque_inspect(x)
    return x.sum()

# Eager
x = torch.randn(3, 3, requires_grad=True)
fn(x).backward()
#   fwd: shape=torch.Size([3, 3]), norm=2.5700
#   bwd: shape=torch.Size([3, 3]), norm=3.0000

# Compiled — same function, same output
torch._dynamo.reset()
compiled_fn = torch.compile(fn, backend="aot_eager")
compiled_fn(x).backward()
#   fwd: shape=torch.Size([3, 3]), norm=2.5700
#   bwd: shape=torch.Size([3, 3]), norm=3.0000
```

**Tagging with multiple tensors:**

```python
opaque_inspect = make_opaque_inspect_tensor_fn(
    fwd_f=lambda t: print(f"    {t.shape}"),
    bwd_f=lambda t: print(f"    {t.shape}"),
)

x = torch.randn(2, 4, requires_grad=True)
y = torch.randn(2, 4, requires_grad=True)
opaque_inspect(x, y, tag="inputs")
# [inputs][fwd]
#     torch.Size([2, 4])
#     torch.Size([2, 4])
(x + y).sum().backward()
# [inputs][bwd]
#     torch.Size([2, 4])
# [inputs][bwd]
#     torch.Size([2, 4])
```

**NaN detection:**

```python
nan_check = make_opaque_inspect_tensor_fn(
    fwd_f=lambda t: print(f"  NaN detected!") if t.isnan().any() else None,
    bwd_f=lambda t: print(f"  grad NaN detected!") if t.isnan().any() else None,
)

@torch.compile(backend="aot_eager")
def fn(x):
    y = x * 2
    nan_check(y, tag="after_mul")
    return y.sum()
```

**Inside a custom autograd function (using `phase`):**

Use `phase="fwd"` in forward and `phase="bwd"` in backward so that each call
runs the right callback without double-registering hooks:

```python
from torch.autograd import Function

opaque_inspect = make_opaque_inspect_tensor_fn(
    fwd_f=lambda t: print(f"  shape={t.shape}, norm={t.norm():.4f}"),
    bwd_f=lambda t: print(f"  shape={t.shape}, norm={t.norm():.4f}"),
)

class MyReLU(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        opaque_inspect(x, tag="my_relu", phase="fwd")
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        opaque_inspect(grad_output, tag="my_relu", phase="bwd")
        return grad_output * (x > 0).float()

def fn(x):
    return MyReLU.apply(x).sum()

# Works in both eager and compiled
fn(torch.randn(3, 3, requires_grad=True)).backward()
# [my_relu][fwd]
#   shape=torch.Size([3, 3]), norm=2.4495
# [my_relu][bwd]
#   shape=torch.Size([3, 3]), norm=3.0000
```

**Filtering by distributed rank:**

```python
opaque_inspect = make_opaque_inspect_tensor_fn(
    fwd_f=lambda t: print(f"  norm={t.norm():.4f}"),
    bwd_f=lambda t: print(f"  grad norm={t.norm():.4f}"),
)

# Only print on rank 0
opaque_inspect(x, tag="layer1", ranks=0)

# Only print on ranks 0 and 1
opaque_inspect(x, tag="layer1", ranks={0, 1})
```

**Instrument all modules in a model:**

```python
import torch.nn as nn
import torch.utils._pytree as pytree

def install_debug_prints(model: nn.Module) -> None:
    for name, module in model.named_modules():
        tag = f"{module.__class__.__name__}:{name}"
        opaque_inspect = make_opaque_inspect_tensor_fn(
            fwd_f=lambda t: print(f"  {t.shape} mean={t.mean():.4f}"),
            bwd_f=lambda t: print(f"  {t.shape} mean={t.mean():.4f}"),
        )
        orig_forward = module.forward
        def wrapped(*args, _orig=orig_forward, _opaque_inspect=opaque_inspect, _tag=tag, **kwargs):
            out = _orig(*args, **kwargs)
            pytree.tree_map_only(torch.Tensor, lambda t: _opaque_inspect(t, tag=_tag), out)
            return out
        module.forward = wrapped

model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
install_debug_prints(model)
compiled_model = torch.compile(model, backend="aot_eager")
out = compiled_model(torch.randn(2, 8, requires_grad=True))
# [Linear:0][fwd]
#   torch.Size([2, 16]) mean=...
# [ReLU:1][fwd]
#   torch.Size([2, 16]) mean=...
# [Linear:2][fwd]
#   torch.Size([2, 4]) mean=...
out.sum().backward()
# [Linear:2][bwd]
#   torch.Size([2, 4]) mean=...
# [ReLU:1][bwd]
#   torch.Size([2, 16]) mean=...
# [Linear:0][bwd]
#   torch.Size([2, 16]) mean=...
```

### Key difference: computations are opaque

With the print HOP, any tensor operations you pass as arguments are traced and
appear in the compiled graph:

```python
# norm() and x + 1 become graph nodes
torch._higher_order_ops.print("norm={}, plus1={}", x.norm(), x + 1)
```

With `make_opaque_inspect_tensor_fn`, the callbacks run at runtime and are
**not** traced. The operations inside your callback (norm, mean, etc.) do not
appear in the compiled graph:

```python
opaque_inspect = make_opaque_inspect_tensor_fn(
    fwd_f=lambda t: print(f"norm={t.norm():.4f}"),  # norm() is NOT in the graph
    bwd_f=lambda t: print(f"grad norm={t.norm():.4f}"),
)
opaque_inspect(x)
```

This makes `make_opaque_inspect_tensor_fn` better suited for heavyweight
inspection logic that you don't want affecting compilation.

---

## When to use which?

| Scenario | Use |
|---|---|
| Quick debug message with scalar values in forward | `torch._higher_order_ops.print` |
| Print a SymInt (e.g., dynamic shape dimension) | `torch._higher_order_ops.print` |
| Inspect full tensor properties (norm, mean, NaN check) | `make_opaque_inspect_tensor_fn` |
| Observe gradients in the backward pass | `make_opaque_inspect_tensor_fn` |
| Heavyweight inspection (log to file, accumulate stats) | `make_opaque_inspect_tensor_fn` |
| Distributed — only print on certain ranks | `make_opaque_inspect_tensor_fn` |
| Keep inspection logic out of the compiled graph | `make_opaque_inspect_tensor_fn` |
| Print something the print HOP doesn't support (tuples, dicts) | `make_opaque_inspect_tensor_fn` |
| Inside a custom autograd Function (fwd + bwd) | `make_opaque_inspect_tensor_fn` with `phase` |

When in doubt, `make_opaque_inspect_tensor_fn` is the more general tool. The
print HOP is simpler for quick one-off forward-only messages.

---

## Alternative Hacks

Beyond the official APIs, there are informal patterns used in practice. These
are not part of the PyTorch public API but can be useful in specific scenarios.

### Global tensor summary storage

Store detached tensor summaries in a global dictionary for later inspection.
This pattern is useful when:

- You want to accumulate statistics across many iterations
- You need post-hoc analysis after training completes
- You want to avoid printing during the hot path

**Example pattern (from sixlib):**

```python
class MetricLogger:
    def __init__(self):
        self._summaries: dict[str, Any] = {}

    def add_summary(self, name: str, value: Any) -> None:
        with torch.no_grad():
            value = tree.map(
                lambda x: x.detach() if isinstance(x, Tensor) else x,
                value,
            )
            self._summaries[name] = value

    def get_summaries(self) -> dict[str, Any]:
        return self._summaries
```

**Key details:**

- **`torch.no_grad()`** — Prevents the storage operation from being traced or
  participating in autograd
- **`.detach()`** — Breaks the connection to the computation graph, avoiding
  memory leaks from holding references to intermediate activations
- **`tree.map`** — Handles nested structures (lists, dicts, tuples) containing
  tensors

**Trade-offs vs official APIs:**

| Aspect | Global storage | Official APIs |
|--------|----------------|---------------|
| Visibility in graph | None (completely opaque) | Print HOP appears in graph |
| When output appears | On-demand (after training) | Immediately during execution |
| Memory overhead | Stores tensor copies | None (prints and discards) |
| Iteration tracking | Easy (keyed by name/step) | Manual (include in format string) |
| Works with compile | Yes | Yes |

**When to use:** Global storage works best when you're collecting metrics for
later analysis (e.g., plotting loss curves, comparing activation distributions
across checkpoints). For immediate debugging output, prefer the official APIs.

### DebugMode

`torch.utils._debug_mode.DebugMode` is a `TorchDispatchMode` that intercepts
and logs runtime calls to a hierarchical string dump. It captures tensor
operations with shapes, dtypes, and placements, and works under `torch.compile`.

```python
from torch.utils._debug_mode import DebugMode

with DebugMode() as debug_mode:
    result = model(x)
print(debug_mode.debug_string())
```

Example output from a DTensor matmul:

```
torch.mm(dt$0: f32[8, 8]| S(0), dt$1: f32[8, 32]| S(0))  ->  dt$6: f32[8, 32]| S(0)
  aten::mm(dt$0: f32[8, 8]| S(0), dt$1: f32[8, 32]| S(0))
    redistribute_input(1, S(0) -> R)
      redistribute_input(t$2: f32[1, 32], trace: S(0)->R)
        _c10d_functional::all_gather_into_tensor(t$2: f32[1, 32], 8, 0)  ->  t$3: f32[8, 32]
        _c10d_functional::wait_tensor(t$3: f32[8, 32])  ->  t$3: f32[8, 32]
    aten::mm(t$4: f32[1, 8], t$3: f32[8, 32])  ->  t$5: f32[1, 32]
```

**Caveat:** DebugMode can be quite verbose — it logs every dispatched operation,
which can produce large outputs for complex models. We're exploring ways to make
it more targeted (e.g., filtering by op type, module boundaries, or tensor
criteria).
