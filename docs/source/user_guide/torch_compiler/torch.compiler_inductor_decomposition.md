(torch.compiler_inductor_decomposition)=

# Operator Decomposition

Operator decomposition refers to replacing a complex or high-level operation
with an equivalent sequence of simpler operations.

## Why Decompose?

Decomposition serves three purposes:

1. **Autograd simplification**: Rather than implementing backward formulas for
   every high-level operation, we can decompose them into simpler operations
   that already have backward formulas.

2. **Operator coverage for backends**: By decomposing high-level operations, we
   reduce the set of operations that backends must support.

3. **Optimization opportunities**: Decomposing an operation can expose more
   opportunities for kernel fusion later in the compiler pipeline.

## Decomposition in AOT Autograd

Operator decompositions are performed by AOT Autograd. As it traces the forward
and backward graphs, it functionalizes and normalizes the IR, then decomposes
torch operators (`torch.*`) to ATen operators (`aten.*`). For TorchInductor,
AOT Autograd is invoked from the
[aot_autograd](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/compile_fx.py)
call in `compile_fx.py`. By the time `aot_autograd` returns, all `torch.*` ops
have been decomposed into `aten.*` ops that TorchInductor can support and
optimize.

### The `__torch_dispatch__` Mechanism

AOT Autograd uses PyTorch's `__torch_dispatch__` mechanism to intercept
operations and replace them with their decomposed forms during tracing. The
process works as follows:

1. During tracing of both the forward and backward graphs, each operation is
   intercepted.
2. The system checks whether the operation has an entry in the decomposition
   table.
3. If it does, the registered decomposition function is invoked and the
   resulting simpler ops are recorded in the trace instead of the original op.

This process occurs for both the forward and backward graphs.

### Decomposition Tables

Decomposition happens at multiple levels. During AOT Autograd tracing, it checks
if an operation has registered in the decomposition table. The decomposition
table structure is defined in
[torch/_decomp/\_\_init\_\_.py](https://github.com/pytorch/pytorch/blob/main/torch/_decomp/__init__.py),
and the actual implementations are registered in
[torch/_decomp/decompositions.py](https://github.com/pytorch/pytorch/blob/main/torch/_decomp/decompositions.py).

Some decompositions are defined as C++ native functions. The list of operations
and their native decompositions is defined in
[aten/src/ATen/native](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native).

### Multi-Level Decomposition

Decomposition can occur across multiple levels. For example, consider
`aten.linear`:

1. `aten.linear` is a C++ op that first decomposes into `aten.addmm`
   (see [Linear.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Linear.cpp)).
2. During AOT Autograd tracing, `aten.addmm` further decomposes into `aten.mm`
   and `aten.add`
   (see [decompositions.py](https://github.com/pytorch/pytorch/blob/main/torch/_decomp/decompositions.py)).

## Inductor-Specific Decompositions

After AOT Autograd produces ATen-level IR, TorchInductor may apply further
decompositions specific to its code generation needs. These are defined in
[torch/_inductor/decomposition.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/decomposition.py).

## Example: `nn.Linear` Decomposition

Consider the following program:

```python
import torch

def fn(x: torch.Tensor):
    layer = torch.nn.Linear(10, 10)
    return layer(x)

x = torch.randn(10)
torch.compile(fn)(x)
```

Running with logging (`aot_graphs`):

```bash
TORCH_LOGS="aot_graphs" python3 example.py
```

**Forward graph:**

```python
def forward(self, primals_1, primals_2, primals_3):
    view = torch.ops.aten.view.default(primals_3, [1, 10])
    t = torch.ops.aten.t.default(primals_1)
    addmm = torch.ops.aten.addmm.default(primals_2, view, t)
    view_1 = torch.ops.aten.view.default(addmm, [10])
    return (view_1, view)
```

**Backward graph:**

```python
def forward(self, view, tangents_1):
    view_2 = torch.ops.aten.view.default(tangents_1, [1, 10])
    t_1 = torch.ops.aten.t.default(view_2)
    mm = torch.ops.aten.mm.default(t_1, view)
    t_2 = torch.ops.aten.t.default(mm)
    sum_1 = torch.ops.aten.sum.dim_IntList(view_2, [0], True)
    view_3 = torch.ops.aten.view.default(sum_1, [10])
    t_3 = torch.ops.aten.t.default(t_2)
    return (t_3, view_3, None)
```

`nn.Linear` decomposes into `aten.view`, `aten.t`, and `aten.addmm` in the
forward pass, and `aten.mm`, `aten.t`, and `aten.sum` in the backward pass.
All operations are decomposed to the ATen level.

## Comparing AOT Autograd and Inductor Decomposition

To understand when operations decompose and when they don't, let's examine
`torch.repeat_interleave`, which behaves differently from `nn.Linear`. Consider
this code:

```python
@torch.compile
def fn(y, repeats, dim, output_size):
    indices = torch.repeat_interleave(
        y, repeats, dim=dim, output_size=output_size
    )
    return indices

y = torch.tensor([[1, 2], [3, 4]])
repeats = torch.tensor([1, 2])
output_size = 3
dim = 0
```

Running with AOT Eager:

```python
torch.compile(fn, backend="aot_eager")(y, repeats, dim, output_size)
```

With AOT Eager, `aten.repeat_interleave` stays in the graph as-is; it is not
decomposed. The operation is used directly, followed by `aten.index_select`.

```python
def forward(self, arg0_1: "i64[2, 2][2, 1]cpu", arg1_1: "i64[2][1]cpu"):
    repeat_interleave: "i64[3][1]cpu" = torch.ops.aten.repeat_interleave.Tensor(arg1_1, output_size=3);  arg1_1 = None
    index_select: "i64[3, 2][2, 1]cpu" = torch.ops.aten.index_select.default(arg0_1, 0, repeat_interleave);  arg0_1 = repeat_interleave = None
    return (index_select,)
```

Running with TorchInductor:

```python
torch.compile(fn, backend="inductor")(y, repeats, dim, output_size)
```

When compiled with TorchInductor, the behavior changes. Inductor applies a
conditional
[decomposition](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/decomposition.py)
to `repeat_interleave`. The decomposed version uses cumulative sum (`cumsum`),
binary search (`searchsorted`) to find insertion points, and indexing to gather
the results. This is fundamentally different from `nn.Linear`, which is fully
decomposed by AOT Autograd.

```python
# torch.compile(fn, backend="inductor")(y, repeats, dim, output_size)
def forward(self, arg0_1, arg1_1):
    cumsum = torch.ops.aten.cumsum.default(arg1_1, 0)
    iota = torch.ops.prims.iota.default(
        3, start=0, step=1, dtype=torch.int64,
        device=device(type='cpu'), requires_grad=False
    )
    searchsorted = torch.ops.aten.searchsorted.Tensor(
        cumsum, iota, right=True
    )
    index = torch.ops.aten.index.Tensor(arg0_1, [searchsorted])
    return (index,)
```

## Decomposition vs. Lowering

Decomposition and [graph lowering](torch.compiler_inductor_ir.md) are distinct
transformations:

- **Decomposition** rewrites ATen ops into *other ATen ops* (or Prims ops).
  The IR stays at the FX graph level. For example, `aten.addmm` decomposes
  into `aten.mm` + `aten.add`.
- **Lowering** converts ATen ops into *Inductor IR* nodes (Pointwise, Reduction,
  etc.) that represent fused computation. For example,
  [aten.mean is lowered](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/lowering.py)
  into Inductor IR that computes a sum and division.

Decomposition happens first (during AOT Autograd and in Inductor's
decomposition pass), and lowering happens afterward during graph lowering.

## Constraints on Decompositions

Decompositions must be **pure functions**: no side effects, no global state
mutations, and no I/O. They must be deterministic, meaning the same inputs
always produce the same outputs.

## Registering Decompositions

You can register decompositions using the `@register_decomposition` decorator
at two levels:

- **AOT Autograd level** — register in
  [torch/_decomp/decompositions.py](https://github.com/pytorch/pytorch/blob/main/torch/_decomp/decompositions.py).
  These apply to all backends that use AOT Autograd.
- **Inductor level** — register in
  [torch/_inductor/decomposition.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/decomposition.py).
  These apply only when TorchInductor is the backend.

For testing and experimentation, you can also pass a custom decomposition
table to
[make_fx](https://github.com/pytorch/pytorch/blob/main/torch/fx/experimental/proxy_tensor.py)
(see the [test examples](https://github.com/pytorch/pytorch/blob/main/test/test_proxy_tensor.py)).

:::{seealso}
For a reference of the Core ATen and Prims operator sets that decomposition
targets, see [IRs](torch.compiler_ir.md).
:::
