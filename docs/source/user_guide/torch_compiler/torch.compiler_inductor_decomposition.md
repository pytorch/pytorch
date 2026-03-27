(torch.compiler_inductor_decomposition)=

# Operator Decomposition

Operator decomposition replaces a complex or high-level operation with an
equivalent sequence of simpler operations. This is a key transformation in the
compilation pipeline that happens primarily during AOT Autograd tracing, before
TorchInductor's own optimizations begin.

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
have been decomposed into `aten.*` ops that TorchInductor can optimize.

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

Decomposition happens at multiple levels. The decomposition table structure is
defined in
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

Running with AOT graph logging:

```bash
TORCH_LOGS="aot_graphs" python3 example.py
```

The output shows the decomposed graph where `nn.Linear` has been broken down
into fundamental ATen operations (`aten.mm`, `aten.add`, etc.) that
TorchInductor can fuse and optimize.

:::{seealso}
For a reference of the Core ATen and Prims operator sets that decomposition
targets, see [IRs](torch.compiler_ir.md).
:::
