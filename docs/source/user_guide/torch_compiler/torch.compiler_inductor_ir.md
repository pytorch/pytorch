(torch.compiler_inductor_ir)=

# Inductor IR and Graph Lowering

This page describes TorchInductor's internal intermediate representation (IR)
and the graph lowering process that converts ATen IR into it.

:::{note}
This page covers TorchInductor's *internal* IR — the representation used after
graph lowering. For the *input* IRs that TorchInductor receives (Core ATen IR
and Prims IR), see [IRs](torch.compiler_ir.md).
:::

## Core Design Principles

The Inductor IR distinguishes among three fundamental concepts:

1. **Tensor metadata**: shapes, strides, views, etc.
2. **Storage**: actual data buffers.
3. **Computation**: operations that produce new buffers.

Each lowering is registered to a particular ATen operator. Instead of
`torch.Tensor` inputs, lowerings work with Inductor `TensorBox` inputs and
produce `TensorBox` outputs. The IR is designed to represent tensor
computations, views, and storage in a way that mirrors PyTorch's tensor model
while enabling efficient compilation.

## IRNode Base Class

All IR nodes inherit from `IRNode`, which provides common functionality:

- Querying tensor properties (size, stride, layout, device, dtype)
- Creating loader and indexer functions
- Tracking dependencies (which buffers are read)
- Materialization (converting unrealized computation to concrete buffers)

## Storage Nodes

### TensorBox

`TensorBox` is the primary IR construct that lowering functions produce and
consume, mapping to a `torch.Tensor` output from an operation. A `TensorBox`
contains either a `View` or directly references a `StorageBox`. Just as PyTorch
tensors can be views or directly own storage, `TensorBox` objects can point to
either.

### StorageBox

`StorageBox` contains either a `Buffer` (the actual computation/storage) or
intermediate computation nodes. It provides:

- **Realization**: converting unrealized computation into materialized buffers
- Determining whether a buffer represents a graph input or module parameter

For tensors that directly own storage, the ownership chain is:

```
TensorBox → StorageBox → Buffer
```

### View

`View` (base class `BaseView`) represents view operations. It contains a
reference to underlying data (an `IRNode`), and multiple `View` objects can
reference the same underlying `StorageBox`.

View types include:

- **ExpandView** — broadcasting/expansion (for example, `x.expand(10, 20)`)
- **PermuteView** — dimension reordering (for example, `x.transpose()`, `x.permute()`)
- **SqueezeView** — dimension removal (for example, `x.squeeze()`)
- **SliceView** — tensor slicing (for example, `x[1:10]`)
- **ReinterpretView** — raw memory reinterpretation (for example, `as_strided()`)
- **DtypeView** — dtype changes
- **GenericView** — general views with custom reindexing functions

For views, the ownership chain is:

```
TensorBox → View → StorageBox → Buffer
```

## Buffer Types

`Buffer` is the base class for all actual storage and computation in the IR. It
contains a name and a `Layout` object with device, dtype, size, stride, and
offset information.

### InputBuffer

An `InputBuffer` represents graph inputs or constants. Unlike other buffers, it
has no corresponding operation because the data comes from outside the graph.

Subtypes:

- **ConstantBuffer** — compile-time constants (like model weights)
- **DonatedBuffer** — saved tensors eligible for in-place reuse during backward

### ComputedBuffer

A `ComputedBuffer` is the result of realizing a computation chain. It holds:

1. **layout** (`ir.Layout`): The properties of the output buffer (shape, stride).
2. **data** (`ir.Loops`): The core of TorchInductor's codegen representation —
   its "loop-nest" representation.

### TemplateBuffer

A `TemplateBuffer` represents buffers computed using pre-written kernel
templates.

Subtypes:

- **TritonTemplateBuffer** — Triton kernel templates
- **CUDATemplateBuffer** — CUDA C++ kernel templates
- **CppTemplateBuffer** — CPU C++ kernel templates
- **CuteDSLTemplateBuffer** — CuTe DSL templates

:::{tip}
For details on writing CuTe DSL templates, see
[torch/_inductor/codegen/cutedsl/README.md](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/codegen/cutedsl/README.md).
:::

### ExternKernel

An `ExternKernel` represents calls to external functions such as ATen operators
or custom kernels. Since Inductor does not fuse `ExternKernel` with other
kernels, inputs and outputs of `ExternKernel` are always immediately realized.

Subtypes:

- **ExternKernelAlloc** — allocates output and calls the function
- **ExternKernelOut** — output is pre-allocated and passed to the function
- **FallbackKernel** — falls back to eager PyTorch execution

## Loops (Computation Representation)

Computation is represented by `Loops` subclasses that describe how to compute
values. A loop nest in TorchInductor is represented conceptually as a function
over index ranges. For example, given:

```python
def f(a, b):
    return a + b
```

The `ir.Loops` representation looks like:

```python
def inner_fn(index):
    i0 = index
    tmp0 = ops.load(arg0_1, i0)
    tmp1 = ops.load(arg1_1, i0)
    tmp2 = tmp0 + tmp1
    return tmp2
```

The indexing range, the buffers used, and the associated function are the primary
components of the `ir.Loops` representation.

:::{note}
The function is not literally a Python lambda — it is actually an FX module.
This design facilitates indexing optimizations.
:::

### Loops Subclasses

- **Pointwise** — element-wise operations
- **Reductions** — operations that reduce along certain dimensions
  - `MultiOutputReduction` — multiple outputs (for example, `argmax` returns both
    value and index)
  - `WelfordReduction` — numerically stable mean/variance computation
  - `OnlineSoftmaxReduction` — fused online softmax computation
- **Scan** — cumulative operations with sequential dependencies (for example,
  `cumsum`, `cumprod`)
- **Scatter** — operations that write to non-contiguous output locations
- **Sort** — sorting operations

## Layout

The `Layout` class carries tensor metadata, analogous to PyTorch's storage
metadata. It describes how to interpret a buffer as a multi-dimensional tensor
with specific device, dtype, dimensions, strides, and offset. All dimensions
and strides are of type `sympy.Expr`, enabling support for symbolic/dynamic
shapes.

Layout types:

- **FlexibleLayout** — not yet fixed; can be optimized. Used during IR
  construction before layout decisions are made, allowing TorchInductor to
  choose stride order.
- **FixedLayout** — frozen and cannot change. Used after layout optimization or
  when a layout must match a specific pattern.
- **NonOwningLayout** — used for views that don't own storage. Contains a
  reference to the viewed buffer.
- **NoneLayout** — represents no tensor to return.
- **MultiOutputLayout** — represents multiple distinct return values from an IR
  node.
- **MutationLayoutSHOULDREMOVE** — represents the result of an in-place
  modification as a fresh IR node.

## Key Operations and Mechanisms

### Realization

Realization is the process of converting an unrealized (lazy) computation into a
materialized buffer. This happens when a computation result needs to be stored
because it cannot be fused with subsequent operations.

Realization is triggered when:

1. **Too many operations read the same value** — the number of consumers allowed
   before realization is controlled by `config.realize_reads_threshold` and
   related configs.
2. **External operations** — `ExternKernel` inputs must be realized because
   external functions need concrete memory to read from.

### Functionalization of Mutations

PyTorch has in-place operations like `relu_()` and `add_()`, but compiler IRs
typically prefer functional semantics. *Functionalization* rewrites in-place
tensor operations into their purely functional equivalents. This happens during
AOT Autograd tracing, before the IR reaches TorchInductor.

### View Unwrapping

View unwrapping traverses the view chain to find the underlying storage. This is
needed when determining the storage that backs a view.

### Loaders and Indexers

- **Loaders** are functions that return values at given positions. For
  `ComputedBuffer`s, loaders execute the computation; for stored buffers, they
  load from memory.
- **Indexers** are functions that compute linear memory offsets from
  multi-dimensional indices.

These are kept separate because loaders can do arbitrary computation while
indexers only compute memory layout. Views compose by transforming indices
before calling the underlying loader.

## Graph Lowering

Graph lowering converts an FX graph of ATen operators into Inductor IR. This
transformation happens in the `GraphLowering` class
([torch/_inductor/graph.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/graph.py))
and uses `fx.Interpreter` to traverse the graph.

The `fx.Interpreter` takes an FX graph and emulates "running" it: it takes
tensor inputs (called `example_inputs`) and processes the graph by passing
around `ir.TensorBox` objects. The result is a list of `ir.Buffer` objects.

### Lowering Process

The lowering process follows these steps:

```python
# Iterate through all nodes in the FX graph in topological order.
# For each node:

# 1. Map the FX node arguments to their corresponding IR representations.

# 2. Call the lowering function to convert the operation into Inductor IR.

# 3. Store the result in the environment for future nodes to reference.

# 4. Check if this intermediate result needs to be "realized".

# Return the list of buffers to be scheduled and code-generated.
```

### Examples

**Graph input** — becomes an `InputBuffer`:

```python
# PyTorch code
x = torch.randn(100, 100)

# In Inductor IR:
# TensorBox(StorageBox(InputBuffer(name='arg0_1', layout=...)))
```

**View operation** — no new storage, just a view wrapper:

```python
# PyTorch code
y = x.t()  # transpose

# In Inductor IR:
# x_box = TensorBox(StorageBox(...))  — original tensor
# y_box = TensorBox(PermuteView(..., StorageBox(...)))  — points to same storage
```

**Lazy computation** — computation is deferred until realization:

```python
# PyTorch code
y = x + 1

# In Inductor IR (before realization):
# StorageBox contains a DESCRIPTION of the computation (an ir.Loops subclass).
# No memory allocated yet, no computation performed yet.

# After calling realize():
# StorageBox now contains a ComputedBuffer with actual memory allocated.
```

**Buffer types in context**:

```python
# 1. InputBuffer — data from outside the graph
def compiled_model(x):  # x becomes InputBuffer

# 2. ComputedBuffer — result of realized computation
    y = x.sin()       # lazy Pointwise
    z = y + 1
    z.realize()        # becomes ComputedBuffer

# 3. ConstantBuffer — compile-time known values
    scale = torch.tensor(2.0)  # becomes ConstantBuffer
    z = x * scale

# 4. ExternKernel — external op implementation
    # Inputs and outputs are always immediately realized
    x = torch.randn(100, 200)
    y = torch.randn(200, 300)
    result = torch.mm(x, y)    # ExternKernel
```
