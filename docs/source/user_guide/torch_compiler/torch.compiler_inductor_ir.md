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

`ops` is an implementation detail used by lowering; you can ignore that detail.
Note, however, how we load from `arg0_1` and `arg1_1` with an indexing range
(`index`, or `i0`). That indexing range, the buffers used, and the associated
function are the primary components of the `ir.Loops` representation.

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
  reference to the viewed buffer and is used for view operations that share
  underlying storage.
- **NoneLayout** — represents no tensor to return.
- **MultiOutputLayout** — represents multiple distinct return values from an IR
  node.
- **MutationLayoutSHOULDREMOVE** — represents the result of an in-place
  modification as a fresh IR node.

## Operation Base Class

The `Operation` class represents any computation that produces buffers. It is
the bridge between the IR and the [scheduler](torch.compiler_inductor_scheduler.md),
providing the information needed for dependency analysis and scheduling:

- Which buffers are **read** and **written**
- What outputs are produced
- What symbols are needed for code generation

After graph lowering, `Operation` objects (typically `ComputedBuffer` and
`ExternKernel` instances) are what the scheduler converts into
`SchedulerNode` objects for fusion and code generation.

## Key Operations and Mechanisms

### Realization

Realization is the process of converting an unrealized computation into a
materialized buffer. This happens when a computation result needs to be stored
because it cannot be fused with subsequent operations.

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
and uses `fx.Interpreter` to traverse the graph and generate IR nodes.

### Entry Point

The main entry point is in
[compile_fx.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/compile_fx.py):

```python
graph = GraphLowering(gm)
with V.set_graph_handler(graph):
    graph.run(*example_inputs)       # Performs lowering
    compiled_fn = graph.compile_to_fn()  # Generates code
```

The `fx.Interpreter` takes an FX graph and emulates "running" it: it takes
tensor inputs (called `example_inputs`) and processes the graph by passing
around `ir.TensorBox` objects. The result is a list of `ir.Buffer` objects.

The call to `graph.compile_to_fn()` converts those `ir.Buffer` objects into
an actual runnable function via the
[scheduler](torch.compiler_inductor_scheduler.md) and
[code generation](torch.compiler_inductor_codegen.md) steps. In this section,
we focus on what happens in `graph.run(*example_inputs)` — that is,
*graph lowering*.

### Lowering Process

The high-level pseudocode for graph lowering:

```python
import torch._inductor.ir as ir
from torch._inductor.lowerings import lowerings

def lower(fx_g: fx.Graph, inputs: torch.Tensor):
    # Map FX nodes to their Inductor IR representations (TensorBox); it stores
    # the lowered result for each node.
    env: Dict[fx.Node, ir.TensorBox] = {}

    # List of buffers that need to be realized.
    # These represent actual memory allocations in the generated code.
    buffers: List[ir.Buffer] = []

    # Iterate through all nodes in the FX graph in topological order.
    for node in fx_g.graph.nodes:
        if node.op == 'placeholder':
            # Placeholder nodes are graph inputs. Wrap them as InputBuffer IR nodes.
            # InputBuffer represents a tensor that already exists.
            env[node] = ir.TensorBox(ir.InputBuffer(get_input(inputs)))

        elif node.op == 'call_function':
            # Look up the lowering function for this operation from the registry.
            # The lowerings dict maps torch ops (e.g., aten.add) to their IR lowering
            # functions.
            lowering_fn = lowerings[node.target]

            # Map the FX node arguments to their corresponding IR representations.
            # tree_map recursively traverses the args structure and looks up each node.
            arg_boxed = tree_map(lambda n: env[n], args)

            # Call the lowering function to convert the operation into Inductor IR.
            # This produces a TensorBox wrapping the IR node (e.g., Pointwise, Reduction).
            output_box: ir.TensorBox = lowering_fn(*arg_boxed)

            # Store the result in the environment for future nodes to reference.
            env[node] = output_box

        else:
            pass  # Skip other node types (e.g., output, get_attr)

        # Check if this intermediate result needs to be "realized".
        # Realization happens when:
        # - The buffer has too many reads (would be recomputed too many times).
        # - It's mutated later.
        # - It's needed by operations that require contiguous inputs.
        if env[node].needs_to_be_realized():
            # Create a ComputedBuffer which represents an actual memory allocation
            # that will appear in the generated kernel code.
            buffers.append(ComputedBuffer(env[node]))

    # Return the list of buffers to be scheduled and code-generated.
    return buffers
```

Essentially, we initialize our environment with
`ir.TensorBox(ir.InputBuffer(...))` nodes (these are inputs to the whole
graph). Then, we call our lowerings which take `ir.TensorBox` objects and return
new `ir.TensorBox` objects. Finally, whenever an `ir.TensorBox` needs to be
realized (such as if it is being passed to an `ir.ExternKernel`), we create a
`ComputedBuffer` object and append it to our list of buffers.

### The Big Picture: A Lazy Compiler's View of Tensors

Think of TorchInductor as a **lazy**, but smart compiler. When you write:

```python
y = x.sin() + 1
z = y * 2
```

PyTorch eager mode would:

1. Compute `sin(x)` and store it in memory as `y`.
2. Add 1 to `y` and store the result.
3. Multiply by 2 and store that result in memory as `z`.

TorchInductor instead operates like this:

1. "Remember that `y` should be `sin(x) + 1`".
2. "Remember that `z` should be `(sin(x) + 1) * 2`".
3. "When someone actually needs `z`, compute the whole thing in one go".

This laziness enables powerful optimizations, but requires a sophisticated IR
to track what needs to be computed and how. The IR was presented in an earlier
section, but the following examples tie it together with graph lowering:

**Example 1: Simple TensorBox**

```python
# PyTorch code
x = torch.randn(100, 100)

# In Inductor IR, this becomes the following.
tensor_box = TensorBox(
    StorageBox(
        InputBuffer(
            name="arg0_1",
            layout=FixedLayout(size=[100, 100], stride=[100, 1])
        )
    )
)

# Note that the TensorBox is just a wrapper - all the interesting stuff is inside.
```

**Example 2: TensorBox pointing to a View**

```python
# PyTorch code
x = torch.randn(100, 100)
y = x.t()  # transpose

# In Inductor IR:
x_box = TensorBox(StorageBox(...))  # Original tensor
y_box = TensorBox(
    PermuteView(
        base=x_box,  # Points to original
        dims=[1, 0]  # Swap dimensions
    )
)

# y_box doesn't have its own storage - it's a view of x's storage
```

**Example 3: Lazy StorageBox**

```python
# PyTorch code
y = x + 1

# In Inductor IR:
lazy_storage = StorageBox(
    Pointwise(
        device="cuda",
        dtype=torch.float32,
        inner_fn=lambda idx: ops.load("x", idx) + 1.0,
        ranges=[100, 100]  # Output shape
    )
)

# This StorageBox contains a DESCRIPTION of the computation, i.e. an ir.Loops
# subclass object. No memory allocated yet, no computation performed yet.
```

**Example 4: Realized StorageBox**

```python
# After calling realize():
realized_storage = StorageBox(
    ComputedBuffer(
        name="buf0",
        layout=FixedLayout(size=[100, 100], stride=[100, 1]),
        data=Pointwise(...)  # Original computation preserved
    )
)

# Now it has actual memory allocated!
```

**Example 5: Different Buffer types in action**

```python
# 1. InputBuffer - data from outside, being passed into the graph
def compiled_model(x):  # x becomes InputBuffer
    return x + 1

# 2. ComputedBuffer - result of computation
y = x.sin()       # Lazy Pointwise
z = y + 1
z.realize()        # Becomes ComputedBuffer, realizing the computation chain

# 3. ConstantBuffer - compile-time known values
scale = torch.tensor(2.0)  # Becomes ConstantBuffer
z = x * scale

# 4. ExternKernel - when Inductor uses external implementation of an op
# NOTE: since we do not fuse ExternKernel with other kernels, inputs and
# outputs of ExternKernel are always immediately realized
x = torch.randn(100, 200)
y = torch.randn(200, 300)
result = torch.mm(x, y)    # ExternKernel
```

### When Does Computation Actually Happen?

Computation happens when we "realize" a buffer. Realization converts the lazy
description into actual memory with values.

**Realization triggers:**

```python
# Trigger 1: Too many operations reading the same value
x = expensive_op(a)
b1 = x + 1     # read 1
b2 = x * 2     # read 2
b3 = x - 3     # read 3
b4 = x / 4     # read 4
b5 = x * 3     # read 5
b6 = x ** 2    # x gets realized (too many reads)
               # number of consumers allowed is controlled by
               # config.realize_reads_threshold and other related configs

# Trigger 2: External operations (ExternKernel)
x = a + 1                            # Lazy
y = torch.ops.aten.mm.default(x, x)  # x must be realized
```

Realization can also be triggered by **mutation** — when the buffer is mutated
later and needs a concrete memory location — or by **contiguous input
requirements** when downstream operations need concrete, contiguous memory to
read from.

After graph lowering, in downstream steps of the compilation (for example, the
[scheduler](torch.compiler_inductor_scheduler.md)), you can assume that we are
mostly dealing with `ComputedBuffer` nodes. (Although, we also need to make sure
to handle cases where we are passing into an `ir.ExternKernel`.)

#### An E2E Example

The following example traces the full lowering process from source code
through IR to generated Triton code.

**Starting code:**

```python
inps = [torch.randn(100, 256, device='cuda'), torch.randn(256, 50, device='cuda')]

def fn(a, b):
    a = a[:50].sin()
    b = b.t().cos()
    return (a * b,)
```

**FX-ATen graph** (after decomposition):

```python
def forward(self, a_1, b_1):
    slice_tensor = torch.ops.aten.slice.Tensor(a_1, 0, 0, 50);  a_1 = None
    sin_default = torch.ops.aten.sin.default(slice_tensor);  slice_tensor = None
    permute_default = torch.ops.aten.permute.default(b_1, [1, 0]);  b_1 = None
    cos_default = torch.ops.aten.cos.default(permute_default);  permute_default = None
    mul_tensor = torch.ops.aten.mul.Tensor(sin_default, cos_default);  cos_default = sin_default = None
    return (mul_tensor,)
```

**IR during lowering** — each FX node is lowered to an IR representation:

```
a_1 → TensorBox(StorageBox(
    InputBuffer(name='a_1', layout=FixedLayout('cuda', f32, size=[s0, s1], stride=[s1, 1]))
  ))

b_1 → TensorBox(StorageBox(
    InputBuffer(name='b_1', layout=FixedLayout('cuda', f32, size=[s1, s2], stride=[s2, 1]))
  ))

slice_tensor → TensorBox(
    ReinterpretView(
      StorageBox(InputBuffer(name='a_1', ...)),
      FixedLayout('cuda', f32, size=[50, s1], stride=[s1, 1])
    ))

sin_default → TensorBox(StorageBox(
    Pointwise('cuda', f32, sin(load(a_1, i0*s1 + i1)), ranges=[50, s1])
  ))

permute_default → TensorBox(
    ReinterpretView(
      StorageBox(InputBuffer(name='b_1', ...)),
      FixedLayout('cuda', f32, size=[s2, s1], stride=[1, s2])
    ))

cos_default → TensorBox(StorageBox(
    Pointwise('cuda', f32, cos(load(b_1, i0 + i1*s2)), ranges=[s2, s1])
  ))

mul_tensor → TensorBox(StorageBox(
    Pointwise('cuda', f32,
      sin(load(a_1, i0*s1 + i1)) * cos(load(b_1, i0 + i1*s2)),
      ranges=[50, s1])
  ))
```

Several things to observe:

- **`slice_tensor`** uses a `ReinterpretView` — no computation, just a
  reinterpretation of `a_1`'s memory with a smaller first dimension.
- **`sin_default`** is a lazy `Pointwise` that loads from `a_1` using the
  slice's indexing. No buffer is allocated yet.
- **`permute_default`** is also a `ReinterpretView` — the transpose swaps
  strides without copying.
- **`mul_tensor`** composes both computation chains. Because everything is
  lazy, the final `Pointwise` contains `sin(load(...)) * cos(load(...))` —
  both operations fused into one expression.

**Buffers after lowering** — a single `ComputedBuffer` that captures the
entire fused computation:

```
[ComputedBuffer(name='buf0',
    layout=FixedLayout('cuda', f32, size=[50, s1], stride=[s1, 1]),
    data=Pointwise('cuda', f32,
      sin(load(a_1, i0*s1 + i1)) * cos(load(b_1, i0 + i1*s2)),
      ranges=[50, s1])
)]
```

**LoopBody** (the defined-by-run IR used for code generation):

```python
index0 = s1*z0 + z1
index1 = z0 + 50*z1

def forward(self, ops):
    index0 = self.index0
    load = ops.load('a_1', index0, False);  index0 = None
    sin = ops.sin(load);  load = None
    index1 = self.index1
    load_1 = ops.load('b_1', index1, False);  index1 = None
    cos = ops.cos(load_1);  load_1 = None
    mul = ops.mul(sin, cos);  sin = cos = None
    index0_1 = self.index0
    store = ops.store('buf0', index0_1, mul, None);  ops = index0_1 = mul = None
    return store
```

**Generated Triton code**:

```python
@pointwise_heuristics(size_hints=[64, 256])
@triton.jit
def kernel0(in_ptr0, in_ptr1, out_ptr0, ks0, xnumel, ynumel,
            XBLOCK: tl.constexpr, YBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.reshape(tl.arange(0, YBLOCK), [1, YBLOCK])
    ymask = yindex < ynumel
    x0 = xindex
    y1 = yindex
    tmp0 = tl.load(in_ptr0 + y1 + (ks0 * x0), xmask & ymask)
    tmp2 = tl.load(in_ptr1 + x0 + (50 * y1), xmask & ymask)
    tmp1 = tl.sin(tmp0)
    tmp3 = tl.cos(tmp2)
    tmp4 = tmp1 * tmp3
    tl.store(out_ptr0 + y1 + (ks0 * x0), tmp4, xmask & ymask)

def call(a_1, b_1):
    a_1_size = a_1.size()
    s0 = a_1_size[0]
    s1 = a_1_size[1]
    b_1_size = b_1.size()
    s2 = b_1_size[1]
    buf0 = empty_strided((50, s1), (s1, 1), device='cuda', dtype=torch.float32)
    kernel0[grid(50, s1)](a_1, b_1, buf0, s1, 50, s1)
    return (buf0,)
```

The slice, sin, transpose, cos, and multiply operations have all been fused
into a single Triton kernel — no intermediate buffers are allocated. This is
the core value of TorchInductor's lazy IR and fusion strategy.
