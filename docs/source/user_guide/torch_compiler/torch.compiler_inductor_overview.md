(torch.compiler_inductor_overview)=

# Architecture Overview

This page describes the software architecture of TorchInductor, with examples
illustrating the various optimizations and features of the backend.

TorchInductor is an optimizing compiler for PyTorch that lowers captured
computation graphs into hardware-optimized execution. Its most important
optimization is operator fusion, which reduces memory bandwidth by combining
multiple operations into single fused kernels. TorchInductor primarily generates
Triton kernels for GPUs and vectorized C++ kernels for CPUs.

As the default backend for `torch.compile`, TorchInductor handles both training
and inference. It applies graph-level rewrites to FX graphs composed of PyTorch
IR before and after forward-backward partitioning. Finally, it lowers from
PyTorch IR to Inductor IR, fuses operators, and generates the final fused
kernels.

## Software Architecture

```{image} ../../_static/img/inductor_user_guide/inductor_arch.png
:alt: High-level architecture of the PyTorch compiler showing TorchDynamo, AOT Dispatcher, and TorchInductor
:width: 600px
:align: center
```

The PyTorch compiler receives a PyTorch program and generates optimized
kernels. The high-level components involved in that translation are:

- **TorchDynamo**: A Python bytecode tracer that captures PyTorch operations
  into a graph (Torch IR) using the CPython Frame Evaluation hook. It
  intercepts Python execution and outputs an
  [FX Graph](https://docs.pytorch.org/docs/stable/fx.html) representation of
  the model's forward pass without requiring code changes. Torch IR consists of
  Python functions (for example, `torch.add`) and supports the full
  [2000+ PyTorch operator set](https://dev-discuss.pytorch.org/t/where-do-the-2000-pytorch-operators-come-from-more-than-you-wanted-to-know/373).
  Later in the pipeline, operators are decomposed to a minimal operator set.
  Note that after TorchDynamo, we only have the forward graph.

- **AOT (Ahead-of-Time) Dispatcher**: Historically known as AOT Autograd, the
  AOT Dispatcher takes the FX graph from Dynamo and uses PyTorch's
  `torch_dispatch` to capture the backward graph through the autograd engine.
  In this step, the graph is lowered to
  [ATen IR](https://docs.pytorch.org/docs/stable/torch.compiler_ir.html)
  that is easier for the compiler to handle. It also performs
  [decomposition](torch.compiler_inductor_decomposition.md) and
  functionalization. By generating the backward graph up front, the AOT
  Dispatcher enables whole-program optimization of training loops.

  :::{note}
  "AOT Dispatcher" and "AOT Autograd" are used interchangeably. The codebase
  uses the "AOT Autograd" terminology.
  :::

- **TorchInductor**: Receives FX graphs in ATen IR and generates optimized
  kernels. This is the focus of this documentation section.

:::{note}
Only the JIT compilation flow is described here. *Ahead-of-Time* user flows
with [TorchExport](https://docs.pytorch.org/docs/stable/export.html) and
[AOTInductor](torch.compiler_aot_inductor.md) are outside the scope of this
page.
:::

## TorchInductor Pipeline

The figure below shows the code structure of TorchInductor.

```{image} ../../_static/img/inductor_user_guide/arch_detailed.png
:alt: Detailed TorchInductor compilation pipeline from Pre-grad Passes through Code Generation to hardware targets
:width: 600px
:align: center
```

The entry point into TorchInductor is the `compile_fx` function in
[compile_fx.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/compile_fx.py),
which orchestrates the compilation of an FX graph. Compilation includes calling
AOT Autograd, as well as TorchInductor's optimizations and code generation.

TorchInductor receives FX graphs in Torch IR from TorchDynamo and applies a
series of passes to generate optimized kernels:

1. **[Pre-grad passes](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/fx_passes/pre_grad.py)**:
   Run on high-level Torch IR (for example, `torch.nn.functional.linear`) which
   contains 2000+ ops. The high-level IR makes it easier to perform pattern
   matching and can expose fusion opportunities. However, the IR has neither been
   normalized (i.e., not in a canonicalized form) nor functionalized (i.e., not
   in SSA form), so pre-grad passes must be safe with respect to aliasing and
   mutation.

2. **[AOT Autograd](https://github.com/pytorch/pytorch/blob/main/torch/_functorch/aot_autograd.py)**:
   Runs the FX graph to trace the forward and derive the backward graph. During
   tracing, the IR is functionalized (put in SSA form), normalized
   (canonicalized), and decomposed into simpler, fundamental ATen IR. At the end
   of this step, we have a joint forward-backward graph.

3. **[Joint graph passes](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/fx_passes/joint_graph.py)**:
   Run on the combined forward and backward graphs from AOT Autograd. These
   optimizations are used when both the forward and backward implementation of
   an operator need to change. Some pattern matching is run at this stage, for
   example.

4. **[Partitioner](https://dev-discuss.pytorch.org/t/min-cut-optimal-recomputation-i-e-activation-checkpointing-with-aotautograd/467)**:
   A min-cut partitioner splits the joint graph into separate forward and
   backward graphs, minimizing global memory accesses and improving memory.

5. **[Post-grad passes](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/fx_passes/post_grad.py)**:
   Receive normalized, functionalized, and partitioned forward and backward
   graphs. They run optimizations such as no-op elimination, dead code
   elimination, and pattern matching. Users can also write their own custom
   post-grad passes and use the `post_grad_custom_pre_pass` /
   `post_grad_custom_post_pass` configuration hooks to add those custom passes
   to the compilation pipeline. This is the final stage to run high-level graph
   optimizations before the IR is lowered to Inductor IR.

6. **[Graph lowering](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/graph.py)**:
   Lowers ATen IR into [Inductor IR](torch.compiler_inductor_ir.md).

7. **[Scheduling](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/scheduler.py)**:
   The [scheduler](torch.compiler_inductor_scheduler.md) operates on
   `SchedulerNode` objects to analyze dependencies and make fusion choices, with
   the goal of minimizing global reads and writes. The scheduler makes fusion
   decisions according to a score, which it calculates based on the type of
   fusion, an estimate of the amount of saved memory operations, as well as the
   proximity of the operations in the graph.

8. **[Code generation](https://github.com/pytorch/pytorch/tree/main/torch/_inductor/codegen)**:
   Inductor IR is executed to [generate kernels](torch.compiler_inductor_codegen.md).
   Based on the target hardware, Triton, CUTLASS, CK, and C++ kernels can be
   generated. This step also writes wrapper functions to call the generated
   kernels.

For more details on each stage, see:
[FX Passes](torch.compiler_inductor_fx_passes.md),
[Inductor IR](torch.compiler_inductor_ir.md),
[Scheduler & Fusion](torch.compiler_inductor_scheduler.md), and
[Code Generation](torch.compiler_inductor_codegen.md).

(starter-example)=

## Starter Example

Let's start with a simple example with two elementwise operations — `relu` and
addition. We will show step-by-step how TorchInductor fuses these two operations
and generates Triton code.

:::{tip}
For a more comprehensive example, see the
[ASPLOS Inductor tutorial](https://colab.research.google.com/drive/1FTeYO6sf1Vco8dn0qyWze8WIVWLiXQ3z?usp=sharing).
:::

### The input program

We start with the following simple PyTorch program (`example.py`):

```python
import torch

def f(x):
    y = torch.nn.functional.relu(x)
    return y + 1

f = torch.compile(f)
x = torch.randn(10, 10, device="cuda")
f(x)
```

### Step 1: Post-grad graph

After TorchDynamo and AOT Autograd, we can get the post-grad graph with:

```bash
TORCH_LOGS=post_grad_graphs python3 example.py
```

```python
def forward(self, arg0_1: "f32[10, 10][10, 1]cuda:0"):
    relu: "f32[10, 10][10, 1]cuda:0" = torch.ops.aten.relu.default(arg0_1);  arg0_1 = None
    add: "f32[10, 10][10, 1]cuda:0" = torch.ops.aten.add.Tensor(relu, 1);  relu = None
    return (add,)
```

This output shows an FX graph with two
[ATen ops](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen):
`torch.ops.aten.relu.default` and `torch.ops.aten.add.Tensor`. We also observe
the placeholder `arg0_1`, output `add`, and intermediate tensors `relu` and
`add`. For each tensor, we can see the dtype, shape, device, and other meta
information.

### Step 2: Graph lowering

The next step in the example is
[graph lowering](torch.compiler_inductor_ir.md), which converts the FX graph
into Inductor IR. Graph lowering visits every FX node in the graph and generates
the corresponding IR node. From the debugger, let's check a few fields after
lowering:

```pycon
>>> graph.graph_inputs
{'arg0_1': TensorBox(StorageBox(
  InputBuffer(name='arg0_1', layout=FixedLayout('cuda:0', torch.float32, size=[10, 10], stride=[10, 1]))
))}

>>> graph.graph_outputs
[StorageBox(
  ComputedBuffer(name='buf0', layout=FixedLayout('cuda:0', torch.float32, size=[10, 10], stride=[10, 1]),
    data=Pointwise(device=device(type='cuda', index=0), dtype=torch.float32,
      inner_fn=<function make_pointwise.<locals>.inner.<locals>.inner_fn at 0x7f6e416553a0>,
      ranges=[10, 10]))
)]

>>> graph.operations
[ComputedBuffer(name='buf0', layout=FixedLayout('cuda:0', torch.float32, size=[10, 10], stride=[10, 1]),
  data=Pointwise(device=device(type='cuda', index=0), dtype=torch.float32,
    inner_fn=<function make_pointwise.<locals>.inner.<locals>.inner_fn at 0x7f6e416553a0>,
    ranges=[10, 10]))]

>>> graph.buffers
[ComputedBuffer(name='buf0', layout=FixedLayout('cuda:0', torch.float32, size=[10, 10], stride=[10, 1]),
  data=Pointwise(device=device(type='cuda', index=0), dtype=torch.float32,
    inner_fn=<function make_pointwise.<locals>.inner.<locals>.inner_fn at 0x7f6e416553a0>,
    ranges=[10, 10]))]
```

On the data side, we can see graph inputs `arg0_1` and graph outputs `buf0`.
Since `buf0` is a tensor computed by an operation, we also record `buf0` in
`graph.operations`. Since `buf0` is an intermediate tensor, we allocate a buffer
for it and record it in `graph.buffers`. In this starter example, `buf0` happens
to be the only buffer and operation, which is also a graph output. For more
complicated code, there could be many buffers and operations that are not graph
outputs.

Graph lowering handles some simple optimizations such as fusing elementwise
operations. Advanced optimizations, such as horizontal fusion and reordering for
peak memory, are mostly done in the
[scheduler](torch.compiler_inductor_scheduler.md). Please check the
[Graph Lowering](torch.compiler_inductor_ir.md) and
[Scheduler and Fusion](torch.compiler_inductor_scheduler.md) sections for more
details.

### Step 3: Scheduling

After graph lowering, we enter the
[scheduler](torch.compiler_inductor_scheduler.md), which performs the most
advanced optimizations. In particular, it converts Inductor IR into
[BaseSchedulerNode](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/scheduler.py)
and
[SchedulerBuffer](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/scheduler.py),
analyzes dependencies and mutations, conducts fusion, and performs reordering.

### Step 4: Code generation

Finally, TorchInductor performs
[codegen](torch.compiler_inductor_codegen.md) to generate code that can be run
by users. Our example operates on CUDA tensors, so TorchInductor generates
Triton code. TorchInductor could also generate other code, such as C++ and
CUTLASS, depending on the data and configuration. The codegen accepts
`nodes: list[BaseSchedulerNode]`, iterates through each node, and generates code
for each node.

Let's breakpoint at the start of `_codegen` and check what information is
available in a `BaseSchedulerNode`:

```pycon
>>> # def _codegen(self, nodes: list[BaseSchedulerNode]) -> None
>>> nodes
[SchedulerNode(name='op0')]

>>> nodes[0].node
ComputedBuffer(name='buf0', layout=FixedLayout('cuda:0', torch.float32, size=[10, 10], stride=[10, 1]),
  data=Pointwise(device=device(type='cuda', index=0), dtype=torch.float32,
    inner_fn=<function make_pointwise.<locals>.inner.<locals>.inner_fn at 0x7fb89e965c60>,
    ranges=[10, 10]))

>>> nodes[0].read_writes
ReadWrites(
  reads=OrderedSet([MemoryDep('arg0_1', c0, {c0: 100})]),
  writes=OrderedSet([MemoryDep('buf0', c0, {c0: 100})]),
  index_exprs=OrderedSet([]),
  range_vars=[],
  var_ranges={d0: 100})
```

As we can see, `_codegen` takes a list of `BaseSchedulerNode`, which is
generated and fused in the scheduler. In this example, we have one
`SchedulerNode`, which wraps the Inductor IR `buf0` as we discussed earlier. It
also contains the dependency information in `read_writes`. Here, the node reads
from `arg0_1` and writes to `buf0`.

### Generated Triton kernel

Finally, we generate runnable Triton code, which contains the generated Triton
kernels, a wrapper call function to call the generated Triton kernels, and
`benchmark_compiled_module`, which is convenient for benchmarking the generated
Triton kernels. In the Triton code, we can find the fused `relu` and `add`
operations:

```python
def triton_poi_fused_add_relu_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 100
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)  # relu
    tmp3 = 1.0
    tmp4 = tmp2 + tmp3  # add
    tl.store(out_ptr0 + (x0), tmp4, xmask)
```

Both `relu` and `add` are fused into a single kernel. The input tensor is loaded
once, both operations are applied, and the result is stored once — eliminating
the intermediate memory read/write that would occur if the operations ran
separately. This is the core value of TorchInductor's operator fusion.

## Exploring Compilation Artifacts with tlparse

[tlparse](https://github.com/pytorch/tlparse) is a tool that analyzes
structured torch trace logs and generates an HTML report to help explore
compilation artifacts. It is often the first step in debugging a `torch.compile`
job.

Using `tlparse` is straightforward: set the `TORCH_TRACE` environment variable
and then point `tlparse` to the output directory. For example:

```bash
TORCH_TRACE=/tmp/my_traced_log_dir python3 example.py
tlparse /tmp/my_traced_log_dir -o tl_out/
```

The generated HTML organizes log artifacts by compile ID. For example:

```{image} ../../_static/img/inductor_user_guide/tlparse_output.png
:alt: tlparse output showing compile artifacts organized by compile ID
:width: 600px
:align: center
```

Among other things, you can observe the post-grad graph discussed earlier (in
`0_0_0/inductor_post_grad_graph_6.txt`) and the generated code (in
`_0_0_0/inductor_output_code_<hash>_7.txt`). The `tlparse` output also includes
other information such as the dynamo graph, the pre-grad graph, cache hit/miss
information, and the dynamo C++ guards.

:::{seealso}
- [Provenance Tracking](torch.compiler_inductor_provenance) for using
  `tlparse` with `INDUCTOR_PROVENANCE=1` to visualize how original operations
  map to generated code.
- The {doc}`Troubleshooting guide <torch.compiler_troubleshooting>` for a
  detailed walkthrough of `tlparse` setup and usage.
:::
