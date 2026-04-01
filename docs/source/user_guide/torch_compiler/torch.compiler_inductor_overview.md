(torch.compiler_inductor_overview)=

# Architecture Overview

This page describes the software architecture of TorchInductor, with a
step-by-step example illustrating how it compiles a simple PyTorch program into
an optimized Triton kernel.

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

```{image} ../../_static/img/inductor_user_guide/arch_detailed.png
:alt: Detailed TorchInductor compilation pipeline from Pre-grad Passes through Code Generation to hardware targets
:width: 600px
:align: center
```

The entry point into TorchInductor is the `compile_fx` function in
[compile_fx.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/compile_fx.py),
which orchestrates compilation of an FX graph. This includes calling AOT
Autograd as well as TorchInductor's optimizations and code generation.

TorchInductor receives FX graphs in Torch IR from TorchDynamo and applies a
series of passes to generate optimized kernels:

1. **[Pre-grad passes](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/fx_passes/pre_grad.py)**:
   Run on high-level Torch IR (for example, `torch.nn.functional.linear`) which
   contains 2000+ ops. The high-level IR makes it easier to perform pattern
   matching and can expose fusion opportunities. However, the IR has not been
   normalized or functionalized, so pre-grad passes must be safe with respect to
   aliasing and mutation.

2. **[AOT Autograd](https://github.com/pytorch/pytorch/blob/main/torch/_functorch/aot_autograd.py)**:
   Traces the forward graph and derives the backward graph. During tracing, the
   IR is functionalized (put in SSA form), normalized (canonicalized), and
   decomposed into simpler ATen IR. At the end of this step, we have a joint
   forward-backward graph.

3. **[Joint graph passes](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/fx_passes/joint_graph.py)**:
   Run on the combined forward and backward graphs from AOT Autograd. These
   optimizations are used when both the forward and backward implementation of
   an operator need to change.

4. **[Partitioner](https://dev-discuss.pytorch.org/t/min-cut-optimal-recomputation-i-e-activation-checkpointing-with-aotautograd/467)**:
   A min-cut partitioner splits the joint graph into separate forward and
   backward graphs, minimizing global memory accesses.

5. **[Post-grad passes](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/fx_passes/post_grad.py)**:
   Receive normalized, functionalized, and partitioned forward and backward
   graphs. They run optimizations such as no-op elimination, dead code
   elimination, and pattern matching. Users can add custom passes via the
   `post_grad_custom_pre_pass` / `post_grad_custom_post_pass` configuration
   hooks. This is the final stage for high-level graph optimizations before
   lowering to Inductor IR.

6. **[Graph lowering](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/graph.py)**:
   Lowers ATen IR into [Inductor IR](torch.compiler_inductor_ir.md).

7. **[Scheduling](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/scheduler.py)**:
   The [scheduler](torch.compiler_inductor_scheduler.md) operates on
   `SchedulerNode` objects to analyze dependencies and make fusion choices, with
   the goal of minimizing global reads and writes. Fusion decisions are based on
   a score calculated from the type of fusion, an estimate of saved memory
   operations, and the proximity of operations in the graph.

8. **[Code generation](https://github.com/pytorch/pytorch/tree/main/torch/_inductor/codegen)**:
   Inductor IR is used to [generate kernels](torch.compiler_inductor_codegen.md).
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

Let's walk through a simple example with two elementwise operations — `relu` and
addition — to see step-by-step how TorchInductor fuses them and generates Triton
code.

:::{tip}
For a more comprehensive example, see the
[ASPLOS Inductor tutorial](https://colab.research.google.com/drive/1FTeYO6sf1Vco8dn0qyWze8WIVWLiXQ3z?usp=sharing).
:::

### The input program

We start with the following PyTorch program (`example.py`):

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

After TorchDynamo and AOT Autograd, we can inspect the post-grad graph:

```bash
TORCH_LOGS=post_grad_graphs python3 example.py
```

```python
def forward(self, arg0_1: "f32[10, 10][10, 1]cuda:0"):
    relu: "f32[10, 10][10, 1]cuda:0" = torch.ops.aten.relu.default(arg0_1);  arg0_1 = None
    add: "f32[10, 10][10, 1]cuda:0" = torch.ops.aten.add.Tensor(relu, 1);  relu = None
    return (add,)
```

This FX graph contains two
[ATen ops](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen):
`torch.ops.aten.relu.default` and `torch.ops.aten.add.Tensor`. The placeholder
`arg0_1` is the input and `add` is the output. For each tensor, we can see the
dtype, shape, device, and stride metadata.

### Step 2: Graph lowering

Next, [graph lowering](torch.compiler_inductor_ir.md) converts the FX graph
into Inductor IR. Graph lowering visits every FX node and generates the
corresponding IR node. Inspecting a few fields after lowering:

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

The graph input is `arg0_1` and the graph output is `buf0`. Since `buf0` is a
tensor computed by an operation, it also appears in `graph.operations`. Since
`buf0` is an intermediate tensor, a buffer is allocated for it and recorded in
`graph.buffers`. In this simple example, `buf0` is the only buffer, operation,
and graph output. In more complex programs, there can be many buffers and
operations that are not graph outputs.

Graph lowering handles some simple optimizations such as fusing elementwise
operations. Advanced optimizations — horizontal fusion, reordering for peak
memory — are performed by the [scheduler](torch.compiler_inductor_scheduler.md).

### Step 3: Scheduling

After graph lowering, the
[scheduler](torch.compiler_inductor_scheduler.md) converts Inductor IR into
[BaseSchedulerNode](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/scheduler.py)
and
[SchedulerBuffer](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/scheduler.py)
objects, analyzes dependencies and mutations, performs fusion, and reorders
operations.

### Step 4: Code generation

Finally, TorchInductor
[generates code](torch.compiler_inductor_codegen.md) that can be executed. The
codegen accepts `nodes: list[BaseSchedulerNode]`, iterates through each node,
and generates code for it.

Breakpointing at `_codegen` shows the information available in a
`BaseSchedulerNode`:

```pycon
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

The `SchedulerNode` wraps the Inductor IR `buf0` and contains dependency
information in `read_writes`: the node reads from `arg0_1` and writes to `buf0`.

### Generated Triton kernel

Since our example operates on CUDA tensors, TorchInductor generates a Triton
kernel. The output includes the generated kernel, a wrapper function to call it,
and a `benchmark_compiled_module` for benchmarking. In the kernel below, we can
see the fused `relu` and `add` operations:

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

Basic usage:

```bash
TORCH_TRACE=/tmp/my_trace_log_dir python3 example.py
tlparse /tmp/my_trace_log_dir -o tl_out/
```

The generated HTML organizes artifacts by compile ID, including the post-grad
graph, generated output code, dynamo graph, pre-grad graph, cache hit/miss
information, and dynamo C++ guards.

:::{seealso}
- [Provenance Tracking](torch.compiler_inductor_provenance) for using
  `tlparse` with `INDUCTOR_PROVENANCE=1` to visualize how original operations
  map to generated code.
- The {doc}`Troubleshooting guide <torch.compiler_troubleshooting>` for a
  detailed walkthrough of `tlparse` setup and usage.
:::
