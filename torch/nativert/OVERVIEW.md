# NativeRT Technical Overview

NativeRT is a flexible C++ inference engine for torch-exported models. For
high-performance CPU inference on the PT2 stack, it's designed to be a drop-in
replacement for
[Static Runtime](https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/runtime/static/README.md).

However, it's support doesn't end there; by default it integrates with the torch
dispatcher, inheriting it's backend
[support matrix](https://docs.pytorch.org/docs/stable/backends.html). Moreover,
it supports the execution of
[AOTInductor](https://github.com/pytorch/pytorch/blob/main/docs/source/torch.compiler_aot_inductor.rst)-lowered
artifacts, and can be extended to support other delegate backends as seen fit.

This document is intended to provide an overview of the foundational NativeRT
components, features, and their interactions.

## Table of Contents

<!-- toc -->

- [Getting Started](#getting-started)
- [Components](#components)
  - [Graph](#graph)
  - [Node](#node)
  - [Value](#value)
  - [IValue](#ivalue)
  - [OpKernel](#opkernel)
  - [Weights](#weights)
  - [Pytree](#pytree)
- [Threading Model](#threading-model)
  - [Execution Frame](#execution-frame)
- [Features](#features)
  - [Delegation](#delegation)
  - [Static Dispatch Kernels](#static-dispatch-kernels)
  - [Constant Folding](#constant-folding)
  - [Quantization](#quantization)
  - [Memory Planning](#memory-planning)
  - [Inter-op parallelism](#inter-op-parallelism)

<!-- tocstop -->

## Getting Started

To get up and running, there isn't much you need to do if you have the path to
your PT2 archive. Below is the minimal code you will need to run inference on
the sample inputs included with the archive.

```cpp
#include <nativert/core/ModelRunner.h>

int main(int argc, char** argv) {
  auto model_name = "my_model";
  auto model_path = "/path/to/my/model";
  const auto device = torch::Device(torch::kCUDA, 0);
  ExecutorType executor_type = ExecutorType::INTERPRETER;

  RuntimeConfigs cfg;
  auto reader =
      std::make_shared<caffe2::serialize::PyTorchStreamReader>(
          std::make_unique<caffe2::serialize::FileAdapter>(
              build::getResourcePath(std::move(model_path)).string()));

  auto runner = ModelRunner(
    std::move(reader),
    std::move(model_name),
    executor_type,
    std::move(cfg),
    Placement(device));

  const auto [args, kwargs] =
      runner.loadSampleInputs(std::move(reader), Placement(device));

  auto output = runner.run(args, kwargs);

  return 0;
}
```

## Components

### Graph

NativeRT is a graph-centric runtime.

In the front-end, torch.export produces an 'ExportedProgram' which contains an
[FX Graph](https://github.com/pytorch/pytorch/tree/main/torch/fx#graph). At this
point, it's possible for additional lowering (e.g., to AOTInductor),
transformations, and optimizations to take place.

[This graph](https://github.com/pytorch/pytorch/blob/main/torch/csrc/utils/generated_serialization_types.h#L1822)
is then serialized into a PT2 archive, which can be deserialized into our
internal, in-memory representation. Following the deserialization stage, upon
initialization, additional graph-passes may be applied; these passes include but
are not limited to [constant folding](#constant-folding), and
[quantization](#quantization).

Here is an example of how we would register a custom pass.

```cpp
#include <nativert/core/passes/pass_manager/GraphPassRegistry.h>

GraphPassRegistry::add_pass("MyPass", [](Graph* graph) {
    bool mutated = do_my_pass(graph);
    return mutated;
});
```

> :warning: **TODO** the graph must be added to the ModelRunner Pipeline s.t.,
> it can be executed during initialization.

After the graph passes have concluded, the graph is deemed immutable and is
ready to run inference.

### Node

A node represents a vertex in the in-memory [graph representation](#graph).
Nodes have a designated target (i.e., the op name), and a list of inputs and
output values.

### Value

Values are a internal construct that represent the edges of our graph. In the
graph, a value could only carry one of these permissible types: Tensor,
TensorList, SymInt, SymIntList and CustomObj. Values are uniquely identifiable
by an integer. Values also have a string name for logging purposes, but we don’t
use it as the identifier. In short, a node consumes some values as inputs, and
produces some values as outputs. Nodes are connected by values to form a graph.

### IValue

Not to be confused with value, which is a graph concept, an IValue (more
formally known as an interpreter value), is a pytorch class. It is a union class
that can hold many datatypes (e.g., at::Tensor, TensorList, SymInt,
CustomClassObject) generically.

Both graph inputs and outputs are IValue's, and execution frames store the state
of the graphs execution using IValues.

### OpKernel

An OpKernel is an internal abstraction representing the computation unit for a
particular graph vertex. As such, each OpKernel has an associated [Node](#node).
OpKernel's are callable; the implementation is responsible for executing the
computation for the associated Node.

The most common OpKernel variant is the C10Kernel, which will by default offload
the computation to the C10 dispatcher. For many performance-sensitive CPU
operators, we provide [static-dispatch kernels](#static-dispatch-kernels) that
can be executed without invoking the dispatcher.

The computation of an OpKernel requires an [ExecutionFrame](#execution-frame) to
be supplied. The frame contains the backed values, and we can use the associated
Node's spec to map inputs/outputs to their runtime values during the kernel
execution.

### Weights

Weights is a class that manages the static states of a model, which remains
immutable throughout the execution. For example module parameters, buffers, and
constants are all managed in Weights. These tensors are "read-only constants" in
a single inference run(). As such, they can be shared across all threads.
Weights is also an user-facing class that provides some APIs for advanced weight
management, such as customized weight loading, weight swapping, and/or in-place
updates.

### Pytree

> :warning: **TODO**

## Threading Model

Our threading model relies heavily on the concept of an 'Execution Frame.'

### Execution Frame

An execution frame encapsulates the state of a particular graph execution. We
maintain a pool of them, and during each execution the calling thread will
acquire one from the pool, execute the graph using that frame, and then return
it back to the pool.

In practice, it's a bit more complicated than this, but here is a pseudo-graphic
attempting to explain the flow.

```
                                         ┌──────────────────── thread_n
                                         │                        ▲
┌────────────────────────────────────────┼────────────────────────┼─────────────────────────────────────────────┐
│ModelRunner(name, ...)                  │run(args)               │                                             │
│┌───────────────────────────────────────┼────────────────────────┼────────────────────────────────────────────┐│
││Executor(Graph, Weights, ...)          │                        │                                            ││
││                                       1                        5                                            ││
││ ┌───────────────────┐                 │                        │                                            ││
││ │ Execution Frames  │                 │                        └─────┐                                      ││
││ │                   │                 │                              │                                      ││
││ │ ┌──────────────┐  │                 │                              │   ┌─────────────────────────────────┐││
││ │ │   Frame_0    │◀─┼──get_frame()──2─┴────3───execute(args,frame)───┼──▶│frame.parse_inputs(args)         │││
││ │ ├──────────────┤  │                                                │   │                                 │││
││ │ │   Frame_1    │  │                                                │   │for kernel in op_kernels:        │││
││ │ ├──────────────┤  │                                                │   │    kernel.compute(frame)        │││
││ │ │   Frame_N    │◀─┼────return_frame()──────┐                       │   │                                 │││
││ │ └──────────────┘  │                        └─────────4─────────────┴───┤return frame.outputs()           │││
││ └───────────────────┘                                                    └─────────────────────────────────┘││
│└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
└───────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

To keep the memory footprint at bay, we have a garbage-collection mechanism to
clean up frames that haven't been used in a while since it's possible that a
traffic spike causes a bunch of frames to get created that will be left dormant
after the spike has subsided.

To set the maximum number of frames that NativeRT should create, you should use
the following configuration.

```cpp
RuntimeConfigs {
    .maxNumConcurrentThreads = n > 0,
};
```

## Features

### Delegation

Compiled graphs/subgraphs are modeled as delegates in PT2 inference.

For AOTInductor, it tends to compile the whole graph. In this case, the
resulting lowered graph contains a single call_delegate node. We commonly call
this full graph delegation.

For other delegates, it would compile some selected regions of the graph, and
leave the remaining uncompiled regions running on CPU. In the resulting lowered
graph, each compiled subgraph region will be fused as a call_delegate node. We
refer to this as partial graph delegation.

NativeRT supports both full graph and partial graph delegation. As the name
suggests, the runtime would execute the "call_delegate" node by delegating the
computation to its compiled binary.

### Static Dispatch Kernels

For CPU kernels, it is extremely inefficient to go through the dispatcher. For
one, the dispatcher doesn't deal with kernel out-variants.

> **_NOTE:_** an out-variant of a kernel is one that takes the outputs as
> mutable references. this has a few benefits... namely, it allows us to reuse
> the storage/manage from the previous execution.

In addition, the dispatcher acts as a stack machine. You push the inputs to the
stack, run the op on the specified device, and then pop the outputs. This in
itself is much more inefficient then accessing the values directly without
having to play musical chairs.

Registering statically-dispatched cpu kernels is pretty easy. Here is an example
of how we would override the default relu kernel.

```cpp
#include <nativert/core/kernels/KernelRegistry.h>

REGISTER_CPU_KERNEL("torch.ops.aten.relu.default", aten_relu, {
  const auto& in0_t = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = create_empty_from(in0_t);
  }
  auto& out_t = KernelOutput(0).toTensor();
  at::cpu::threshold_out(out_t, in0_t, 0, 0);
});
```

Static dispatch kernel registration can be enabled using the following
configurations.

```cpp
RuntimeConfigs {
    .enableStaticCPUKernels = true,
};
```

### Constant Folding

Constant folding is the process of finding all of the constant-evaluable
subgraphs, evaluating them at startup, and then storing their results as
constants as opposed to re-evaluating them every time.

To enable constant folding, you can set the following configurations.

```cpp
RuntimeConfigs {
    .enableRuntimeConstFolding = true,
};
```

### Quantization

For performance-sensitive models, we add the option to swap

```
torch.ops.aten.linear.default
```

with

```
torch.ops.quantized.linear_prepack_fp16.default
+
torch.ops.quantized.linear_dynamic_fp16.default
```

which should give a ~2x speedup over the fp32 variant with minimal effect on
correctness.

The linear_prepack_fp16 op will be constant-folded, so it's imperative that
these two features are used together.

To enable this feature, use the following configurations.

```cpp
RuntimeConfigs {
    .enableQuantizationPasses = true,
    .enableRuntimeConstFolding = true,
};
```

### Memory Planning

> :warning: **This is an experimental feature**

The main upside of memory planning comes from the efficient reuse of tensor
buffers, which is extremely important in memory-bound services. That is, if two
tensors don’t have an overlapping lifetime during execution, and the first
tensor is larger than the second, then the second tensor can share the same
chunk of memory as the first. As such, the main goal of our planning mechanism
is to pack tensors efficiently in memory with minimal impact on E2E execution
latency.

That said, there is a caveat -- the planning is best-effort for
dynamically-sized tensors. Because we slab-allocate a buffer for all tensors
before the graph is executed, and we cannot infer the exact size of some tensors
(namely those with data-dependent shapes), we opt to plan based on their
historical maximums.

Keeping in mind the [threading model](#threading-model), we make the following
additions to enable memory planning.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                                                                                                 │
│                                                                                      ┌──────────────────── thread_n                                             │
│                                                                                      │                        ▲                                                 │
│                                            ┌─────────────────────────────────────────┼────────────────────────┼───────────────────────────────────────────────┐ │
│    frame-local layout manager ─────────┐   │ModelRunner(name, ...)                   │run(args)               │                                               │ │
│                                        │   │ ┌───────────────────────────────────────┼────────────────────────┼─────────────────────────────────────────────┐ │ │
│    responsible for:                    │   │ │Executor(Graph, Weights, ...)          │                        │                                             │ │ │
│                                        │   │ │                                       1                        6                                             │ │ │
│    1. managing a buffer created        │   │ │ ┌───────────────────┐                 │                        │                                             │ │ │
│    from the most-recent layout         │   │ │ │ Execution Frames  │                 │                        └─────┐                                       │ │ │
│    plan                                │   │ │ │                   │                 │                              │                                       │ │ │
│                                        │   │ │ │ ┌──────────────┐  │                 │                              │   ┌─────────────────────────────────┐ │ │ │
│    2. ensuring tensor storages         └───┼─┼─┼─▶   Frame_0    │◀─┼──get_frame()──2─┴────3───execute(args,frame)───┼──▶│frame.layout_manager.allocate()  │ │ │ │
│    are mapped to the correct,              │ │ │ ├──────────────┤  │                                                │   │...                              │ │ │ │
│    owned buffer offsets                    │ │ │ │   Frame_1    │  │                                                │   │executor.execute(args, frame)    │ │ │ │
│                                            │ │ │ ├──────────────┤  │                                                │   │...                              │ │ │ │
│    3. telling the layout planner           │ │ │ │   Frame_N    │◀─┼────return_frame()────────────5─────────────────┴───│frame.layout_manager.deallocate()│ │ │ │
│    when a tensor have outgrown             │ │ │ └──────────────┘  │                                   ┌───────4────────┴─────────────────────────────────┘ │ │ │
│                                            │ │ └───────────────────┘                                   │                                                    │ │ │
│                                            │ │ ┌───────────────────┐                                   │                                                    │ │ │
│                                            │ │ │                   │                                   │                                                    │ │ │
│                                       ┌────┼─┼─▶  Layout Planner   │◀─────update_max_tensor_sizes()────┘                                                    │ │ │
│                                       │    │ │ │                   │                                                                                        │ │ │
│                                       │    │ │ └─────────▲─────────┘                                                                                        │ │ │
│                                       │    │ └───────────┼──────────────────────────────────────────────────────────────────────────────────────────────────┘ │ │
│                                       │    └─────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                       │                  │                                                                                                      │
│                                       │                  └──┐                                                                                                   │
│                                       │                     │                                                                                                   │
│                                       │                     │                                                                                                   │
│      cross-frame asynchronous ────────┘                     │                                                                                                   │
│      best-effort layout planner                             │                                                                                                   │
│                                                             │                                                                                                   │
│      responsible for:                                                                                                                                           │
│                                               plan is updated on interval.                                                                                      │
│      1. aggregating historical maximum      most up-to-date plan accessible                                                                                     │
│      tensor sizes across associated                 from each frame.                                                                                            │
│      layout managers                                                                                                                                            │
│                                                                                                                                                                 │
│      2. re-planning based on these                                                                                                                              │
│      historical maximums on some                                                                                                                                │
│      predefined interval                                                                                                                                        │
│                                                                                                                                                                 │
│      3. giving associated layout managers                                                                                                                       │
│      access to the most up-to-date plan                                                                                                                         │
│                                                                                                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Inter-op parallelism

> :warning: **This is an experimental feature**

Inter-op parallelism allows us to execute graph nodes concurrently as long as
all node inputs have been realized. This is an experimental feature, and its
effectiveness is highly dependent on the graph shape, the ops being executed,
and the model traffic patterns. This functionality does not currently work with
memory planning enabled, as the planner makes assumptions about lifetimes that
do not hold when node execution order is undefined.

To enable this feature, use the following configurations.

```cpp
RuntimeConfigs {
    .maxParallelOps = n > 1,
};
```
