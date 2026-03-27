(torch.compiler_inductor_scheduler)=

# Scheduler and Fusion

After [graph lowering](torch.compiler_inductor_ir.md) converts ATen IR into
Inductor IR, the scheduler takes over. The scheduler is TorchInductor's most
important optimization stage — it analyzes dependencies, fuses operations to
minimize memory traffic, and reorders computations for optimal performance.

**Source**: [torch/_inductor/scheduler.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/scheduler.py)

## Overview

The scheduler performs the following steps:

1. Converts Inductor IR into `SchedulerNode` and `SchedulerBuffer` objects.
2. Analyzes dependencies and mutations between nodes.
3. Performs fusion to combine multiple operations into single kernels.
4. Reorders operations to minimize peak memory usage.

The primary goal is to **minimize global memory reads and writes**. By fusing
operations that share data dependencies, the scheduler eliminates intermediate
memory allocations and transfers — the same optimization shown in the
{ref}`starter example <starter-example>` where
`relu` and `add` are fused into a single Triton kernel.

## SchedulerNode and SchedulerBuffer

The scheduler wraps Inductor IR objects in its own node types:

- **[BaseSchedulerNode](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/scheduler.py)**:
  The base class for all scheduler nodes. It wraps an Inductor IR operation and
  carries dependency information.

- **SchedulerNode**: Wraps a `ComputedBuffer` or other IR operation. Contains
  the `read_writes` attribute which tracks which buffers the node reads from
  and writes to.

- **SchedulerBuffer**: Wraps an `ir.Buffer` and tracks metadata about the
  buffer's usage in the schedule.

Each `SchedulerNode` contains:

- **node**: The underlying Inductor IR (for example, a `ComputedBuffer`)
- **read_writes**: A `ReadWrites` object recording memory dependencies — which
  buffers are read and which are written, along with indexing information.

## Dependency Analysis

Before making fusion decisions, the scheduler analyzes the dependency graph to
understand:

- **Read-after-write (RAW) dependencies**: A node that reads a buffer must be
  scheduled after the node that writes it.
- **Write-after-read (WAR) dependencies**: A node that overwrites a buffer must
  be scheduled after all nodes that read the previous value.
- **Mutations**: Nodes that mutate buffers in-place create ordering constraints.

This analysis ensures that fusion never changes the semantics of the program.

## Fusion

Fusion is the core optimization performed by the scheduler. It combines multiple
`SchedulerNode` objects into a single fused node that generates one kernel,
eliminating intermediate memory traffic.

### Fusion Strategy

The scheduler makes fusion decisions according to a **score**, calculated based
on:

1. **Type of fusion**: Different fusion types (pointwise-pointwise,
   reduction-pointwise, etc.) have different baseline scores.
2. **Estimated memory savings**: The scheduler estimates how many memory
   operations would be saved by fusing two nodes. Larger savings produce higher
   scores.
3. **Proximity in the graph**: Operations that are close together in the
   dependency graph are preferred fusion candidates.

### Fusion Iteration

The scheduler runs fusion in an iterative loop:

- Up to **10 iterations** of fusion attempts are performed.
- In each iteration, the scheduler examines pairs of nodes and fuses those with
  the highest scores.
- The loop **exits early** if no more fusions can occur in a given iteration.
- After fusion, the dependency graph is updated to reflect the new fused nodes.

### Types of Fusion

The scheduler can fuse various combinations of operations:

- **Pointwise + Pointwise**: Combining element-wise operations (for example,
  `relu` followed by `add`).
- **Pointwise + Reduction**: Fusing an element-wise operation with a subsequent
  reduction.
- **Horizontal fusion**: Combining independent operations that operate on the
  same data to improve data locality.

The `max_autotune` configuration enables additional fusion strategies, including
template-based fusions for GEMM prologue/epilogue operations.

## Memory Optimization

Beyond fusion, the scheduler also performs **reordering** to minimize peak
memory usage. By scheduling operations in an order that frees intermediate
buffers as early as possible, the scheduler reduces the maximum amount of memory
required during execution.

This is particularly important for large models where memory pressure can be
a bottleneck.
