(torch.compiler_inductor_scheduler)=

# Scheduler and Fusion

**Source**: [torch/_inductor/scheduler.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/scheduler.py)

## Glossary

### Inductor IR

The Inductor IR is the define-by-run IR defined in
[ir.py](https://github.com/pytorch/pytorch/blob/d27d36136ce35d5d6dc3faa818ba840ba61d4357/torch/_inductor/ir.py).
This IR is generated during GraphLowering, described in
[Graph Lowering](torch.compiler_inductor_ir.md).

- **TemplateBuffer**: A scheduler node which will use a template to perform code
  generation for its kernel. Examples of these are Triton templates and CUDA
  templates.
- **RealizedBuffer**: A buffer which will be written (materialized) into global
  memory and be read separately by each of its users. There is an inherent
  tradeoff here where if a buffer has several disparate users, it may be
  beneficial to materialize the buffer into global memory to avoid redundant
  recomputations. Otherwise, the code used to compute the buffer is inlined into
  its users and recomputed at all of its uses.
- **Operation**: A computation which populates one or more buffers.
- **Buffer**: An abstraction representing data which will be read or written.
- **ComputedBuffer**: A buffer which is computed from a looped computation,
  either a reduction or a pointwise op.

### Scheduler IR

The scheduler is defined in
[scheduler.py](https://github.com/pytorch/pytorch/blob/d27d36136ce35d5d6dc3faa818ba840ba61d4357/torch/_inductor/scheduler.py).

- **SchedulerBuffer**: A Scheduler-specific wrapper of an `ir.Buffer`. It tracks
  additional metadata to make Scheduling easier.
- **SchedulerNode**: A Node in the Scheduler IR representing a kernel which is
  not fused.
- **FusedSchedulerNode**: A SchedulerNode representing a group of
  SchedulerNodes that have been fused together.
- **GroupedSchedulerNode**: A SchedulerNode representing a group of
  SchedulerNodes where fusion should only occur within the group, not outside
  it. This is used for comm reordering and finer-grained control of peak memory.
- **ForeachKernelSchedulerNode**: A more specific FusedSchedulerNode which
  encapsulates a Foreach op: a group of fused nodes which can be horizontally
  fused.
- **ExternKernelSchedulerNode**: A node representing a call to a
  non-codegenerated kernel.

## Background

The scheduling phase of compilation occurs after graph lowering, and determines
which **realized buffers** can be fused into single kernels to save memory
bandwidth and increase utilization of the GPU.

**Graph lowering** does its own fusion analysis, where it decides to inline or
realize a **buffer**. Graph lowering runs the FX graph output by AOTAutograd and
generates the Inductor IR from the FX graph while also tracking realized buffers
as a side effect of this process. After lowering, the Scheduler iterates over
all of the registered buffers and converts these into **SchedulerBuffers**, the
main abstraction for performing fusion analysis. This analysis is performed in
the
[create_scheduler_node](https://github.com/pytorch/pytorch/blob/9620994067b18e846a097d1e99af85ec2426ef0a/torch/_inductor/scheduler.py#L2198C9-L2198C30)
method. There are a few important fields on the SchedulerNode, in addition to
various use/def metadata with which you might be familiar.

```python
class SchedulerNode:
    _sizes: tuple[Sequence[sympy.Expr], ...]
    _body: LoopBody
    read_writes: dependencies.ReadWrites
```

`_sizes` correspond to the iteration variables TorchInductor will codegen. The
first sequence is the pointwise iteration sizes, and the second sequence is the
reduction iteration size.

`_body` is the persistent representation of TorchInductor IR on which we will
do loop-level modifications. For instance,
[applying a new loop order](https://github.com/pytorch/pytorch/blob/8cb6957e019086c1f6e39941047ead6e29f6f402/torch/_inductor/scheduler.py#L1108-L1113).
This contains the ops IR which compose the SchedulerNode (e.g. any ops inlined
during lowering).

## Fusion

At a high level, the main objective of the scheduler is to reduce the number of
global memory read/writes. This goal is achieved through fusion. The fusion
process consists of grouping possible **SchedulerNodes** into groups which may
or may not be fused, and iterating over these groups to evaluate if fusion can
occur between constituent nodes. If fusion is permitted (described below), these
nodes are combined into a **FusedSchedulerNode**, an abstraction with the same
interface as a SchedulerNode, but internally consists of a set of child
SchedulerNodes. This process continues for **10 iterations** or until no more
fusions can occur (whichever comes first).

### Evaluating Fusions

The main function of interest for evaluating fusions is the
[can_fuse](https://github.com/pytorch/pytorch/blob/9620994067b18e846a097d1e99af85ec2426ef0a/torch/_inductor/scheduler.py#L3646C1-L3647C1)
method of the scheduler. This method is broken into two phases: First
backend-agnostic checks are run, then each scheduling backend makes
device-dependent fusion decisions. This method first checks heuristics on the
type of nodes that we are fusing. For example, template nodes (nodes which use
a template to generate the output code) have specific heuristics for when fusion
is allowed. For brevity, we will omit these heuristics here and the code will
be the best reference. The next set of heuristics that are checked are
encapsulated in
[InductorChoices](https://github.com/pytorch/pytorch/blob/d27d36136ce35d5d6dc3faa818ba840ba61d4357/torch/_inductor/choices.py#L43)
which performs fusion checks in its own
[can_fuse](https://github.com/pytorch/pytorch/blob/d27d36136ce35d5d6dc3faa818ba840ba61d4357/torch/_inductor/choices.py#L313)
implementation.

### Dependency Tracking

During fusion, it is critical to track the dependencies between the scheduler
nodes to make sure that fusions are not just profitable, but also legal. Each
scheduler node tracks information about the reads and writes of the computation
it contains (see
[dependencies.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/dependencies.py)).
Specifically, it tracks three kinds of dependencies, found in
`scheduler_node.read_writes`:

1. **MemoryDeps**, which represent reading or writing the result of a
   preceding/succeeding computation, and come associated with the indexing that
   is used to access them.
2. **StarDeps**, which also represent reading or writing of a tensor, but
   without making indexing assumptions.
3. **WeakDeps**, which are used for ordering of mutation, and make sure that
   reads of a tensor occur before it is overwritten, or that successive writes
   are ordered correctly.

### FusedSchedulerNode

At a high level, if fusion is allowed, individual **SchedulerNodes** are fused
into **FusedSchedulerNodes**, which internally store an `OrderedSet` of nodes
which have been fused together in topologically sorted order. During codegen,
the code for each node is generated in this order. When considering future
fusions with a **FusedSchedulerNode**, the dependencies of the constituent
nodes are union'd together and considered as if the fused node were a single
node.
