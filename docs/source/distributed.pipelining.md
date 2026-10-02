```{eval-rst}
.. role:: hidden
    :class: hidden-section
```

# Pipeline Parallelism

:::{note}
`torch.distributed.pipelining` is currently in alpha state and under
development. API changes may be possible. It was migrated from the [PiPPy](https://github.com/pytorch/PiPPy) project.
:::

## Why Pipeline Parallel?

Pipeline Parallelism is one of the primitive forms of parallelism for deep learning.
It allows the **execution** of a model to be partitioned such that multiple
**micro-batches** can execute different parts of the model code concurrently.
Pipeline parallelism can be an effective technique for:

- large-scale training
- bandwidth-limited clusters
- large model inference

The above scenarios share a commonality that the computation per device cannot
hide the communication of conventional parallelism, for example, the weight
all-gather of FSDP.

## What is `torch.distributed.pipelining`?

While promising for scaling, pipelining is often difficult to implement because
it needs to **partition the execution** of a model in addition to model weights.
The partitioning of execution often requires intrusive code changes to your
model. Another aspect of complexity comes from **scheduling micro-batches in a
distributed environment**, with **data flow dependency** considered.

The `pipelining` package provides a toolkit that does said things
**automatically** which allows easy implementation of pipeline parallelism
on **general** models.

It consists of two parts: a
**splitting frontend** and a **distributed runtime**.
The splitting frontend takes your model code as-is, splits it up into "model
partitions", and captures the data-flow relationship. The distributed runtime
executes the pipeline stages on different devices in parallel, handling things
like micro-batch splitting, scheduling, communication, and gradient propagation,
etc.

Overall, the `pipelining` package provides the following features:

- Splitting of model code based on simple specification.
- Rich support for pipeline schedules, including GPipe, 1F1B,
  Interleaved 1F1B and Looped BFS, and providing the infrastructure for writing
  customized schedules.
- First-class support for cross-host pipeline parallelism, as this is where PP
  is typically used (over slower interconnects).
- Composability with other PyTorch parallel techniques such as data parallel
  (DDP, FSDP) or tensor parallel. The [TorchTitan](https://github.com/pytorch/torchtitan) project demonstrates a "3D parallel"
  application on the Llama model.

## Step 1: build `PipelineStage`

Before we can use a `PipelineSchedule`, we need to create `PipelineStage`
objects that wrap the part of the model running in that stage. The
`PipelineStage` is responsible for allocating communication buffers and
creating send/recv ops to communicate with its peers. It manages intermediate
buffers e.g. for the outputs of forward that have not been consumed yet, and it
provides a utility for running the backwards for the stage model.

A `PipelineStage` needs to know the input and output shapes for the stage
model, so that it can correctly allocate communication buffers. The shapes must
be static, e.g. at runtime the shapes can not change from step to step. A class
`PipeliningShapeError` will be raised if runtime shapes do not match the
expected shapes. When composing with other parallelisms or applying mixed
precision, these techniques must be taken into account so the `PipelineStage`
knows the correct shape (and dtype) for the output of the stage module at
runtime.

Users may construct a `PipelineStage` instance directly, by passing in an
`nn.Module` representing the portion of the model that should run on the
stage. This may require changes to the original model code. See the example
in {ref}`option_1_manual`.

Alternatively, the splitting frontend can use graph partitioning to split your
model into a series of `nn.Module` automatically. This technique requires the
model is traceable with `torch.Export`. Composability of the resulting
`nn.Module` with other parallelism techniques is experimental, and may require
some workarounds. Usage of this frontend may be more appealing if the user
cannot easily change the model code. See {ref}`option_2_tracer` for more
information.

## Step 2: use `PipelineSchedule` for execution

We can now attach the `PipelineStage` to a pipeline schedule, and run the
schedule with input data. Here is a GPipe example:

```python
from torch.distributed.pipelining import ScheduleGPipe

# Create a schedule
schedule = ScheduleGPipe(stage, n_microbatches)

# Input data (whole batch)
x = torch.randn(batch_size, in_dim, device=device)

# Run the pipeline with input `x`
# `x` will be divided into microbatches automatically
if rank == 0:
    schedule.step(x)
else:
    output = schedule.step()
```

If your input pipeline already produces microbatches, pass them directly:

```python
arg_mbs = [(x0,), (x1,)]
kwarg_mbs = [{"mask": mask0}, {"mask": mask1}]
target_mbs = [target0, target1]

schedule.step(
    arg_mbs=arg_mbs,
    kwarg_mbs=kwarg_mbs,
    target_mbs=target_mbs,
)
```

Note that the above code needs to be launched for each worker, thus we use a
launcher service to launch multiple processes:

```bash
torchrun --nproc_per_node=2 example.py
```

## Options for Splitting a Model

(option_1_manual)=

### Option 1: splitting a model manually

To directly construct a `PipelineStage`, the user is responsible for providing
a single `nn.Module` instance that owns the relevant `nn.Parameters` and
`nn.Buffers`, and defines a `forward()` method that executes the operations
relevant for that stage. For example, a condensed version of the Transformer
class defined in Torchtitan shows a pattern of building an easily partitionable
model.

```python
class Transformer(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()

        self.tok_embeddings = nn.Embedding(...)

        # Using a ModuleDict lets us delete layers without affecting names,
        # ensuring checkpoints will correctly save and load.
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(...)

        self.output = nn.Linear(...)

    def forward(self, tokens: torch.Tensor):
        # Handling layers being 'None' at runtime enables easy pipeline splitting
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        for layer in self.layers.values():
            h = layer(h, self.freqs_cis)

        h = self.norm(h) if self.norm else h
        output = self.output(h).float() if self.output else h
        return output
```

A model defined in this manner can be easily configured per stage by first
initializing the whole model (using meta-device to avoid OOM errors), deleting
undesired layers for that stage, and then creating a PipelineStage that wraps
the model. For example:

```python
with torch.device("meta"):
    assert num_stages == 2, "This is a simple 2-stage example"

    # we construct the entire model, then delete the parts we do not need for this stage
    # in practice, this can be done using a helper function that automatically divides up layers across stages.
    model = Transformer()

    if stage_index == 0:
        # prepare the first stage model
        del model.layers["1"]
        model.norm = None
        model.output = None

    elif stage_index == 1:
        # prepare the second stage model
        model.tok_embeddings = None
        del model.layers["0"]

    from torch.distributed.pipelining import PipelineStage
    stage = PipelineStage(
        model,
        stage_index,
        num_stages,
        device,
    )
```

When composing with other Data or Model parallelism techniques, `output_args`
may also be required, if the output shape/dtype of the model chunk will be
affected.

(option_2_tracer)=

### Option 2: splitting a model automatically

If you have a full model and do not want to spend time on modifying it into a
sequence of "model partitions", the `pipeline` API is here to help.
Here is a brief example:

```python
class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(10, 3)
        self.layers = torch.nn.ModuleList(
            Layer() for _ in range(2)
        )
        self.lm = LMHead()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x)
        x = self.lm(x)
        return x
```

If we print the model, we can see multiple hierarchies, which makes it hard to split by hand:

```python
Model(
  (emb): Embedding(10, 3)
  (layers): ModuleList(
    (0-1): 2 x Layer(
      (lin): Linear(in_features=3, out_features=3, bias=True)
    )
  )
  (lm): LMHead(
    (proj): Linear(in_features=3, out_features=3, bias=True)
  )
)
```

Let us see how the `pipeline` API works:

```python
from torch.distributed.pipelining import pipeline, SplitPoint

# An example micro-batch input
x = torch.LongTensor([1, 2, 4, 5])

pipe = pipeline(
    module=mod,
    mb_args=(x,),
    split_spec={
        "layers.1": SplitPoint.BEGINNING,
    }
)
```

The `pipeline` API splits your model given a `split_spec`, where
`SplitPoint.BEGINNING` stands for adding a split point
*before* execution of certain submodule in the `forward` function, and
similarly, `SplitPoint.END` for split point *after* such.

If we `print(pipe)`, we can see:

```python
GraphModule(
  (submod_0): GraphModule(
    (emb): InterpreterModule()
    (layers): Module(
      (0): InterpreterModule(
        (lin): InterpreterModule()
      )
    )
  )
  (submod_1): GraphModule(
    (layers): Module(
      (1): InterpreterModule(
        (lin): InterpreterModule()
      )
    )
    (lm): InterpreterModule(
      (proj): InterpreterModule()
    )
  )
)

def forward(self, x):
    submod_0 = self.submod_0(x);  x = None
    submod_1 = self.submod_1(submod_0);  submod_0 = None
    return (submod_1,)
```

The "model partitions" are represented by submodules (`submod_0`,
`submod_1`), each of which is reconstructed with original model operations, weights
and hierarchies. In addition, a "root-level" `forward` function is
reconstructed to capture the data flow between those partitions. Such data flow
will be replayed by the pipeline runtime later, in a distributed fashion.

The `Pipe` object provides a method for retrieving the "model partitions":

```python
stage_mod : nn.Module = pipe.get_stage_module(stage_idx)
```

The returned `stage_mod` is a `nn.Module`, with which you can create an
optimizer, save or load checkpoints, or apply other parallelisms.

`Pipe` also allows you to create a distributed stage runtime on a device given
a `ProcessGroup`:

```python
stage = pipe.build_stage(stage_idx, device, group)
```

Alternatively, if you would like to build the stage runtime later after some
modification to the `stage_mod`, you can use a functional version of the
`build_stage` API. For example:

```python
from torch.distributed.pipelining import build_stage
from torch.nn.parallel import DistributedDataParallel

dp_mod = DistributedDataParallel(stage_mod)
info = pipe.info()
stage = build_stage(dp_mod, stage_idx, info, device, group)
```

:::{note}
The `pipeline` frontend uses a tracer (`torch.export`) to capture your
model into a single graph. If your model is not full-graph'able, you can use
our manual frontend below.
:::

## Hugging Face Examples

In the [PiPPy](https://github.com/pytorch/PiPPy) repo where this package was
original created, we kept examples based on unmodified Hugging Face models.
See the [examples/huggingface](https://github.com/pytorch/PiPPy/tree/main/examples/huggingface) directory.

Examples include:

- [GPT2](https://github.com/pytorch/PiPPy/tree/main/examples/huggingface/pippy_gpt2.py)
- [Llama](https://github.com/pytorch/PiPPy/tree/main/examples/llama)

## Technical Deep Dive

### How does the `pipeline` API split a model?

First, the `pipeline` API turns our model into a directed acyclic graph (DAG)
by tracing the model. It traces the model using `torch.export` -- a PyTorch 2
full-graph capturing tool.

Then, it groups together the **operations and parameters** needed by a stage
into a reconstructed submodule: `submod_0`, `submod_1`, ...

Different from conventional submodule access methods like `Module.children()`,
the `pipeline` API does not only cut the module structure of your model, but
also the **forward** function of your model.

This is necessary because model structure like `Module.children()` merely
captures information during `Module.__init__()`, and does not capture any
information about `Module.forward()`. Said differently, `Module.children()`
lacks information about the following aspects key to pipelininig:

- Execution order of child modules in `forward`
- Activation flows between child modules
- Whether there are any functional operators between child modules (for example,
  `relu` or `add` operations will not be captured by `Module.children()`).

The `pipeline` API, on the contrary, makes sure that the `forward` behavior
is truly preserved. It also captures the activation flow between the partitions,
helping the distributed runtime to make correct send/receive calls without human
intervention.

Another flexibility of the `pipeline` API is that split points can be at
arbitrary levels within your model hierarchy. In the split partitions, the original model
hierarchy related to that partition will be reconstructed at no cost to you.
At a result, fully-qualified names (FQNs) pointing to a submodule or parameter
would be still valid, and services that relies on FQNs (such as FSDP, TP or
checkpointing) can still run with your partitioned modules with almost zero code
change.

## Implementing Your Own Schedule

You can implement your own pipeline schedule by extending one of the following two classes:

- `PipelineScheduleSingle`
- `PipelineScheduleMulti`

`PipelineScheduleSingle` is for schedules that assigns *only one* stage per rank.
`PipelineScheduleMulti` is for schedules that assigns multiple stages per rank.
All stages assigned to one rank must execute on one device; adjacent same-rank
stages pass activations and gradients directly without device copies.

For example, `ScheduleGPipe` and `Schedule1F1B` are subclasses of `PipelineScheduleSingle`.
Whereas, `ScheduleInterleaved1F1B`, `ScheduleLoopedBFS`, `ScheduleInterleavedZeroBubble`, and `ScheduleZBVZeroBubble`
are subclasses of `PipelineScheduleMulti`.

### Accessing Stage Forward Information

A runtime surrounding a pipeline stage may need to select resources by the
global logical stage and current microbatch. Physical pipeline rank is not a
substitute for logical stage identity because one rank may own several stages
in an interleaved schedule.

Register a context factory on the stage without changing the wrapped module's
forward signature:

```python
from contextlib import contextmanager

from torch.distributed.pipelining import PipelineStageInfo


@contextmanager
def stage_forward_context(info: PipelineStageInfo):
    planner.enter(
        stage_index=info.stage_index,
        microbatch_index=info.microbatch_index,
        is_metadata_inference=info.is_metadata_inference,
    )
    try:
        yield
    finally:
        planner.exit()


handle = stage.register_forward_context(stage_forward_context)
```

The factory is called around each built-in stage forward. Dynamic metadata
inference uses microbatch zero and sets `is_metadata_inference=True`, allowing a
consumer to distinguish the representative probe from real microbatch-zero
execution. Static metadata setup does not execute the module and therefore does
not enter the context.

Only one context can be registered on a stage at a time. Remove its handle
before registering another context. Register before the first schedule step if
the consumer must observe dynamic metadata inference.

The context executes outside a compiled or exported stage module, so it does
not add graph inputs or change the module signature. CUDA graph capture runs
the Python context while recording the stage computation, but replay does not
re-enter Python; consumers must bind replay-stable state during capture. A
custom schedule action receives this context only when it delegates execution
to `stage.forward_one_chunk()`.

```{eval-rst}
.. autoclass:: torch.distributed.pipelining.PipelineStageInfo
  :members:

.. automethod:: torch.distributed.pipelining.PipelineStage.register_forward_context
```

### Controlling FSDP Unshard Lookahead

Multi-stage runtime schedules lower a compute-only schedule into explicit FSDP
`UNSHARD` and `RESHARD` actions. Two parameters control different parts of
that lowering:

- `max_active_stages` is the target parameter-residency window. It determines
  which stages remain unsharded and where `RESHARD` actions are inserted.
- `unshard_lookahead` is the issue-distance window. It determines how many
  upcoming distinct logical stages may begin unsharding.

Separating these windows allows a schedule to issue fewer all-gathers early
without evicting parameters sooner or adding another unshard/reshard cycle.
An asynchronous unshard is still real GPU work:

```text
pre-all-gather cast or quantization
  -> copy-in and packing
  -> all-gather
  -> copy-out
  -> post-all-gather quantization or layout preparation
  -> parameter ready
```

`async_op=True` avoids a host-side wait, but these kernels, copies, collective
traffic, allocations, and stream dependencies can still contend with the
forward. Issuing the entire residency window at once can therefore put
non-critical parameter preparation ahead of useful compute.

The `"auto"` policy estimates how much of this work fits into each rank's
pipeline startup bubble. Let a balanced stage forward take `F`, and let the
composite critical-path cost of preparing one stage's unsharded parameters be
`U`. Ignoring pipeline transfer latency, rank `r` waits approximately
`U + rF` before its first useful forward. This can complete approximately
`floor((U + rF) / U)` unshards; issuing one more allows the next unshard to
overlap that first forward. With the simplifying assumption `F = U = T`, the
lookahead is `r + 2`, capped by `max_active_stages`.

For PP4 with `max_active_stages=4`, `"auto"` resolves to `(2, 3, 4, 4)`:

```text
interval           | 0..T    | T..2T   | 2T..3T  | 3T..4T  | 4T..5T
-------------------+---------+----------+----------+----------+---------
rank 0 forward     | blocked | F(first) |          |          |
rank 0 preparation | U0      | U1       |          |          |  => 2
-------------------+---------+----------+----------+----------+---------
rank 1 forward     | blocked | blocked  | F(first) |          |
rank 1 preparation | U0      | U1       | U2       |          |  => 3
-------------------+---------+----------+----------+----------+---------
rank 2 forward     | blocked | blocked  | blocked  | F(first) |
rank 2 preparation | U0      | U1       | U2       | U3       |  => 4
-------------------+---------+----------+----------+----------+---------
rank 3 forward     | blocked | blocked  | blocked  | blocked  | F(first)
rank 3 preparation | U0      | U1       | U2       | U3       |  => 4
```

The diagram is an analytical starting point, not an exact CUDA-stream model.
`Uk` denotes preparation of the kth upcoming rank-local stage, not a global
stage index. Vertically aligned preparation and forward cells are intended to
overlap.
Real stages may be unbalanced, unshard phases may overlap only partially, and
network or memory-bandwidth contention may change the best distance. Choose a
policy accordingly:

| Policy | Per-rank issue distance | Intended use |
| --- | --- | --- |
| `"full"` | `max_active_stages` | Compatibility default matching the original full-window behavior. |
| `"auto"` | `min(pp_rank + 2, max_active_stages)` | Deterministic startup-bubble estimate that avoids recipe-level tuning; not a universal optimum. |
| Tuple | The corresponding positive integer for each PP rank | Expert tuning for measured model, topology, and fabric behavior. |

A custom compute-only schedule may be combined with a tuple before lowering.
For exact `UNSHARD` and communication placement, supply an already lowered
`compute_comms` schedule; `unshard_lookahead` cannot retune actions that are
already present.

An atomic compound action remains indivisible, so it may extend either window
by up to the action's number of stages minus one. Non-full policies can also
move absolute P2P action positions because unshards consume lowering rounds.
They do not change compute order, `RESHARD` placement, residency episodes, or
collective counts. Applications should benchmark an explicit tuple when stage
costs differ materially from the balanced model.

## Logging

You can turn on additional logging using the `TORCH_LOGS` environment variable from [torch.\_logging](https://pytorch.org/docs/main/logging.html#module-torch._logging):

- `TORCH_LOGS=+pp` will display `logging.DEBUG` messages and all levels above it.
- `TORCH_LOGS=pp` will display `logging.INFO` messages and above.
- `TORCH_LOGS=-pp` will display `logging.WARNING` messages and above.

## API Reference

```{eval-rst}
.. automodule:: torch.distributed.pipelining
```

### Model Split APIs

The following set of APIs transform your model into a pipeline representation.

```{eval-rst}
.. currentmodule:: torch.distributed.pipelining
```

```{eval-rst}
.. autoclass:: SplitPoint
```

```{eval-rst}
.. autofunction:: pipeline
```

```{eval-rst}
.. autoclass:: Pipe
```

```{eval-rst}
.. autofunction:: pipe_split
```

### Microbatch Utilities

```{eval-rst}
.. automodule:: torch.distributed.pipelining.microbatch
```

```{eval-rst}
.. currentmodule:: torch.distributed.pipelining.microbatch
```

```{eval-rst}
.. autoclass:: TensorChunkSpec
```

```{eval-rst}
.. autofunction:: split_args_kwargs_into_chunks
```

```{eval-rst}
.. autofunction:: merge_chunks
```

### Pipeline Stages

```{eval-rst}
.. automodule:: torch.distributed.pipelining.stage
```

```{eval-rst}
.. currentmodule:: torch.distributed.pipelining.stage
```

```{eval-rst}
.. autoclass:: PipelineStage
```

```{eval-rst}
.. autofunction:: build_stage
```

### Pipeline Schedules

#### Activation-liveness analysis

Pipeline runtimes and memory planners can use
`analyze_pipeline_activation_liveness` to determine how many reusable logical
slots are needed for activations retained from forward through backward. The
analysis does not allocate tensors. It returns a
`PipelineActivationLiveness` plan whose
`slot_by_stage_and_microbatch[(stage_index, microbatch_index)]` values are slot
IDs that a caller may map to buffers or arena regions.

An activation becomes live at its forward (`F`) action. Full backward (`B`)
releases it. For schedules that separate input backward (`I`) from weight
backward (`W`), `I` does not release the activation because `W` may still need
the saved forward state; `W` releases it. Lifetimes include both endpoint
positions, so actions grouped into the same compound schedule position overlap.

For example, consider two stages and two microbatches on one pipeline rank:

```text
position:  0     1     2     3     4     5     6     7
action:   F0,0  F1,0  F0,1  B1,0  B0,0  F1,1  B1,1  B0,1
```

With `granularity="stage_microbatch"`, the four activation lifetimes are
`(0, 4)`, `(1, 3)`, `(2, 7)`, and `(5, 6)`; the final lifetime may reuse the
first slot. With `granularity="microbatch"`, the selected stages for each
microbatch share one conservative lifetime: `(0, 4)` for microbatch 0 and
`(2, 7)` for microbatch 1. The latter mode is useful when a consumer manages
all selected stages for one microbatch as one storage unit. Stage indices are
global logical indices and commonly identify virtual stages hosted by the same
pipeline rank.

```{eval-rst}
.. currentmodule:: torch.distributed.pipelining
```

```{eval-rst}
.. autofunction:: torch.distributed.pipelining.schedules.analyze_pipeline_activation_liveness
```

```{eval-rst}
.. autoclass:: torch.distributed.pipelining.schedules.PipelineActivationLiveness
  :members:
```

```{eval-rst}
.. automodule:: torch.distributed.pipelining.schedules
```

```{eval-rst}
.. currentmodule:: torch.distributed.pipelining.schedules
```

```{eval-rst}
.. autoclass:: ScheduleGPipe
```

```{eval-rst}
.. autoclass:: Schedule1F1B
```

```{eval-rst}
.. autoclass:: ScheduleInterleaved1F1B
```

```{eval-rst}
.. autoclass:: ScheduleLoopedBFS
```

```{eval-rst}
.. autoclass:: ScheduleInterleavedZeroBubble
```

```{eval-rst}
.. autoclass:: ScheduleZBVZeroBubble
```

```{eval-rst}
.. autoclass:: ScheduleDualPipeV
```

```{eval-rst}
.. autoclass:: PipelineScheduleSingle
  :members:
```

```{eval-rst}
.. autoclass:: PipelineScheduleMulti
  :members:
```

```{eval-rst}
.. autofunction:: get_schedule_class
```
