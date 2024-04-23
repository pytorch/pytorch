# Pipeline Parallelism for PyTorch

> [!NOTE]
> `torch.distributed.pipelining` is a package migrated from the [PiPPy](https://github.com/pytorch/PiPPy) project. It is currently in alpha state and under extensive development. If you need examples that work with our APIs, please refer to PiPPy's [examples](https://github.com/pytorch/PiPPy/tree/main/examples) directory.

[**Why Pipeline Parallel?**](#why-pipeline-parallel)
| [**What is `torch.distributed.pipelining`?**](#what-is-torchdistributedpipelining)
| [**Examples**](#examples)
| [**Techniques Explained**](#techniques-explained)

# Why Pipeline Parallel?

One of the most important techniques for advancing the state of the art in deep learning is scaling. Common techniques for scaling neural networks include _data parallelism_, _tensor/operation parallelism_, and _pipeline parallelism_. In many cases, pipeline parallelism in particular can be an effective technique for scaling, however it is often difficult to implement, requiring intrusive code changes to model code and difficult-to-implement runtime orchestration code. `torch.distributed.pipelining` aims to provide a toolkit that does said things automatically to allow high-productivity scaling of models.

# What is `torch.distributed.pipelining`?

`torch.distributed.pipelining` consists of a compiler and runtime stack for automated pipelining of PyTorch models. Pipelining, or _pipeline parallelism_, is a technique in which the _code_ of the model is partitioned and multiple _micro-batches_ execute different parts of the model code concurrently. To learn more about pipeline parallelism, see [this article](https://www.deepspeed.ai/tutorials/pipeline/).

![pipeline_diagram_web](https://github.com/pytorch/PiPPy/assets/6676466/c93e2fe7-1cd4-49a2-9fd8-231ec9905e0c)

Figure: Pipeline parallel. "F", "B" and "U" denote forward, backward and weight update, respectively. Different colors represent different micro-batches.

`torch.distributed.pipelining` provides the following features that make pipeline parallelism easier:

* Automatic splitting of model code based on your specification. The goal is for the user to provide model code as-is to the system for parallelization, without having to make heavyweight modifications to make parallelism work. The specification is also simple.
* Support for rich pipeline scheduling paradigms, including GPipe, 1F1B, Interleaved 1F1B and Looped BFS. More schedules will be added and it will be easy to customize your own schedule under `torch.distributed.pipelining`'s framework.
* First-class support for cross-host pipeline parallelism, as this is where PP is typically used (over slower interconnects).
* Composability with other PyTorch parallel schemes such as data parallelism (DDP, FSDP) or tensor  parallelism (overall, known as "3d parallelism").

# Examples

In the [PiPPy](https://github.com/pytorch/PiPPy) repo where this package is migrated from, we provide rich examples based on realistic models. In particular, we show how to apply pipelining without any model code change. You can refer to the [HuggingFace examples directory](https://github.com/pytorch/PiPPy/tree/main/examples/huggingface). Popular examples include: [GPT2](https://github.com/pytorch/PiPPy/tree/main/examples/huggingface/pippy_gpt2.py), and [LLaMA](https://github.com/pytorch/PiPPy/tree/main/examples/llama).

# Techniques Explained

`torch.distributed.pipelining` consists of two parts: a _compiler_ and a _runtime_. The compiler takes your model code, splits it up, and transforms it into a `Pipe`, which is a wrapper that describes the model at each pipeline stage and their data-flow relationship. The runtime executes the `PipelineStage`s in parallel, handling things like micro-batch splitting, scheduling, communication, and gradient propagation, etc. We will cover the APIs for these concepts in this section.

## Splitting a Model with `pipeline`

To see how we can split a model into a pipeline, let's first take an example trivial neural network:

```python
import torch

class MyNetworkBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.lin(x)
        x = torch.relu(x)
        return x


class MyNetwork(torch.nn.Module):
    def __init__(self, in_dim, layer_dims):
        super().__init__()

        prev_dim = in_dim
        for i, dim in enumerate(layer_dims):
            setattr(self, f'layer{i}', MyNetworkBlock(prev_dim, dim))
            prev_dim = dim

        self.num_layers = len(layer_dims)
        # 10 output classes
        self.output_proj = torch.nn.Linear(layer_dims[-1], 10)

    def forward(self, x):
        for i in range(self.num_layers):
            x = getattr(self, f'layer{i}')(x)

        return self.output_proj(x)


in_dim = 512
layer_dims = [512, 1024, 256]
mn = MyNetwork(in_dim, layer_dims).to(device)
```

This network is written as free-form Python code; it has not been modified for any specific parallelism technique.

Let us see our first usage of the `torch.distributed.pipelining` interfaces:

```python
from torch.distributed.pipelining import annotate_split_points, pipeline, Pipe, SplitPoint

annotate_split_points(mn, {'layer0': SplitPoint.END,
                           'layer1': SplitPoint.END})

batch_size = 32
example_input = torch.randn(batch_size, in_dim, device=device)
chunks = 4

pipe = pipeline(mn, chunks, example_args=(example_input,))
print(pipe)

"""
************************************* pipe *************************************
GraphModule(
  (submod_0): GraphModule(
    (layer0): InterpreterModule(
      (lin): InterpreterModule()
    )
  )
  (submod_1): GraphModule(
    (layer1): InterpreterModule(
      (lin): InterpreterModule()
    )
  )
  (submod_2): GraphModule(
    (layer2): InterpreterModule(
      (lin): InterpreterModule()
    )
    (output_proj): InterpreterModule()
  )
)

def forward(self, arg8_1):
    submod_0 = self.submod_0(arg8_1);  arg8_1 = None
    submod_1 = self.submod_1(submod_0);  submod_0 = None
    submod_2 = self.submod_2(submod_1);  submod_1 = None
    return (submod_2,)
"""
```

So what's going on here? First, `pipeline` turns our model into a directed acyclic graph (DAG) by tracing the model. Then, it groups together the operations and parameters into _pipeline stages_. Stages are represented as `submod_N` submodules, where `N` is a natural number.

We used `annotate_split_points` to specify that the code should be split and the end of `layer0` and `layer1`. Our code has thus been split into _three_ pipeline stages. Our library also provides `SplitPoint.BEGINNING` if a user wants to split before certain annotation point.

While the `annotate_split_points` API gives users a way to specify the split points without modifying the model, our library also provides an API for in-model annotation: `pipe_split()`. For details, you can read [this example](https://github.com/pytorch/PiPPy/blob/main/test/test_pipe.py).

This covers the basic usage of the `Pipe` API. For more information, please see the documentation.

<!-- (TODO: link to docs when live) -->

## Using PipelineStage for Pipelined Execution

Given the above `Pipe` object, we can use one of the `PipelineStage` classes to execute our model in a pipelined fashion. First off, let us instantiate a `PipelineStage` instance:

```python
# We are using `torchrun` to run this example with multiple processes.
# `torchrun` defines two environment variables: `RANK` and `WORLD_SIZE`.
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

# Initialize distributed environment
import torch.distributed as dist
dist.init_process_group(rank=rank, world_size=world_size)

# Pipeline stage is our main pipeline runtime. It takes in the pipe object,
# the rank of this process, and the device.
from torch.distributed.pipelining import PipelineStage
stage = PipelineStage(pipe, rank, device)
```

We can now run the pipeline by attaching the `PipelineStage` to a pipeline schedule, GPipe for example:

```python
from torch.distributed.pipelining import ScheduleGPipe
schedule = ScheduleGPipe(stage, chunks)

# Input data
x = torch.randn(batch_size, in_dim, device=device)

# Run the pipeline with input `x`. Divide the batch into 4 micro-batches
# and run them in parallel on the pipeline
if rank == 0:
    schedule.step(x)
else:
    output = schedule.step()
```

Note that since we split our model into three stages, we must run this script with three workers. For this example, we will use `torchrun` to run multiple processes within a single machine for demonstration purposes. We can collect up all of the code blocks above into a file named [example.py](https://github.com/pytorch/PiPPy/tree/main/examples/basic) and then run it with `torchrun` like so:

```
torchrun --nproc_per_node=3 example.py
```
