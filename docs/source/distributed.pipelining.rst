.. role:: hidden
    :class: hidden-section

Pipeline Parallelism
####################

.. note:: ``torch.distributed.pipelining`` is a package migrated from the `PiPPy <https://github.com/pytorch/PiPPy>`_ project. It is currently in alpha state and under extensive development. For examples that work with our APIs, please refer to PiPPy's `examples <https://github.com/pytorch/PiPPy/tree/main/examples>`_ directory.

Why Pipeline Parallel?
**********************

One of the most important techniques for advancing the state of the art in deep learning is scaling. Common techniques for scaling neural networks include *data parallelism*, *tensor/operation parallelism*, and *pipeline parallelism* (or *pipelining*). Pipelining is a technique in which the *code* of the model is partitioned and multiple *micro-batches* execute different parts of the model code concurrently. In many cases, pipeline parallelism can be an effective technique for scaling, in particular for large-scale jobs or bandwidth-limited interconnects. To learn more about pipeline parallelism in deep learning, see `this article <https://www.deepspeed.ai/tutorials/pipeline/>`_.

What is ``torch.distributed.pipelining``?
*****************************************

.. automodule:: torch.distributed.pipelining

.. currentmodule:: torch.distributed.pipelining

While promising for scaling, pipelining is often difficult to implement, requiring intrusive code changes to model code and difficult-to-implement runtime orchestration code. ``torch.distributed.pipelining`` aims to provide **a toolkit that does said things automatically to allow high-productivity scaling of models.**
``torch.distributed.pipelining`` consists of two parts: a *compiler* and a *runtime*. The compiler takes your model code, splits it up, and transforms it into a ``Pipe``, which is a wrapper that describes the model at each pipeline stage and their data-flow relationship. The runtime executes the ``PipelineStage`` in parallel, handling things like micro-batch splitting, scheduling, communication, and gradient propagation, etc. We will cover the APIs for these concepts in this section.
Overall, the ``pipelining`` package provides the following features:

* Splitting of model code based on your specification. The goal is for the user to provide model code as-is to the system for parallelization, without having to make heavyweight modifications to make parallelism work. The specification is also simple.
* Support for rich pipeline scheduling paradigms, including GPipe, 1F1B, Interleaved 1F1B and Looped BFS. It will be also easy to customize your own schedule under this framework.
* First-class support for cross-host pipeline parallelism, as this is where PP is typically used (over slower interconnects).
* Composability with other PyTorch parallel schemes such as data parallelism (DDP, FSDP) or tensor  parallelism (overall, known as "3d parallelism").

Step 1: Choosing the Frontend that Fits Your Need
*************************************************

The ``pipelining`` package provides two frontends for two different scenarios. Depending on whether you have (i) a full model or (ii) module constructor for each stage, you can choose one of the frontends below.

Frontend 1: the ``pipeline`` API -- If You Have a Full Model
============================================================

If you don't want to modify model code into a sequence of "model partitions", the ``pipeline`` API is here to help. Here is a brief example:

.. code-block:: python

  import torch

  class Layer(torch.nn.Module):
      def __init__(self) -> None:
          super().__init__()
          self.lin = torch.nn.Linear(3, 3)

      def forward(self, x: torch.Tensor) -> torch.Tensor:
          return self.lin(x)

  class LMHead(torch.nn.Module):
      def __init__(self) -> None:
          super().__init__()
          self.proj = torch.nn.Linear(3, 3)

      def forward(self, x: torch.Tensor) -> torch.Tensor:
          return self.proj(x)

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

  mod = Model()
  print(mod)

::
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

This network is written as free-form Python code; it has not been modified for any specific parallelism technique.

Let us see how the ``pipeline`` works:

.. code-block:: python

  from torch.distributed.pipelining import pipeline, SplitPoint

  x = torch.LongTensor([1, 2, 4, 5])

  pipe = pipeline(
      module=mod,
      num_chunks=1,
      example_args=(x,),
      split_spec={
          "layers.1": SplitPoint.BEGINNING,
      }
  )
  print(pipe)

::
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


The ``pipeline`` API accepts a ``nn.Module``  -- your original full model -- and a set of split points per your specification.
It splits your model into multiple "model partitions" -- each of which to be executed at a stage, and put them in a container called `Pipe`.

Different from conventional submodule access methods like ``Module.children()``, the ``pipeline`` API cuts the ``forward`` function of your model, at the places where those split-point FQNs are called.

This provides a few safety guarantees and flexibility:

* All operations of your ``forward`` function are preserved.
* Split points can be at arbitrary hierarchy of your model.

Each model partition is a reconstructed ``nn.Module``, and the ``Pipe`` container provides a ``get_stage_module`` method for you to retrieve them:

Frontend 2: the ``ManualPipelineStage`` API -- If You Already Have the Module for Each Stage
============================================================================================

If you already have the module for each stage, you can skip the pipeline split step and directly use the runtime offering of the ``pipelining`` package.

This can be done by creating a ``ManualPipelineStage`` to wrap your stage module:

.. currentmodule:: torch.distributed.pipelining.PipelineStage

.. autoclass:: ManualPipelineStage


Step 2: Using ``PipelineSchedule`` for Execution
************************************************

After transforming the model into a ``Pipe`` representation, we can run its stages in a distributed *runtime*. This can be done in two steps:
* instantiate a ``PipelineStage`` from a stage module of ``Pipe``;
* run the ``PipelineStage`` according to a ``PipelineSchedule``.

First off, let us instantiate a ``PipelineStage`` instance:

.. code-block:: python

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

We can now attach the ``PipelineStage`` to a pipeline schedule, GPipe for example, and run with data:

.. code-block:: python

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

Note that since we split our model into three stages, we must run this script with three workers. For this example, we will use ``torchrun`` to run multiple processes within a single machine for demonstration purposes. We can collect up all of the code blocks above into a file named `example.py <https://github.com/pytorch/PiPPy/tree/main/examples/basic>`_ and then run it with ``torchrun`` like so:

.. code-block:: bash

  torchrun --nproc_per_node=3 example.py

Examples
********

In the `PiPPy <https://github.com/pytorch/PiPPy>`_ repo where this package is migrated from, we provide rich examples based on realistic models. In particular, we show how to apply pipelining without any model code change. You can refer to the `HuggingFace examples directory <https://github.com/pytorch/PiPPy/tree/main/examples/huggingface>`_. Popular examples include: `GPT2 <https://github.com/pytorch/PiPPy/tree/main/examples/huggingface/pippy_gpt2.py>`_, and `LLaMA <https://github.com/pytorch/PiPPy/tree/main/examples/llama>`_.

Techniques Deep Dive
********************

How does the ``pipeline`` API split a model?
============================================

First, the ``pipeline`` API turns our model into a directed acyclic graph (DAG) by tracing the model.
We trace the model via ``torch.export`` -- a PyTorch 2 full-graph capturing tool.
Then, it groups together the operations and parameters into *pipeline stages*.
Stages are represented as ``submod_N`` submodules, where ``N`` is a natural number.

For the `split_spec`, our library also provides ``SplitPoint.BEGINNING`` for a user to specify a split point *before* execution of certain module, and similarly, ``SplitPoint.END``  for split point *after* execution of certain module.

API Reference
*************

Pipeline Transformation APIs
============================

The following set of APIs transform your model into a pipeline representation.

.. currentmodule:: torch.distributed.pipelining

.. autoclass:: SplitPoint

.. autofunction:: pipeline

.. autoclass:: Pipe

.. autofunction:: annotate_split_points

.. autofunction:: pipe_split

.. autoclass:: ArgsChunkSpec

.. autoclass:: KwargsChunkSpec

Microbatch Utilities
====================

.. automodule:: torch.distributed.pipelining.microbatch

.. currentmodule:: torch.distributed.pipelining.microbatch

.. autoclass:: TensorChunkSpec

.. autofunction:: split_args_kwargs_into_chunks

.. autofunction:: merge_chunks

Pipeline Stages
===============

.. automodule:: torch.distributed.pipelining.PipelineStage

.. currentmodule:: torch.distributed.pipelining.PipelineStage

.. autoclass:: PipelineStage

.. autoclass:: ManualPipelineStage

Pipeline Schedules
==================

.. automodule:: torch.distributed.pipelining.PipelineSchedule

.. currentmodule:: torch.distributed.pipelining.PipelineSchedule

.. autoclass:: ScheduleGPipe

.. autoclass:: Schedule1F1B

.. autoclass:: ScheduleInterleaved1F1B

.. autoclass:: ScheduleLoopedBFS

Implementing Your Own Schedule
==============================

You can implement your own pipeline schedule by extending one of the following two class:

* ``PipelineScheduleSingle``
* ``PipelineScheduleMulti``

``PipelineScheduleSingle`` is for schedules that assigns *only one* stage per rank.
``PipelineScheduleMulti`` is for schedules that assigns multiple stages per rank.

For example, ``ScheduleGPipe`` and ``Schedule1F1B`` are subclasses of ``PipelineScheduleSingle``.
Whereas, ``ScheduleInterleaved1F1B`` and ``ScheduleLoopedBFS`` are subclasses of ``PipelineScheduleMulti``.

.. currentmodule:: torch.distributed.pipelining.PipelineSchedule

.. autoclass:: PipelineScheduleSingle

.. autoclass:: PipelineScheduleMulti
