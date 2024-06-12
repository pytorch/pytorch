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

While promising for scaling, pipelining is often difficult to implement, requiring intrusive code changes to model code and difficult-to-implement runtime orchestration code. ``torch.distributed.pipelining`` aims to provide **a toolkit that does said things automatically to allow high-productivity scaling of models.** It consists of a **compiler** and a **runtime** stack for easy pipelining of PyTorch models. In particular, it provides the following features:

* Splitting of model code based on your specification. The goal is for the user to provide model code as-is to the system for parallelization, without having to make heavyweight modifications to make parallelism work. The specification is also simple.
* Support for rich pipeline scheduling paradigms, including GPipe, 1F1B, Interleaved 1F1B and Looped BFS. It will be also easy to customize your own schedule under this framework.
* First-class support for cross-host pipeline parallelism, as this is where PP is typically used (over slower interconnects).
* Composability with other PyTorch parallel schemes such as data parallelism (DDP, FSDP) or tensor  parallelism (overall, known as "3d parallelism").

Examples
********

In the `PiPPy <https://github.com/pytorch/PiPPy>`_ repo where this package is migrated from, we provide rich examples based on realistic models. In particular, we show how to apply pipelining without any model code change. You can refer to the `HuggingFace examples directory <https://github.com/pytorch/PiPPy/tree/main/examples/huggingface>`_. Popular examples include: `GPT2 <https://github.com/pytorch/PiPPy/tree/main/examples/huggingface/pippy_gpt2.py>`_, and `LLaMA <https://github.com/pytorch/PiPPy/tree/main/examples/llama>`_.

Techniques Explained
********************

``torch.distributed.pipelining`` consists of two parts: a *compiler* and a *runtime*. The compiler takes your model code, splits it up, and transforms it into a ``Pipe``, which is a wrapper that describes the model at each pipeline stage and their data-flow relationship. The runtime executes the ``PipelineStage`` in parallel, handling things like micro-batch splitting, scheduling, communication, and gradient propagation, etc. We will cover the APIs for these concepts in this section.

Splitting a Model with ``pipeline``
===================================

To see how we can split a model into a pipeline, let's first take an example trivial neural network:

.. code-block:: python

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

This network is written as free-form Python code; it has not been modified for any specific parallelism technique.

Let us see our usage of the ``pipeline`` interface:

.. code-block:: python

  from torch.distributed.pipelining import annotate_split_points, pipeline, Pipe, SplitPoint

  annotate_split_points(mn, {'layer0': SplitPoint.END,
                            'layer1': SplitPoint.END})

  batch_size = 32
  example_input = torch.randn(batch_size, in_dim, device=device)
  chunks = 4

  pipe = pipeline(mn, chunks, example_args=(example_input,))
  print(pipe)

::

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

So what's going on here? First, ``pipeline`` turns our model into a directed acyclic graph (DAG) by tracing the model. Then, it groups together the operations and parameters into *pipeline stages*. Stages are represented as ``submod_N`` submodules, where ``N`` is a natural number.

We used ``annotate_split_points`` to specify that the code should be split and the end of ``layer0`` and ``layer1``. Our code has thus been split into *three* pipeline stages. Our library also provides ``SplitPoint.BEGINNING`` if a user wants to split before certain annotation point.

While the ``annotate_split_points`` API gives users a way to specify the split points without modifying the model, our library also provides an API for in-model annotation: ``pipe_split()``. For details, you can read `this example <https://github.com/pytorch/PiPPy/blob/main/test/test_pipe.py>`_.

This covers the basic usage of the ``Pipe`` API. For more information, please see the documentation.

Using ``PipelineSchedule`` for Execution
========================================

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
