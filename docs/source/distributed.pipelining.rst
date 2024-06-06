.. role:: hidden
    :class: hidden-section

Pipeline Parallelism
####################

.. note::
  ``torch.distributed.pipelining`` is currently in alpha state and under
  development. API changes may be possible. It was migrated from the `PiPPy
  <https://github.com/pytorch/PiPPy>`_ project.


Why Pipeline Parallel?
**********************

Pipeline Parallelism is one of the **primitive** parallelism for deep learning.
It allows the **execution** of a model to be partitioned such that multiple
**micro-batches** can execute different parts of the model code concurrently.
Pipeline parallelism can be an effective technique for large-scale jobs or
bandwidth-limited clusters, where popular techniques like FSDP face **increased
overhead** from all-gathering model weights.


What is ``torch.distributed.pipelining``?
*****************************************

While promising for scaling, pipelining is often difficult to implement because
it needs to partition the **execution** of a model in addition to model weights.
The partitioning of execution often requires intrusive code changes to your
model. Another aspect of complexity comes from the need for scheduling
micro-batches in a **distributed** environment, with **data flow dependency**
considered.

The ``pipelining`` package provides a toolkit that does said things
**automatically** and allows easy implementation of pipeline parallelism
on **general** models.

It consists of two parts: a
**transformation frontend** and a **distributed runtime**.
The transformation frontend takes your model code as-is, splits it up into
"model partitions", and capture the data-flow relationship.  The distributed
runtime executes the pipeline stages on different devices in parallel, handling
things like micro-batch splitting, scheduling, communication, and gradient
propagation, etc.

Overall, the ``pipelining`` package provides the following features:

* Splitting of model code based on simple specification. The goal is for the
  user to provide model code with **zero code change**, and make
  parallelism work.
* Rich support for pipeline schedules, including GPipe, 1F1B,
  Interleaved 1F1B and Looped BFS, and provide the infrastruture for writing
  customized schedule if desired.
* First-class support for cross-host pipeline parallelism, as this is where PP
  is typically used (over slower interconnects).
* Composability with other PyTorch parallel techniques such as data parallel
  (DDP, FSDP) or tensor  parallel. The `TorchTitan
  <https://github.com/pytorch/torchtitan>`_ project demonstrates such a "3D
  parallel" application on the Llama model.


Step 1: Choosing the Frontend that Fits Your Need
*************************************************

The ``pipelining`` package provides two frontends for two different use cases.
You can make your choice based on whether you have (i) a full model or (ii)
module constructors for each stage.


Frontend 1: the ``pipeline`` API -- If You Have a Full Model
============================================================

If you have a full model and do not want to spend time on modifying it into a
sequence of "model partitions", the ``pipeline`` API is here to help.
Here is a brief example:

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


If we print the model, we can see multiple hierarchies, which makes it hard to split by hand::

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

Let us see how the ``pipeline`` API works:

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

The ``pipeline`` API splits your model given a ``split_spec``, where
``SplitPoint.BEGINNING`` stands for adding a split point
*before* execution of certain submodule in the ``forward`` function, and
*similarly, ``SplitPoint.END`` for
split point *after* such.

If we ``print(pipe)``, we can see::

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


The "model partitions" are represented by submodules (``submod_0``,
``submod_1``), each of which are reconstructed with original model operations
and hierarchies.  In addition, a "root-level" ``forward`` function is
reconstructed to capture the data flow between those partitions. Such data flow
will be replayed by the pipeline runtime later, in a distributed fashion.

The ``Pipe`` object provides a method for retrieving the "model partitions":

.. code-block:: python

  stage_mod : nn.Module = pipe.get_stage_module(stage_idx)

You can also create a distributed stage runtime on a device using ``Pipe``:

.. code-block:: python

  from torch.distributed.pipelining import PipelineStage

  stage = PipelineStage(pipe, stage_idx, device)


Frontend 2: ``ManualPipelineStage`` -- If You Already Have the Module for Each Stage
====================================================================================

If you already have the module for each stage, you can skip the pipeline split
step above and directly connect to the runtime offering of the ``pipelining``
package.
This can be done by creating a ``ManualPipelineStage`` to wrap your stage
module:

.. currentmodule:: torch.distributed.pipelining.PipelineStage

.. autoclass:: ManualPipelineStage

TODO: manual example here


Step 2: Using ``PipelineSchedule`` for Execution
************************************************

We can now attach the ``PipelineStage`` to a pipeline schedule, GPipe for
example, and run with data:

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

Note that the above code needs to be launched for each worker, thus we can a
launcher service like ``torchrun`` to launch multiple processes.

.. code-block:: bash

  torchrun --nproc_per_node=2 example.py


Examples
********

In the `PiPPy <https://github.com/pytorch/PiPPy>`_ repo where this package was
original created, we kept examples based on unmodified Hugging Face models.
You can refer to the `examples/huggingface
<https://github.com/pytorch/PiPPy/tree/main/examples/huggingface>`_ directory.
See for example:
* `GPT2 <https://github.com/pytorch/PiPPy/tree/main/examples/huggingface/pippy_gpt2.py>`_
* `Llama <https://github.com/pytorch/PiPPy/tree/main/examples/llama>`_


Technical Deep Dive
*******************

How does the ``pipeline`` API split a model?
============================================

First, the ``pipeline`` API turns our model into a directed acyclic graph (DAG)
by tracing the model.  It traces the model using ``torch.export`` -- a PyTorch 2
full-graph capturing tool.

Then, it groups together the **operations** and **parameters** needed by a stage
into a reconstructed submodule: ``submod_0``, ``submod_1``, ...

Different from conventional submodule access methods like ``Module.children()``,
the ``pipeline`` API does not only cut the module structure of your model, but
also the **``forward``** function of your model.

This is necessary because model structure like ``Module.children()`` merely
captures information during ``Module.__init__()``, and does not capture any
information about ``Module.forward()``. Said differently, ``Module.children()``
lacks information on the following aspects which are key to pipelininig:

* Exectuion order of child modules in ``forward``
* Activation flows between child modules
* Whether there are any functional operators between child modules (for example,
  ``relu`` or ``add`` operations would be be captured by ``Module.children()``).

The ``pipeline`` API, on the contrary, makes sure that the ``forward`` behavior
is truly preserved. It also captures the activation flow between the partitions,
helping the distributed runtime to make correct send/receive calls without human
intervention.

Another flexibility of the ``pipeline`` API is that split points can be at
arbitrary hierarchy of your model. In the split partitions, the original model
hierarchy related to that partition will be reconstructed at no cost of yours.
At a result, fully-qualified names (FQNs) pointing to a submodule or parameter
would be still valid, and services that relies on FQNs (such as FSDP, TP or
checkpointing) can still run with your partitioned modules at almost zero code
change.



Implementing Your Own Schedule
******************************

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
