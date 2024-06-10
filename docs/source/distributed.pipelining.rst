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
Pipeline parallelism can be an effective technique for:

* large-scale training
* bandwidth-limited clusters
* large model inference.

The above scenarios share a commonality that the computation per device cannot
hide the communication of conventional parallelism, for example, the weight
all-gather of FSDP.


What is ``torch.distributed.pipelining``?
*****************************************

While promising for scaling, pipelining is often difficult to implement because
it needs to **partition the execution** of a model in addition to model weights.
The partitioning of execution often requires intrusive code changes to your
model. Another aspect of complexity comes from **scheduling micro-batches in a
distributed environment**, with **data flow dependency** considered.

The ``pipelining`` package provides a toolkit that does said things
**automatically** which allows easy implementation of pipeline parallelism
on **general** models.

It consists of two parts: a
**splitting frontend** and a **distributed runtime**.
The splitting frontend takes your model code as-is, splits it up into "model
partitions", and captures the data-flow relationship.  The distributed runtime
executes the pipeline stages on different devices in parallel, handling things
like micro-batch splitting, scheduling, communication, and gradient propagation,
etc.

Overall, the ``pipelining`` package provides the following features:

* Splitting of model code based on simple specification.
* Rich support for pipeline schedules, including GPipe, 1F1B,
  Interleaved 1F1B and Looped BFS, and providing the infrastruture for writing
  customized schedules.
* First-class support for cross-host pipeline parallelism, as this is where PP
  is typically used (over slower interconnects).
* Composability with other PyTorch parallel techniques such as data parallel
  (DDP, FSDP) or tensor  parallel. The `TorchTitan
  <https://github.com/pytorch/torchtitan>`_ project demonstrates a "3D parallel"
  application on the Llama model.


Step 1: build ``PipelineStage`` for execution
*********************************************

Before we can use a ``PipelineSchedule``, we need to create ``PipelineStage``
objects that wrap the part of the model running in that stage.  The
``PipelineStage`` is responsible for allocating communication buffers and
creating send/recv ops to communicate with its peers.  It manages intermediate
buffers e.g. for the outputs of forward that have not been consumed yet, and it
provides a utility for running the backwards for the stage model.

A ``PipelineStage`` needs to know the input and output shapes for the stage
model, so that it can correctly allocate communication buffers.  The shapes must
be static, e.g. at runtime the shapes can not change from step to step.  A class
``PipeliningShapeError`` will be raised if runtime shapes do not match the
expected shapes.  When composing with other paralleisms or applying mixed
precision, these techniques must be taken into account so the ``PipelineStage``
knows the correct shape (and dtype) for the output of the stage module at
runtime.

Users may construct a ``PipelineStage`` instance directly, by passing in an
``nn.Module`` representing the portion of the model that should run on the
stage.  This may require changes to the original model code.  See the example
in :ref:`option_1_manual`.

Alternatively, the splitting frontend can use graph partitioning to split your
model into a series of ``nn.Module`` automatically.  This technique requires the
model is traceable with ``torch.Export``. Composability of the resulting
``nn.Module`` with other parallelism techniques is experimental, and may require
some workarounds.  Usage of this frontend may be more appealing if the user
cannot easily change the model code.  See :ref:`option_2_tracer` for more
information.


Step 2: use ``PipelineSchedule`` for execution
**********************************************

We can now attach the ``PipelineStage`` to a pipeline schedule, and run the
schedule with input data. Here is a GPipe example:

.. code-block:: python

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

Note that the above code needs to be launched for each worker, thus we use a
launcher service to launch multiple processes:

.. code-block:: bash

  torchrun --nproc_per_node=2 example.py


Options for Splitting a Model
*****************************

.. _option_1_manual:

Option 1: splitting a model manually
====================================

To directly construct a ``PipelineStage``, the user is responsible for providing
a single ``nn.Module`` instance that owns the relevant ``nn.Parameters`` and
``nn.Buffers``, and defines a ``forward()`` method that executes the operations
relevant for that stage.  For example, a condensed version of the Transformer
class defined in Torchtitan shows a pattern of building an easily partitionable
model.

.. code-block:: python

  class Transformer(nn.Module):
      def __init__(self, model_args: ModelArgs):
          super().__init__()

          self.tok_embeddings = nn.Embedding(...)

          # Using a ModuleDict lets us delete layers witout affecting names,
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

A model defined in this manner can be easily configured per stage by first
initializing the whole model (using meta-device to avoid OOM errors), deleting
undesired layers for that stage, and then creating a PipelineStage that wraps
the model.  For example:

.. code-block:: python

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
          input_args=example_input_microbatch,
      )


The ``PipelineStage`` requires an example argument ``input_args`` representing
the runtime input to the stage, which would be one microbatch worth of input
data.  This argument is passed through the forward method of the stage module to
determine the input and output shapes required for communication.

When composing with other Data or Model parallelism techniques, ``output_args``
may also be required, if the output shape/dtype of the model chunk will be
affected.


.. _option_2_tracer:

Option 2: splitting a model automatically
=========================================

If you have a full model and do not want to spend time on modifying it into a
sequence of "model partitions", the ``pipeline`` API is here to help.
Here is a brief example:

.. code-block:: python

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
similarly, ``SplitPoint.END`` for split point *after* such.

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
``submod_1``), each of which is reconstructed with original model operations
and hierarchies.  In addition, a "root-level" ``forward`` function is
reconstructed to capture the data flow between those partitions. Such data flow
will be replayed by the pipeline runtime later, in a distributed fashion.

The ``Pipe`` object provides a method for retrieving the "model partitions":

.. code-block:: python

  stage_mod : nn.Module = pipe.get_stage_module(stage_idx)

You can also create a distributed stage runtime on a device using ``Pipe``:

.. code-block:: python

  stage = pipe.build_stage(stage_idx, device, group)

.. note::
  The ``pipeline`` frontend uses a tracer (``torch.export``) to capture your
  model into a single graph. If your model is not full-graph'able, you can use
  our manual frontend below.


Hugging Face Examples
*********************

In the `PiPPy <https://github.com/pytorch/PiPPy>`_ repo where this package was
original created, we kept examples based on unmodified Hugging Face models.
See the `examples/huggingface
<https://github.com/pytorch/PiPPy/tree/main/examples/huggingface>`_ directory.

Examples include:

* `GPT2 <https://github.com/pytorch/PiPPy/tree/main/examples/huggingface/pippy_gpt2.py>`_
* `Llama <https://github.com/pytorch/PiPPy/tree/main/examples/llama>`_


Technical Deep Dive
*******************

How does the ``pipeline`` API split a model?
============================================

First, the ``pipeline`` API turns our model into a directed acyclic graph (DAG)
by tracing the model.  It traces the model using ``torch.export`` -- a PyTorch 2
full-graph capturing tool.

Then, it groups together the **operations and parameters** needed by a stage
into a reconstructed submodule: ``submod_0``, ``submod_1``, ...

Different from conventional submodule access methods like ``Module.children()``,
the ``pipeline`` API does not only cut the module structure of your model, but
also the **forward** function of your model.

This is necessary because model structure like ``Module.children()`` merely
captures information during ``Module.__init__()``, and does not capture any
information about ``Module.forward()``. Said differently, ``Module.children()``
lacks information about the following aspects key to pipelininig:

* Execution order of child modules in ``forward``
* Activation flows between child modules
* Whether there are any functional operators between child modules (for example,
  ``relu`` or ``add`` operations will not be captured by ``Module.children()``).

The ``pipeline`` API, on the contrary, makes sure that the ``forward`` behavior
is truly preserved. It also captures the activation flow between the partitions,
helping the distributed runtime to make correct send/receive calls without human
intervention.

Another flexibility of the ``pipeline`` API is that split points can be at
arbitrary levels within your model hierarchy. In the split partitions, the original model
hierarchy related to that partition will be reconstructed at no cost to you.
At a result, fully-qualified names (FQNs) pointing to a submodule or parameter
would be still valid, and services that relies on FQNs (such as FSDP, TP or
checkpointing) can still run with your partitioned modules with almost zero code
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


API Reference
*************

.. automodule:: torch.distributed.pipelining

Model Split APIs
============================

The following set of APIs transform your model into a pipeline representation.

.. currentmodule:: torch.distributed.pipelining

.. autoclass:: SplitPoint

.. autofunction:: pipeline

.. autoclass:: Pipe

.. autofunction:: pipe_split

Microbatch Utilities
====================

.. automodule:: torch.distributed.pipelining.microbatch

.. currentmodule:: torch.distributed.pipelining.microbatch

.. autoclass:: TensorChunkSpec

.. autofunction:: split_args_kwargs_into_chunks

.. autofunction:: merge_chunks

Pipeline Stages
===============

.. automodule:: torch.distributed.pipelining.stage

.. currentmodule:: torch.distributed.pipelining.stage

.. autoclass:: PipelineStage

.. autofunction:: build_stage

Pipeline Schedules
==================

.. automodule:: torch.distributed.pipelining.schedules

.. currentmodule:: torch.distributed.pipelining.schedules

.. autoclass:: ScheduleGPipe

.. autoclass:: Schedule1F1B

.. autoclass:: ScheduleInterleaved1F1B

.. autoclass:: ScheduleLoopedBFS

.. autoclass:: PipelineScheduleSingle
  :members:

.. autoclass:: PipelineScheduleMulti
  :members:
