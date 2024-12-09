.. role:: hidden
    :class: hidden-section

Distributed Checkpoint - torch.distributed.checkpoint
=====================================================


Distributed Checkpoint (DCP) support loading and saving models from multiple ranks in parallel.
It handles load-time resharding which enables saving in one cluster topology and loading into another.

DCP is different than `torch.save` and `torch.load` in a few significant ways:

* It produces multiple files per checkpoint, with at least one per rank.
* It operates in place, meaning that the model should allocate its data first and DCP uses that storage instead.

The entrypoints to load and save a checkpoint are the following:

Additional resources:
---------------------

* `Getting Started with Distributed Checkpoint (DCP) <https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html>`__
* `Asynchronous Saving with Distributed Checkpoint (DCP) <https://pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html>`__
* `TorchTitan Checkpointing Docs <https://github.com/pytorch/torchtitan/blob/main/docs/checkpoint.md>`__
* `TorchTitan DCP Implementation <https://github.com/pytorch/torchtitan/blob/main/torchtitan/checkpoint.py>`__

.. automodule:: torch.distributed.checkpoint

.. currentmodule:: torch.distributed.checkpoint.state_dict_saver

.. autofunction::  save
.. autofunction::  async_save
.. autofunction::  save_state_dict

.. currentmodule:: torch.distributed.checkpoint.state_dict_loader

.. autofunction::  load
.. autofunction::  load_state_dict

The following module is also useful for additional customization of the staging mechanisms used for asynchronous checkpointing (`torch.distributed.checkpoint.async_save`):

.. automodule:: torch.distributed.checkpoint.staging

.. autoclass:: torch.distributed.checkpoint.staging.AsyncStager
  :members:

.. autoclass:: torch.distributed.checkpoint.staging.BlockingAsyncStager
  :members:

In addition to the above entrypoints, `Stateful` objects, as described below, provide additional customization during saving/loading
.. automodule:: torch.distributed.checkpoint.stateful

.. autoclass:: torch.distributed.checkpoint.stateful.Stateful
  :members:

This `example <https://github.com/pytorch/pytorch/blob/main/torch/distributed/checkpoint/examples/fsdp_checkpoint_example.py>`_ shows how to use Pytorch Distributed Checkpoint to save a FSDP model.

The following types define the IO interface used during checkpoint:

.. autoclass:: torch.distributed.checkpoint.StorageReader
  :members:

.. autoclass:: torch.distributed.checkpoint.StorageWriter
  :members:

The following types define the planner interface used during checkpoint:

.. autoclass:: torch.distributed.checkpoint.LoadPlanner
  :members:

.. autoclass:: torch.distributed.checkpoint.LoadPlan
  :members:

.. autoclass:: torch.distributed.checkpoint.ReadItem
  :members:

.. autoclass:: torch.distributed.checkpoint.SavePlanner
  :members:

.. autoclass:: torch.distributed.checkpoint.SavePlan
  :members:

.. autoclass:: torch.distributed.checkpoint.planner.WriteItem
  :members:

We provide a filesystem based storage layer:

.. autoclass:: torch.distributed.checkpoint.FileSystemReader
  :members:

.. autoclass:: torch.distributed.checkpoint.FileSystemWriter
  :members:

We provide default implementations of `LoadPlanner` and `SavePlanner` that
can handle all of torch.distributed constructs such as FSDP, DDP, ShardedTensor and DistributedTensor.

.. autoclass:: torch.distributed.checkpoint.DefaultSavePlanner
  :members:

.. autoclass:: torch.distributed.checkpoint.DefaultLoadPlanner
  :members:


Due to legacy design decisions, the state dictionaries of `FSDP` and `DDP` may have different keys or fully qualified names (e.g., layer1.weight) even when the original unparallelized model is identical. Moreover, `FSDP` offers various types of model state dictionaries, such as full and sharded state dictionaries. Additionally, optimizer state dictionaries employ parameter IDs instead of fully qualified names to identify parameters, potentially causing issues when parallelisms are used (e.g., pipeline parallelism).

To tackle these challenges, we offer a collection of APIs for users to easily manage state_dicts. `get_model_state_dict` returns a model state dictionary with keys consistent with those returned by the unparallelized model state dictionary. Similarly, `get_optimizer_state_dict` provides the optimizer state dictionary with keys uniform across all parallelisms applied. To achieve this consistency, `get_optimizer_state_dict` converts parameter IDs to fully qualified names identical to those found in the unparallelized model state dictionary.

Note that results returned by these APIs can be used directly with the `torch.distributed.checkpoint.save()` and `torch.distributed.checkpoint.load()` methods without requiring any additional conversions.

Note that this feature is experimental, and API signatures might change in the future.


.. autofunction:: torch.distributed.checkpoint.state_dict.get_state_dict

.. autofunction:: torch.distributed.checkpoint.state_dict.get_model_state_dict

.. autofunction:: torch.distributed.checkpoint.state_dict.get_optimizer_state_dict

.. autofunction:: torch.distributed.checkpoint.state_dict.set_state_dict

.. autofunction:: torch.distributed.checkpoint.state_dict.set_model_state_dict

.. autofunction:: torch.distributed.checkpoint.state_dict.set_optimizer_state_dict

.. autoclass:: torch.distributed.checkpoint.state_dict.StateDictOptions
   :members:

For users which are used to using and sharing models in the `torch.save` format, the following methods are provided which provide offline utilities for converting betweeing formats.

.. automodule:: torch.distributed.checkpoint.format_utils

.. currentmodule:: torch.distributed.checkpoint.format_utils

.. autofunction:: dcp_to_torch_save
.. autofunction:: torch_save_to_dcp

The following classes can also be utilized for online loading and resharding of models from the torch.save format.

.. autoclass:: torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader
   :members:

.. autoclass:: torch.distributed.checkpoint.format_utils.DynamicMetaLoadPlanner
   :members:

The following experimental interfaces are provided for improved observability in production environments:

.. py:module:: torch.distributed.checkpoint.logger
.. py:module:: torch.distributed.checkpoint.logging_handlers
