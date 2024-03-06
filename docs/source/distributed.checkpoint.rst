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


.. automodule:: torch.distributed.checkpoint

.. currentmodule:: torch.distributed.checkpoint

.. autofunction::  load
.. autofunction::  save
.. autofunction::  load_state_dict
.. autofunction::  save_state_dict

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


Due to the legacy design decisions, the state dictionary of `FSDP` and `DDP` have different keys, or fully qualified names (e.g., layer1.weight) even if the original unparallelized model is the same. Moreover, `FSDP` provides different types of model state dictionary, like full state_dict and sharded state_dict. An even more confusing case is the optimizer state dictionary, which does not use the fully qualified name but uses a parameter ID to indiciates a parameter. The design of optimizer state dictionary is fine when there is only one trainer but can cause issues when different parallelisms are used (e.g., pipeline parallelism).

To address these issues, we provide a set of APIs to help users to get and set state_dict easily. `get_model_state_dict` returns a model state dictionary, which always has the same keys as the keys returned by the unparallelized model state dictionary. Similarly, `get_optimizer_state_dict` returns the optimizer state dictionary, which always has the same key regardless what parallelisms are applized. To achieve this consistency, `get_optimizer_state_dict` converts the parameter IDs to the fully qualified names that are the same as the fully qualified names in the unparallelized model state dictionary.

This is an experimental feature and the signatures of APIs are subject to change.

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
