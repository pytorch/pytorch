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

We provide a set of APIs to help users do get and set state_dict easily. This is
an experimental feature and is subject to change.

.. autofunction:: torch.distributed.checkpoint.state_dict.get_state_dict

.. autofunction:: torch.distributed.checkpoint.state_dict.get_model_state_dict

.. autofunction:: torch.distributed.checkpoint.state_dict.get_optimizer_state_dict

.. autofunction:: torch.distributed.checkpoint.state_dict.set_state_dict

.. autofunction:: torch.distributed.checkpoint.state_dict.set_model_state_dict

.. autofunction:: torch.distributed.checkpoint.state_dict.set_optimizer_state_dict

.. autoclass:: torch.distributed.checkpoint.state_dict.StateDictOptions
   :members:

For users which are used to using and sharing models in the `torch.save` format, the following utilities are pvoided:

.. autofunction:: torch.distributed.checkpoint.format_utils.dcp_to_torch_save

.. autofunction:: torch.distributed.checkpoint.format_utils.torch_save_to_dcp
