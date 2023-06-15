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

.. autofunction::  load_state_dict
.. autofunction::  save_state_dict

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

.. autoclass:: torch.distributed.checkpoint.WriteItem
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
