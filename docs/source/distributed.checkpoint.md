```{eval-rst}
.. role:: hidden
    :class: hidden-section
```

# Distributed Checkpoint - torch.distributed.checkpoint

Distributed Checkpoint (DCP) support loading and saving models from multiple ranks in parallel.
It handles load-time resharding which enables saving in one cluster topology and loading into another.

DCP is different than `torch.save` and `torch.load` in a few significant ways:

- It produces multiple files per checkpoint, with at least one per rank.
- It operates in place, meaning that the model should allocate its data first and DCP uses that storage instead.

The entrypoints to load and save a checkpoint are the following:

## Additional resources:

- [Getting Started with Distributed Checkpoint (DCP)](https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html)
- [Asynchronous Saving with Distributed Checkpoint (DCP)](https://pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html)
- [TorchTitan Checkpointing Docs](https://github.com/pytorch/torchtitan/blob/main/docs/checkpoint.md)
- [TorchTitan DCP Implementation](https://github.com/pytorch/torchtitan/blob/main/torchtitan/components/checkpoint.py)

```{eval-rst}
.. automodule:: torch.distributed.checkpoint
```

```{eval-rst}
.. currentmodule:: torch.distributed.checkpoint.state_dict_saver
```

```{eval-rst}
.. autoclass:: torch.distributed.checkpoint.state_dict_saver.AsyncCheckpointerType
  :members:
```

```{eval-rst}
.. autoclass:: torch.distributed.checkpoint.state_dict_saver.AsyncSaveResponse
  :members:
```

```{eval-rst}
.. autofunction::  save
```

```{eval-rst}
.. autofunction::  async_save
```

```{eval-rst}
.. autofunction::  save_state_dict
```

```{eval-rst}
.. currentmodule:: torch.distributed.checkpoint.state_dict_loader
```

```{eval-rst}
.. autofunction::  load
```

```{eval-rst}
.. autofunction::  load_state_dict
```

The following module is also useful for additional customization of the staging mechanisms used for asynchronous checkpointing (`torch.distributed.checkpoint.async_save`):

```{eval-rst}
.. automodule:: torch.distributed.checkpoint.staging
```

```{eval-rst}
.. autoclass:: torch.distributed.checkpoint.staging.AsyncStager
  :members:
```

```{eval-rst}
.. autoclass:: torch.distributed.checkpoint.staging.DefaultStager
  :members:
```

```{eval-rst}
.. autoclass:: torch.distributed.checkpoint.staging.StagingOptions
  :members:
```

```{eval-rst}
.. autoclass:: torch.distributed.checkpoint.staging.BlockingAsyncStager
  :members:
```

In addition to the above entrypoints, `Stateful` objects, as described below, provide additional customization during saving/loading

```{eval-rst}
.. automodule:: torch.distributed.checkpoint.stateful
   :noindex:
```

```{eval-rst}
.. autoclass:: torch.distributed.checkpoint.stateful.Stateful
  :members:
```

This [example](https://github.com/pytorch/pytorch/blob/main/torch/distributed/checkpoint/examples/fsdp_checkpoint_example.py) shows how to use Pytorch Distributed Checkpoint to save a FSDP model.

The following types define the IO interface used during checkpoint:

```{eval-rst}
.. autoclass:: torch.distributed.checkpoint.StorageReader
  :members:
```

```{eval-rst}
.. autoclass:: torch.distributed.checkpoint.StorageWriter
  :members:
```

The following types define the planner interface used during checkpoint:

```{eval-rst}
.. autoclass:: torch.distributed.checkpoint.LoadPlanner
  :members:
```

```{eval-rst}
.. autoclass:: torch.distributed.checkpoint.LoadPlan
  :members:
```

```{eval-rst}
.. autoclass:: torch.distributed.checkpoint.ReadItem
  :members:
```

```{eval-rst}
.. autoclass:: torch.distributed.checkpoint.SavePlanner
  :members:
```

```{eval-rst}
.. autoclass:: torch.distributed.checkpoint.SavePlan
  :members:
```

```{eval-rst}
.. autoclass:: torch.distributed.checkpoint.planner.WriteItem
  :members:
```

```{eval-rst}
.. autoclass:: torch.distributed.checkpoint.planner.BytesIOWriteData
  :members:
```

We provide a filesystem based storage layer:

```{eval-rst}
.. autoclass:: torch.distributed.checkpoint.FileSystemReader
  :members:
```

```{eval-rst}
.. autoclass:: torch.distributed.checkpoint.FileSystemWriter
  :members:
```

We also provide other storage layers, including ones to interact with HuggingFace safetensors:

.. autoclass:: torch.distributed.checkpoint.HuggingFaceStorageReader
  :members:

.. autoclass:: torch.distributed.checkpoint.HuggingFaceStorageWriter
  :members:

We provide default implementations of `LoadPlanner` and `SavePlanner` that
can handle all of torch.distributed constructs such as FSDP, DDP, ShardedTensor and DistributedTensor.

```{eval-rst}
.. autoclass:: torch.distributed.checkpoint.DefaultSavePlanner
  :members:
```

```{eval-rst}
.. autoclass:: torch.distributed.checkpoint.DefaultLoadPlanner
  :members:

```

Due to legacy design decisions, the state dictionaries of `FSDP` and `DDP` may have different keys or fully qualified names (e.g., layer1.weight) even when the original unparallelized model is identical. Moreover, `FSDP` offers various types of model state dictionaries, such as full and sharded state dictionaries. Additionally, optimizer state dictionaries employ parameter IDs instead of fully qualified names to identify parameters, potentially causing issues when parallelisms are used (e.g., pipeline parallelism).

To tackle these challenges, we offer a collection of APIs for users to easily manage state_dicts. `get_model_state_dict()` returns a model state dictionary with keys consistent with those returned by the unparallelized model state dictionary. Similarly, `get_optimizer_state_dict()` provides the optimizer state dictionary with keys uniform across all parallelisms applied. To achieve this consistency, `get_optimizer_state_dict()` converts parameter IDs to fully qualified names identical to those found in the unparallelized model state dictionary.

Note that results returned by these APIs can be used directly with the `torch.distributed.checkpoint.save()` and `torch.distributed.checkpoint.load()` methods without requiring any additional conversions.

`set_model_state_dict()` and `set_optimizer_state_dict()` are provided to load the model and optimizer state_dict generated by by their respective getter APIs.

Note that `set_optimizer_state_dict()` can only be called before `backward()` or after `step()` is called on optimizers.

Note that this feature is experimental, and API signatures might change in the future.

```{eval-rst}
.. autofunction:: torch.distributed.checkpoint.state_dict.get_state_dict
```

```{eval-rst}
.. autofunction:: torch.distributed.checkpoint.state_dict.get_model_state_dict
```

```{eval-rst}
.. autofunction:: torch.distributed.checkpoint.state_dict.get_optimizer_state_dict
```

```{eval-rst}
.. autofunction:: torch.distributed.checkpoint.state_dict.set_state_dict
```

```{eval-rst}
.. autofunction:: torch.distributed.checkpoint.state_dict.set_model_state_dict
```

```{eval-rst}
.. autofunction:: torch.distributed.checkpoint.state_dict.set_optimizer_state_dict
```

```{eval-rst}
.. autoclass:: torch.distributed.checkpoint.state_dict.StateDictOptions
   :members:
```

For users which are used to using and sharing models in the `torch.save` format, the following methods are provided which provide offline utilities for converting betweeing formats.

```{eval-rst}
.. automodule:: torch.distributed.checkpoint.format_utils
```

```{eval-rst}
.. currentmodule:: torch.distributed.checkpoint.format_utils
```

```{eval-rst}
.. autofunction:: dcp_to_torch_save
```

```{eval-rst}
.. autofunction:: torch_save_to_dcp
```

The following classes can also be utilized for online loading and resharding of models from the torch.save format.

```{eval-rst}
.. autoclass:: torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader
   :members:
```

```{eval-rst}
.. autoclass:: torch.distributed.checkpoint.format_utils.DynamicMetaLoadPlanner
   :members:
```

The following experimental interfaces are provided for improved observability in production environments:

```{eval-rst}
.. py:module:: torch.distributed.checkpoint.logger
```

```{eval-rst}
.. py:module:: torch.distributed.checkpoint.logging_handlers
```
