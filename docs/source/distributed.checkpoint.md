```{eval-rst}
.. role:: hidden
    :class: hidden-section
```

# Distributed Checkpoint - torch.distributed.checkpoint

Distributed Checkpoint (DCP) supports loading and saving models from multiple ranks in parallel.
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
.. currentmodule:: torch.distributed.checkpoint.optimizer
```

```{eval-rst}
.. autofunction:: load_sharded_optimizer_state_dict
```

```{eval-rst}
.. currentmodule:: torch.distributed.checkpoint.planner_helpers
```

```{eval-rst}
.. autofunction:: create_read_items_for_chunk_list
```

```{eval-rst}
.. currentmodule:: torch.distributed.checkpoint.default_planner
```

```{eval-rst}
.. autofunction:: create_default_global_load_plan
```

```{eval-rst}
.. autofunction:: create_default_global_save_plan
```

```{eval-rst}
.. autofunction:: create_default_local_save_plan
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
.. autofunction:: save_state_dict
```

```{eval-rst}
.. currentmodule:: torch.distributed.checkpoint.state_dict_loader
```

```{eval-rst}
.. autofunction::  load
```

```{eval-rst}
.. autofunction:: load_state_dict
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

The following types define the metadata used during checkpoint:

```{eval-rst}
.. autoclass:: torch.distributed.checkpoint.metadata.StorageMeta
  :members:
```

```{eval-rst}
.. autoclass:: torch.distributed.checkpoint.metadata.TensorProperties
  :members:
```

```{eval-rst}
.. autoclass:: torch.distributed.checkpoint.metadata.TensorStorageMetadata
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

```{eval-rst}
.. autoclass:: torch.distributed.checkpoint.planner.TensorWriteData
  :members:
```

We provide a filesystem based storage layer:

```{eval-rst}
.. autoclass:: torch.distributed.checkpoint.filesystem.FileSystemBase
  :members:
```

```{eval-rst}
.. autoclass:: torch.distributed.checkpoint.filesystem.FileSystem
  :members:
```

```{eval-rst}
.. autoclass:: torch.distributed.checkpoint.filesystem.SerializationFormat
  :members:
```

```{eval-rst}
.. autoclass:: torch.distributed.checkpoint.FileSystemReader
  :members:
```

```{eval-rst}
.. autoclass:: torch.distributed.checkpoint.FileSystemWriter
  :members:
```

For object stores and other non-POSIX backends, DCP also ships an
[fsspec](https://filesystem-spec.readthedocs.io/) backed storage layer.
`FsspecReader` and `FsspecWriter` accept any URL that fsspec can resolve
(for example `s3://bucket/path`) and forward any extra keyword arguments to
`fsspec.core.url_to_fs`. For an `s3://` URL those arguments reach the
[s3fs](https://s3fs.readthedocs.io/) filesystem, which is how you point DCP at
Amazon S3 or any S3-compatible object store (for example Backblaze B2,
Cloudflare R2, or MinIO) by setting credentials and, for a non-AWS store, a
custom endpoint. Install the matching backend first (`pip install s3fs`).

s3fs reads credentials from the standard AWS environment variables:

```bash
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
```

For a non-AWS S3-compatible store, also pass its endpoint through the s3fs
`client_kwargs` argument (AWS S3 needs no endpoint):

```python
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FsspecWriter

writer = FsspecWriter(
    "s3://my-bucket/run-42/step-1000",
    client_kwargs={"endpoint_url": "https://your-s3-endpoint.example.com"},
)
dcp.save(state_dict=app_state, storage_writer=writer)
```

Credentials can also be passed explicitly instead of through the environment,
using the s3fs `key` and `secret` arguments:

```python
import os
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FsspecWriter

writer = FsspecWriter(
    "s3://my-bucket/run-42/step-1000",
    client_kwargs={"endpoint_url": "https://your-s3-endpoint.example.com"},
    key=os.environ["AWS_ACCESS_KEY_ID"],
    secret=os.environ["AWS_SECRET_ACCESS_KEY"],
)
dcp.save(state_dict=app_state, storage_writer=writer)
```

`FsspecReader` takes the same arguments for distributed loads:

```python
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FsspecReader

reader = FsspecReader(
    "s3://my-bucket/run-42/step-1000",
    client_kwargs={"endpoint_url": "https://your-s3-endpoint.example.com"},
)
dcp.load(state_dict=app_state, storage_reader=reader)
```

We also provide other storage layers, including ones to interact with HuggingFace safetensors:

.. autoclass:: torch.distributed.checkpoint.HuggingFaceStorageReader
  :members:

.. autoclass:: torch.distributed.checkpoint.HuggingFaceStorageWriter
  :members:

.. autoclass:: torch.distributed.checkpoint.QuantizedHuggingFaceStorageReader
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

`set_model_state_dict()` and `set_optimizer_state_dict()` are provided to load the model and optimizer state_dict generated by their respective getter APIs.

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

For users which are used to using and sharing models in the `torch.save` format, the following methods are provided which provide offline utilities for converting between formats.

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
