
import io
from typing import Any, Dict, List, Tuple

import torch
from torch.distributed._shard.sharded_tensor import (
    ShardedTensor,
    ShardedTensorMetadata
)
from torch.distributed._shard.sharding_spec._internals import (
    validate_non_overlapping_shards_metadata,
    _check_shard_metadata_pair_overlap,
)

from .metadata import (
    BytesReadRequest,
    BytesStorageMetadata,
    TensorReadRequest,
    Metadata,
    ShardedTensorStorageMetadata,
    TensorStorageMetadata,
)
from .resharding import (
    _prepare_sharded_tensor_read,
    _shards_get_overlap_region_wrt_saved_tensor
)
from .storage import StorageReader


def _reshard_and_prepare_read_request(
    state_dict: Dict[str, Any], metadata_from_storage: Metadata
) -> Tuple[List[BytesReadRequest], List[TensorReadRequest]]:
    """
    Use the loaded metadata and the current state dict to map the saved tensors to current tensor
    """
    tensor_read_requests = []
    bytes_read_requests = []
    for fqn, obj in state_dict.items():
        if isinstance(obj, ShardedTensor):
            md = metadata_from_storage.state_dict_metadata[fqn]
            if isinstance(md, ShardedTensorStorageMetadata):
                tensor_read_requests += _prepare_sharded_tensor_read(md, obj)
            else:
                raise ValueError(
                    f"Invalid checkpoint metadata for {fqn}, " +
                    f"expected ShardedTensorStorageMetadata but found {type(md)}"
                )
        elif isinstance(obj, torch.Tensor):
            tensor = obj.detach()
            md = metadata_from_storage.state_dict_metadata[fqn]
            if isinstance(md, TensorStorageMetadata):
                rr = TensorReadRequest(
                    tensor=tensor,
                    storage_key=md.storage_key,
                    offsets=tuple([0] * len(tensor.size())),
                    lengths=md.size,
                )

                tensor_read_requests.append(rr)
            else:
                raise ValueError(
                    f"Invalid checkpoint metadata for {fqn}, " +
                    f"expected TensorStorageMetadata but found {type(md)}"
                )
        else:
            md = metadata_from_storage.state_dict_metadata[fqn]
            # This is actually hard to handle correctly
            # If the value is not a tensor but any random obj,
            # we cannot just write whatever memory it points to inplace
            # the best we can to is to replace it with an object of the same type
            if isinstance(md, BytesStorageMetadata):
                bytes_io = io.BytesIO()
                brr = BytesReadRequest(
                    bytes=bytes_io,
                    storage_key=md.storage_key,
                    fqn=fqn
                )
                bytes_read_requests.append(brr)
            else:
                raise ValueError(
                    f"Invalid checkpoint metadata for {fqn}, " +
                    f"expected BytesStorageMetadata but found {type(md)}"
                )


    return (bytes_read_requests, tensor_read_requests)


def load_state_dict(
    state_dict: Dict[str, Any],
    storage_reader: StorageReader,
) -> None:
    """
    Load a distributed state_dict in SPMD style.

    Each rank will try to read the least amount of data necessary
    to fullfill the requested `state_dict`.

    When loading ShardedTensor instances, each rank only
    reads data for their local shards.

    All tensors in ``state_dict`` must be allocated on their
    destination device prior to calling this function.

    All non-tensor data is loaded using `torch.load()` and modified in place
    on state_dict.

    Users must call `load_state_dict` on the root module to ensure load
    pos-processing and non-tensor data properly propagates.

    Args:
        state_dict (Dict[str, Any]) : The state_dict to load. Note that this
            state dict will updated in places.
        storage_reader (StorageReader): StorageReader used to load data from.

    Returns:
        None.

    Examples
        >>> my_model = MyModule()
        >>> optimizer = Adagrad(my_model.parameters())
        >>> model_state_dict = my_model.state_dict()
        >>> fs_storage_loader = torch.distributed._shard.checkpoint.FileSystemLoader("/checkpoint/1")

        >>> torch.distributed._shard.checkpoint.load_state_dict(
        >>>     state_dict=model_state_dict,
        >>>     storage_reader=fs_storage_loader,
        >>> )

        >>> # module.load_state_dict() function might have customized steps
        >>> # to flush the state_dict, must call it to
        >>> # ensure correct behavior.
        >>> my_model.load_state_dict(model_state_dict)
    """

    metadata = storage_reader.read_metadata()
    bytes_read_requests, tensor_read_requests = _reshard_and_prepare_read_request(
        state_dict=state_dict, metadata_from_storage=metadata
    )
    bytes_futures = storage_reader.read_bytes(bytes_read_requests)
    tensor_futures = storage_reader.read_tensors(tensor_read_requests)

    bytes_futures.wait()

    # Addtional steps are required to convert the bytes to its original type
    # Note that this is NOT inplace,
    # it creating a new object and replace what's in the state dict
    for req in bytes_read_requests:
        # Ensure the BytesIO is rewound
        req.bytes.seek(0)
        state_dict[req.fqn] = torch.load(req.bytes)

    tensor_futures.wait()


def _validate_sharded_tensor(
    tensor_md: ShardedTensorMetadata, checkpoint_md: ShardedTensorStorageMetadata
) -> None:
    # We assume the incoming tensor has being validated during construction

    # To ensure a checkpoint can satisfy loading a ST, we compute the loading
    # plans for all shards and see if they are doable.
    validate_non_overlapping_shards_metadata(
        checkpoint_md.tensor_metadata.shards_metadata
    )

    for shard_md in tensor_md.shards_metadata:
        read_volume = 0
        for storage_md in checkpoint_md.storage_metadata:
            shard_md_from_storage = storage_md.shard_metadata

            if not _check_shard_metadata_pair_overlap(shard_md, shard_md_from_storage):
                continue

            shard_volume = 1
            for (_, _, _, length,) in _shards_get_overlap_region_wrt_saved_tensor(
                saved_shard=shard_md_from_storage, current_shard=shard_md
            ):
                shard_volume *= length
            read_volume += shard_volume

        shard_volume = 1
        for size in shard_md.shard_sizes:
            shard_volume *= size
        if read_volume != shard_volume:
            raise ValueError(
                f"Shard {shard_md} only has {read_volume} available" +
                "elements but needs {shard_volume}"
            )

def validate_metadata(
    state_dict: Dict[str, Any], metadata: Metadata
) -> None:
    """
    Verify if it's possible to correctly load `state_dict` from `metadata`.

    This method validate if a checkpoint is usable with a given model
    state_dict without loading it. It will raise ValueError if it finds
    anything problematic.

    Args:
        state_dict: A state_dict to verify if it's loadable.
        metadata: Checkpoint metadata to verify against.

    Returns:
        None

    Example:
        >>> my_model: torch.nn.Model = ....
        >>> my_reader: torch.distributed._shard.checkpoint.StorageReader = ...

        >>> torch.distributed._shard.checkpoint.validate_metadata(my_model.state_dict(), my_reader.read_metadata())
        None
    ```

    """
    for fqn, obj in state_dict.items():
        if isinstance(obj, ShardedTensor):
            if fqn not in metadata.state_dict_metadata:
                raise ValueError(f"{fqn}: Could not find ShardedTensor metadata")

            md = metadata.state_dict_metadata[fqn]
            if not isinstance(md, ShardedTensorStorageMetadata):
                raise ValueError(f"{fqn}: Expected ShardedTensorStorageMetadata but found: {type(md)}")

            # Check if the overall ShardedTensor size is the same. Individual shards don't matter as we can reshard.
            md_size = list(md.tensor_metadata.size)
            tensor_size = list(obj.metadata().size)
            if md_size != tensor_size:
                raise ValueError(
                    f"{fqn}: Incompatible ShardedTensor size: expectected {tensor_size} but found {md_size}"
                )

            _validate_sharded_tensor(obj.metadata(), md)
        elif isinstance(obj, torch.Tensor):
            if fqn not in metadata.state_dict_metadata:
                raise ValueError(f"{fqn}: Could not find Tensor metadata")

            md = metadata.state_dict_metadata[fqn]
            if not isinstance(md, TensorStorageMetadata):
                raise ValueError(f"{fqn}: Expected TensorStorageMetadata but found: {type(md)}")

            if md.size != obj.size():
                raise ValueError(
                    f"{fqn}: Incompatible tensor size: expected {obj.size()} but found {md.size}"
                )
