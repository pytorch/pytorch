from typing import Any, Callable, Dict, List, Tuple
import torch
from torch import Tensor
from .metadata import Metadata, ReadWriteRequest, ExtendedTensorMetadata, StorageMetadata
from .storage_writer import StorageWriter
from torch.futures import Future
from torch.distributed._sharded_tensor import (
    ShardedTensor,
    ShardedTensorMetadata,
    TensorProperties,
    ShardMetadata,
)
import torch.distributed as dist

# -------------- private functions --------------
def _populate_inplace_with_a_tensor(
    fqn: str,
    tensor: Tensor,
    metadata: Metadata,
    size_for_storage_handles: Dict[str, int],
    write_requests: List[ReadWriteRequest],
):
    # --- Step 1, populate write request ---
    # The reshaping should not be required here
    target_tensor = torch.reshape(tensor, (-1,))
    storage_size = target_tensor.nelement() * target_tensor.element_size()

    wr = ReadWriteRequest(
        target_tensor=target_tensor,
        storage_key=fqn,
        offset=0,
        length=storage_size,
    )

    write_requests.append(wr)

    # --- Step 2, populate the size_for_storage_handles ---
    #
    size_for_storage_handles[fqn] = storage_size

    # --- Step 3, populate the metadata ---
    #
    # Since torch.Tensor does not have a standard set of metadata we can operate on
    # We wrap troch.Tensor's metadata with ShardMetadata
    # This is frankly a bad idea, I will need to change this
    tensor_size = list(tensor.detach().size())
    shard_metadata = ShardMetadata(
        shard_offsets=[0] * len(tensor_size),
        shard_sizes=tensor_size,
        # Not entirely sure how to deal with placement for regular tensor yet
        # Since they are not shared, it makes sense to assume they are always replicated
        #
        placement=tensor.device,
    )
    stm = ShardedTensorMetadata(
        shards_metadata=[shard_metadata],
        size=tensor.size(),
        tensor_properties=TensorProperties(
            dtype=tensor.dtype,
            layout=tensor.layout,
            requires_grad=tensor.requires_grad,
            memory_format=torch.contiguous_format,
            pin_memory=tensor.is_pinned(),
        )
    )

    etmd = ExtendedTensorMetadata(
        tensor_metadata=stm,
        storage_metadata=[StorageMetadata(
            shard_metadata=shard_metadata,
            storage_key=fqn,
            length=storage_size,
            offset=0,
        )]
    )
    metadata.state_dict_metadata[fqn] = etmd

def _populate_inplace_with_a_sharded_tensor(
    fqn: str,
    tensor: ShardedTensor,
    metadata: Metadata,
    size_for_storage_handles: Dict[str, int],
    write_requests: List[ReadWriteRequest],
):
    smd = []
    for shard in tensor.local_shards():
        # each shard is in it own file.
        # Most network file system is optimized with single write, multiple read
        # Unless we can group tensors locally into one big chunk
        # It might be best to write each shard as one key
        suffix = "_".join([str(i) for i in shard.metadata.shard_offsets + shard.metadata.shard_sizes])
        storage_key = f"{fqn}_{suffix}"

        # The reshaping is not required here
        target_tensor = shard.tensor.detach().reshape(-1,)
        storage_size = target_tensor.nelement() * target_tensor.element_size()

        one_smd = StorageMetadata(
            shard_metadata=shard.metadata,
            storage_key=storage_key,
            length=storage_size,
            offset=0,
        )
        smd.append(one_smd)

        size_for_storage_handles[storage_key] = storage_size

        wr = ReadWriteRequest(
            target_tensor=target_tensor,
            storage_key=storage_key,
            offset=0,
            length=storage_size,
        )
        write_requests.append(wr)


    etmd = ExtendedTensorMetadata(
        tensor_metadata=tensor.metadata(),
        storage_metadata=smd,
    )
    metadata.state_dict_metadata[fqn] = etmd


def _prepare(state_dict: Dict[str, Any]) -> Tuple[Metadata, Dict[str, int], List[ReadWriteRequest]]:
    """
    Uses the state_dict to build three things.

    metadata: Metadata
        The metatdata discribing the tensor / sharded tensor.
        And it is storage meta data. See "../metadata.py" for detail

    size_for_storage_handles: Dict[str, int]
        Key is the handle name, value is the handle's size
        It can used to pre allocate the storage for parallel and non sequential writes.

    write_requests: List[ReadWriteRequest]
        List of write requests that should br perfromed by the writer.


    Subclasses can optionally overwrite the implementation here,
    if the default does not meet its requirement.
    """
    metadata = Metadata(state_dict_metadata={})
    storage_handles = {}
    write_requests = []

    for fqn, tensor in state_dict.items():
        if isinstance(tensor, Tensor):
            # The assumption is that non ShardedTensors are full replicated across all ranks
            # So we just need one from Rank 0.
            # If that's not the case, we will update later.
            if dist.is_initialized() and dist.get_rank() != 0:
                pass
            else:
                _populate_inplace_with_a_tensor(fqn, tensor, metadata, storage_handles, write_requests)
        elif isinstance(tensor, ShardedTensor):
            _populate_inplace_with_a_sharded_tensor(fqn, tensor, metadata, storage_handles, write_requests)
        else:
            raise RuntimeError("The input need to be either Tensor or ShardedTensor")

    return (metadata, storage_handles, write_requests)


# These two public functions defined the default behavior to save a state_dict
# Note this is a WIP, the state_dict save with different version of the code might not be
# compatible.
#
# This code defines/determines these schema
# 1. The metadata that discribe the state_dict.
# 2. How we map each tensor/sharded_tensor to storage handles.
def save_state_dict(
    state_dict: Dict[str, Any],
    storage_writer: StorageWriter,
    metadata_prepare_fn : Callable = _prepare,
) -> None:
    """
    The same as save_state_dict_async, but blocking until everything is finished.

    Args:
        state_dict (Dict[str, Any]) : A state_dict
        storage_writer (StorageWriter): An instance of storage writer that
            performance the writes.
        metadata_prepare_fn (Callable): The function what creates
            the metadata, write request and others.
    """
    futures = save_state_dict_async(
        state_dict=state_dict,
        storage_writer=storage_writer,
        metadata_prepare_fn=metadata_prepare_fn,
    )
    torch.futures.wait_all(futures)


def save_state_dict_async(
    state_dict: Dict[str, Any],
    storage_writer: StorageWriter,
    metadata_prepare_fn: Callable = _prepare,
) -> List[Future]:
    """
    Write the state dict with the storage_writer, and return futures to wait on.

    Args:
        state_dict (Dict[str, Any]) : A state_dict
        storage_writer (StorageWriter): An instance of storage writer that
            performance the writes.
        metadata_prepare_fn (Callable): The function what creates
            the metadata, write request and others.
    """
    (metadata, storage_handles, write_requests) = metadata_prepare_fn(state_dict=state_dict)
    storage_writer.prepare_storage(storage_handles=storage_handles)
    storage_writer.write_metadata(metadata=metadata)
    futures = [storage_writer.write(wr) for wr in write_requests]
    return futures
