from typing import Any, List

import torch

from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharded_tensor.metadata import TensorProperties
from torch.distributed._tensor import DTensor
from torch.distributed._tensor._utils import compute_local_shape_and_global_offset

from .metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    MetadataIndex,
    STATE_DICT_TYPE,
    STORAGE_TYPES,
    TensorStorageMetadata,
)

from .planner import (
    LoadItemType,
    ReadItem,
    SavePlan,
    TensorWriteData,
    WriteItem,
    WriteItemType,
)

from .resharding import (
    _check_shard_metadata_pair_overlap,
    _shards_get_overlap_region_wrt_saved_tensor,
)

__all__: List[str] = ["create_read_items_for_chunk_list"]


def _create_chunk_from_tensor(tensor: torch.Tensor) -> ChunkStorageMetadata:
    return ChunkStorageMetadata(
        offsets=torch.Size([0] * len(tensor.size())), sizes=tensor.size()
    )


def _chunk_for_shard(shard_md: ShardMetadata) -> ChunkStorageMetadata:
    return ChunkStorageMetadata(
        offsets=torch.Size(shard_md.shard_offsets),
        sizes=torch.Size(shard_md.shard_sizes),
    )


def _sharded_tensor_metadata(
    sharded_tensor: ShardedTensor, shard_md: ShardMetadata
) -> TensorWriteData:
    return TensorWriteData(
        chunk=_chunk_for_shard(shard_md),
        properties=sharded_tensor.metadata().tensor_properties,
        size=sharded_tensor.metadata().size,
    )


def _create_write_items_for_dtensor(fqn: str, tensor: DTensor) -> WriteItem:
    sizes, offsets = compute_local_shape_and_global_offset(
        tensor.shape, tensor.device_mesh, tensor.placements
    )
    sizes, offsets = torch.Size(sizes), torch.Size(offsets)

    return WriteItem(
        index=MetadataIndex(fqn, offsets),
        type=WriteItemType.SHARD,
        tensor_data=TensorWriteData(
            chunk=ChunkStorageMetadata(
                offsets=offsets,
                sizes=sizes,
            ),
            # TODO:update this to not use TensorProperties from ST.
            properties=TensorProperties.create_from_tensor(tensor.to_local()),
            size=tensor.size(),
        ),
    )


def _create_write_item_for_shard(
    fqn: str, sharded_tensor: ShardedTensor, shard_md: ShardMetadata
) -> WriteItem:
    offsets = torch.Size(shard_md.shard_offsets)
    return WriteItem(
        index=MetadataIndex(fqn, offsets),
        type=WriteItemType.SHARD,
        tensor_data=_sharded_tensor_metadata(sharded_tensor, shard_md),
    )


def _create_write_item_for_tensor(fqn: str, tensor: torch.Tensor) -> WriteItem:
    offsets = torch.Size([0] * len(tensor.size()))
    return WriteItem(
        index=MetadataIndex(fqn, offsets),
        type=WriteItemType.TENSOR,
        tensor_data=TensorWriteData(
            chunk=ChunkStorageMetadata(offsets=offsets, sizes=tensor.size()),
            properties=TensorProperties.create_from_tensor(tensor),
            size=tensor.size(),
        ),
    )


def _create_write_item_for_bytesio(fqn: str, bytes: Any):
    return WriteItem(
        index=MetadataIndex(fqn),
        type=WriteItemType.BYTE_IO,
    )


def _create_read_item_for_byteio(
    dest_index, dest_offset, storage_index, storage_offset, length
):
    return ReadItem(
        type=LoadItemType.BYTE_IO,
        dest_index=dest_index,
        dest_offsets=torch.Size((dest_offset,)),
        storage_index=storage_index,
        storage_offsets=torch.Size((storage_offset,)),
        lengths=torch.Size((length,)),
    )


def _create_read_item_for_tensor(
    dest_index, dest_offsets, storage_index, storage_offsets, lengths
):
    return ReadItem(
        type=LoadItemType.TENSOR,
        dest_index=dest_index,
        dest_offsets=torch.Size(dest_offsets),
        storage_index=storage_index,
        storage_offsets=torch.Size(storage_offsets),
        lengths=torch.Size(lengths),
    )


def create_read_items_for_chunk_list(
    fqn: str,
    checkpoint_md: TensorStorageMetadata,
    local_chunks: List[ChunkStorageMetadata],
) -> List[ReadItem]:
    """
    Create a list of ``ReadItem`` based on the checkpoint and local chunks.

    This applies the resharding algorithm and computes the reads needed
    to satisfy ``local_chunks`` with a checkpoint described by ``checkpoint_md``.

    Args:
        fqn (str) : The state_dict FQN to pass to ``ReadItem``.
        checkpoint_md (TensorStorageMetadata): metadata for a given tensor
            from a checkpoint.
        local_chunks (List[ChunkStorageMetadata]): Local chunks that needs to be
            loaded.

    Returns:
        A list of ``ReadItem`` that will satisfy all input chunks.
    """
    read_items = []
    # this is a naive quadratic algo that can be optimized later
    for idx, shard in enumerate(local_chunks):
        for storage_idx, storage_md in enumerate(checkpoint_md.chunks):
            if not _check_shard_metadata_pair_overlap(shard, storage_md):
                continue

            storage_offsets = []
            dest_offsets = []
            lengths = []
            for (
                dim,
                offset_for_saved_tensor,
                offset_for_current_tensor,
                length,
            ) in _shards_get_overlap_region_wrt_saved_tensor(
                saved_shard=storage_md, current_shard=shard
            ):
                storage_offsets.append(offset_for_saved_tensor)
                dest_offsets.append(offset_for_current_tensor)
                lengths.append(length)

            read_items.append(
                _create_read_item_for_tensor(
                    dest_index=MetadataIndex(fqn, shard.offsets, idx),
                    dest_offsets=dest_offsets,
                    storage_index=MetadataIndex(fqn, storage_md.offsets, storage_idx),
                    storage_offsets=storage_offsets,
                    lengths=lengths,
                )
            )
    return read_items


def _create_default_metadata_only_plan(state_dict: STATE_DICT_TYPE) -> SavePlan:
    requests = []
    for fqn, obj in state_dict.items():
        if isinstance(obj, DTensor):
            requests.append(_create_write_items_for_dtensor(fqn, obj))
        elif isinstance(obj, ShardedTensor):
            for shard_md in obj.metadata().shards_metadata:
                requests.append(_create_write_item_for_shard(fqn, obj, shard_md))
        elif isinstance(obj, torch.Tensor):
            requests.append(_create_write_item_for_tensor(fqn, obj))
        else:
            requests.append(_create_write_item_for_bytesio(fqn, obj))
    return SavePlan(requests)


def _create_write_items(fqn: str, object: Any) -> List[WriteItem]:
    if isinstance(object, DTensor):
        return [_create_write_items_for_dtensor(fqn, object)]
    elif isinstance(object, ShardedTensor):
        return [
            _create_write_item_for_shard(fqn, object, shard.metadata)
            for shard in object.local_shards()
        ]
    elif isinstance(object, torch.Tensor):
        return [_create_write_item_for_tensor(fqn, object)]
    else:
        return [_create_write_item_for_bytesio(fqn, object)]


def _create_chunk_from_dtensor(tensor: DTensor) -> ChunkStorageMetadata:
    sizes, offsets = compute_local_shape_and_global_offset(
        tensor.shape, tensor.device_mesh, tensor.placements
    )
    sizes, offsets = torch.Size(sizes), torch.Size(offsets)
    return ChunkStorageMetadata(
        offsets=offsets,
        sizes=sizes,
    )


def _create_read_items(fqn: str, md: STORAGE_TYPES, obj: Any) -> List[ReadItem]:
    if not isinstance(md, BytesStorageMetadata):
        if isinstance(obj, DTensor):
            local_chunks = [_create_chunk_from_dtensor(obj)]
        elif isinstance(obj, ShardedTensor):
            local_chunks = [
                _chunk_for_shard(shard.metadata) for shard in obj.local_shards()
            ]
        elif isinstance(obj, torch.Tensor):
            local_chunks = [_create_chunk_from_tensor(obj)]
        else:
            raise ValueError(
                f"Invalid checkpoint metadata for {fqn}, "
                + f"expected BytesStorageMetadata but found {type(md)}"
            )
        return create_read_items_for_chunk_list(fqn, md, local_chunks)
    else:
        return [
            _create_read_item_for_byteio(
                dest_index=MetadataIndex(fqn),
                dest_offset=0,
                storage_index=MetadataIndex(fqn),
                storage_offset=0,
                length=0,
            )
        ]
