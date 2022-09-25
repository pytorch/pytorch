from typing import List, Any

import torch

from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharded_tensor.metadata import TensorProperties
from torch.distributed._shard.sharded_tensor.shard import Shard

from torch.distributed._shard.sharding_spec._internals import (
    _check_shard_metadata_pair_overlap,
)

from .planner import (
    LoadItemType,
    SavePlan,
    ReadItem,
    WriteItem,
    WriteItemType,
    TensorWriteData,
)

from .metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    TensorStorageMetadata,
    MetadataIndex,
    STATE_DICT_TYPE,
    STORAGE_TYPES
)

from .resharding import (
    _shards_get_overlap_region_wrt_saved_tensor
)

def _create_shard_metadata(size: torch.Size) -> ShardMetadata:
    return ShardMetadata(
        shard_offsets=[0] * len(size),
        shard_sizes=list(size),
    )

def _create_shard_from_tensor(tensor: torch.Tensor) -> Shard:
    return Shard(
        tensor=tensor,
        metadata=_create_shard_metadata(tensor.size())
    )

def _chunk_for_shard(shard_md: ShardMetadata) -> ChunkStorageMetadata:
    return ChunkStorageMetadata(
        offsets=torch.Size(shard_md.shard_offsets),
        sizes=torch.Size(shard_md.shard_sizes)
    )

def _sharded_tensor_metadata(sharded_tensor: ShardedTensor, shard_md: ShardMetadata) -> TensorWriteData:
    return TensorWriteData(
        chunk=_chunk_for_shard(shard_md),
        properties=sharded_tensor.metadata().tensor_properties,
        size=sharded_tensor.metadata().size,
    )

def _create_write_item_for_shard(fqn: str, sharded_tensor: ShardedTensor, shard_md: ShardMetadata) -> WriteItem:
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
            chunk=ChunkStorageMetadata(
                offsets=offsets,
                sizes=tensor.size()
            ),
            properties=TensorProperties.create_from_tensor(tensor),
            size=tensor.size(),
        )
    )

def _create_write_item_for_bytesio(fqn: str, bytes: Any):
    return WriteItem(
        index=MetadataIndex(fqn),
        type=WriteItemType.BYTE_IO,
    )

def _create_read_item_for_byteio(dest_index, dest_offset, storage_index, storage_offset, length):
    return ReadItem(
        type=LoadItemType.BYTE_IO,
        dest_index=dest_index,
        dest_offsets=torch.Size((dest_offset,)),
        storage_index=storage_index,
        storage_offsets=torch.Size((storage_offset,)),
        lengths=torch.Size((length,)),
    )

def _create_read_item_for_tensor(dest_index, dest_offsets, storage_index, storage_offsets, lengths):
    return ReadItem(
        type=LoadItemType.TENSOR,
        dest_index=dest_index,
        dest_offsets=torch.Size(dest_offsets),
        storage_index=storage_index,
        storage_offsets=torch.Size(storage_offsets),
        lengths=torch.Size(lengths),
    )

def _create_sharded_read_items(
    fqn: str,
    checkpoint_md: TensorStorageMetadata,
    local_shards: List[Shard],
) -> List[ReadItem]:

    read_items = []
    # this is a naive quadratic algo that can be optimized later
    for idx, shard in enumerate(local_shards):
        for storage_idx, storage_md in enumerate(checkpoint_md.chunks):
            shard_md_from_storage = ShardMetadata(
                shard_sizes=list(storage_md.sizes),
                shard_offsets=list(storage_md.offsets),
            )

            if not _check_shard_metadata_pair_overlap(
                shard.metadata, shard_md_from_storage
            ):
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
                saved_shard=shard_md_from_storage, current_shard=shard.metadata
            ):
                storage_offsets.append(offset_for_saved_tensor)
                dest_offsets.append(offset_for_current_tensor)
                lengths.append(length)

            read_items.append(
                _create_read_item_for_tensor(
                    dest_index=MetadataIndex(fqn, shard.metadata.shard_offsets, idx),
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
        if isinstance(obj, ShardedTensor):
            for shard_md in obj.metadata().shards_metadata:
                requests.append(_create_write_item_for_shard(fqn, obj, shard_md))
        elif isinstance(obj, torch.Tensor):
            requests.append(_create_write_item_for_tensor(fqn, obj))
        else:
            requests.append(_create_write_item_for_bytesio(fqn, obj))
    return SavePlan(requests)

def _create_write_items(fqn: str, object: Any) -> List[WriteItem]:
    if isinstance(object, ShardedTensor):
        return [_create_write_item_for_shard(fqn, object, shard.metadata) for shard in object.local_shards()]
    elif isinstance(object, torch.Tensor):
        return [_create_write_item_for_tensor(fqn, object)]
    else:
        return [_create_write_item_for_bytesio(fqn, object)]

def _create_read_items(fqn: str, md: STORAGE_TYPES, obj: Any) -> List[ReadItem]:
    if isinstance(md, BytesStorageMetadata):
        return [_create_read_item_for_byteio(
            dest_index=MetadataIndex(fqn),
            dest_offset=0,
            storage_index=MetadataIndex(fqn),
            storage_offset=0,
            length=0
        )]
    elif isinstance(obj, ShardedTensor):
        local_shards = obj.local_shards()
    elif isinstance(obj, torch.Tensor):
        local_shards = [_create_shard_from_tensor(obj)]
    else:
        raise ValueError(
            f"Invalid checkpoint metadata for {fqn}, " +
            f"expected BytesStorageMetadata but found {type(md)}"
        )

    return _create_sharded_read_items(
        fqn,
        md,
        local_shards
    )
