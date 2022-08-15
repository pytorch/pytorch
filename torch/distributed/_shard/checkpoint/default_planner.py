import dataclasses
import io
from typing import List, Tuple, Dict, Any, Union, cast

import torch

from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharded_tensor.metadata import TensorProperties
from torch.distributed._shard.sharded_tensor.shard import Shard

from torch.distributed._shard.sharding_spec._internals import (
    _check_shard_metadata_pair_overlap,
)
from .resharding import (
    create_default_local_save_plan,
    create_default_global_save_plan,
    create_default_local_load_plan,
    create_default_global_load_plan,
)

from .planner import (
    LoadItemType,
    SavePlanner,
    LoadPlanner,
    SavePlan,
    LoadPlan,
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
    Metadata,
    STATE_DICT_TYPE,
    STORAGE_TYPES
)

from .resharding import (
    _shards_get_overlap_region_wrt_saved_tensor
)
from .utils import (
    find_state_dict_object
)


class DefaultSavePlanner(SavePlanner):
    def init(self, state_dict: Dict[str, Any], is_coordinator: bool) -> None:
        self.state_dict = state_dict
        self.is_coordinator = is_coordinator

    def create_local_plan(self) -> SavePlan:
        self.plan = create_default_local_save_plan(self.state_dict, self.is_coordinator)
        return self.plan

    def create_global_plan(self, all_plans: List[SavePlan]) -> Tuple[List[SavePlan], Metadata]:
        self.global_plan, self.metadata = create_default_global_save_plan(all_plans)
        return self.global_plan, self.metadata

    def finish_plan(self, new_plan: SavePlan) -> SavePlan:
        self.plan = new_plan
        return new_plan

    def resolve_data(self, write_item: WriteItem) -> Union[torch.Tensor, io.BytesIO]:
        object = self.lookup_object(write_item.index)
        return self.transform_object(write_item, object)

    def lookup_object(self, index: MetadataIndex) -> Any:
        """
        This is an extension from the planner interface to make it easy to extend the default planner
        """
        return find_state_dict_object(self.state_dict, index)

    def transform_object(self, write_item: WriteItem, object: Any):
        """
        This is an extension from the planner interface to make it easy to extend the default planner
        """
        if write_item.type == WriteItemType.BYTE_IO:
            bytes = io.BytesIO()
            torch.save(object, bytes)
            object = bytes
        return object


class DefaultLoadPlanner(LoadPlanner):
    def init(self, state_dict: STATE_DICT_TYPE, metadata: Metadata, is_coordinator: bool) -> None:
        self.state_dict = state_dict
        self.metadata = metadata
        self.is_coordinator = is_coordinator

    def create_local_plan(self) -> LoadPlan:
        return create_default_local_load_plan(self.state_dict, self.metadata)

    def create_global_plan(self, global_plan: List[LoadPlan]) -> List[LoadPlan]:
        return create_default_global_load_plan(global_plan)

    def finish_plan(self, new_plan: LoadPlan) -> LoadPlan:
        return new_plan

    def load_bytes(self, read_item: ReadItem, value: io.BytesIO) -> None:
        self.state_dict[read_item.dest_index.fqn] = torch.load(value)

    def resolve_tensor(self, read_item: ReadItem):
        tensor = self.lookup_tensor(read_item.dest_index)
        return self.transform_tensor(read_item, tensor)

    def commit_tensor(self, read_item: ReadItem, tensor: torch.Tensor) -> None:
        pass

    def lookup_tensor(self, index: MetadataIndex) -> torch.Tensor:
        """
        This is an extension from the planner interface to make it easy to extend the default planner
        """
        return find_state_dict_object(self.state_dict, index)

    def transform_tensor(self, read_item: ReadItem, tensor: torch.Tensor):
        """
        This is an extension from the planner interface to make it easy to extend the default planner
        """
        return narrow_tensor_by_index(tensor, read_item.dest_offsets, read_item.lengths)


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

def _chunk_for_sharmd(shard_md: ShardMetadata) -> ChunkStorageMetadata:
    return ChunkStorageMetadata(
        offsets=torch.Size(shard_md.shard_offsets),
        sizes=torch.Size(shard_md.shard_sizes)
    )

def _sharded_tensor_metadata(sharded_tensor: ShardedTensor, shard_md: ShardMetadata) -> TensorWriteData:
    return TensorWriteData(
        chunk=_chunk_for_sharmd(shard_md),
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

def create_default_metadata_only_plan(state_dict: STATE_DICT_TYPE) -> SavePlan:
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

def create_default_local_metadata(state_dict: STATE_DICT_TYPE) -> Metadata:
    plan = create_default_metadata_only_plan(state_dict)
    _, md = create_default_global_save_plan([plan])
    return md

def create_write_items(fqn: str, object: Any) -> List[WriteItem]:
    if isinstance(object, ShardedTensor):
        return [_create_write_item_for_shard(fqn, object, shard.metadata) for shard in object.local_shards()]
    elif isinstance(object, torch.Tensor):
        return [_create_write_item_for_tensor(fqn, object)]
    else:
        return [_create_write_item_for_bytesio(fqn, object)]

def create_read_items(fqn: str, md: STORAGE_TYPES, obj: Any) -> List[ReadItem]:
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
        local_shards)

def create_default_local_load_plan(
    state_dict: Dict[str, Any],
    metadata: Metadata,
) -> LoadPlan:
    requests = []

    """
    Use the loaded metadata and the current state dict to map the saved tensors to current tensor
    """
    for fqn, obj in state_dict.items():
        md = metadata.state_dict_metadata[fqn]
        requests += create_read_items(fqn, md, obj)

    return LoadPlan(requests)

def create_default_global_load_plan(all_plans: List[LoadPlan]) -> List[LoadPlan]:
    return all_plans

def create_default_local_save_plan(state_dict: Dict[str, Any], is_coordinator: bool):
    requests = []
    for fqn, obj in state_dict.items():
        if isinstance(obj, ShardedTensor) or is_coordinator:
            requests += create_write_items(fqn, obj)
    return SavePlan(requests)

def create_default_global_save_plan(all_plans: List[SavePlan]) -> Tuple[List[SavePlan], Metadata]:
    md: Dict[str, STORAGE_TYPES] = dict()
    new_plans = []
    for plan in all_plans:
        new_items = []
        for item in plan.items:
            if not item.type == WriteItemType.SHARD:
                assert item.index.fqn not in md

            if item.type == WriteItemType.BYTE_IO:
                md[item.index.fqn] = BytesStorageMetadata()
                new_items.append(item)
            else:
                assert item.tensor_data is not None
                tensor_md = cast(
                    TensorStorageMetadata,
                    md.setdefault(item.index.fqn, TensorStorageMetadata(
                        properties=item.tensor_data.properties,
                        size=item.tensor_data.size,
                        chunks=[],
                    ))
                )
                new_index = dataclasses.replace(item.index, index=len(tensor_md.chunks))
                new_item = dataclasses.replace(item, index=new_index)
                new_items.append(new_item)

                assert item.tensor_data.chunk is not None, f"Cannot create MD for tensor without bounds. FQN: {item.index.fqn}"
                tensor_md.chunks.append(item.tensor_data.chunk)
        new_plans.append(dataclasses.replace(plan, items=new_items))
    return (new_plans, Metadata(md))
