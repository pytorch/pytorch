# mypy: allow-untyped-defs
import io
from collections.abc import Callable
from typing import Any, cast

import torch
import torch.distributed as dist
from torch._utils import _get_device_module
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from .metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    MetadataIndex,
    STATE_DICT_TYPE,
    STORAGE_TYPES,
    TensorProperties,
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


__all__: list[str] = ["create_read_items_for_chunk_list"]


def _compare_save_plans(plan: SavePlan, other_plan: SavePlan) -> bool:
    """
    Compare the two Save plans and return True if they are equal.

    Args:
        plan (SavePlan): First SavePlan to compare.
        other_plan (SavePlan): Second SavePlan to compare.

    Returns:
       True if the two plans are equal, False otherwise.
    """
    if plan.usable != other_plan.usable:
        return False

    # Both the plans should have the same number of items
    if len(plan.items) != len(other_plan.items):
        return False

    # Both the plans should have the same write items.
    for plan_item, other_plan_item in zip(plan.items, other_plan.items):
        # Write item type should be same
        if plan_item.type != other_plan_item.type:
            return False

        plan_metadata_index = plan_item.index
        other_plan_metadata_index = other_plan_item.index

        # Write item metadata_index should be same
        if (
            plan_metadata_index.fqn != other_plan_metadata_index.fqn
            or plan_metadata_index.offset != other_plan_metadata_index.offset
            or plan_metadata_index.index != other_plan_metadata_index.index
        ):
            return False

        # Write item tensor_data should be present in both the write items plans, if it exists in either of them.
        tensor_data = plan_item.tensor_data
        other_tensor_data = other_plan_item.tensor_data
        if (tensor_data and not other_tensor_data) or (
            not tensor_data and other_tensor_data
        ):
            return False

        if tensor_data and other_tensor_data:
            # Write item tensor_data size should be same
            if tensor_data.size != other_tensor_data.size:
                return False

            # Write item tensor_data chunk should be present in both the write items, if it exists in either of them.
            chunk = tensor_data.chunk
            other_chunk = other_tensor_data.chunk
            if (chunk and not other_chunk) or (not chunk and other_chunk):
                return False

            # Write item tensor_data chunk offsets and sizes should be same
            if chunk and other_chunk:
                if (
                    chunk.offsets != other_chunk.offsets
                    or chunk.sizes != other_chunk.sizes
                ):
                    return False

    return True


def _contains_usable_plan(delta_plans: list[SavePlan]) -> bool:
    """
    Check if any delta plan is usable, indicating the plan has changed.

    Args:
        delta_plans (List[SavePlan]): A list of delta plans to check.
    Returns:
        True if any delta plan is usable, False otherwise.
    """
    return any(delta_plan and delta_plan.usable for delta_plan in delta_plans)


def _merge_delta_local_plans(
    cached_plans: list[SavePlan],
    delta_plans: list[SavePlan],
) -> list[SavePlan]:
    """
    Merge a list of delta plans into a single plan.

    Args:
        cached_plans (List[SavePlan]): A list of cached plans.
        delta_plans (List[SavePlan]): A list of delta plans to merge. It can contain empty plans

    Returns:
        A single merged plan. If a delta plan is not usable, use the cached plan. Otherwise, use the delta plan.
    """
    merged_plans = []

    for cached_plan, delta_plan in zip(cached_plans, delta_plans):
        if delta_plan and not delta_plan.usable:
            merged_plans.append(cached_plan)
        else:
            merged_plans.append(delta_plan)

    return merged_plans


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
    shard_properties = sharded_tensor.metadata().tensor_properties

    properties = TensorProperties(
        dtype=shard_properties.dtype,
        layout=shard_properties.layout,
        requires_grad=shard_properties.requires_grad,
        memory_format=shard_properties.memory_format,
        pin_memory=shard_properties.pin_memory,
    )

    return TensorWriteData(
        chunk=_chunk_for_shard(shard_md),
        properties=properties,
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
    local_chunks: list[ChunkStorageMetadata],
) -> list[ReadItem]:
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
                _dim,
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
            requests.extend(
                _create_write_item_for_shard(fqn, obj, shard_md)
                for shard_md in obj.metadata().shards_metadata
            )
        elif isinstance(obj, torch.Tensor):
            requests.append(_create_write_item_for_tensor(fqn, obj))
        else:
            requests.append(_create_write_item_for_bytesio(fqn, obj))
    return SavePlan(requests)


def _create_write_items(fqn: str, object: Any) -> list[WriteItem]:
    if hasattr(object, "__create_write_items__"):
        # DTensor implements _Checkpointable
        return object.__create_write_items__(fqn, object)
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


def _create_chunk_list(tensor: torch.Tensor) -> list[ChunkStorageMetadata]:
    if hasattr(tensor, "__create_chunk_list__"):
        # DTensor implements _Checkpointable
        local_chunks = tensor.__create_chunk_list__()  # type: ignore[attr-defined]
    elif isinstance(tensor, ShardedTensor):
        local_chunks = [
            _chunk_for_shard(shard.metadata) for shard in tensor.local_shards()
        ]
    elif isinstance(tensor, torch.Tensor):
        local_chunks = [_create_chunk_from_tensor(tensor)]
    else:
        raise ValueError(
            "Unsupported Type, expecting one of [Tensor, DTensor, ShardedTensor] "
            f",but got {type(tensor)}"
        )

    return local_chunks


def _create_read_items(fqn: str, md: STORAGE_TYPES, obj: Any) -> list[ReadItem]:
    if not isinstance(md, BytesStorageMetadata):
        try:
            local_chunks = _create_chunk_list(obj)
        except ValueError as ex:
            raise ValueError(
                f"Invalid checkpoint metadata for {fqn}, "
                + f"expected BytesStorageMetadata but found {type(md)}",
            ) from ex

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


def _init_state_dict(state_dict: dict[str, Any]) -> Any:
    """
    Initializes meta tensor if the meta tensor is DTensor or torch.Tensor.
    """

    def dtensor_func(value: DTensor):
        device = getattr(value, "device", None)
        if device == torch.device("meta"):
            device_type = dist.distributed_c10d._get_pg_default_device().type
            device = cast(
                torch.device, _get_device_module(device_type).current_device()
            )
            new_local_tensor = torch.empty_like(value.to_local(), device=device)
            # We need to pass shape and stride explicitly, since DTensor might be
            # sharded unevenly.
            dtensor = DTensor.from_local(
                new_local_tensor,
                device_mesh=value.device_mesh,
                placements=value.placements,
                shape=value.size(),
                stride=value.stride(),
            )
            return dtensor
        else:
            return value

    def sharded_tensor_func(value: Any):
        device = getattr(value, "device", None)
        if device == torch.device("meta"):
            raise RuntimeError(
                f"Found unsupported type {type(value)} for meta device loading."
            )
        else:
            return value

    def tensor_func(value: torch.Tensor):
        device = getattr(value, "device", None)
        if device == torch.device("meta"):
            device_type = dist.distributed_c10d._get_pg_default_device().type
            device = cast(
                torch.device, _get_device_module(device_type).current_device()
            )
            tensor = torch.empty_like(value, device=device)
            return tensor
        else:
            return value

    _iterate_state_dict(
        state_dict,
        dtensor_func,
        sharded_tensor_func,
        tensor_func,
    )


def _iterate_state_dict(
    iter_object: Any,
    dtensor_func: Callable,
    sharded_tensor_func: Callable,
    tensor_func: Callable,
):
    """
    Iterate through the state dict, applying the given functions to each tensor type
    and update the state dict in place.

    Args:
        iter_object (Any): the target state_dict.
        sharded_tensor_func (Callable): the function to apply to ShardedTensor
        dtensor_func (Callable): the function to apply to DTensor
        tensor_func (Callable): the function to apply to Tensor

    # TODO: let state_dict_util._iterate_state_dict() to support in place option
    so we don't need to have two versions of _iterate_state_dict.
    """

    if isinstance(iter_object, DTensor):
        return dtensor_func(iter_object)
    elif isinstance(iter_object, ShardedTensor):
        return sharded_tensor_func(iter_object)
    elif isinstance(iter_object, torch.Tensor):
        return tensor_func(iter_object)
    elif (
        isinstance(iter_object, (int, float, str, bytes, io.BytesIO))
        or iter_object is None
    ):
        return iter_object
    elif isinstance(iter_object, dict):
        for key, value in iter_object.items():
            iter_object[key] = _iterate_state_dict(
                value, dtensor_func, sharded_tensor_func, tensor_func
            )
        return iter_object
    elif isinstance(iter_object, (list, tuple)):
        ret = [
            _iterate_state_dict(v, dtensor_func, sharded_tensor_func, tensor_func)
            for v in iter_object
        ]
        if isinstance(iter_object, tuple):
            ret = tuple(ret)  # type: ignore[assignment]
        return ret
