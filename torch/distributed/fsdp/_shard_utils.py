import copy
import itertools
import math
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed import distributed_c10d
from torch.distributed._shard.sharded_tensor import (
    Shard,
    ShardedTensor,
    ShardedTensorMetadata,
    TensorProperties,
)
from torch.distributed._shard.sharding_spec import ShardMetadata
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard as DShard
from torch.distributed.fsdp._debug_utils import SimpleProfiler


def _all_gather_sharded_tensor(
    sharded_tensor: ShardedTensor, pg: Optional[dist.ProcessGroup] = None
) -> torch.Tensor:
    if pg is None:
        pg = distributed_c10d._get_default_group()
    world_size = dist.get_world_size(pg)
    shards = sharded_tensor.local_shards()
    dim_0_size = sharded_tensor.size()[0]  # type: ignore[index]
    tensor_numel = sharded_tensor.size().numel()  # type: ignore[union-attr]
    chunk_size = math.ceil(dim_0_size / world_size) * tensor_numel // dim_0_size
    pg_device = distributed_c10d._get_pg_default_device(pg)
    if shards:
        local_tensor = shards[0].tensor.flatten()
        with SimpleProfiler.profile(SimpleProfiler.Type.D2H):
            if local_tensor.device.type != pg_device.type:
                local_tensor = local_tensor.to(pg_device)
        num_padding = chunk_size - local_tensor.numel()
        if num_padding > 0:
            local_tensor = F.pad(local_tensor, [0, num_padding])
    else:
        local_tensor = torch.zeros(
            chunk_size, dtype=sharded_tensor.dtype, device=pg_device
        )

    tensor = torch.empty(
        chunk_size * world_size,
        dtype=local_tensor.dtype,
        device=pg_device,
    )
    dist._all_gather_base(tensor, local_tensor, group=pg)

    tensor = tensor.narrow(0, 0, tensor_numel).reshape(sharded_tensor.size())
    return tensor


# TODO: Make this API work for both FSDP, and 2D. Move it outside of FSDP.
# External users are interesting in using this API.
def _gather_state_dict(
    state_dict: Dict[str, Any],
    pg: Optional[dist.ProcessGroup] = None,
) -> Dict[str, Any]:
    """
    Given a state_dict, this API gathers all the ShardedTensors or DTensors in the state_dict.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if isinstance(value, ShardedTensor):
            output_tensor = _all_gather_sharded_tensor(value, pg)
            local_shard_device = (
                value.local_shards()[0].tensor.device
                if value.local_shards()
                else torch.device("cpu")
            )
            with SimpleProfiler.profile(SimpleProfiler.Type.H2D):
                if output_tensor.device != local_shard_device:
                    value = output_tensor.to(local_shard_device)
                else:
                    value = output_tensor
        elif isinstance(value, DTensor):
            if value.device != value.device_mesh.device_type:
                value = value.to(value.device_mesh.device_type)
            # FSDP all_gather: [Shard(0)] -> [Replicate()]
            # HSDP all_gather: [Replicate(), Shard(0)] -> [Replicate(), Replicate()]
            placements = list(copy.deepcopy(value.placements))
            placements[-1] = Replicate()
            value = value.redistribute(
                device_mesh=value.device_mesh,
                placements=placements,
            )
            value = value.to_local()
        elif isinstance(value, dict):
            value = _gather_state_dict(value, pg)

        new_state_dict[key] = value
    return new_state_dict


def _get_remove_device_str(rank, device_type, num_devices_per_node):
    if device_type.lower() == "cpu":
        return f"rank:{rank}/{device_type}"
    else:
        return f"rank:{rank}/{device_type}:{rank % num_devices_per_node}"


def _create_chunk_sharded_tensor(
    tensor: torch.Tensor,
    rank: int,
    world_size: int,
    num_devices_per_node: int,
    pg: dist.ProcessGroup,
) -> ShardedTensor:
    """
    Shard a tensor to chunks along the first dimension. The local rank will gets its
    corresponding chunk as the local shard to create a ShardedTensor.
    """
    chunks = tensor.chunk(world_size, dim=0)
    if len(chunks) > rank:
        local_shard = chunks[rank].clone()
        offsets = [0 for _ in tensor.size()]
        offsets[0] = math.ceil(tensor.size()[0] / world_size) * rank
        local_shards = [Shard.from_tensor_and_offsets(local_shard, offsets, rank)]
    else:
        local_shards = []

    # Create a ShardedTensor without invoking communication.
    chunk_sizes = [list(chunk.size()) for chunk in chunks]
    dim0_offsets = [0] + list(
        itertools.accumulate([chunk_size[0] for chunk_size in chunk_sizes])
    )[:-1]
    offsets = [0] * (len(chunk_sizes[0]) - 1)
    chunk_offsets = [[d0] + offsets for d0 in dim0_offsets]
    device_type = distributed_c10d._get_pg_default_device(pg).type
    placements = [
        _get_remove_device_str(r, device_type, num_devices_per_node)
        for r in range(len(chunk_sizes))
    ]
    assert len(chunk_sizes) == len(chunk_offsets) == len(placements)
    shard_metadata = [
        ShardMetadata(offset, size, placement)
        for offset, size, placement in zip(chunk_offsets, chunk_sizes, placements)
    ]
    sharded_tensor_metadata = ShardedTensorMetadata(
        shards_metadata=shard_metadata,
        size=tensor.size(),
        tensor_properties=TensorProperties(
            dtype=tensor.dtype,
            layout=tensor.layout,
            requires_grad=False,
            memory_format=torch.contiguous_format,
            pin_memory=tensor.is_pinned(),
        ),
    )
    return ShardedTensor._init_from_local_shards_and_global_metadata(
        local_shards, sharded_tensor_metadata=sharded_tensor_metadata, process_group=pg
    )


def _create_chunk_dtensor(
    tensor: torch.Tensor,
    rank: int,
    device_mesh: DeviceMesh,
) -> DTensor:
    """
    Shard a tensor to chunks along the first dimension. The local rank will gets its
    corresponding chunk as the local tensor to create a DTensor.
    """
    inner_dim = device_mesh.ndim - 1
    shard_placement = DShard(0)
    tensor_list, _ = shard_placement._split_tensor(
        tensor,
        device_mesh.size(dim=inner_dim),
        with_padding=False,
        contiguous=True,
    )
    # We need to explicitly call .clone() here as tensor.chunks() splits a tensor into the specified number of chunks.
    # Each chunk is a view of the input tensor. If the original tensor change, the view will also be changed.
    # We need to explicitly call .detach() to return a new tensor detached from the current graph.
    local_tensor = tensor_list[rank].clone().detach()

    # FSDP placements: [Shard(0)]
    # HSDP placements: [Replicate(), Shard(0)]
    placements = [Replicate() for _ in range(device_mesh.ndim)]
    placements[-1] = shard_placement  # type: ignore[call-overload]
    return DTensor.from_local(local_tensor, device_mesh, placements)
