import bisect
import itertools
import math
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharding_spec import (
    ChunkShardingSpec,
    EnumerableShardingSpec,
    ShardingSpec,
)

def _sharding_spec_to_offsets(
    sharding_spec: ShardingSpec, tensor_numel: int, world_size: int
) -> List[int]:
    r"""
    Translates the sharding spec to a list of offsets along dim 0. If the
    sharding spec is ChunkShardingSpec, only the ``dim`` is used and the
    placement is not used.
    """
    offsets: List[int] = []
    if isinstance(sharding_spec, EnumerableShardingSpec):
        for shard in sharding_spec.shards:
            offsets.append(shard.shard_offsets[0])
    elif isinstance(sharding_spec, ChunkShardingSpec):
        assert sharding_spec.dim == 0
        chunk_size = math.ceil(tensor_numel / world_size)
        if chunk_size == 1:
            offsets = [
                rank if rank < tensor_numel else tensor_numel
                for rank in range(world_size)
            ]
        else:
            offsets = [chunk_size if rank > 0 else 0 for rank in range(world_size)]
            offsets = list(itertools.accumulate(offsets))
    else:
        raise ValueError(f"Un-recognized sharding spec type {type(sharding_spec)}.")

    return offsets


def _offsets_to_split_sizes(
    input_offsets: List[int],
    output_offsets: List[int],
    tensor_numel: int,
    world_size: int,
    my_rank: int,
) -> Tuple[List[int], List[int]]:
    r"""
    Given the shard offsets for each rank of the input tensor and output tensor,
    this API returns the corresponding split sizes that can be passed to
    all_to_all_single().
    """

    def _get_interval(offsets):
        if my_rank != world_size - 1:
            return offsets[my_rank], offsets[my_rank + 1] - 1
        else:
            return offsets[my_rank], tensor_numel - 1

    def _offsets_to_sizes(offsets, begin, end):
        sizes = []
        for i, offset in enumerate(offsets):
            next_offset = offsets[i + 1] if i < len(offsets) - 1 else end + 1
            sizes.append(
                (next_offset - offset)
                - max(begin - offset, 0)
                - max(next_offset - end - 1, 0)
            )
        return sizes

    def _convert(from_offsets, to_offsets, split_sizes):
        begin, end = _get_interval(from_offsets)
        to_begin_rank = bisect.bisect(to_offsets, begin) - 1
        to_end_rank = bisect.bisect(to_offsets, end) - 1
        _split_sizes = _offsets_to_sizes(
            to_offsets[to_begin_rank : to_end_rank + 1], begin, end
        )
        split_sizes[to_begin_rank : to_end_rank + 1] = _split_sizes

    input_split_sizes = [0 for _ in range(world_size)]
    output_split_sizes = [0 for _ in range(world_size)]
    _convert(input_offsets, output_offsets, input_split_sizes)
    _convert(output_offsets, input_offsets, output_split_sizes)

    return input_split_sizes, output_split_sizes


def _reshard_flatten_tensor(
    input_tensor: ShardedTensor,
    output_spec: ShardingSpec,
    world_size: int,
    my_rank: int,
    device: torch.device,
    process_group: Optional[dist.ProcessGroup],
) -> torch.Tensor:
    """
    Resharded a sharded flatten tensor, this is used by FSDP to do sharded
    state_dict. But the functionaility is not supported by ShardedTensor.
    This API is designed to be used for FSDP; therefore this API supports only
    1-D ShardedTensor (hence the naming, reshard_flatten_tensor).

    This API uses the ChunkShardingSpec and EnumerableShardingSpec from
    torch.distributed.sharding_spec but ignores the placement field in
    ChunkShardingSpec, as the placement requires the callees understand the
    number of GPUs per node. The API simply uses the semantics of the sharding
    specs.

    Args:
        input_tensor (ShardedTensor): the original ShardedTensor. Must be 1D.
        output_spec (ShardingSpec): the sharding spect for the output tensor.
        world_size (int): total trainer count.
        my_rank (int): the rank for this trainer.

    Returns:
        The local shard for the new ShardedTensor.
    """

    input_spec = input_tensor.sharding_spec()
    size = input_tensor.size()
    if isinstance(size, int):
        raise ValueError("The input tensor has no dimensions.")
    tensor_numel = size.numel()
    input_offsets = _sharding_spec_to_offsets(input_spec, tensor_numel, world_size)
    output_offsets = _sharding_spec_to_offsets(output_spec, tensor_numel, world_size)
    input_split_sizes, output_split_sizes = _offsets_to_split_sizes(
        input_offsets, output_offsets, tensor_numel, world_size, my_rank
    )
    output_size = sum(output_split_sizes)
    local_shard = torch.empty(output_size, dtype=input_tensor.dtype, device=device)
    dist.all_to_all_single(
        local_shard,
        input_tensor.local_shards()[0].tensor,
        input_split_sizes=input_split_sizes,
        output_split_sizes=output_split_sizes,
        group=process_group,
    )
    return local_shard


def _gather_state_dict(
    state_dict: Dict[str, Any], curr_rank: int, output_rank: int = 0
) -> Dict[str, Any]:
    """
    Given a state_dict, this API gathers all the ShardedTensor in the state_dict
    to the output_rank, and creates a new state_dict which the values are either
    the gathered tensors (rank == output_rank) or None (rank != output_rank).
    """
    new_state_dict = {}
    for key, tensor in state_dict.items():
        if isinstance(tensor, ShardedTensor):
            output_tensor = (
                torch.empty(tensor.shape).cuda() if curr_rank == output_rank else None
            )
            tensor.gather(0, output_tensor)
            tensor = output_tensor
        new_state_dict[key] = tensor
    return new_state_dict
