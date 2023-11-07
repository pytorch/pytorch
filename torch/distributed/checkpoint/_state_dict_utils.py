import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed import distributed_c10d
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._tensor import DTensor, Replicate


def _all_gather_sharded_tensor(
    sharded_tensor: ShardedTensor,
    pg: Optional[dist.ProcessGroup] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if pg is None:
        pg = distributed_c10d._get_default_group()
    world_size = dist.get_world_size(pg)
    shards = sharded_tensor.local_shards()
    dim_0_size = sharded_tensor.size()[0]  # type: ignore[index]
    tensor_numel = sharded_tensor.size().numel()  # type: ignore[union-attr]
    chunk_size = math.ceil(dim_0_size / world_size) * tensor_numel // dim_0_size
    pg_device = (
        distributed_c10d._get_pg_default_device(pg) if device is None else device
    )
    if shards:
        local_tensor = shards[0].tensor.flatten()
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
    dist.all_gather_into_tensor(tensor, local_tensor, group=pg)

    tensor = tensor.narrow(0, 0, tensor_numel).reshape(sharded_tensor.size())
    return tensor


def _gather_state_dict(
    state_dict: Dict[str, Any],
    *,
    pg: Optional[dist.ProcessGroup] = None,
    device: Optional[torch.device] = None,
    cpu_offload: bool = False,
    ranks_only: Tuple[int, ...] = tuple(),
) -> Dict[str, Any]:
    """
    Given a state_dict, this API gathers all the ShardedTensors or DTensors in
    the state_dict.


    Args:
        state_dict (Dict[str, Any]): the target sharded state_dict.
        pg (Optional[dist.ProcessGroup]): the process group that is used to
            gather ShardedTensor.
        device: (Optional[torch.device]): the device that is used to
            perform allgather for ShardedTensor.
        cpu_offload (bool): whether to offload the tensors to CPU memory. The
            default value is False.
        ranks_only: (Tuple[int, ...]): if this tuple is empty, all ranks will
            have the same state_dicts. Otherwise only ranks that in ``ranks_only``
            have the same state_dicts. Other ranks will get empty state_dicts.

    Returns:
        The gathered state dictionary.
    """
    new_state_dict = {}
    cpu_device = torch.device("cpu")
    for key, value in state_dict.items():
        if isinstance(value, ShardedTensor):
            # ShardedTensor does not seem to record the original device type.
            # So if the tensor is moved to CPU, we won't know the original type.
            # As a result, we have to rely on the user to tell us the correct one.
            output_tensor = _all_gather_sharded_tensor(value, pg, device)
            local_shard_device = (
                value.local_shards()[0].tensor.device
                if value.local_shards()
                else cpu_device
            )
            if output_tensor.device != local_shard_device:
                value = output_tensor.to(local_shard_device)
            else:
                value = output_tensor
        elif isinstance(value, DTensor):
            if value.device != value.device_mesh.device_type:
                value = value.to(value.device_mesh.device_type)
            # FSDP all_gather: [Shard(0)] -> [Replicate()]
            # HSDP all_gather: [Replicate(), Shard(0)] -> [Replicate(), Replicate()]
            # 2D FSDP + TP all_gather:
            # - [Shard(0), Shard(n)] -> [Replicate(), Replicate()]
            # - [Shard(0), Replicate()] -> [Replicate(), Replicate()]
            placements = [Replicate() for _ in value.placements]
            value = value.redistribute(
                device_mesh=value.device_mesh,
                placements=placements,
            )
            value = value.to_local()
        elif isinstance(value, dict):
            value = _gather_state_dict(value, pg=pg, device=device)

        if isinstance(value, torch.Tensor) and cpu_offload:
            value = value.to(cpu_device)

        if not cpu_offload or len(ranks_only) == 0 or dist.get_rank(pg) in ranks_only:
            new_state_dict[key] = value
    return new_state_dict
