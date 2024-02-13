import math
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed._functional_collectives import AsyncCollectiveTensor

if dist.is_available() or TYPE_CHECKING:
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


def _iterate_state_dict(
    iter_object: Any,
    sharded_tensor_func: Callable,
    dtensor_func: Callable,
    *,
    pg: Optional[dist.ProcessGroup] = None,
    device: Optional[torch.device] = None,
    cpu_offload: bool = False,
    ranks_only: Tuple[int, ...] = tuple(),
) -> Dict[str, Any]:
    # TODO: should we use pytree?
    cpu_device = torch.device("cpu")
    if isinstance(iter_object, ShardedTensor):
        ret = sharded_tensor_func(iter_object, pg, device)
    elif isinstance(iter_object, DTensor):
        ret = dtensor_func(iter_object, pg, device)
    elif (
        isinstance(iter_object, (torch.Tensor, int, float, str)) or iter_object is None
    ):
        ret = iter_object
    elif isinstance(iter_object, dict):
        ret = {
            key: _iterate_state_dict(
                value,
                sharded_tensor_func,
                dtensor_func,
                pg=pg,
                device=device,
                cpu_offload=cpu_offload,
                ranks_only=ranks_only,
            )
            for key, value in iter_object.items()
        }
    elif isinstance(iter_object, (list, tuple)):
        ret = [
            _iterate_state_dict(
                v,
                sharded_tensor_func,
                dtensor_func,
                pg=pg,
                device=device,
                cpu_offload=cpu_offload,
                ranks_only=ranks_only,
            )
            for v in iter_object
        ]
        if isinstance(iter_object, tuple):
            ret = tuple(ret)
    else:
        raise ValueError(f"Unexpected value type {type(iter_object)}")

    if not ranks_only or dist.get_rank(pg) in ranks_only:
        if isinstance(ret, torch.Tensor) and cpu_offload:
            ret = ret.to(cpu_device)
    else:
        ret = {} if isinstance(ret, dict) else None

    return ret


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
            gather ShardedTensor. Note that gathering a DTensor will use
            the DeviceMesh. So this argument will be ignored when gathering a
            DTensor.
        device: (Optional[torch.device]): the device that is used to
            perform allgather for ShardedTensor. Note that gathering a DTensor
            will use the DeviceMesh. So this argument will be ignored when
            gathering a DTensor.
        cpu_offload (bool): whether to offload the tensors to CPU memory. The
            default value is False.
        ranks_only: (Tuple[int, ...]): if this tuple is empty, all ranks will
            have the same state_dicts. Otherwise only ranks that in ``ranks_only``
            have the same state_dicts. Other ranks will get empty state_dicts.

    Returns:
        The gathered state dictionary.
    """

    def sharded_tensor_func(value, pg, device):
        # ShardedTensor does not seem to record the original device type.
        # So if the tensor is moved to CPU, we won't know the original type.
        # As a result, we have to rely on the user to tell us the correct one.
        cpu_device = torch.device("cpu")
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
        return value

    def dtensor_func(value, pg, device):
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
        # Call `wait()` to force the tensor is synchronous with respect
        # to the main stream.
        # See the discussion in https://github.com/pytorch/pytorch/pull/117799.
        value = value.to_local()
        if isinstance(value, AsyncCollectiveTensor):
            value = value.wait()
        return value

    return _iterate_state_dict(
        state_dict,
        sharded_tensor_func,
        dtensor_func,
        pg=pg,
        device=device,
        cpu_offload=cpu_offload,
        ranks_only=ranks_only,
    )


def _offload_state_dict_to_cpu(
    state_dict: Dict[str, Any],
    *,
    pg: Optional[dist.ProcessGroup] = None,
    device: Optional[torch.device] = None,
    ranks_only: Tuple[int, ...] = tuple(),
) -> Dict[str, Any]:
    return _iterate_state_dict(
        state_dict,
        lambda value, pg, device: value,
        lambda value, pg, device: value,
        pg=pg,
        device=device,
        cpu_offload=True,
        ranks_only=ranks_only,
    )
