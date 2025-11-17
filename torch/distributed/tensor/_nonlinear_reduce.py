import torch
import torch.distributed as dist
import torch.distributed.tensor._api as dtensor
from torch.distributed.tensor.placement_types import Shard, Replicate


def argmax_handler(
        op_call: torch._ops.OpOverload,
        args: tuple[object, ...],
        kwargs: dict[str, object]
):
    """
    Handles reduces on sharded dimensions locally to limit calls to replicate.
    """
    op_info = dtensor.DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)
    dtensor.DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding
    assert output_sharding is not None, "output sharding should not be None"
    
    input_dtensor = args[0]
    if not isinstance(input_dtensor, dtensor.DTensor):
        raise NotImplementedError
    
    dim = args[1] if len(args) > 1 else None
    keepdim = args[2] if len(args) > 2 else False
    
    placements = input_dtensor.placements
    local_tensor = input_dtensor.to_local()

    shard_dim = None
    for p in placements:
        if isinstance(p, Shard):
            if dim is None or p.dim == (dim if dim >= 0 else local_tensor.ndim + dim):
                shard_dim = p.dim
                break
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if dim is None:
        local_max, local_idx = torch.max(local_tensor).view(1)
    else:
        local_max, local_idx = torch.max(local_tensor, dim=dim, keepdim=keepdim)
    
    if shard_dim is not None:
        # find the correct offset for sharded dim
        shard_size = local_tensor.shape[shard_dim]
        global_offset = rank * shard_size
        global_idx = local_idx + global_offset
    else:
        return dtensor.DTensor._op_dispatcher.wrap(local_idx, output_sharding.output_spec)

    gathered_maxes = [torch.zeros_like(local_max) for _ in range(world_size)]
    gathered_idxs = [torch.zeros_like(local_idx) for _ in range(world_size)]
    dist.all_gather(gathered_maxes, local_max)
    dist.all_gather(gathered_idxs, global_idx)

    all_maxes = torch.stack(gathered_maxes, dim=0)
    rank_winner = op_call(all_maxes, dim=0)
    all_idxs = torch.stack(gathered_idxs, dim=0)

    final_idx = torch.gather(all_idxs, dim=0, index=rank_winner.view(1, *rank_winner.shape))
    final_idx = final_idx.view(*final_idx.shape[1:])

    if not keepdim and dim is not None:
        final_idx = final_idx.squeeze(0)

    return dtensor.DTensor._op_dispatcher.wrap(final_idx, output_sharding.output_spec)
