import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as distributed_c10d
from torch.distributed._shard.sharded_tensor import (
    ShardedTensor,
    sharded_op_impl
)

def _communicate_result(result, pg):
    # Gather results from all ranks.
    if result:
        result_tensor = torch.ones(1, device=torch.device(torch.cuda.current_device()))
    else:
        result_tensor = torch.zeros(1, device=torch.device(torch.cuda.current_device()))

    dist.all_reduce(result_tensor, group=pg)

    expected_result = torch.ones(1, device=torch.device(torch.cuda.current_device())) * dist.get_world_size(pg)

    return torch.equal(result_tensor, expected_result)

def binary_cmp(cmp_fun, types, args, kwargs=None, process_group=None):
    if len(args) != 2:
        raise ValueError(f'Expected two arguments for torch.{cmp_fun.__name__}')

    result = True
    st1 = args[0]
    st2 = args[1]
    if not(isinstance(st1, ShardedTensor) and isinstance(st2, ShardedTensor)):
        raise TypeError(f'Both arguments to torch.{cmp_fun.__name__} need to be of type ShardedTensor')

    # Verify same PG
    if st1._process_group != st2._process_group:
        return False

    if distributed_c10d._rank_not_in_group(st1._process_group) or distributed_c10d._rank_not_in_group(st2._process_group):
        return distributed_c10d._rank_not_in_group(st1._process_group) == distributed_c10d._rank_not_in_group(st2._process_group)

    # Verify metadata
    if st1.metadata() != st2.metadata():
        return _communicate_result(False, st1._process_group)

    # Verify number of local shards
    st1_local_shards = st1.local_shards()
    st2_local_shards = st2.local_shards()
    if len(st1_local_shards) != len(st2_local_shards):
        return _communicate_result(False, st1._process_group)

    # kwargs must be dict-like
    if kwargs is None:
        kwargs = {}
    # Verify each local shard
    for idx in range(len(st1_local_shards)):
        if st1_local_shards[idx].metadata != st2_local_shards[idx].metadata:
            return _communicate_result(False, st1._process_group)
        if not cmp_fun(st1_local_shards[idx].tensor, st2_local_shards[idx].tensor, **kwargs):
            return _communicate_result(False, st1._process_group)


    return _communicate_result(True, st1._process_group)

@sharded_op_impl(torch.equal)
def equal(types, args, kwargs, process_group):
    return binary_cmp(torch.equal, types, args, kwargs, process_group)

@sharded_op_impl(torch.allclose)
def allclose(types, args, kwargs, process_group):
    return binary_cmp(torch.allclose, types, args, kwargs, process_group)
