import torch
import torch.distributed as dist

def get_world_size(group):
    return dist.get_world_size(group=group)

def all_reduce(tensor, group, op="sum"):
    if op == "sum":
        op = dist.ReduceOp.SUM
    elif op == "max":
        op = dist.ReduceOp.MAX
    else:
        raise NotImplementedError
    dist.all_reduce(tensor, op=op, group=group)
    return tensor
