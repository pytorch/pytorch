# mypy: allow-untyped-defs

import copy
import random

import torch
from torch.distributed._shard import sharded_tensor
from torch.distributed._shard.sharding_spec import ChunkShardingSpec


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


def _placement(rank, dt=None):
    """Generate a placement string for the given rank.

    CPU does not support device indices, so placements omit the index.
    If dt is None, uses the module-level device type (current accelerator).
    """
    if dt is None:
        dt = device_type
    if dt == "cpu":
        return f"rank:{rank}/cpu"
    return f"rank:{rank}/{dt}:{rank}"


PLACEMENTS = [_placement(i) for i in range(4)]

DEFAULT_GPU_NUM = 4


def _chunk_sharding_specs_list_for_test(sharding_dims, seed=0, device_type=None):
    spec_list = []
    placements = [_placement(i, device_type) for i in range(4)]
    for i in range(len(sharding_dims)):
        random.Random(seed + i).shuffle(placements)
        spec_list.append(
            ChunkShardingSpec(
                dim=sharding_dims[i],
                placements=copy.deepcopy(placements),
            )
        )
    return spec_list


class MyShardedModel2(torch.nn.Module):
    def __init__(self, spec=None, group=None, init_rrefs=True) -> None:
        super().__init__()
        if spec is not None:
            self.sharded_tensor2 = sharded_tensor.rand(
                spec, 10, 20, process_group=group, init_rrefs=init_rrefs
            )
        else:
            self.sharded_tensor2 = None
        self.random_tensor2 = torch.nn.Parameter(torch.rand(2, 2))


class MyShardedModel1(torch.nn.Module):
    def __init__(self, spec=None, group=None, init_rrefs=True) -> None:
        super().__init__()
        if spec is not None:
            self.sharded_tensor1 = sharded_tensor.rand(
                spec, 10, 20, process_group=group, init_rrefs=init_rrefs
            )
        else:
            self.sharded_tensor1 = None
        self.random_tensor1 = torch.nn.Parameter(torch.rand(2, 2))
        self.submodule = MyShardedModel2(spec, group, init_rrefs)
