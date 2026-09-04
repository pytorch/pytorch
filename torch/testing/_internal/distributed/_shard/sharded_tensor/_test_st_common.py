# mypy: allow-untyped-defs

import copy
import random

import torch
from torch.distributed._shard import sharded_tensor
from torch.distributed._shard.sharding_spec import ChunkShardingSpec


def _get_default_placements(device_type=None):
    """Generate placements dynamically based on device type.

    If device_type is None, uses the current accelerator type.
    CPU does not support device indices, so placements omit the index.
    """
    if device_type is None:
        import torch
        acc = torch.accelerator.current_accelerator()
        device_type = acc.type if acc is not None else "cpu"
    if device_type == "cpu":
        return [
            "rank:0/cpu",
            "rank:1/cpu",
            "rank:2/cpu",
            "rank:3/cpu",
        ]
    return [
        f"rank:0/{device_type}:0",
        f"rank:1/{device_type}:1",
        f"rank:2/{device_type}:2",
        f"rank:3/{device_type}:3",
    ]


# Default placements for backward compatibility
PLACEMENTS = _get_default_placements()

DEFAULT_GPU_NUM = 4


def _chunk_sharding_specs_list_for_test(sharding_dims, seed=0, device_type=None):
    spec_list = []
    placements = _get_default_placements(device_type)
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
