import copy
import random

from torch.distributed._shard.sharding_spec import (
    ChunkShardingSpec,
)

PLACEMENTS = [
    "rank:0/cuda:0",
    "rank:1/cuda:1",
    "rank:2/cuda:2",
    "rank:3/cuda:3",
]

DEFAULT_GPU_NUM = 4


def _chunk_sharding_specs_list_for_test(sharding_dims, seed=0):
    spec_list = []
    for i in range(len(sharding_dims)):
        random.Random(seed + i).shuffle(PLACEMENTS)
        spec_list.append(
            ChunkShardingSpec(
                dim=sharding_dims[i],
                placements=copy.deepcopy(PLACEMENTS),
            )
        )
    return spec_list
