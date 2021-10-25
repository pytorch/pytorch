from torch.distributed._shard import (
    ChunkShardingSpec,
)

def generate_chunk_sharding_specs_for_test(sharding_dim):
    return [
        ChunkShardingSpec(
            dim=sharding_dim,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        ),
        # Test different ordering. (Case 1)
        ChunkShardingSpec(
            dim=sharding_dim,
            placements=[
                "rank:2/cuda:2",
                "rank:3/cuda:3",
                "rank:0/cuda:0",
                "rank:1/cuda:1",
            ],
        ),
        # Test different ordering. (Case 2)
        ChunkShardingSpec(
            dim=sharding_dim,
            placements=[
                "rank:3/cuda:3",
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
            ],
        )
    ]
