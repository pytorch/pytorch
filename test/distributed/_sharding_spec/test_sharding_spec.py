import torch
from torch.testing._internal.common_utils import TestCase
from torch.distributed._sharding_spec import (
    ChunkShardingSpec,
    DevicePlacementSpec,
    EnumerableShardingSpec,
    ShardMetadata,
)

class TestShardingSpec(TestCase):

    def test_device_placement(self):
        # valid devices
        DevicePlacementSpec("cuda:0")
        DevicePlacementSpec(0)
        DevicePlacementSpec(torch.device("cuda:0"))
        DevicePlacementSpec("rank:0/cuda:0")
        DevicePlacementSpec("rank:0/cpu")
        DevicePlacementSpec("rank:0")

        # invalid devices
        with self.assertRaisesRegex(ValueError, "not a valid device"):
            DevicePlacementSpec("cuda:foo")
        with self.assertRaisesRegex(ValueError, "not a valid device"):
            DevicePlacementSpec("foo:0")
        with self.assertRaisesRegex(ValueError, "not a valid device"):
            DevicePlacementSpec("rank:0/cuda:foo")
        with self.assertRaisesRegex(ValueError, "not a valid device"):
            DevicePlacementSpec("rank:0/cpu2")

    def test_chunked_sharding_spec(self):
        # Test valid specs.
        ChunkShardingSpec(0, [0, 1])
        # Named dimension.
        ChunkShardingSpec("N", ["cuda:0", "cuda:1"])
        ChunkShardingSpec(0, [torch.device("cuda:0"), torch.device("cuda:1")])
        ChunkShardingSpec(-1, ["cuda:0", "cuda:1"])
        ChunkShardingSpec(0, ["rank:0/cuda:0", "rank:0/cuda:1"])
        ChunkShardingSpec(0, ["rank:0", "rank:1"])
        ChunkShardingSpec(0, ["rank:0/cpu", "rank:1/cpu"])

        # Test invalid specs
        with self.assertRaisesRegex(ValueError, "int or str"):
            ChunkShardingSpec(None, ["cuda:0", "cuda:1"])
        with self.assertRaisesRegex(ValueError, "int or str"):
            ChunkShardingSpec({}, ["cuda:0", "cuda:1"])
        with self.assertRaisesRegex(ValueError, "not a valid device"):
            ChunkShardingSpec(0, ["random:0", "cuda:1"])
        with self.assertRaisesRegex(ValueError, "not a valid device"):
            ChunkShardingSpec(0, ["cuda:foo", "cuda:1"])
        with self.assertRaisesRegex(ValueError, "not a valid device"):
            ChunkShardingSpec(0, ["rank:foo", "cuda:1"])
        with self.assertRaisesRegex(ValueError, "not a valid device"):
            ChunkShardingSpec(0, ["rank:0/foo", "cuda:1"])
        with self.assertRaisesRegex(ValueError, "not a valid device"):
            ChunkShardingSpec(0, ["rank:0/random:0", "cuda:1"])
        with self.assertRaisesRegex(ValueError, "not a valid device"):
            ChunkShardingSpec(0, ["rank:0/cuda:foo", "cuda:1"])

    def test_enumerable_sharding_spec(self):
        # test valid specs

        # test row-wise sharding
        spec = EnumerableShardingSpec([
            ShardMetadata(
                shard_offsets=[0, 0],
                shard_lengths=[5, 5],
                placement="cuda:0",
            ),
            ShardMetadata(
                shard_offsets=[5, 0],
                shard_lengths=[5, 5],
                placement="cuda:1",
            )
        ])
        spec.check_tensor(torch.rand(10, 5).size())

        # test row and column sharding
        spec = EnumerableShardingSpec([
            ShardMetadata(
                shard_offsets=[0, 0],
                shard_lengths=[3, 3],
                placement="cuda:0",
            ),
            ShardMetadata(
                shard_offsets=[0, 3],
                shard_lengths=[3, 3],
                placement="cuda:1",
            ),
            ShardMetadata(
                shard_offsets=[3, 0],
                shard_lengths=[3, 3],
                placement="cuda:2",
            ),
            ShardMetadata(
                shard_offsets=[3, 3],
                shard_lengths=[3, 3],
                placement="cuda:3",
            ),
        ])
        spec.check_tensor(torch.rand(6, 6).size())

        # test uneven shard sizes.
        spec = EnumerableShardingSpec([
            ShardMetadata(
                shard_offsets=[0, 0],
                shard_lengths=[2, 4],
                placement="cuda:0",
            ),
            ShardMetadata(
                shard_offsets=[0, 4],
                shard_lengths=[4, 2],
                placement="cuda:1",
            ),
            ShardMetadata(
                shard_offsets=[2, 0],
                shard_lengths=[4, 4],
                placement="cuda:2",
            ),
            ShardMetadata(
                shard_offsets=[4, 4],
                shard_lengths=[2, 2],
                placement="cuda:3",
            ),
        ])
        spec.check_tensor(torch.rand(6, 6).size())

        # test invalid sharding
        with self.assertRaisesRegex(ValueError, 'not a valid device'):
            ShardMetadata(shard_offsets=[0], shard_lengths=[1], placement="cuda:foo")

        with self.assertRaisesRegex(ValueError, 'same number of elements'):
            ShardMetadata(shard_offsets=[0, 0], shard_lengths=[1], placement="cuda:0")

        with self.assertRaisesRegex(ValueError, 'shard_offsets should be >=0'):
            ShardMetadata(shard_offsets=[-1, 0], shard_lengths=[1, 1], placement="cuda:0")

        with self.assertRaisesRegex(ValueError, 'shard_lengths should be > 0'):
            ShardMetadata(shard_offsets=[0, 0], shard_lengths=[0, 1], placement="cuda:0")

        with self.assertRaisesRegex(ValueError, 'Empty shard list provided'):
            EnumerableShardingSpec([])

        with self.assertRaisesRegex(ValueError, 'Found inconsistent ranks for shards'):
            EnumerableShardingSpec([
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_lengths=[1, 1],
                    placement="cpu"
                ),
                ShardMetadata(
                    shard_offsets=[0, 0, 0],
                    shard_lengths=[1, 1, 1],
                    placement="cpu"
                ),
            ])

        with self.assertRaisesRegex(ValueError, 'Shards.*overlap'):
            EnumerableShardingSpec([
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_lengths=[3, 3],
                    placement="cpu"
                ),
                ShardMetadata(
                    shard_offsets=[2, 0],
                    shard_lengths=[3, 3],
                    placement="cpu"
                ),
            ])

        spec = EnumerableShardingSpec([
            ShardMetadata(
                shard_offsets=[0, 0],
                shard_lengths=[5, 5],
                placement="cuda:0",
            ),
            ShardMetadata(
                shard_offsets=[5, 0],
                shard_lengths=[5, 5],
                placement="cuda:1",
            )
        ])

        with self.assertRaisesRegex(ValueError, 'Rank of tensor is.*but shards rank'):
            spec.check_tensor(torch.rand(10, 10, 10).size())

        spec = EnumerableShardingSpec([
            ShardMetadata(
                shard_offsets=[0, 0],
                shard_lengths=[5, 5],
                placement="cuda:0",
            ),
            ShardMetadata(
                shard_offsets=[5, 0],
                shard_lengths=[5, 5],
                placement="cuda:1",
            )
        ])

        with self.assertRaisesRegex(ValueError, 'exceeds tensor dim'):
            spec.check_tensor(torch.rand(10, 3).size())

        spec = EnumerableShardingSpec([
            ShardMetadata(
                shard_offsets=[0, 0],
                shard_lengths=[5, 5],
                placement="cuda:0",
            ),
            ShardMetadata(
                shard_offsets=[5, 5],
                shard_lengths=[5, 5],
                placement="cuda:1",
            )
        ])

        with self.assertRaisesRegex(ValueError, 'does not match tensor volume'):
            spec.check_tensor(torch.rand(10, 10).size())
