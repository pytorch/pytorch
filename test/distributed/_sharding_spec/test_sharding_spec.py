import torch
from torch.testing._internal.common_utils import TestCase
from torch.distributed._sharding_spec import (
    ChunkShardingSpec,
    DevicePlacement,
    GenericShardingSpec,
)

class TestShardingSpec(TestCase):

    def test_device_placement(self):
        # valid devices
        DevicePlacement("cuda:0")
        DevicePlacement(0)
        DevicePlacement(torch.device("cuda:0"))
        DevicePlacement("rank:0/cuda:0")
        DevicePlacement("rank:0/cpu")
        DevicePlacement("rank:0")

        # invalid devices
        with self.assertRaisesRegex(ValueError, "not a valid device"):
            DevicePlacement("cuda:foo")
        with self.assertRaisesRegex(ValueError, "not a valid device"):
            DevicePlacement("foo:0")
        with self.assertRaisesRegex(ValueError, "not a valid device"):
            DevicePlacement("rank:0/cuda:foo")

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

    def test_generic_sharding_spec(self):
        Shard = GenericShardingSpec.Shard
        # test valid specs

        # test row-wise sharding
        spec = GenericShardingSpec([
            Shard(
                shard_offsets=[0, 0],
                shard_lengths=[5, 5],
                placement="cuda:0",
            ),
            Shard(
                shard_offsets=[5, 0],
                shard_lengths=[5, 5],
                placement="cuda:1",
            )
        ])
        spec.check_tensor(torch.rand(10, 5))

        # test row and column sharding
        spec = GenericShardingSpec([
            Shard(
                shard_offsets=[0, 0],
                shard_lengths=[3, 3],
                placement="cuda:0",
            ),
            Shard(
                shard_offsets=[0, 3],
                shard_lengths=[3, 3],
                placement="cuda:1",
            ),
            Shard(
                shard_offsets=[3, 0],
                shard_lengths=[3, 3],
                placement="cuda:2",
            ),
            Shard(
                shard_offsets=[3, 3],
                shard_lengths=[3, 3],
                placement="cuda:3",
            ),
        ])
        spec.check_tensor(torch.rand(6, 6))

        # test uneven shard sizes.
        spec = GenericShardingSpec([
            Shard(
                shard_offsets=[0, 0],
                shard_lengths=[2, 4],
                placement="cuda:0",
            ),
            Shard(
                shard_offsets=[0, 4],
                shard_lengths=[4, 2],
                placement="cuda:1",
            ),
            Shard(
                shard_offsets=[2, 0],
                shard_lengths=[4, 4],
                placement="cuda:2",
            ),
            Shard(
                shard_offsets=[4, 4],
                shard_lengths=[2, 2],
                placement="cuda:3",
            ),
        ])
        spec.check_tensor(torch.rand(6, 6))

        # test invalid sharding
        with self.assertRaisesRegex(ValueError, 'not a valid device'):
            Shard(shard_offsets=[0], shard_lengths=[1], placement="cuda:foo")

        with self.assertRaisesRegex(ValueError, 'same number of elements'):
            Shard(shard_offsets=[0, 0], shard_lengths=[1], placement="cuda:0")

        with self.assertRaisesRegex(ValueError, 'shard_offsets should be >=0'):
            Shard(shard_offsets=[-1, 0], shard_lengths=[1, 1], placement="cuda:0")

        with self.assertRaisesRegex(ValueError, 'shard_lengths should be > 0'):
            Shard(shard_offsets=[0, 0], shard_lengths=[0, 1], placement="cuda:0")

        with self.assertRaisesRegex(ValueError, 'Empty shard list provided'):
            GenericShardingSpec([])

        with self.assertRaisesRegex(ValueError, 'Found inconsistent ranks for shards'):
            GenericShardingSpec([
                Shard(
                    shard_offsets=[0, 0],
                    shard_lengths=[1, 1],
                    placement="cpu"
                ),
                Shard(
                    shard_offsets=[0, 0, 0],
                    shard_lengths=[1, 1, 1],
                    placement="cpu"
                ),
            ])

        with self.assertRaisesRegex(ValueError, 'Shards.*overlap'):
            GenericShardingSpec([
                Shard(
                    shard_offsets=[0, 0],
                    shard_lengths=[3, 3],
                    placement="cpu"
                ),
                Shard(
                    shard_offsets=[2, 0],
                    shard_lengths=[3, 3],
                    placement="cpu"
                ),
            ])

        spec = GenericShardingSpec([
            Shard(
                shard_offsets=[0, 0],
                shard_lengths=[5, 5],
                placement="cuda:0",
            ),
            Shard(
                shard_offsets=[5, 0],
                shard_lengths=[5, 5],
                placement="cuda:1",
            )
        ])

        with self.assertRaisesRegex(ValueError, 'Rank of tensor is.*but shards rank'):
            spec.check_tensor(torch.rand(10, 10, 10))

        spec = GenericShardingSpec([
            Shard(
                shard_offsets=[0, 0],
                shard_lengths=[5, 5],
                placement="cuda:0",
            ),
            Shard(
                shard_offsets=[5, 0],
                shard_lengths=[5, 5],
                placement="cuda:1",
            )
        ])

        with self.assertRaisesRegex(ValueError, 'exceeds tensor dim'):
            spec.check_tensor(torch.rand(10, 3))

        spec = GenericShardingSpec([
            Shard(
                shard_offsets=[0, 0],
                shard_lengths=[5, 5],
                placement="cuda:0",
            ),
            Shard(
                shard_offsets=[5, 5],
                shard_lengths=[5, 5],
                placement="cuda:1",
            )
        ])

        with self.assertRaisesRegex(ValueError, 'does not match tensor volume'):
            spec.check_tensor(torch.rand(10, 10))
