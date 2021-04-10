import torch
from torch.testing._internal.common_utils import TestCase
from torch.distributed._sharding_spec import (
    ChunkShardingSpec,
    DevicePlacement
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
