# Owner(s): ["oncall: distributed"]

import copy
import torch
from torch.testing._internal.common_utils import TestCase
from torch.distributed._shard.sharding_spec import (
    ChunkShardingSpec,
    DevicePlacementSpec,
    EnumerableShardingSpec,
    ShardMetadata,
    _infer_sharding_spec_from_shards_metadata,
)
from torch.distributed._shard.sharding_spec._internals import (
    check_tensor,
    get_split_size,
    get_chunked_dim_size,
    get_chunk_sharding_params,
)

from torch.testing._internal.common_utils import (
    run_tests,
    sandcastle_skip_if,
)
from torch.testing._internal.distributed._shard.sharded_tensor._test_st_common import (
    _chunk_sharding_specs_list_for_test,
)

class TestShardingSpec(TestCase):

    @sandcastle_skip_if(torch.cuda.device_count() < 2, '2 CUDA GPUs are needed')
    def test_device_placement(self):
        # valid devices
        DevicePlacementSpec("cuda:0")
        DevicePlacementSpec(torch.device(0))
        DevicePlacementSpec(torch.device("cuda:0"))
        DevicePlacementSpec("rank:0/cuda:0")
        DevicePlacementSpec("rank:0/cpu")
        DevicePlacementSpec("rank:0")

        # invalid devices
        with self.assertRaisesRegex(ValueError, "Could not parse remote_device"):
            DevicePlacementSpec("cuda:foo")
        with self.assertRaisesRegex(ValueError, "Could not parse remote_device"):
            DevicePlacementSpec("foo:0")
        with self.assertRaisesRegex(RuntimeError, "Invalid device string"):
            DevicePlacementSpec("rank:0/cuda:foo")
        with self.assertRaisesRegex(RuntimeError, "Invalid device string"):
            DevicePlacementSpec("rank:0/cpu2")

    @sandcastle_skip_if(torch.cuda.device_count() < 2, '2 CUDA GPUs are needed')
    def test_chunked_sharding_spec(self):
        # Test valid specs.
        ChunkShardingSpec(0, [torch.device(0), torch.device(1)])
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
        with self.assertRaisesRegex(ValueError, "Could not parse remote_device"):
            ChunkShardingSpec(0, ["random:0", "cuda:1"])
        with self.assertRaisesRegex(ValueError, "Could not parse remote_device"):
            ChunkShardingSpec(0, ["cuda:foo", "cuda:1"])
        with self.assertRaisesRegex(ValueError, "Could not parse remote_device"):
            ChunkShardingSpec(0, ["rank:foo", "cuda:1"])
        with self.assertRaisesRegex(RuntimeError, "Expected one of"):
            ChunkShardingSpec(0, ["rank:0/foo", "cuda:1"])
        with self.assertRaisesRegex(RuntimeError, "Expected one of"):
            ChunkShardingSpec(0, ["rank:0/random:0", "cuda:1"])
        with self.assertRaisesRegex(RuntimeError, "Invalid device string"):
            ChunkShardingSpec(0, ["rank:0/cuda:foo", "cuda:1"])

    @sandcastle_skip_if(torch.cuda.device_count() < 2, '2 CUDA GPUs are needed')
    def test_enumerable_sharding_spec(self):
        # test valid specs

        # test row-wise sharding
        spec = EnumerableShardingSpec([
            ShardMetadata(
                shard_offsets=[0, 0],
                shard_sizes=[5, 5],
                placement="cuda:0",
            ),
            ShardMetadata(
                shard_offsets=[5, 0],
                shard_sizes=[5, 5],
                placement="cuda:1",
            )
        ])
        check_tensor(spec.shards, torch.rand(10, 5).size())

        # test row and column sharding
        spec = EnumerableShardingSpec([
            ShardMetadata(
                shard_offsets=[0, 0],
                shard_sizes=[3, 3],
                placement="cuda:0",
            ),
            ShardMetadata(
                shard_offsets=[0, 3],
                shard_sizes=[3, 3],
                placement="cuda:1",
            ),
            ShardMetadata(
                shard_offsets=[3, 0],
                shard_sizes=[3, 3],
                placement="cuda:2",
            ),
            ShardMetadata(
                shard_offsets=[3, 3],
                shard_sizes=[3, 3],
                placement="cuda:3",
            ),
        ])
        check_tensor(spec.shards, torch.rand(6, 6).size())

        # test uneven shard sizes.
        spec = EnumerableShardingSpec([
            ShardMetadata(
                shard_offsets=[0, 0],
                shard_sizes=[2, 4],
                placement="cuda:0",
            ),
            ShardMetadata(
                shard_offsets=[0, 4],
                shard_sizes=[4, 2],
                placement="cuda:1",
            ),
            ShardMetadata(
                shard_offsets=[2, 0],
                shard_sizes=[4, 4],
                placement="cuda:2",
            ),
            ShardMetadata(
                shard_offsets=[4, 4],
                shard_sizes=[2, 2],
                placement="cuda:3",
            ),
        ])
        check_tensor(spec.shards, torch.rand(6, 6).size())

        # test invalid sharding
        with self.assertRaisesRegex(ValueError, 'Could not parse remote_device'):
            ShardMetadata(shard_offsets=[0], shard_sizes=[1], placement="cuda:foo")

        with self.assertRaisesRegex(ValueError, 'same number of elements'):
            ShardMetadata(shard_offsets=[0, 0], shard_sizes=[1], placement="cuda:0")

        with self.assertRaisesRegex(ValueError, 'shard_offsets should be >=0'):
            ShardMetadata(shard_offsets=[-1, 0], shard_sizes=[1, 1], placement="cuda:0")

        with self.assertRaisesRegex(ValueError, 'shard_sizes should be >= 0'):
            ShardMetadata(shard_offsets=[0, 0], shard_sizes=[-1, 1], placement="cuda:0")

        with self.assertRaisesRegex(ValueError, 'Empty shard list provided'):
            EnumerableShardingSpec([])

        with self.assertRaisesRegex(ValueError, 'Found inconsistent ranks for shards'):
            EnumerableShardingSpec([
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_sizes=[1, 1],
                    placement="cpu"
                ),
                ShardMetadata(
                    shard_offsets=[0, 0, 0],
                    shard_sizes=[1, 1, 1],
                    placement="cpu"
                ),
            ])

        with self.assertRaisesRegex(ValueError, 'Shards.*overlap'):
            EnumerableShardingSpec([
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_sizes=[3, 3],
                    placement="cpu"
                ),
                ShardMetadata(
                    shard_offsets=[2, 0],
                    shard_sizes=[3, 3],
                    placement="cpu"
                ),
            ])

        spec = EnumerableShardingSpec([
            ShardMetadata(
                shard_offsets=[0, 0],
                shard_sizes=[5, 5],
                placement="cuda:0",
            ),
            ShardMetadata(
                shard_offsets=[5, 0],
                shard_sizes=[5, 5],
                placement="cuda:1",
            )
        ])

        with self.assertRaisesRegex(ValueError, 'Rank of tensor is.*but shards rank'):
            check_tensor(spec.shards, torch.rand(10, 10, 10).size())

        spec = EnumerableShardingSpec([
            ShardMetadata(
                shard_offsets=[0, 0],
                shard_sizes=[5, 5],
                placement="cuda:0",
            ),
            ShardMetadata(
                shard_offsets=[5, 0],
                shard_sizes=[5, 5],
                placement="cuda:1",
            )
        ])

        with self.assertRaisesRegex(ValueError, 'exceeds tensor dim'):
            check_tensor(spec.shards, torch.rand(10, 3).size())

        spec = EnumerableShardingSpec([
            ShardMetadata(
                shard_offsets=[0, 0],
                shard_sizes=[5, 5],
                placement="cuda:0",
            ),
            ShardMetadata(
                shard_offsets=[5, 5],
                shard_sizes=[5, 5],
                placement="cuda:1",
            )
        ])

        with self.assertRaisesRegex(ValueError, 'does not match tensor volume'):
            check_tensor(spec.shards, torch.rand(10, 10).size())

    def test_get_split_size(self):
        self.assertEqual(3, get_split_size(11, 4))
        self.assertEqual(3, get_split_size(12, 4))
        self.assertEqual(4, get_split_size(13, 4))
        self.assertEqual(2, get_split_size(5, 4))

        self.assertEqual(11, get_split_size(11, 1))
        self.assertEqual(1, get_split_size(11, 11))

    def test_get_chunked_dim_size(self):
        self.assertEqual(3, get_chunked_dim_size(11, 3, 0))
        self.assertEqual(2, get_chunked_dim_size(11, 3, 3))
        self.assertEqual(4, get_chunked_dim_size(13, 4, 0))
        self.assertEqual(1, get_chunked_dim_size(13, 4, 3))
        self.assertEqual(0, get_chunked_dim_size(5, 2, 3))

    def test_get_chunk_sharding_params(self):
        ranks = [
            "rank:0/cuda:0",
            "rank:1/cuda:1",
            "rank:2/cuda:2",
            "rank:3/cuda:3",
        ]
        spec = ChunkShardingSpec(
            dim=0,
            placements=ranks,
        )
        result = get_chunk_sharding_params(21, 4, spec, 1)
        self.assertEqual(6, result[0])
        self.assertEqual(6, result[1])
        result = get_chunk_sharding_params(21, 4, spec, 3)
        self.assertEqual(18, result[0])
        self.assertEqual(3, result[1])
        ranks[1], ranks[2] = ranks[2], ranks[1]
        ranks[0], ranks[3] = ranks[3], ranks[0]
        spec.placements = ranks
        result = get_chunk_sharding_params(21, 4, spec, 1)
        self.assertEqual(12, result[0])
        self.assertEqual(6, result[1])
        result = get_chunk_sharding_params(21, 4, spec, 3)
        self.assertEqual(0, result[0])
        self.assertEqual(6, result[1])

    def _infer_enum_sharding_spec_case(self):
        shards_metadata = [
            ShardMetadata(
                shard_offsets=[0, 0],
                shard_sizes=[5, 5],
                placement="cuda:0",
            ),
            ShardMetadata(
                shard_offsets=[5, 0],
                shard_sizes=[10, 5],
                placement="cuda:1",
            )
        ]
        spec = _infer_sharding_spec_from_shards_metadata(shards_metadata)
        self.assertTrue(isinstance(spec, EnumerableShardingSpec))
        self.assertEqual(spec.shards, shards_metadata)

        shards_metadata = [
            ShardMetadata(
                shard_offsets=[0, 0],
                shard_sizes=[5, 5],
                placement="rank:0/cuda:0",
            ),
            ShardMetadata(
                shard_offsets=[5, 0],
                shard_sizes=[5, 5],
                placement="rank:1/cuda:1",
            ),
            ShardMetadata(
                shard_offsets=[0, 5],
                shard_sizes=[5, 5],
                placement="rank:2/cuda:2",
            ),
            ShardMetadata(
                shard_offsets=[5, 5],
                shard_sizes=[5, 5],
                placement="rank:3/cuda:3",
            ),
        ]
        spec = _infer_sharding_spec_from_shards_metadata(shards_metadata)
        self.assertTrue(isinstance(spec, EnumerableShardingSpec))
        self.assertEqual(spec.shards, shards_metadata)

    def _infer_chunk_sharding_spec_case(self, placements, sharding_dim, st_size):
        world_size = len(placements)
        split_size = get_split_size(st_size[sharding_dim], world_size)
        shards_metadata = [None] * world_size
        for idx, placement in enumerate(placements):
            shard_size = copy.deepcopy(st_size)
            offsets = [0] * len(st_size)
            offsets[sharding_dim] = split_size * idx
            shard_size[sharding_dim] = get_chunked_dim_size(st_size[sharding_dim], split_size, idx)
            shards_metadata[placement.rank()] = ShardMetadata(
                shard_offsets=offsets,
                shard_sizes=shard_size,
                placement=placement,
            )

        spec = _infer_sharding_spec_from_shards_metadata(shards_metadata)
        self.assertTrue(isinstance(spec, ChunkShardingSpec))
        self.assertEqual(spec.dim, sharding_dim)
        self.assertEqual(spec.placements, placements)

    def test_infer_sharding_spec_from_shards_metadata(self):
        self._infer_enum_sharding_spec_case()
        chunk_specs = _chunk_sharding_specs_list_for_test([0, 0, 1, 1], seed=31)
        for spec in chunk_specs:
            self._infer_chunk_sharding_spec_case(spec.placements, 0, [4, 16])
            self._infer_chunk_sharding_spec_case(spec.placements, 0, [5, 15, 16])
            self._infer_chunk_sharding_spec_case(spec.placements, 1, [12, 16])
            self._infer_chunk_sharding_spec_case(spec.placements, 2, [4, 18, 15])
            self._infer_chunk_sharding_spec_case(spec.placements, 3, [7, 12, 16, 37])
            self._infer_chunk_sharding_spec_case(spec.placements, 4, [50, 4, 18, 15, 77])

if __name__ == '__main__':
    run_tests()
