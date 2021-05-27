import torch
import torch.distributed as dist
from torch.distributed import _sharded_tensor
from torch.distributed._sharding_spec import ChunkShardingSpec
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    TEST_WITH_ASAN,
)
import unittest

@unittest.skipIf(
    TEST_WITH_ASAN, "Skip ASAN as torch + multiprocessing spawn have known issues"
)
class TestShardedTensorChunked(MultiProcessTestCase):

    @property
    def world_size(self):
        return 4

    def init_pg(self):
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            init_method=f"file://{self.file_name}",
        )

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_complete_world_size(self):
        self.init_pg()

        for dim in [0, -2]:
            spec = ChunkShardingSpec(
                dim=dim,
                placements=[
                    "rank:0/cuda:0",
                    "rank:1/cuda:1",
                    "rank:2/cuda:2",
                    "rank:3/cuda:3",
                ],
            )
            sharded_tensor = _sharded_tensor.empty(spec, 10, 20)

            # Validate local shard.
            local_shards = sharded_tensor.local_shards()
            self.assertEqual(1, len(local_shards))
            local_shard = local_shards[0].tensor
            self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.device)
            if self.rank == 3:
                self.assertEqual((1, 20), local_shard.size())
            else:
                self.assertEqual((3, 20), local_shard.size())

            # Validate global metadata.
            sharding_metadata = sharded_tensor.sharding_metadata()
            self.assertEqual(4, len(sharding_metadata))

            for rank, shard_metadata in enumerate(sharding_metadata):
                self.assertEqual([rank * 3, 0], shard_metadata.shard_offsets)
                if rank == 3:
                    self.assertEqual([1, 20], shard_metadata.shard_lengths)
                else:
                    self.assertEqual([3, 20], shard_metadata.shard_lengths)
                self.assertEqual(f'rank:{rank}/cuda:{rank}', shard_metadata.placement)

    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_partial_world_size(self):
        self.init_pg()

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        sharded_tensor = _sharded_tensor.empty(spec, 10, 20)

        # Validate local shard.
        local_shards = sharded_tensor.local_shards()
        if self.rank >= 2:
            self.assertEqual(1, len(local_shards))
            local_shard = local_shards[0].tensor
            self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.device)
            self.assertEqual((5, 20), local_shard.size())
        else:
            self.assertEqual(0, len(local_shards))

        # Validate global metadata.
        sharding_metadata = sharded_tensor.sharding_metadata()
        self.assertEqual(2, len(sharding_metadata))

        for shard_rank, shard_metadata in enumerate(sharding_metadata):
            self.assertEqual([shard_rank * 5, 0], shard_metadata.shard_offsets)
            self.assertEqual([5, 20], shard_metadata.shard_lengths)
            self.assertEqual(f'rank:{shard_rank + 2}/cuda:{shard_rank + 2}', shard_metadata.placement)

    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_new_group(self):
        self.init_pg()

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:1/cuda:2",
                "rank:2/cuda:3",
            ],
        )

        pg = dist.new_group(ranks=[1, 2, 3])
        if self.rank >= 1:
            sharded_tensor = _sharded_tensor.empty(spec, 10, 20, process_group=pg)

            # Validate local shard.
            local_shards = sharded_tensor.local_shards()
            if self.rank >= 2:
                self.assertEqual(1, len(local_shards))
                local_shard = local_shards[0].tensor
                self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.device)
                self.assertEqual((5, 20), local_shard.size())
            else:
                self.assertEqual(0, len(local_shards))

            # Validate global metadata.
            sharding_metadata = sharded_tensor.sharding_metadata()
            self.assertEqual(2, len(sharding_metadata))

            for shard_rank, shard_metadata in enumerate(sharding_metadata):
                self.assertEqual([shard_rank * 5, 0], shard_metadata.shard_offsets)
                self.assertEqual([5, 20], shard_metadata.shard_lengths)
                self.assertEqual(f'rank:{shard_rank + 1}/cuda:{shard_rank + 2}', shard_metadata.placement)

    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_multiple_local_shards(self):
        self.init_pg()

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        sharded_tensor = _sharded_tensor.empty(spec, 16, 20)

        # Validate local shards.
        local_shards = sharded_tensor.local_shards()
        self.assertEqual(2, len(local_shards))
        for local_shard in local_shards:
            self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.tensor.device)
            self.assertEqual((2, 20), local_shard.tensor.size())

        # Validate global metadata.
        sharding_metadata = sharded_tensor.sharding_metadata()
        self.assertEqual(8, len(sharding_metadata))

        for shard_idx, shard_metadata in enumerate(sharding_metadata):
            self.assertEqual([shard_idx * 2, 0], shard_metadata.shard_offsets)
            self.assertEqual([2, 20], shard_metadata.shard_lengths)
            self.assertEqual(f'rank:{shard_idx % 4}/cuda:{shard_idx % 4}', shard_metadata.placement)

    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_sharding_columns(self):
        self.init_pg()

        for dim in [1, -1]:
            spec = ChunkShardingSpec(
                dim=dim,
                placements=[
                    "rank:0/cuda:0",
                    "rank:1/cuda:1",
                    "rank:2/cuda:2",
                    "rank:3/cuda:3",
                ],
            )

            sharded_tensor = _sharded_tensor.empty(spec, 10, 32)

            # Validate local shard.
            local_shards = sharded_tensor.local_shards()
            self.assertEqual(1, len(local_shards))
            local_shard = local_shards[0].tensor
            self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.device)
            self.assertEqual((10, 8), local_shard.size())

            # Validate global metadata.
            sharding_metadata = sharded_tensor.sharding_metadata()
            self.assertEqual(4, len(sharding_metadata))

            for rank, shard_metadata in enumerate(sharding_metadata):
                self.assertEqual([0, rank * 8], shard_metadata.shard_offsets)
                self.assertEqual([10, 8], shard_metadata.shard_lengths)
                self.assertEqual(f'rank:{rank}/cuda:{rank}', shard_metadata.placement)

    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_invalid_sharding(self):
        self.init_pg()

        spec = ChunkShardingSpec(dim=0, placements=["rank:1/cuda:1"])
        pg = dist.new_group(ranks=[2, 3])
        if self.rank < 2:
            with self.assertRaisesRegex(ValueError, 'not part of process group'):
                _sharded_tensor.empty(spec, 10, 20, process_group=pg)

        spec = ChunkShardingSpec(dim='H', placements=["rank:1/cuda:1"])
        with self.assertRaisesRegex(ValueError, 'needs to be an integer'):
            _sharded_tensor.empty(spec, 10, 20)

        for dim in [2, 3, 4, -3, -4, -5]:
            spec = ChunkShardingSpec(dim=dim, placements=["rank:1/cuda:1"])
            with self.assertRaisesRegex(ValueError, 'Invalid sharding dim'):
                _sharded_tensor.empty(spec, 10, 20)

        spec = ChunkShardingSpec(dim=0, placements=["rank:5/cuda:1"])
        with self.assertRaisesRegex(ValueError, 'Invalid rank'):
            _sharded_tensor.empty(spec, 10, 20)

        spec = ChunkShardingSpec(dim=0, placements=["rank:0/cuda:1"])
        sharded_tensor = _sharded_tensor.empty(spec, 10, 20)
        tensor = torch.empty(10, 20)
        with self.assertRaisesRegex(RuntimeError, "torch function 'add' not supported for ShardedTensor!"):
            torch.add(sharded_tensor, tensor)

        spec = ChunkShardingSpec(dim=0, placements=["rank:0/cuda:1"])
        with self.assertRaisesRegex(ValueError, 'Only torch.strided layout is currently supported'):
            _sharded_tensor.empty(spec, 10, 20, layout=torch.sparse)

        spec = ChunkShardingSpec(dim=0, placements=["rank:0/cuda:1"])
        with self.assertRaisesRegex(ValueError, 'Only torch.contiguous_format memory_format is currently supported'):
            _sharded_tensor.empty(spec, 10, 20, memory_format=torch.channels_last)

    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_insufficient_sharding_dims(self):
        self.init_pg()

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        sharded_tensor = _sharded_tensor.empty(spec, 2, 20)

        # Validate local shard.
        local_shards = sharded_tensor.local_shards()
        if self.rank <= 1:
            self.assertEqual(1, len(local_shards))
            local_shard = local_shards[0].tensor
            self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.device)
            self.assertEqual((1, 20), local_shard.size())
        else:
            self.assertEqual(0, len(local_shards))

        # Validate global metadata.
        sharding_metadata = sharded_tensor.sharding_metadata()
        self.assertEqual(2, len(sharding_metadata))

        for shard_rank, shard_metadata in enumerate(sharding_metadata):
            self.assertEqual([shard_rank, 0], shard_metadata.shard_offsets)
            self.assertEqual([1, 20], shard_metadata.shard_lengths)
            self.assertEqual(f'rank:{shard_rank}/cuda:{shard_rank}', shard_metadata.placement)
