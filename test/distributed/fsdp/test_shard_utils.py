# Owner(s): ["oncall: distributed"]

import torch
from torch.distributed._shard.sharded_tensor import (
    init_from_local_shards,
    Shard,
    ShardMetadata,
)
from torch.distributed._shard.sharding_spec import (
    ChunkShardingSpec,
    EnumerableShardingSpec,
)
from torch.distributed.distributed_c10d import _get_default_group
from torch.distributed.fsdp._shard_utils import (
    _create_chunk_sharded_tensor,
    _offsets_to_split_sizes,
    _reshard_flatten_tensor,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import TestCase


class TestShardUtils(TestCase):
    def test_offsets_to_split_sizes(self):
        tensor_numel = 40

        def _get_and_check_split_sizes(
            world_size,
            in_offsets,
            out_offsets,
            in_split_sizes,
        ):

            for my_rank in range(world_size):
                _in_split_sizes = in_split_sizes[my_rank]
                _out_split_sizes = [
                    in_split_sizes[i][my_rank] for i in range(world_size)
                ]
                res_in_split_sizes, res_out_split_sizes = _offsets_to_split_sizes(
                    in_offsets, out_offsets, tensor_numel, world_size, my_rank
                )
                self.assertEqual(_in_split_sizes, res_in_split_sizes)
                self.assertEqual(_out_split_sizes, res_out_split_sizes)

        # The tensor size can be evenly divided by the world size.
        world_size = 4
        in_offsets = [0, 10, 20, 30]
        out_offsets = [0, 10, 20, 30]
        in_split_sizes = [
            [10, 0, 0, 0],
            [0, 10, 0, 0],
            [0, 0, 10, 0],
            [0, 0, 0, 10],
        ]
        _get_and_check_split_sizes(world_size, in_offsets, out_offsets, in_split_sizes)

        world_size = 4
        in_offsets = [0, 3, 17, 18]
        out_offsets = [0, 10, 20, 30]
        in_split_sizes = [
            [3, 0, 0, 0],
            [7, 7, 0, 0],
            [0, 1, 0, 0],
            [0, 2, 10, 10],
        ]
        _get_and_check_split_sizes(world_size, in_offsets, out_offsets, in_split_sizes)

        world_size = 4
        in_offsets = [0, 10, 20, 30]
        out_offsets = [0, 3, 17, 18]
        in_split_sizes = [
            [3, 7, 0, 0],
            [0, 7, 1, 2],
            [0, 0, 0, 10],
            [0, 0, 0, 10],
        ]
        _get_and_check_split_sizes(world_size, in_offsets, out_offsets, in_split_sizes)

        world_size = 4
        in_offsets = [0, 7, 11, 25]
        out_offsets = [0, 10, 17, 18]
        in_split_sizes = [
            [7, 0, 0, 0],
            [3, 1, 0, 0],
            [0, 6, 1, 7],
            [0, 0, 0, 15],
        ]
        _get_and_check_split_sizes(world_size, in_offsets, out_offsets, in_split_sizes)

        # The tensor size cannot be evenly divided by the world size.
        world_size = 6
        in_offsets = [0, 7, 14, 21, 28, 35]
        out_offsets = [0, 7, 14, 21, 28, 35]
        in_split_sizes = [
            [7, 0, 0, 0, 0, 0],
            [0, 7, 0, 0, 0, 0],
            [0, 0, 7, 0, 0, 0],
            [0, 0, 0, 7, 0, 0],
            [0, 0, 0, 0, 7, 0],
            [0, 0, 0, 0, 0, 5],
        ]
        _get_and_check_split_sizes(world_size, in_offsets, out_offsets, in_split_sizes)

        world_size = 6
        in_offsets = [0, 0, 10, 11, 28, 40]
        out_offsets = [0, 7, 14, 21, 28, 35]
        in_split_sizes = [
            [0, 0, 0, 0, 0, 0],
            [7, 3, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 3, 7, 7, 0, 0],
            [0, 0, 0, 0, 7, 5],
            [0, 0, 0, 0, 0, 0],
        ]
        _get_and_check_split_sizes(world_size, in_offsets, out_offsets, in_split_sizes)


class TestShardUtilsDistributed(FSDPTest):
    @property
    def world_size(self):
        return 2

    def _create_local_chunk(self, tensor):
        chunk = tensor.chunk(2)[self.rank]
        offsets = [0] if self.rank == 0 else [tensor.shape[0] - chunk.shape[0]]
        shard = Shard.from_tensor_and_offsets(chunk, offsets, self.rank)
        return init_from_local_shards([shard], tensor.numel())

    def _create_enumerate_spec(self, tensor):
        # Since placement is not used, always set placement to rank0 to mimic
        # the actual usage.
        metadata = [
            ShardMetadata([0], [101], placement="rank0/cuda:0"),
            ShardMetadata([101], [900], placement="rank0/cuda:0"),
        ]
        return EnumerableShardingSpec(metadata)

    def _create_chunk_spec(self):
        return ChunkShardingSpec(dim=0, placements=["rank0/cuda:0"])

    def _create_tensor(self, *size):
        # Keep everything deterministic.
        torch.manual_seed(0)
        return torch.rand(*size).cuda()

    @skip_if_lt_x_gpu(2)
    def test_reshard_flatten_tensor(self):
        def get_offsets(tensor, shard):
            if self.rank == 0:
                return [0]
            else:
                return [tensor.shape[0] - shard.shape[0]]

        tensor = self._create_tensor(1001)

        shard = _reshard_flatten_tensor(
            self._create_local_chunk(tensor),
            self._create_enumerate_spec(tensor),
            self.world_size,
            self.rank,
            tensor.device,
            _get_default_group(),
        )
        offsets = [0] if self.rank == 0 else [tensor.shape[0] - shard.shape[0]]
        shard = Shard.from_tensor_and_offsets(shard, offsets, self.rank)
        uneven_sharded_tensor = init_from_local_shards([shard], tensor.numel())

        shard = _reshard_flatten_tensor(
            uneven_sharded_tensor,
            self._create_chunk_spec(),
            self.world_size,
            self.rank,
            tensor.device,
            _get_default_group(),
        )
        offsets = [0] if self.rank == 0 else [tensor.shape[0] - shard.shape[0]]
        shard = Shard.from_tensor_and_offsets(shard, offsets, self.rank)
        even_sharded_tensor = init_from_local_shards([shard], tensor.numel())

        output = torch.empty(tensor.shape).cuda() if self.rank == 0 else None
        even_sharded_tensor.gather(0, output)
        if self.rank == 0:
            self.assertEqual(tensor, output)
        output = torch.empty(tensor.shape).cuda() if self.rank == 0 else None
        uneven_sharded_tensor.gather(0, output)
        if self.rank == 0:
            self.assertEqual(tensor, output)

    @skip_if_lt_x_gpu(2)
    def test_create_chunk_sharded_tensor(self):
        for size in ((1,), (1, 6), (12,), (12, 6), (25,), (25, 6)):
            tensor = self._create_tensor(*size)

            sharded_tensor = _create_chunk_sharded_tensor(
                tensor,
                self.rank,
                self.world_size,
                torch.cuda.device_count(),
                _get_default_group(),
            )
            output = torch.empty(*size).cuda() if self.rank == 0 else None
            sharded_tensor.gather(0, output)
            if self.rank == 0:
                self.assertEqual(tensor, output)
