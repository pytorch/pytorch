# Owner(s): ["oncall: distributed"]

import sys

import torch
from torch.distributed._shard import sharded_tensor, _shard_tensor
from torch.testing._internal.common_distributed import (
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    TEST_WITH_DEV_DBG_ASAN,
    run_tests,
)
from torch.testing._internal.distributed._shard.sharded_tensor import (
    TEST_GPU_NUM,
    ShardedTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed._shard.sharded_tensor._test_ops_common import (
    generate_chunk_sharding_specs_for_test,
    generate_enumerable_sharding_specs_for_test,
)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestShardedTensorChunkOps(ShardedTensorTestBase):
    def _compare_chunk_result(self, chunked_list, chunked_st_list):
        self.assertEqual(len(chunked_list), len(chunked_st_list))
        for idx, chunked_st in enumerate(chunked_st_list):
            tensor = chunked_list[idx]
            st = _shard_tensor(tensor.contiguous(), chunked_st.sharding_spec())
            # _shard_tensor generate sharded tensor with metadata ranked by # of rank.
            st._metadata.shards_metadata.sort(
                key=lambda x: x.shard_offsets[chunked_st.sharding_spec().dim],
            )
            self.assertTrue(torch.allclose(chunked_st, st))

    def _run_sharded_chunk_test(self, local_tensor_size, shard_spec, chunk_num):
        torch.manual_seed(0)
        local_tensor = torch.rand(*local_tensor_size).cuda(self.rank)
        st_tensor = _shard_tensor(local_tensor.clone().detach(), shard_spec)
        local_tensor_chunked = torch.chunk(local_tensor, chunk_num, dim=-1)
        chunked_st = torch.chunk(st_tensor, chunk_num, dim=-1)
        self._compare_chunk_result(local_tensor_chunked, chunked_st)
        chunked_st = st_tensor.chunk(chunk_num, dim=-1)
        self._compare_chunk_result(local_tensor_chunked, chunked_st)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_sharded_chunk(self):
        sharding_dims = [0]
        specs = []
        for dim in sharding_dims:
            specs.extend(generate_chunk_sharding_specs_for_test(dim))
        for spec in specs:
            self._run_sharded_chunk_test([17, 14], spec, 3)
            self._run_sharded_chunk_test([17, 15, 20], spec, 5)
            self._run_sharded_chunk_test([17, 16], spec, 2)
            # Large matrix case.
            self._run_sharded_chunk_test([128, 512], spec, 8)
            self._run_sharded_chunk_test([1024, 2048], spec, 4)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_sharded_chunk_error(self):
        chunk_spec = generate_chunk_sharding_specs_for_test(-1)
        with self.assertRaisesRegex(
            NotImplementedError, "Chunk by sharding dim is not supported."
        ):
            st = sharded_tensor.rand(chunk_spec[0], [17, 24])
            torch.chunk(st, 5, dim=-1)
        enumerable_spec = generate_enumerable_sharding_specs_for_test()
        with self.assertRaisesRegex(
            NotImplementedError, "Only ChunkShardingSpec is supported for chunk."
        ):
            st = sharded_tensor.rand(enumerable_spec[0], [10, 10])
            torch.chunk(st, 5, dim=-1)


if __name__ == "__main__":
    run_tests()
