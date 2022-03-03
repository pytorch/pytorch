# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor import (
    _PartialTensor,
)
from torch.distributed._shard.sharding_spec import (
    EnumerableShardingSpec,
    ShardMetadata,
)
from torch.testing._internal.common_distributed import (
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    TEST_WITH_DEV_DBG_ASAN,
    run_tests,
)
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed._shard.sharded_tensor._test_st_common import (
    _chunk_sharding_specs_list_for_test,
)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestPartialTensorReshard(ShardedTensorTestBase):
    def _run_partial_tensor_n_reshard(
        self, reshard_spec, input_size, world_size, reduce_op, dtype=torch.float
    ):
        results = []
        results_compare = []
        for _ in range(0, world_size):
            tensor = torch.rand(*input_size, dtype=dtype).cuda(self.rank)
            results.append(tensor)
            results_compare.append(tensor.clone().detach())
        pg = dist.distributed_c10d._get_default_group()
        parital_tensor = _PartialTensor(torch.cat(results), pg, reduce_op=reduce_op)
        local_sharded_result = parital_tensor.reshard(reshard_spec)
        local_shards = local_sharded_result.local_shards()
        if pg.size() > world_size:
            chunk_mode_res = (input_size[0] * world_size) % pg.size()
            padding = [0] * (len(input_size) * 2)
            padding[-1] = pg.size() - chunk_mode_res
            results_compare = list(
                torch.nn.functional.pad(
                    torch.cat(results_compare),
                    tuple(padding),
                    "constant",
                    0,
                ).chunk(pg.size())
            )
        local_result_compare = torch.empty_like(results_compare[0])
        dist.reduce_scatter(local_result_compare, results_compare, op=reduce_op)
        self.assertEqual(1, len(local_shards))
        self.assertEqual(local_shards[0].tensor, local_result_compare)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_partial_tensor_reshard(self):
        specs = _chunk_sharding_specs_list_for_test([0], seed=7)
        spec = specs[0]
        self._run_partial_tensor_n_reshard(spec, [13, 21], 4, dist.ReduceOp.SUM)
        self._run_partial_tensor_n_reshard(spec, [12, 22], 4, dist.ReduceOp.MAX)
        self._run_partial_tensor_n_reshard(spec, [13, 21], 3, dist.ReduceOp.SUM)
        self._run_partial_tensor_n_reshard(spec, [17, 21], 2, dist.ReduceOp.MAX)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_partial_tensor_reshard_errors(self):
        enumerable_sharding_spec = EnumerableShardingSpec(
            [
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
            ]
        )
        with self.assertRaisesRegex(
            NotImplementedError, "Only ChunkShardingSpec supported for reshard."
        ):
            self._run_partial_tensor_n_reshard(
                enumerable_sharding_spec, [13, 21], 4, dist.ReduceOp.SUM
            )
            self._run_partial_tensor_n_reshard(
                enumerable_sharding_spec, [12, 22], 4, dist.ReduceOp.MAX
            )
        specs = _chunk_sharding_specs_list_for_test([0], seed=7)
        spec = specs[0]
        with self.assertRaisesRegex(
            NotImplementedError, "Only real partial tensor supported for reshard."
        ):
            self._run_partial_tensor_n_reshard(
                spec, [13, 21], 4, dist.ReduceOp.SUM, dtype=torch.cfloat
            )
            self._run_partial_tensor_n_reshard(
                spec, [12, 22], 4, dist.ReduceOp.MAX, dtype=torch.cfloat
            )


if __name__ == "__main__":
    run_tests()
