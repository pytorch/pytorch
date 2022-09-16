# Owner(s): ["oncall: distributed"]

import sys
import torch
from torch.testing._internal.common_utils import (
    TEST_WITH_DEV_DBG_ASAN,
    run_tests,
)
from torch.testing._internal.distributed._shard.sharded_tensor import (
    TEST_GPU_NUM,
    ShardedTensorTestBase,
    with_comms,
)
from torch.testing._internal.common_distributed import (
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed._shard import _shard_tensor

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestShardedSoftmax(ShardedTensorTestBase):

    def _test_sharded_softmax(self, softmax_dim, sharding_dim):
        torch.manual_seed(0)
        local_tensor = torch.rand(10, 10, device=self.rank)
        local_softmax = torch.nn.functional.softmax(local_tensor, softmax_dim)

        spec = ChunkShardingSpec(dim=sharding_dim, placements=[f'rank:{idx}/cuda:{idx}' for idx in range(self.world_size)])
        st = _shard_tensor(local_tensor, spec)
        sharded_softmax = torch.nn.functional.softmax(st, softmax_dim)

        self.assertEqual(local_softmax.chunk(self.world_size, dim=sharding_dim)[self.rank], sharded_softmax.local_tensor())

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_sharded_softmax_basic(self):
        self._test_sharded_softmax(0, 1)
        self._test_sharded_softmax(-2, 1)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_sharded_softmax_on_sharding_dim(self):
        self._test_sharded_softmax(1, 1)
        self._test_sharded_softmax(-1, 1)

if __name__ == "__main__":
    run_tests()
