# Owner(s): ["oncall: distributed"]

import sys
import torch
import torch.distributed as dist

from torch.distributed import _sharded_tensor
from torch.distributed._sharding_spec import (
    ChunkShardingSpec,
)
from torch.testing._internal.common_distributed import (
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.distributed._sharded_tensor import (
    ShardedTensorTestBase,
    with_comms,
)
from torch.testing._internal.common_utils import (
    TEST_WITH_DEV_DBG_ASAN,
    run_tests,
)

from torch.distributed.distributed_c10d import _get_default_group


if TEST_WITH_DEV_DBG_ASAN:
    print("Skip dev-asan as torch + multiprocessing spawn have known issues", file=sys.stderr)
    sys.exit(0)

class TestShardedTensorEquals(ShardedTensorTestBase):
    """ Testing torch.equal, torch.allclose etc. functions for ShardedTensor """
    seed = 42

    def get_random_tensors(self, spec1, spec2, *sizes, pg1=None, pg2=None):
        pg1 = _get_default_group() if pg1 is None else pg1
        pg2 = _get_default_group() if pg2 is None else pg2
        torch.manual_seed(TestShardedTensorEquals.seed)
        st1 = _sharded_tensor.rand(spec1, sizes, process_group=pg1)
        torch.manual_seed(TestShardedTensorEquals.seed)
        st2 = _sharded_tensor.rand(spec2, sizes, process_group=pg2)

        TestShardedTensorEquals.seed += 1
        return st1, st2

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_torch_equal(self):
        """ Test torch.equal(ShardedTensor, ShardedTensor) """

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )

        alt_spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:1/cuda:1",
                "rank:0/cuda:0",
                "rank:3/cuda:3",
                "rank:2/cuda:2",
            ],
        )

        st1, st2 = self.get_random_tensors(spec, spec, 10, 10)
        self.assertTrue(torch.equal(st1, st2))

        st1, st2 = self.get_random_tensors(spec, spec, 10, 10)
        if self.rank == 0:
            torch.nn.init.uniform_(st1.local_shards()[0].tensor)
        self.assertFalse(torch.equal(st1, st2))

        st1 = _sharded_tensor.ones(spec, 10, 10)
        st2 = _sharded_tensor.ones(spec, 10, 5)
        self.assertFalse(torch.equal(st1, st2))

        st1, st2 = self.get_random_tensors(spec, alt_spec, 10, 10)
        self.assertFalse(torch.equal(st1, st2))

        st1 = _sharded_tensor.ones(spec, 10, 10)
        st2 = _sharded_tensor.zeros(spec, 10, 10)
        self.assertFalse(torch.equal(st1, st2))

        st1 = _sharded_tensor.ones(spec, 10, 10)
        st2 = _sharded_tensor.ones(spec, 10, 10, dtype=torch.double)
        self.assertFalse(torch.equal(st1, st2))

        st1 = _sharded_tensor.ones(spec, 10, 10)
        st2 = _sharded_tensor.ones(spec, 10, 10, requires_grad=True)
        self.assertFalse(torch.equal(st1, st2))

        cpu_spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cpu",
                "rank:1/cpu",
                "rank:2/cpu",
                "rank:3/cpu",
            ],
        )
        st1 = _sharded_tensor.ones(cpu_spec, 10, 10)
        st2 = _sharded_tensor.ones(cpu_spec, 10, 10, pin_memory=True)
        self.assertFalse(torch.equal(st1, st2))

        pg = dist.new_group([1, 0, 3, 2])
        st1, st2 = self.get_random_tensors(spec, spec, 10, 10, pg2=pg)
        self.assertFalse(torch.equal(st1, st2))

        pg = dist.new_group([0, 1, 2, 3])
        st1, st2 = self.get_random_tensors(spec, spec, 10, 10, pg2=pg)
        self.assertFalse(torch.equal(st1, st2))

if __name__ == '__main__':
    run_tests()
