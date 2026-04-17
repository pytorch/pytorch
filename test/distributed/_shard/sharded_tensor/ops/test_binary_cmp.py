# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.distributed as dist
from torch.distributed._shard import sharded_tensor
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed.distributed_c10d import _get_default_group
from torch.testing._internal.common_distributed import requires_nccl, skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    with_comms,
)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestShardedTensorBinaryOps(ShardedTensorTestBase):
    """Test base for binary comparison functions such as torch.equal, torch.allclose etc. for ShardedTensor"""

    seed = 42

    def get_random_tensors(
        self, spec1, spec2, *sizes, pg1=None, pg2=None, seed_offset=0
    ):
        pg1 = _get_default_group() if pg1 is None else pg1
        pg2 = _get_default_group() if pg2 is None else pg2
        torch.manual_seed(TestShardedTensorBinaryOps.seed)
        st1 = sharded_tensor.rand(spec1, sizes, process_group=pg1)
        torch.manual_seed(TestShardedTensorBinaryOps.seed + seed_offset)
        st2 = sharded_tensor.rand(spec2, sizes, process_group=pg2)

        TestShardedTensorBinaryOps.seed += 1
        return st1, st2

    def get_gpu_specs(self):
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
        return spec, alt_spec

    def _test_common_failures(self, cmp_op):
        spec, alt_spec = self.get_gpu_specs()

        st1, st2 = self.get_random_tensors(spec, spec, 10, 10)
        if self.rank == 0:
            torch.nn.init.uniform_(st1.local_shards()[0].tensor)
        self.assertFalse(cmp_op(st1, st2))

        st1 = sharded_tensor.ones(spec, 10, 10)
        st2 = sharded_tensor.ones(spec, 10, 5)
        self.assertFalse(cmp_op(st1, st2))

        st1, st2 = self.get_random_tensors(spec, alt_spec, 10, 10)
        self.assertFalse(cmp_op(st1, st2))

        st1 = sharded_tensor.ones(spec, 10, 10)
        st2 = sharded_tensor.zeros(spec, 10, 10)
        self.assertFalse(cmp_op(st1, st2))

        st1 = sharded_tensor.ones(spec, 10, 10)
        st2 = sharded_tensor.ones(spec, 10, 10, dtype=torch.double)
        self.assertFalse(cmp_op(st1, st2))

        st1 = sharded_tensor.ones(spec, 10, 10)
        st2 = sharded_tensor.ones(spec, 10, 10, requires_grad=True)
        self.assertFalse(cmp_op(st1, st2))

        cpu_spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cpu",
                "rank:1/cpu",
                "rank:2/cpu",
                "rank:3/cpu",
            ],
        )
        st1 = sharded_tensor.ones(cpu_spec, 10, 10)
        st2 = sharded_tensor.ones(cpu_spec, 10, 10, pin_memory=True)
        self.assertFalse(cmp_op(st1, st2))

        pg = dist.new_group([1, 0, 3, 2])
        st1, st2 = self.get_random_tensors(spec, spec, 10, 10, pg2=pg)
        with self.assertRaisesRegex(
            RuntimeError, "All distributed tensors should use the same ProcessGroup"
        ):
            cmp_op(st1, st2)

        pg = dist.new_group([0, 1, 2, 3])
        st1, st2 = self.get_random_tensors(spec, spec, 10, 10, pg2=pg)
        with self.assertRaisesRegex(
            RuntimeError, "All distributed tensors should use the same ProcessGroup"
        ):
            cmp_op(st1, st2)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_torch_equal_tensor_specs(self):
        self._test_common_failures(torch.equal)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_torch_equal(self):
        """Test torch.equal(ShardedTensor, ShardedTensor)"""

        spec, _ = self.get_gpu_specs()
        st1, st2 = self.get_random_tensors(spec, spec, 10, 10)
        self.assertTrue(torch.equal(st1, st2))

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_torch_allclose_tensor_specs(self):
        self._test_common_failures(torch.allclose)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_torch_allclose(self):
        """Test torch.allclose(ShardedTensor, ShardedTensor)"""

        spec, _ = self.get_gpu_specs()

        st1, st2 = self.get_random_tensors(spec, spec, 10, 10)
        self.assertTrue(torch.allclose(st1, st2))
        self.assertTrue(torch.allclose(st1, st2, atol=0))

        # compare different arrays
        st1, st2 = self.get_random_tensors(spec, spec, 10, 10, seed_offset=1)
        self.assertFalse(torch.allclose(st1, st2))
        # sharded_tensor.rand produces uniform values in the [0,1] range.
        self.assertTrue(torch.allclose(st1, st2, atol=1))


if __name__ == "__main__":
    run_tests()
