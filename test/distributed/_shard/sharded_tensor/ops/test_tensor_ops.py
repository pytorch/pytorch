# Owner(s): ["oncall: distributed"]

import copy

import torch
import torch.distributed._shard.sharded_tensor as sharded_tensor
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.testing._internal.common_distributed import requires_nccl, skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    TEST_GPU_NUM,
    with_comms,
)


class TestTensorOps(ShardedTensorTestBase):
    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_deep_copy(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        st = sharded_tensor.rand(spec, (12, 5))
        copied_st = copy.deepcopy(st)
        self.assertTrue(type(copied_st) is type(st))
        self.assertEqual(copied_st.local_tensor(), st.local_tensor())
        self.assertFalse(copied_st is st)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_inplace_copy(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        st = sharded_tensor.rand(spec, (12, 5))
        ones_st = sharded_tensor.ones(spec, (12, 5))
        self.assertFalse(torch.equal(ones_st, st))
        st.copy_(ones_st)
        self.assertTrue(torch.equal(st, ones_st))

        # no grad inplace_copy should work between two with different requires_grad
        st_with_grad = sharded_tensor.rand(spec, (12, 5), requires_grad=True)
        self.assertTrue(st_with_grad.requires_grad)
        self.assertFalse(ones_st.requires_grad)
        with torch.no_grad():
            st_with_grad.copy_(ones_st)
            self.assertEqual(st_with_grad.local_tensor(), ones_st.local_tensor())

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_clone(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        st = sharded_tensor.rand(spec, (12, 5))
        copied_st = st.clone()
        self.assertTrue(type(copied_st) is type(st))
        self.assertEqual(copied_st.local_tensor(), st.local_tensor())
        self.assertFalse(copied_st is st)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_detach(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        st = sharded_tensor.rand(spec, (12, 5), requires_grad=True)
        local_shards = st.local_shards()
        # before set requires_grad, all local shards should not require grads
        for local_shard in local_shards:
            self.assertTrue(local_shard.tensor.requires_grad)

        detached_st = st.detach()
        self.assertFalse(detached_st.requires_grad)

        for local_shard in detached_st.local_shards():
            self.assertFalse(local_shard.tensor.requires_grad)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_set_requires_grad(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        st = sharded_tensor.rand(spec, (12, 5))
        local_shards = st.local_shards()
        # before set requires_grad, all local shards should not require grads
        for local_shard in local_shards:
            self.assertFalse(local_shard.tensor.requires_grad)

        st.requires_grad_()
        self.assertTrue(st.requires_grad)

        for local_shard in local_shards:
            self.assertTrue(local_shard.tensor.requires_grad)


if __name__ == "__main__":
    run_tests()
