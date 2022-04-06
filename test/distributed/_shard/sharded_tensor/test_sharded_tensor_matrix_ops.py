# Owner(s): ["oncall: distributed"]

import sys

import torch
from torch.distributed._shard import (
    sharded_tensor,
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


class TestShardedTernsorMatrixOps(ShardedTensorTestBase):
    @with_comms(init_rpc=True)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_sharded_tensor_transpose(self):
        specs = _chunk_sharding_specs_list_for_test([0], seed=7)
        for spec in specs:
            st = sharded_tensor.rand(spec, 10, 20, 5, init_rrefs=True)
            self.assertEqual(3, st.dim())
            self.assertEqual(torch.Size([10, 20, 5]), st.size())
            local_tensor = st.local_tensor()
            st = st.transpose(1, 0)
            spec.dim = 1
            local_tensor = local_tensor.transpose(1, 0)
            st_2 = sharded_tensor.rand(spec, 20, 10, 5, init_rrefs=True)
            self.assertEqual(st_2.size(), st.size())
            self.assertEqual(local_tensor, st.local_tensor())
            self.assertEqual(st_2.metadata(), st.metadata())

    @with_comms(init_rpc=True)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_sharded_tensor_contiguous(self):
        specs = _chunk_sharding_specs_list_for_test([0], seed=7)
        for spec in specs:
            st = sharded_tensor.rand(spec, 10, 22, 5, init_rrefs=True)
            st = st.transpose(1, 0)
            st = st.contiguous()
            self.assertTrue(st.is_contiguous())
            self.assertTrue(st.local_tensor().is_contiguous())

    @with_comms(init_rpc=True)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_sharded_tensor_masked_fill(self):
        specs = _chunk_sharding_specs_list_for_test([0], seed=7)
        for spec in specs:
            st = sharded_tensor.rand(spec, 16, 30, 5, init_rrefs=True)
            local_tensor = st.local_tensor().clone().detach()
            mask = torch.zeros(st.size())
            mask[:, 1, :] = 1
            mask = mask.type(torch.ByteTensor).cuda()
            local_tensor = local_tensor.masked_fill(mask.narrow(0, 0, 4).cuda(), 5.0)
            st = st.masked_fill(mask, 5.0)
            self.assertEqual(local_tensor, st.local_tensor())

    @with_comms(init_rpc=True)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_sharded_tensor_type_as(self):
        specs = _chunk_sharding_specs_list_for_test([0], seed=7)
        for spec in specs:
            st = sharded_tensor.rand(spec, 16, 30, 5, init_rrefs=True, dtype=torch.double)
            st_2 = sharded_tensor.rand(spec, 16, 30, 5, init_rrefs=True, dtype=torch.float)
            st_3 = st.type_as(st_2)
            self.assertEqual(torch.float, st_3.dtype)
            self.assertEqual(torch.float, st_3.local_tensor().dtype)
            st_3 = st.type_as(torch.zeros(10).type(torch.ByteTensor).cuda())
            self.assertEqual(torch.uint8, st_3.dtype)
            self.assertEqual(torch.uint8, st_3.local_tensor().dtype)

    @with_comms(init_rpc=True)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_sharded_tensor_view(self):
        specs = _chunk_sharding_specs_list_for_test([0], seed=7)
        for spec in specs:
            st = sharded_tensor.rand(spec, 16, 30, 5, init_rrefs=True)
            st_2 = st.view(16, 15, 2, 5)
            st_3 = st_2.view(16, 15, -1)
            self.assertEqual(torch.Size([16, 15, 2, 5]), st_2.size())
            self.assertEqual(torch.Size([16, 15, 10]), st_3.size())


if __name__ == "__main__":
    run_tests()
