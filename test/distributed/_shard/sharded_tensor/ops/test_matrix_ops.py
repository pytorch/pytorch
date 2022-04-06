# Owner(s): ["oncall: distributed"]

import copy
import sys

import torch
from torch.distributed._shard import sharded_tensor
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
)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestShardedTensorMatrixOps(ShardedTensorTestBase):
    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_sharded_matrix(self):
        specs = generate_chunk_sharding_specs_for_test(0)
        st = sharded_tensor.rand(specs[0], 16, 5)
        st_2 = sharded_tensor.rand(specs[0], 16, 5)
        st = st.view(-1, 4, 5).contiguous()
        st_2 = st_2.view(-1, 4, 5).contiguous()
        st_3 = torch.bmm(st, st_2.transpose(1, 2))
        self.assertEqual(st_3.size(), torch.Size([4, 4, 4]))
        new_spec = copy.deepcopy(specs[0])
        new_spec.dim = -1
        st_3 = sharded_tensor.rand(new_spec, 3, 4, 11)
        st_4 = sharded_tensor.rand(new_spec, 3, 4, 11)
        st_5 = torch.bmm(st_3, st_4.transpose(1, 2))
        self.assertEqual(st_5.size(), torch.Size([3, 4, 16]))
        local_tensor = st_3.local_tensor()
        dropout = torch.nn.Dropout(0.2)
        local_tensor = torch.nn.functional.softmax(
            local_tensor, dim=-1, dtype=torch.float32
        )
        st = torch.nn.functional.softmax(st_3, dim=-1, dtype=torch.float32)
        self.assertEqual(st.local_tensor(), local_tensor)
        torch.manual_seed(0)
        local_tensor = dropout(local_tensor)
        torch.manual_seed(0)
        st_dropout = dropout(st)
        self.assertEqual(st_dropout.local_tensor(), local_tensor)


if __name__ == "__main__":
    run_tests()
