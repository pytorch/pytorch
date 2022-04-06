# Owner(s): ["oncall: distributed"]

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


class TestShardedTensorChunkOps(ShardedTensorTestBase):
    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_sharded_chunk(self):
        specs = generate_chunk_sharding_specs_for_test(0)
        st = sharded_tensor.rand(specs[0], 12, 16)
        st.contiguous().view(-1, 4, 4).transpose(1, 2).view(12, 16).contiguous()
        st = st * 0.2
        local_tensor = st.local_tensor()
        local_tensor_chunked = torch.chunk(local_tensor, 4, dim=-1)
        new_st = torch.chunk(st, 4, dim=-1)
        self.assertEqual(len(local_tensor_chunked), len(new_st))
        for idx in range(len(local_tensor_chunked)):
            self.assertEqual(
                local_tensor_chunked[idx],
                new_st[idx].local_tensor(),
            )


if __name__ == "__main__":
    run_tests()
