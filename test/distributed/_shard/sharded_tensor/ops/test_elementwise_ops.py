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


class TestShardedTensorElementWiseOps(ShardedTensorTestBase):
    def _run_sharded_elementwise_ops(self, spec, input_size, op):
        torch.manual_seed(self.rank)
        st = sharded_tensor.rand(spec, *input_size)
        new_st = op(st)
        local_shard = st.local_tensor()
        new_st_local_shard = new_st.local_tensor()
        self.assertEqual(
            op(local_shard),
            new_st_local_shard,
        )

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_sharded_gelu(self):
        specs = generate_chunk_sharding_specs_for_test(
            0
        ) + generate_chunk_sharding_specs_for_test(1)
        for spec in specs:
            self._run_sharded_elementwise_ops(spec, [12, 17], torch.nn.functional.gelu)
            self._run_sharded_elementwise_ops(spec, [18, 21], torch.nn.functional.gelu)
            self._run_sharded_elementwise_ops(spec, [17, 23], torch.nn.functional.gelu)
            self._run_sharded_elementwise_ops(spec, [14, 15], torch.nn.functional.gelu)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_sharded_relu(self):
        specs = generate_chunk_sharding_specs_for_test(
            0
        ) + generate_chunk_sharding_specs_for_test(1)
        for spec in specs:
            self._run_sharded_elementwise_ops(spec, [12, 17], torch.nn.functional.relu)
            self._run_sharded_elementwise_ops(spec, [18, 21], torch.nn.functional.relu)
            self._run_sharded_elementwise_ops(spec, [17, 23], torch.nn.functional.relu)
            self._run_sharded_elementwise_ops(spec, [14, 15], torch.nn.functional.relu)


if __name__ == "__main__":
    run_tests()
