# Owner(s): ["oncall: distributed"]

import sys

import torch
from torch.distributed._shard import (
    shard_parameter,
)
from torch.testing._internal.common_distributed import (
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    TEST_WITH_DEV_DBG_ASAN,
    run_tests,
)
from torch.testing._internal.distributed._sharded_tensor import (
    ShardedTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed._sharded_tensor._test_ops_common import (
    generate_chunk_sharding_specs_for_test,
)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

class TestShardedTensorOpsLinear(ShardedTensorTestBase):
    def _run_sharded_linear(self, spec, input_size, linear_size, sharded_dim):
        # Use same seed.
        torch.manual_seed(0)
        local_linear = torch.nn.Linear(*linear_size).cuda(self.rank)

        sharded_linear = torch.nn.Linear(*linear_size)

        # Copy the weights and bias from local linear
        sharded_linear.weight = local_linear.weight
        sharded_linear.bias = local_linear.bias

        # Shard the parameter.
        shard_parameter(sharded_linear, "weight", spec)

        # Run sharded computation
        torch.manual_seed(self.rank)  # inputs different on each rank
        inp = torch.rand(*input_size).cuda(self.rank)
        sharded_output = sharded_linear(inp)

        # Run local computation
        local_output = local_linear(inp)

        # Verify
        self.assertEqual(local_output, sharded_output)

        # Validate for torch.nn.functional.linear version.
        local_output = torch.nn.functional.linear(
            inp, local_linear.weight, local_linear.bias
        )
        sharded_output = torch.nn.functional.linear(
            inp, sharded_linear.weight, sharded_linear.bias
        )
        self.assertEqual(local_output, sharded_output)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_sharded_linear_colwise(self):
        for spec in generate_chunk_sharding_specs_for_test(0):
            self._run_sharded_linear(spec, [5, 17], [17, 12], 0)
            self._run_sharded_linear(spec, [5, 21], [21, 11], 0)
            self._run_sharded_linear(spec, [5, 23], [23, 13], 0)
            self._run_sharded_linear(spec, [5, 15], [15, 14], 0)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_sharded_linear_rowwise(self):
        for spec in generate_chunk_sharding_specs_for_test(1):
            # Test even split.
            self._run_sharded_linear(spec, [5, 16], [16, 11], 1)

            # Test uneven split.
            self._run_sharded_linear(spec, [5, 19], [19, 11], 1)
            self._run_sharded_linear(spec, [5, 21], [21, 11], 1)

if __name__ == '__main__':
    run_tests()
