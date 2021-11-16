# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.distributed as dist
from torch.distributed._sharded_tensor import (
    shard_parameter,
)
from torch.distributed._sharding_spec._internals import (
    get_chunked_dim_size,
    get_split_size,
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
        inp = torch.rand(*input_size, requires_grad=True).cuda(self.rank)
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

        # Generate expected for loss calculation.
        expected_result = torch.randint(0, 2, (input_size[0],)).cuda(self.rank)
        external_grad = (
            torch.ones(input_size[0], dtype=torch.float32).cuda(self.rank)
        )

        # Compute loss.
        local_loss = expected_result - torch.sum(local_output, dim=1)
        sharded_loss = expected_result - torch.sum(sharded_output, dim=1)
        self.assertEqual(local_loss, sharded_loss)

        # Run backward pass
        local_loss.backward(gradient=local_loss)
        sharded_loss.backward(gradient=sharded_loss)

        # Verify that both weight and bias in the sharded linear has non-None grad.
        sharded_weight = sharded_linear.weight.local_shards()[0].tensor
        self.assertEqual(sharded_weight.requires_grad, True)
        self.assertNotEqual(sharded_linear.bias.grad, None)
        self.assertNotEqual(sharded_weight.grad, None)

        # Shard the local linear's weight grad so that we can compare.
        local_grad = local_linear.weight.grad
        dist.all_reduce(local_grad)
        sharding_dim_size = local_grad.size(sharded_dim)
        split_size = get_split_size(sharding_dim_size, 4)
        current_offsets = 0
        start_pos = current_offsets
        for idx, placement in enumerate(spec.placements):
            chunked_dim_size = get_chunked_dim_size(sharding_dim_size, split_size, idx)
            if self.rank == placement.rank():
                start_pos = current_offsets
                break
            current_offsets += chunked_dim_size
        local_grad_narrowed = local_grad.narrow(
            sharded_dim, start_pos, chunked_dim_size
        )
        local_weight_narrowed = local_linear.weight.narrow(
            sharded_dim, start_pos, chunked_dim_size
        )

        # Test backward gradient calculation.
        self.assertEqual(sharded_linear.bias.grad, local_linear.bias.grad)
        self.assertEqual(sharded_weight, local_weight_narrowed)
        self.assertEqual(sharded_weight.grad, local_grad_narrowed * 0.5)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_sharded_linear_colwise(self):
        for spec in generate_chunk_sharding_specs_for_test(0):
            self._run_sharded_linear(spec, [5, 17], [17, 12], 0)
            self._run_sharded_linear(spec, [8, 21], [21, 11], 0)
            self._run_sharded_linear(spec, [7, 23], [23, 13], 0)
            self._run_sharded_linear(spec, [4, 15], [15, 14], 0)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_sharded_linear_rowwise(self):
        for spec in generate_chunk_sharding_specs_for_test(1):
            # Test even split.
            self._run_sharded_linear(spec, [8, 16], [16, 11], 1)

            # Test uneven split.
            self._run_sharded_linear(spec, [5, 19], [19, 11], 1)
            self._run_sharded_linear(spec, [10, 21], [21, 11], 1)


if __name__ == "__main__":
    run_tests()
