# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.distributed as dist
from torch.distributed._sharded_tensor import (
    aggregate_partial_tensor_list,
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
    TEST_GPU_NUM,
    ShardedTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed._sharded_tensor._test_ops_common import (
    generate_chunk_sharding_specs_for_test,
    generate_local_weight_sharding_params_for_test,
)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestShardedTensorMegatronLinear(ShardedTensorTestBase):
    def _run_megatron_linear(self, spec, input_size, linear_size):
        # Use same seed.
        torch.manual_seed(0)
        linear_size_first = linear_size[0]
        local_linear_first_layer = torch.nn.Linear(*linear_size_first).cuda(self.rank)
        sharded_linear_first_layer = torch.nn.Linear(*linear_size_first)

        torch.manual_seed(1024)
        linear_size_second = linear_size[1]
        local_linear_second_layer = torch.nn.Linear(*linear_size_second).cuda(self.rank)
        sharded_linear_second_layer = torch.nn.Linear(*linear_size_second)

        # Copy the weights and bias from local linear
        sharded_linear_first_layer.weight = torch.nn.Parameter(local_linear_first_layer.weight.detach().clone())
        sharded_linear_first_layer.bias = torch.nn.Parameter(local_linear_first_layer.bias.detach().clone())
        sharded_linear_second_layer.weight = torch.nn.Parameter(local_linear_second_layer.weight.detach().clone())
        sharded_linear_second_layer.bias = torch.nn.Parameter(local_linear_second_layer.bias.detach().clone())

        # Shard the parameter. First col-wise sharding and then row-wise
        shard_parameter(sharded_linear_first_layer, "weight", spec[0])
        shard_parameter(sharded_linear_second_layer, "weight", spec[1])
        act_func = torch.nn.GELU()

        # Run sharded computation
        torch.manual_seed(self.rank)  # inputs different on each rank
        inp = torch.rand(*input_size).cuda(self.rank)
        temp = sharded_linear_first_layer(inp)
        temp2 = act_func(sharded_linear_first_layer(inp))
        sharded_output = sharded_linear_second_layer(act_func(sharded_linear_first_layer(inp)))
        sharded_output = aggregate_partial_tensor_list(sharded_output, self.rank)

        # Run local computation
        local_output = local_linear_second_layer(act_func(local_linear_first_layer(inp)))

        # Verify
        self.assertEqual(local_output, sharded_output)

        # Compute loss and run backward pass.
        local_output.sum().backward()
        sharded_output.sum().backward()
        local_grad_first = local_linear_first_layer.weight.grad
        local_grad_second = local_linear_second_layer.weight.grad

        # Verify that weights in both layers and biases in the sharded linear has non-None grad.
        sharded_weight_first = sharded_linear_first_layer.weight.local_shards()[0].tensor
        sharded_weight_second = sharded_linear_second_layer.weight.local_shards()[0].tensor
        self.assertNotEqual(sharded_weight_first.grad, None)
        self.assertNotEqual(sharded_weight_second.grad, None)
        self.assertNotEqual(sharded_linear_first_layer.bias.grad, None)
        self.assertNotEqual(sharded_linear_second_layer.bias.grad, None)

        # Shard the local linear's weight grad so that we can compare.
        dist.all_reduce(local_grad_first)
        dist.all_reduce(local_grad_second)
        (start_pos, chunk_size) = generate_local_weight_sharding_params_for_test(
            local_linear_first_layer.weight, 0, TEST_GPU_NUM, spec[0], self.rank
        )
        local_grad_narrowed_first = local_grad_first.narrow(0, start_pos, chunk_size)
        (start_pos, chunk_size) = generate_local_weight_sharding_params_for_test(
            local_linear_second_layer.weight, 1, TEST_GPU_NUM, spec[1], self.rank
        )
        local_grad_narrowed_second = local_grad_second.narrow(1, start_pos, chunk_size)

        # Test backward gradient calculation.
        self.assertEqual(sharded_weight_first.grad, local_grad_narrowed_first)
        self.assertEqual(sharded_weight_second.grad, local_grad_narrowed_second)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_megatron_two_layer_prototype(self):
        colwise_sharding_spec = generate_chunk_sharding_specs_for_test(0)
        rowwise_sharding_spec = generate_chunk_sharding_specs_for_test(1)
        for spec in zip(colwise_sharding_spec, rowwise_sharding_spec):
            self._run_megatron_linear(spec, [22, 17], [[17, 12], [12, 29]])
            self._run_megatron_linear(spec, [28, 21], [[21, 11], [11, 29]])
            self._run_megatron_linear(spec, [37, 23], [[23, 13], [13, 24]])
            self._run_megatron_linear(spec, [24, 15], [[15, 14], [14, 20]])


if __name__ == "__main__":
    run_tests()
