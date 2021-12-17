# Owner(s): ["oncall: distributed"]

import copy
import sys

import torch
import torch.distributed as dist
from torch.distributed._sharded_optim import (
    ShardedOptimizer,
    named_params_with_sharded_tensor,
)
from torch.distributed._sharded_tensor import (
    shard_parameter,
    reshard_output,
)
from torch.distributed._sharding_spec import (
    ReshardingSpec,
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
    clone_module_parameter,
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
    class SimpleMegatronLM(torch.nn.Module):
        def __init__(self, linear_size, rank=None):
            super().__init__()
            self.fc1 = torch.nn.Linear(*linear_size[0])
            self.gelu = torch.nn.GELU()
            self.fc2 = torch.nn.Linear(*linear_size[1])
            if rank:
                self.fc1.cuda(rank)
                self.fc2.cuda(rank)

        def forward(self, inp):
            return self.fc2(self.gelu(self.fc1(inp)))

        def weight_override(self, module):
            self.fc1.weight = clone_module_parameter(module.fc1, "weight")
            self.fc1.bias = clone_module_parameter(module.fc1, "bias")
            self.fc2.weight = clone_module_parameter(module.fc2, "weight")
            self.fc2.bias = clone_module_parameter(module.fc2, "bias")

        def shard_parameter(self, spec):
            shard_parameter(self.fc1, "weight", spec[0])
            shard_parameter(self.fc2, "weight", spec[1])

        def get_weight_grad(self):
            return (self.fc1.weight.grad, self.fc2.weight.grad)

        def get_bias_grad(self):
            return (self.fc1.bias.grad, self.fc2.bias.grad)

        def get_weights(self):
            return (self.fc1.weight, self.fc2.weight)

        def get_bias(self):
            return (self.fc1.bias, self.fc2.bias)

        def get_weight_local_shard(self):
            return (
                self.fc1.weight.local_shards()[0].tensor,
                self.fc2.weight.local_shards()[0].tensor,
            )

    def _run_megatron_linear(self, spec, input_size, linear_size):
        # Use same seed.
        torch.manual_seed(0)
        local_megatron_lm = self.SimpleMegatronLM(linear_size, rank=self.rank).cuda(
            self.rank
        )
        sharded_megatron_lm = self.SimpleMegatronLM(linear_size)
        sharded_megatron_lm.weight_override(local_megatron_lm)

        # Shard the parameter. First col-wise sharding and then row-wise
        sharded_megatron_lm.shard_parameter(spec)

        # Run sharded computation
        torch.manual_seed(self.rank)  # inputs different on each rank
        inp = torch.rand(*input_size).cuda(self.rank)
        resharding_spec = ReshardingSpec(copy.deepcopy(spec[1]), True)
        sharded_megatron_lm = reshard_output(sharded_megatron_lm, resharding_spec)
        sharded_output = sharded_megatron_lm(inp)

        # Run local computation
        local_output = local_megatron_lm(inp)

        # Verify
        self.assertEqual(local_output, sharded_output)

        # Compute loss and run backward pass.
        local_output.sum().backward()
        sharded_output.sum().backward()
        (
            local_weight_grad_fc1,
            local_weight_grad_fc2,
        ) = local_megatron_lm.get_weight_grad()
        local_bias_grad_fc1, local_bias_grad_fc2 = local_megatron_lm.get_bias_grad()

        # Verify that weights in both layers and biases in the sharded linear has non-None grad.
        sharded_megatron_lm_original = sharded_megatron_lm.original_module
        (
            sharded_weight_fc1,
            sharded_weight_fc2,
        ) = sharded_megatron_lm_original.get_weight_local_shard()
        bias_grad_fc1, bias_grad_fc2 = sharded_megatron_lm_original.get_bias_grad()
        self.assertNotEqual(sharded_weight_fc1.grad, None)
        self.assertNotEqual(sharded_weight_fc2.grad, None)
        self.assertNotEqual(bias_grad_fc1, None)
        self.assertNotEqual(bias_grad_fc2, None)

        # Shard the local linear's weight grad so that we can compare.
        dist.all_reduce(local_weight_grad_fc1)
        dist.all_reduce(local_weight_grad_fc2)
        dist.all_reduce(local_bias_grad_fc1)
        dist.all_reduce(local_bias_grad_fc2)
        local_weight_fc1, local_weight_fc2 = local_megatron_lm.get_weights()
        (
            start_pos_fc1,
            chunk_size_fc1,
        ) = generate_local_weight_sharding_params_for_test(
            local_weight_fc1, 0, TEST_GPU_NUM, spec[0], self.rank
        )
        local_grad_narrowed_fc1 = local_weight_grad_fc1.narrow(
            0, start_pos_fc1, chunk_size_fc1
        )
        (
            start_pos_fc2,
            chunk_size_fc2,
        ) = generate_local_weight_sharding_params_for_test(
            local_weight_fc2, 1, TEST_GPU_NUM, spec[1], self.rank
        )
        local_grad_narrowed_fc2 = local_weight_grad_fc2.narrow(
            1, start_pos_fc2, chunk_size_fc2
        )

        # Test backward gradient calculation.
        self.assertEqual(sharded_weight_fc1.grad, local_grad_narrowed_fc1)
        self.assertEqual(sharded_weight_fc2.grad, local_grad_narrowed_fc2)
        self.assertEqual(bias_grad_fc1, local_bias_grad_fc1)
        self.assertEqual(bias_grad_fc2, local_bias_grad_fc2)

        # Test optimizer.
        bias_fc1, bias_fc2 = sharded_megatron_lm_original.get_bias()
        local_bias_fc1, local_bias_fc2 = local_megatron_lm.get_bias()
        self.assertEqual(bias_fc1, local_bias_fc1)
        self.assertEqual(bias_fc2, local_bias_fc2)
        self.assertEqual(bias_fc1.grad, local_bias_fc1.grad)
        self.assertEqual(bias_fc2.grad, local_bias_fc2.grad)
        previous_sharded_weight_fc1 = sharded_weight_fc1.clone()
        previous_sharded_weight_fc2 = sharded_weight_fc2.clone()
        previous_bias_fc1 = bias_fc1.clone()
        previous_bias_fc2 = bias_fc2.clone()
        optim = torch.optim.SGD(local_megatron_lm.parameters(), lr=0.1)
        optim.step()
        sharded_optim = ShardedOptimizer(
            dict(named_params_with_sharded_tensor(sharded_megatron_lm_original)),
            torch.optim.SGD,
            lr=0.1,
        )
        sharded_optim.step()
        local_weight_fc1_narrowed = local_weight_fc1.narrow(
            0, start_pos_fc1, chunk_size_fc1
        )
        local_weight_fc2_narrowed = local_weight_fc2.narrow(
            1, start_pos_fc2, chunk_size_fc2
        )

        # Test weight value after optimizer.
        self.assertEqual(sharded_weight_fc1.size(), local_weight_fc1_narrowed.size())
        self.assertEqual(sharded_weight_fc2.size(), local_weight_fc2_narrowed.size())
        self.assertNotEqual(previous_sharded_weight_fc1, sharded_weight_fc1)
        self.assertNotEqual(previous_sharded_weight_fc2, sharded_weight_fc2)
        self.assertEqual(sharded_weight_fc1, local_weight_fc1_narrowed)
        self.assertEqual(sharded_weight_fc2, local_weight_fc2_narrowed)

        # Test bias value after optimizer.
        local_bias_fc1, local_bias_fc2 = local_megatron_lm.get_bias()
        self.assertNotEqual(previous_bias_fc1, bias_fc1)
        self.assertEqual(bias_fc1, local_bias_fc1)
        self.assertNotEqual(previous_bias_fc2, bias_fc2)
        # Sharded bias get double updated here for fc2.
        # self.assertEqual(bias_fc2, local_bias_fc2)

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
