# Owner(s): ["oncall: distributed"]

import copy
import sys

import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_optim import (
    ShardedOptimizer,
    named_params_with_sharded_tensor,
)
from torch.distributed._shard import (
    shard_parameter,
)
from torch.distributed._shard.sharded_tensor import (
    _collect_local_shard,
    _reshard_output,
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
    TEST_GPU_NUM,
    ShardedTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed._shard.sharded_tensor._test_ops_common import (
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

    def _run_megatron_linear(self, spec, input_size, linear_size):
        def _weight_override(module_dst, module_src):
            module_dst.fc1.weight = clone_module_parameter(module_src.fc1, "weight")
            module_dst.fc1.bias = clone_module_parameter(module_src.fc1, "bias")
            module_dst.fc2.weight = clone_module_parameter(module_src.fc2, "weight")
            module_dst.fc2.bias = clone_module_parameter(module_src.fc2, "bias")

        def _shard_parameter(module, spec):
            shard_parameter(module.fc1, "weight", spec[0])
            shard_parameter(module.fc2, "weight", spec[1])

        def _get_weight_grad(module):
            return (module.fc1.weight.grad, module.fc2.weight.grad)

        def _get_bias_grad(module):
            return (module.fc1.bias.grad, module.fc2.bias.grad)

        def _get_weights(module):
            return (module.fc1.weight, module.fc2.weight)

        def _get_bias(module):
            return (module.fc1.bias, module.fc2.bias)

        def _get_weight_local_shard(module):
            return (
                module.fc1.weight.local_tensor(),
                module.fc2.weight.local_tensor(),
            )

        # Use same seed.
        torch.manual_seed(0)
        local_megatron_lm = self.SimpleMegatronLM(linear_size, rank=self.rank).cuda(
            self.rank
        )
        sharded_megatron_lm = self.SimpleMegatronLM(linear_size)
        _weight_override(sharded_megatron_lm, local_megatron_lm)

        # Shard the parameter. First col-wise sharding and then row-wise
        _shard_parameter(sharded_megatron_lm, spec)

        # Run sharded computation
        torch.manual_seed(self.rank)  # inputs different on each rank
        inp = torch.rand(*input_size).cuda(self.rank)
        reshard_spec = copy.deepcopy(spec[1])
        reshard_spec.placements.sort(key=lambda placement: placement.rank())
        reshard_spec.dim = 0

        sharded_megatron_lm = _collect_local_shard(
            _reshard_output(sharded_megatron_lm, reshard_spec)
        )
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
        ) = _get_weight_grad(local_megatron_lm)
        local_bias_grad_fc1, local_bias_grad_fc2 = _get_bias_grad(local_megatron_lm)

        # Verify that weights in both layers and biases in the sharded linear has non-None grad.
        (
            sharded_weight_fc1,
            sharded_weight_fc2,
        ) = _get_weight_local_shard(sharded_megatron_lm)
        bias_grad_fc1, bias_grad_fc2 = _get_bias_grad(sharded_megatron_lm)
        self.assertNotEqual(sharded_weight_fc1.grad, None)
        self.assertNotEqual(sharded_weight_fc2.grad, None)
        self.assertNotEqual(bias_grad_fc1, None)
        self.assertNotEqual(bias_grad_fc2, None)

        # Shard the local linear's weight grad so that we can compare.
        dist.all_reduce(local_weight_grad_fc1)
        dist.all_reduce(local_weight_grad_fc2)
        dist.all_reduce(local_bias_grad_fc1)
        dist.all_reduce(local_bias_grad_fc2)
        local_weight_fc1, local_weight_fc2 = _get_weights(local_megatron_lm)
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
        bias_fc1, bias_fc2 = _get_bias(sharded_megatron_lm)
        local_bias_fc1, local_bias_fc2 = _get_bias(local_megatron_lm)
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
            dict(named_params_with_sharded_tensor(sharded_megatron_lm)),
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
        local_bias_fc1, local_bias_fc2 = _get_bias(local_megatron_lm)
        self.assertNotEqual(previous_bias_fc1, bias_fc1)
        self.assertEqual(bias_fc1, local_bias_fc1)
        self.assertNotEqual(previous_bias_fc2, bias_fc2)
        self.assertEqual(bias_fc2, local_bias_fc2)

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
