# Owner(s): ["oncall: distributed"]

import copy
import sys

import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_optim import (
    ShardedOptimizer,
)
from torch.distributed._shard.api import (
    shard_parameter,
    _reshard_output,
    _collect_local_shard
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
from torch.testing._internal.distributed._shard.test_common import SimpleMegatronLM

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

class TestShardedTensorMegatronLinear(ShardedTensorTestBase):
    def assertEdistNorm(self, t1, t2):
        """
        Use a normalized euclidean distance measure to validate two tensors
        are close since comparing each element individually is not a good
        measure where majority of elements are similar and maybe only a few
        elements are slightly off.
        """
        dist = torch.sqrt(((t1 - t2) ** 2).sum() / t1.numel())
        self.assertTrue(dist.item() <= 0.5)

    def _run_megatron_linear(self, spec, input_size, linear_size, dtype):
        def _weight_override(module_dst, module_src):
            module_dst.fc1.weight = clone_module_parameter(module_src.fc1, "weight")
            module_dst.fc1.bias = clone_module_parameter(module_src.fc1, "bias")
            module_dst.fc2.weight = clone_module_parameter(module_src.fc2, "weight")
            module_dst.fc2.bias = clone_module_parameter(module_src.fc2, "bias")

        def _shard_parameter(module, spec):
            shard_parameter(module.fc1, "weight", spec[0])
            shard_parameter(module.fc2, "weight", spec[1])

        # Use same seed.
        torch.manual_seed(0)
        local_megatron_lm = SimpleMegatronLM(linear_size, rank=self.rank, dtype=dtype)
        sharded_megatron_lm = SimpleMegatronLM(linear_size, dtype=dtype)
        _weight_override(sharded_megatron_lm, local_megatron_lm)

        # Shard the parameter. First col-wise sharding and then row-wise
        _shard_parameter(sharded_megatron_lm, spec)

        # Setup resharding of output.
        reshard_spec = copy.deepcopy(spec[1])
        reshard_spec.placements.sort(key=lambda placement: placement.rank())
        reshard_spec.dim = 0

        sharded_megatron_lm = _collect_local_shard(
            _reshard_output(sharded_megatron_lm, reshard_spec)
        )


        torch.manual_seed(self.rank)  # inputs different on each rank
        inp = torch.rand(*input_size, requires_grad=True, device=self.rank, dtype=dtype)

        # Run local computation
        local_output = local_megatron_lm(inp)

        # Compute loss and run backward pass.
        local_output.sum().backward()

        # Save and reset input grads.
        local_input_grad = inp.grad
        self.assertIsNotNone(inp.grad)
        inp.grad = None

        # Run sharded computation
        sharded_output = sharded_megatron_lm(inp)

        # Verify local and sharded results
        self.assertEqual(local_output, sharded_output, atol=1e-3, rtol=1e-6)

        sharded_output.sum().backward()
        sharded_input_grad = inp.grad
        self.assertIsNotNone(inp.grad)

        # Verify sharded and local grads.
        self.assertEqual(local_input_grad, sharded_input_grad, atol=1e-3, rtol=1e-6)

        (
            local_weight_grad_fc1,
            local_weight_grad_fc2,
        ) = local_megatron_lm.get_weight_grads()
        local_bias_grad_fc1, local_bias_grad_fc2 = local_megatron_lm.get_bias_grads()

        # Verify that weights in both layers and biases in the sharded linear has non-None grad.
        (
            sharded_weight_fc1,
            sharded_weight_fc2,
        ) = sharded_megatron_lm.get_weights()
        bias_grad_fc1, bias_grad_fc2 = sharded_megatron_lm.get_bias_grads()
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
        self.assertEdistNorm(sharded_weight_fc1.grad, local_grad_narrowed_fc1)
        self.assertEdistNorm(sharded_weight_fc2.grad, local_grad_narrowed_fc2)
        self.assertEdistNorm(bias_grad_fc1, local_bias_grad_fc1)
        self.assertEdistNorm(bias_grad_fc2, local_bias_grad_fc2)

        # Test optimizer.
        bias_fc1, bias_fc2 = sharded_megatron_lm.get_biases()
        local_bias_fc1, local_bias_fc2 = local_megatron_lm.get_biases()
        self.assertEdistNorm(bias_fc1, local_bias_fc1)
        self.assertEdistNorm(bias_fc2, local_bias_fc2)
        self.assertEdistNorm(bias_fc1.grad, local_bias_fc1.grad)
        self.assertEdistNorm(bias_fc2.grad, local_bias_fc2.grad)
        previous_sharded_weight_fc1 = sharded_weight_fc1.clone()
        previous_sharded_weight_fc2 = sharded_weight_fc2.clone()
        previous_bias_fc1 = bias_fc1.clone()
        previous_bias_fc2 = bias_fc2.clone()
        optim = torch.optim.SGD(local_megatron_lm.parameters(), lr=0.1)
        optim.step()
        sharded_optim = ShardedOptimizer(
            dict(sharded_megatron_lm.named_parameters()),
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
        self.assertEdistNorm(sharded_weight_fc1, local_weight_fc1_narrowed)
        self.assertEdistNorm(sharded_weight_fc2, local_weight_fc2_narrowed)

        # Test bias value after optimizer.
        local_bias_fc1, local_bias_fc2 = local_megatron_lm.get_biases()
        self.assertNotEqual(previous_bias_fc1, bias_fc1)
        self.assertEdistNorm(bias_fc1, local_bias_fc1)
        self.assertNotEqual(previous_bias_fc2, bias_fc2)
        self.assertEdistNorm(bias_fc2, local_bias_fc2)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_megatron_two_layer_prototype(self):
        colwise_sharding_spec = generate_chunk_sharding_specs_for_test(0)
        rowwise_sharding_spec = generate_chunk_sharding_specs_for_test(1)
        for spec in zip(colwise_sharding_spec, rowwise_sharding_spec):
            self._run_megatron_linear(spec, [22, 17], [[17, 12], [12, 29]], torch.float16)
            self._run_megatron_linear(spec, [28, 21], [[21, 11], [11, 29]], torch.float32)
            self._run_megatron_linear(spec, [37, 23], [[23, 13], [13, 24]], torch.float64)
            self._run_megatron_linear(spec, [24, 15], [[15, 14], [14, 20]], torch.float16)

            # Test multiple input dims
            self._run_megatron_linear(spec, [10, 22, 17], [[17, 12], [12, 29]], torch.float32)
            self._run_megatron_linear(spec, [13, 28, 21], [[21, 11], [11, 29]], torch.float16)
            self._run_megatron_linear(spec, [27, 37, 23], [[23, 13], [13, 24]], torch.float32)
            self._run_megatron_linear(spec, [100, 24, 15], [[15, 14], [14, 20]], torch.float64)

            # Test single input dim
            self._run_megatron_linear(spec, [17], [[17, 12], [12, 29]], torch.float16)
            self._run_megatron_linear(spec, [21], [[21, 11], [11, 29]], torch.float32)
            self._run_megatron_linear(spec, [23], [[23, 13], [13, 24]], torch.float64)
            self._run_megatron_linear(spec, [15], [[15, 14], [14, 20]], torch.float16)


if __name__ == "__main__":
    run_tests()
