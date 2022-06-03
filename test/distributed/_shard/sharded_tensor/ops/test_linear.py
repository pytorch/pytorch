# Owner(s): ["oncall: distributed"]

import copy
import sys

import torch
import torch.distributed as dist
from torch.distributed._shard.api import (
    shard_parameter,
    _collect_local_shard,
    _reshard_output,
)
from torch.distributed._shard.sharded_optim import (
    ShardedOptimizer,
    named_params_with_sharded_tensor,
)
from torch.distributed._shard.sharded_tensor import (
    empty,
)
from torch.distributed._shard.sharding_spec import (
    ChunkShardingSpec,
    EnumerableShardingSpec,
    ShardMetadata,
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


class TestShardedTensorOpsLinear(ShardedTensorTestBase):
    def _run_sharded_linear(
        self, spec, input_size, linear_size, sharded_dim
    ):
        # Use same seed.
        torch.manual_seed(0)
        local_linear = torch.nn.Linear(*linear_size).cuda(self.rank)
        sharded_linear = torch.nn.Linear(*linear_size)

        # Copy the weights and bias from local linear
        sharded_linear.weight = clone_module_parameter(local_linear, "weight")
        sharded_linear.bias = clone_module_parameter(local_linear, "bias")

        # Shard the parameter.
        shard_parameter(sharded_linear, "weight", spec)

        # Run sharded computation
        torch.manual_seed(self.rank)  # inputs different on each rank
        inp = torch.rand(*input_size).cuda(self.rank)
        reshard_spec = copy.deepcopy(spec)
        reshard_spec.dim = 0
        reshard_spec.placements.sort(key=lambda placement: placement.rank())
        sharded_linear = _collect_local_shard(
            _reshard_output(sharded_linear, reshard_spec)
        )
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
        sharded_output = sharded_output.reshard(reshard_spec).local_tensor()
        # When local tensor only has one dimension, we increase one more dimension
        # for reshard. We need to squeeze the # of dimensions manually.
        if inp.dim() == 1:
            sharded_output = sharded_output.squeeze(reshard_spec.dim)
        self.assertEqual(local_output, sharded_output)

        # Compute loss and run backward pass.
        local_output.sum().backward()
        sharded_output.sum().backward()
        local_grad = local_linear.weight.grad

        # Verify that both weight and bias in the sharded linear has non-None grad.
        sharded_weight = sharded_linear.weight.local_tensor()
        self.assertNotEqual(sharded_linear.bias.grad, None)
        self.assertNotEqual(sharded_weight.grad, None)

        # Shard the local linear's weight grad so that we can compare.
        dist.all_reduce(local_grad)
        (start_pos, chunk_size) = generate_local_weight_sharding_params_for_test(
            local_linear.weight, sharded_dim, TEST_GPU_NUM, spec, self.rank
        )
        local_grad_narrowed = local_grad.narrow(sharded_dim, start_pos, chunk_size)
        local_bias_grad = local_linear.bias.grad
        dist.all_reduce(local_bias_grad)

        # Test backward gradient calculation.
        self.assertEqual(sharded_linear.bias.grad, local_bias_grad)
        self.assertEqual(sharded_weight.grad, local_grad_narrowed)

        # Test optimizer.
        previous = local_linear.weight.clone().detach()
        optim = torch.optim.SGD(local_linear.parameters(), lr=0.1)
        optim.step()
        self.assertNotEqual(previous, local_linear.weight)
        previous_sharded_weight = sharded_weight.clone()
        previous_sharded_bias = sharded_linear.bias.clone()
        sharded_optim = ShardedOptimizer(
            dict(named_params_with_sharded_tensor(sharded_linear)),
            torch.optim.SGD,
            lr=0.1,
        )
        sharded_optim.step()
        sharded_weight = sharded_linear.weight.local_tensor()
        local_weight_narrowed = local_linear.weight.narrow(
            sharded_dim, start_pos, chunk_size
        )
        self.assertEqual(sharded_weight.size(), local_weight_narrowed.size())
        self.assertNotEqual(previous_sharded_weight, sharded_weight)
        self.assertEqual(sharded_weight, local_weight_narrowed)
        self.assertNotEqual(previous_sharded_bias, sharded_linear.bias)
        self.assertEqual(sharded_linear.bias, local_linear.bias)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_sharded_linear_colwise(self):
        for spec in generate_chunk_sharding_specs_for_test(0):
            self._run_sharded_linear(spec, [2, 17], [17, 12], 0)
            self._run_sharded_linear(spec, [8, 21], [21, 11], 0)
            self._run_sharded_linear(spec, [7, 23], [23, 13], 0)
            self._run_sharded_linear(spec, [4, 15], [15, 14], 0)

            # Test multiple input dims
            self._run_sharded_linear(spec, [10, 2, 17], [17, 12], 0)
            self._run_sharded_linear(spec, [13, 8, 21], [21, 11], 0)
            self._run_sharded_linear(spec, [27, 7, 23], [23, 13], 0)
            self._run_sharded_linear(spec, [100, 12, 4, 15], [15, 14], 0)

            # Test single input dim
            self._run_sharded_linear(spec, [17], [17, 12], 0)
            self._run_sharded_linear(spec, [21], [21, 11], 0)
            self._run_sharded_linear(spec, [23], [23, 13], 0)
            self._run_sharded_linear(spec, [15], [15, 14], 0)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_sharded_linear_rowwise(self):
        for spec in generate_chunk_sharding_specs_for_test(1):
            # Test even split.
            self._run_sharded_linear(spec, [8, 16], [16, 11], 1)

            # Test uneven split.
            self._run_sharded_linear(spec, [5, 19], [19, 11], 1)
            self._run_sharded_linear(spec, [10, 21], [21, 11], 1)

            # Test multiple input dims
            self._run_sharded_linear(spec, [13, 8, 16], [16, 11], 1)
            self._run_sharded_linear(spec, [10, 5, 19], [19, 11], 1)
            self._run_sharded_linear(spec, [12, 15, 10, 21], [21, 11], 1)

            # Test single input dim
            self._run_sharded_linear(spec, [16], [16, 11], 1)
            self._run_sharded_linear(spec, [19], [19, 11], 1)
            self._run_sharded_linear(spec, [21], [21, 11], 1)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_sharded_linear_errors(self):
        for spec in generate_chunk_sharding_specs_for_test(0):
            fc1 = torch.nn.Linear(10, 10).cuda(self.rank)
            shard_parameter(fc1, "bias", spec)
            with self.assertRaisesRegex(TypeError, 'bias needs to be torch.Tensor'):
                fc1(torch.rand(10, 10).cuda(self.rank))

            fc2 = torch.nn.Linear(10, 10).cuda(self.rank)
            shard_parameter(fc2, "weight", spec)
            with self.assertRaisesRegex(ValueError, 'Input needs to have at least 1 dim'):
                fc2(torch.tensor(1).cuda(self.rank))

            fc3 = torch.nn.Linear(10, 10).cuda(self.rank)
            fc3.weight = torch.nn.Parameter(torch.rand(10, 10, 10).cuda(self.rank))
            shard_parameter(fc3, "weight", spec)
            with self.assertRaisesRegex(ValueError, 'Weight needs to have exactly 2 dims'):
                fc3(torch.rand(10, 10).cuda(self.rank))

            fc4 = torch.nn.Linear(10, 10).cuda(self.rank)
            fc4.bias = torch.nn.Parameter(torch.rand(10, 10).cuda(self.rank))
            shard_parameter(fc4, "weight", spec)
            with self.assertRaisesRegex(ValueError, 'Bias needs to have exactly 1 dim'):
                fc4(torch.rand(10, 10).cuda(self.rank))

            fc5 = torch.nn.Linear(7, 10).cuda(self.rank)
            shard_parameter(fc5, "weight", spec)
            with self.assertRaisesRegex(ValueError, 'Input dim: 13 does not match appropriate weight dim: 7'):
                fc5(torch.rand(20, 10, 13).cuda(self.rank))

            fc6 = torch.nn.Linear(10, 10).cuda(self.rank)
            del fc6.weight
            enumerable_spec = EnumerableShardingSpec([
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_sizes=[5, 5],
                    placement="rank:0/cuda:0",
                ),
                ShardMetadata(
                    shard_offsets=[0, 5],
                    shard_sizes=[5, 5],
                    placement="rank:1/cuda:1",
                ),
                ShardMetadata(
                    shard_offsets=[5, 0],
                    shard_sizes=[5, 5],
                    placement="rank:2/cuda:2",
                ),
                ShardMetadata(
                    shard_offsets=[5, 5],
                    shard_sizes=[5, 5],
                    placement="rank:3/cuda:3",
                )
            ])

            fc6.weight = empty(enumerable_spec, 10, 10)
            # Sharded Tensor metadata has parenthesis imbalance issue when using re.compile
            error_msg = r"torch function 'linear', with args: (?s).* "
            r"and kwargs: None not supported for ShardedTensor!"
            with self.assertRaisesRegex(RuntimeError, error_msg):
                fc6(torch.rand(10, 10).cuda(self.rank))

            fc7 = torch.nn.Linear(10, 80).cuda(self.rank)
            multiple_local_shard_spec = ChunkShardingSpec(
                dim=0,
                placements=[
                    "rank:0/cuda:0",
                    "rank:0/cuda:0",
                    "rank:1/cuda:1",
                    "rank:1/cuda:1",
                    "rank:2/cuda:2",
                    "rank:2/cuda:2",
                    "rank:3/cuda:3",
                    "rank:3/cuda:3",
                ],
            )
            del fc7.weight
            fc7.weight = empty(multiple_local_shard_spec, 80, 10)
            with self.assertRaisesRegex(ValueError, 'Only one local shard supported!'):
                fc7(torch.rand(10, 10).cuda(self.rank))


if __name__ == "__main__":
    run_tests()
