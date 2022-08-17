
# Owner(s): ["oncall: distributed"]
import sys
import copy

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed._shard.sharded_optim import (
    ShardedOptimizer,
)
from torch.testing._internal.common_distributed import (
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.distributed._shard import shard_module
from torch.distributed._shard.sharding_plan import ShardingPlan, ShardingPlanner
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed._shard.sharded_tensor import ShardedTensor

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
    generate_local_weight_sharding_params_for_test,
)
from torch.testing._internal.distributed._shard.test_common import SimpleMegatronLM

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


# Example ShardingPlanner that chunks every parameter in the module
# to all available devices defined.
class ChunkAllShardingPlanner(ShardingPlanner):
    dim = 0
    devices = []

    def __init__(self, chunk_dim=0, device_count=0):
        self.dim = chunk_dim
        self.devices = [f"rank:{i}/cuda:{i}" for i in range(device_count)]

    def build_plan(self, module: nn.Module) -> ShardingPlan:
        named_params = module.named_parameters()
        plan = {}
        for name, param in named_params:
            plan[name] = ChunkShardingSpec(self.dim, placements=self.devices)

        return ShardingPlan(plan=plan)


class TestShardingPlan(ShardedTensorTestBase):

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_sharding_plan_simple_megatron(self):
        colwise_sharding_spec = generate_chunk_sharding_specs_for_test(0)
        rowwise_sharding_spec = generate_chunk_sharding_specs_for_test(1)
        for spec in zip(colwise_sharding_spec, rowwise_sharding_spec):
            # test each sharding spec pair and see if we can apply sharding
            reshard_spec = copy.deepcopy(spec[1])
            reshard_spec.placements.sort(key=lambda placement: placement.rank())
            reshard_spec.dim = 0

            sharding_plan = ShardingPlan(
                plan={
                    "fc1.weight": spec[0],
                    "fc2.weight": spec[1]
                },
                output_plan={
                    "": reshard_spec
                },
                return_local_tensor=[""])

            # Use same seed.
            torch.manual_seed(0)
            local_megatron_lm = SimpleMegatronLM([[17, 12], [12, 29]]).cuda(self.rank)
            megatron_lm = copy.deepcopy(local_megatron_lm)

            # shard the module with the provided sharding plan
            shard_module(megatron_lm, sharding_plan)

            # check to make sure the module already been sharded
            self.assertTrue(isinstance(megatron_lm.fc1.weight, ShardedTensor))
            self.assertTrue(isinstance(megatron_lm.fc2.weight, ShardedTensor))
            self.assertEqual(megatron_lm.fc1.weight.sharding_spec(), spec[0])
            self.assertEqual(megatron_lm.fc2.weight.sharding_spec(), spec[1])

            # make sure we can run sharded computation
            input = torch.rand(22, 17).cuda(self.rank)
            sharded_output = megatron_lm(input)
            local_output = local_megatron_lm(input)

            # verify and make sure local and sharded output matches
            self.assertEqual(local_output, sharded_output)

            # Compute loss and run backward pass.
            local_output.sum().backward()
            sharded_output.sum().backward()
            (
                local_weight_grad_fc1,
                local_weight_grad_fc2,
            ) = local_megatron_lm.get_weight_grads()
            local_bias_grad_fc1, local_bias_grad_fc2 = local_megatron_lm.get_bias_grads()

            # Verify that weights in both layers and biases in the sharded linear has non-None grad.
            (
                sharded_weight_fc1,
                sharded_weight_fc2,
            ) = megatron_lm.get_weights()
            bias_grad_fc1, bias_grad_fc2 = megatron_lm.get_bias_grads()
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
            bias_fc1, bias_fc2 = megatron_lm.get_biases()
            local_bias_fc1, local_bias_fc2 = local_megatron_lm.get_biases()
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
                dict(megatron_lm.named_parameters()),
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
            local_bias_fc1, local_bias_fc2 = local_megatron_lm.get_biases()
            self.assertNotEqual(previous_bias_fc1, bias_fc1)
            self.assertEqual(bias_fc1, local_bias_fc1)
            self.assertNotEqual(previous_bias_fc2, bias_fc2)
            self.assertEqual(bias_fc2, local_bias_fc2)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_reshard_to_ddp_sharding_plan(self):
        colwise_sharding_spec = generate_chunk_sharding_specs_for_test(0)[0]
        rowwise_sharding_spec = generate_chunk_sharding_specs_for_test(1)[0]

        # test each sharding spec pair and see if we can apply sharding
        output_spec = copy.deepcopy(rowwise_sharding_spec)
        output_spec.placements.sort(key=lambda placement: placement.rank())
        output_spec.dim = 0

        # new module with megatron as submodule
        class MyModule(nn.Module):
            def __init__(self, rank=None):
                super().__init__()
                self.megatron = SimpleMegatronLM([[17, 12], [12, 29]], rank=rank)
                self.relu = nn.ReLU()

            def forward(self, input):
                return self.relu(self.megatron(input))

        sharding_plan = ShardingPlan(
            plan={
                "megatron.fc1.weight": colwise_sharding_spec,
                "megatron.fc2.weight": rowwise_sharding_spec,
            },
            output_plan={
                "megatron": output_spec
            },
            return_local_tensor=[
                "megatron"
            ]
        )

        # Use same seed.
        torch.manual_seed(0)
        local_module = MyModule().cuda(self.rank)
        sharded_module = copy.deepcopy(local_module)

        # shard the module with the provided sharding plan
        shard_module(sharded_module, sharding_plan)

        # check to make sure the module already been sharded
        self.assertTrue(isinstance(sharded_module.megatron.fc1.weight, ShardedTensor))
        self.assertTrue(isinstance(sharded_module.megatron.fc2.weight, ShardedTensor))
        self.assertEqual(sharded_module.megatron.fc1.weight.sharding_spec(), colwise_sharding_spec)
        self.assertEqual(sharded_module.megatron.fc2.weight.sharding_spec(), rowwise_sharding_spec)

        # make sure we can run sharded computation
        input = torch.rand(22, 17).cuda(self.rank)
        sharded_output = sharded_module(input)
        local_output = local_module(input)

        # verify and make sure local and sharded output matches
        self.assertEqual(local_output, sharded_output)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_sharding_plan_errors(self):
        rowwise_sharding_spec = generate_chunk_sharding_specs_for_test(1)[0]
        sharding_plan_wrong_plan = ShardingPlan(
            plan={
                "fc1.weight": torch.randn(3, 4),
            },
            output_plan={
                "": rowwise_sharding_spec
            },
        )

        megatron_lm = SimpleMegatronLM([[17, 12], [12, 29]]).cuda(self.rank)

        with self.assertRaisesRegex(
            TypeError, "Only `ShardingSpec` and `Sharder` are supported to shard"
        ):
            # shard the module with the provided sharding plan
            shard_module(megatron_lm, sharding_plan_wrong_plan)

        sharding_plan_wrong_output_plan = ShardingPlan(
            plan={
                "fc1.weight": rowwise_sharding_spec,
            },
            output_plan={
                "": torch.randn(3, 4)
            },
        )

        with self.assertRaisesRegex(
            TypeError, "Only `ShardingSpec` is supported as output_plan"
        ):
            # shard the module with the provided sharding plan
            shard_module(megatron_lm, sharding_plan_wrong_output_plan)

        sharding_plan_wrong_module_path = ShardingPlan(
            plan={
                "fc3.weight": rowwise_sharding_spec,
            },
        )
        with self.assertRaisesRegex(
            AttributeError, "has no attribute"
        ):
            # shard the module with the provided sharding plan
            shard_module(megatron_lm, sharding_plan_wrong_module_path)

        sharding_plan_wrong_param_path = ShardingPlan(
            plan={
                "fc1.biass": rowwise_sharding_spec,
            },
        )
        with self.assertRaisesRegex(
            AttributeError, "has no attribute"
        ):
            # shard the module with the provided sharding plan
            shard_module(megatron_lm, sharding_plan_wrong_param_path)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_custom_sharding_planner(self):
        megatron_lm = SimpleMegatronLM([[17, 12], [12, 29]], rank=self.rank).cuda(
            self.rank
        )
        planner = ChunkAllShardingPlanner(device_count=TEST_GPU_NUM)
        sharding_plan = planner.build_plan(megatron_lm)

        shard_module(megatron_lm, sharding_plan)

        # check to make sure the module already been sharded
        self.assertTrue(isinstance(megatron_lm.fc1.weight, ShardedTensor))
        self.assertTrue(isinstance(megatron_lm.fc2.weight, ShardedTensor))
        self.assertTrue(isinstance(megatron_lm.fc1.bias, ShardedTensor))
        self.assertTrue(isinstance(megatron_lm.fc2.bias, ShardedTensor))

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_shard_module_sub_process_group(self):
        megatron_lm = SimpleMegatronLM([[17, 12], [12, 29]], rank=self.rank)
        colwise_sharding_spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:2",
                "rank:1/cuda:3",
            ],
        )
        rowwise_sharding_spec = ChunkShardingSpec(
            dim=1,
            placements=[
                "rank:0/cuda:2",
                "rank:1/cuda:3",
            ],
        )
        sharding_plan = ShardingPlan(
            plan={
                "fc1.weight": colwise_sharding_spec,
                "fc2.weight": rowwise_sharding_spec
            }
        )

        pg = dist.new_group([2, 3])

        if self.rank >= 2:
            shard_module(megatron_lm, sharding_plan, process_group=pg)

if __name__ == "__main__":
    run_tests()
