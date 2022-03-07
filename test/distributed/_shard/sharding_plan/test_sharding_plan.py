
# Owner(s): ["oncall: distributed"]
import sys

import torch
import torch.nn as nn
from torch.testing._internal.common_distributed import (
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.distributed._shard import shard_module, ShardedModuleSwapper
from torch.distributed._shard.sharding_plan import ShardingPlan, ShardingPlanner
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed._shard.sharded_tensor import ShardedTensor

from torch.testing._internal.common_utils import TEST_WITH_DEV_DBG_ASAN
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


class SimpleMegatronLM(nn.Module):
    def __init__(self, linear_size, rank=None):
        super().__init__()
        self.fc1 = nn.Linear(*linear_size[0])
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(*linear_size[1])
        if rank:
            self.fc1.cuda(rank)
            self.fc2.cuda(rank)

    def forward(self, inp):
        return self.fc2(self.gelu(self.fc1(inp)))

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
    def test_basic_sharding_plan(self):
        # since we already test the numerics in test_megatron_prototype.py
        # this test is only testing the ShardingPlan and whether the model
        # be sharded with the ShardingPlan, we don't need to verify the
        # numeric correctness.
        colwise_sharding_spec = generate_chunk_sharding_specs_for_test(0)
        rowwise_sharding_spec = generate_chunk_sharding_specs_for_test(1)
        for spec in zip(colwise_sharding_spec, rowwise_sharding_spec):
            # test each sharding spec pair and see if we can apply sharding
            sharding_plan = ShardingPlan(plan={
                "fc1.weight": spec[0],
                "fc2.weight": spec[1],
            })

            megatron_lm = SimpleMegatronLM([[17, 12], [12, 29]], rank=self.rank).cuda(
                self.rank
            )

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


    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_basic_sharding_plan_error(self):
        class DummyModuleSwapper(ShardedModuleSwapper):
            def process(self, module):
                return module

        colwise_sharding_spec = generate_chunk_sharding_specs_for_test(0)[0]
        rowwise_sharding_spec = generate_chunk_sharding_specs_for_test(1)[0]
        sharding_plan = ShardingPlan(plan={
            "fc1.weight": colwise_sharding_spec,
            "fc2.weight": rowwise_sharding_spec,
        })
        megatron_lm = SimpleMegatronLM([[17, 12], [12, 29]], rank=self.rank).cuda(
            self.rank
        )

        # shard the module with the provided sharding plan and module swapper should error
        with self.assertRaisesRegex(NotImplementedError, "not implemented yet!"):
            shard_module(megatron_lm, sharding_plan, sharded_module_swapper=DummyModuleSwapper)

        reshard_plan = ShardingPlan(plan={
            "fc1": colwise_sharding_spec,
            "fc2": rowwise_sharding_spec,
        })
        # shard the module with the provided sharding plan and module swapper should error
        with self.assertRaisesRegex(NotImplementedError, "not implemented yet!"):
            shard_module(megatron_lm, sharding_plan, reshard_output=reshard_plan)


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
