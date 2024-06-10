# Owner(s): ["oncall: distributed"]
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._shard import shard_module
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharding_plan import ShardingPlan, ShardingPlanner
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.testing._internal.common_distributed import requires_nccl, skip_if_lt_x_gpu

from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    TEST_GPU_NUM,
    with_comms,
)
from torch.testing._internal.distributed._shard.sharded_tensor._test_ops_common import (
    generate_chunk_sharding_specs_for_test,
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
    def test_sharding_plan_errors(self):
        rowwise_sharding_spec = generate_chunk_sharding_specs_for_test(1)[0]
        sharding_plan_wrong_plan = ShardingPlan(
            plan={
                "fc1.weight": torch.randn(3, 4),
            },
            output_plan={"": rowwise_sharding_spec},
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
            output_plan={"": torch.randn(3, 4)},
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
        with self.assertRaisesRegex(AttributeError, "has no attribute"):
            # shard the module with the provided sharding plan
            shard_module(megatron_lm, sharding_plan_wrong_module_path)

        sharding_plan_wrong_param_path = ShardingPlan(
            plan={
                "fc1.biass": rowwise_sharding_spec,
            },
        )
        with self.assertRaisesRegex(AttributeError, "has no attribute"):
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
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        rowwise_sharding_spec = ChunkShardingSpec(
            dim=1,
            placements=[
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        sharding_plan = ShardingPlan(
            plan={
                "fc1.weight": colwise_sharding_spec,
                "fc2.weight": rowwise_sharding_spec,
            }
        )

        pg = dist.new_group([2, 3])

        if self.rank >= 2:
            shard_module(megatron_lm, sharding_plan, process_group=pg)


if __name__ == "__main__":
    run_tests()
