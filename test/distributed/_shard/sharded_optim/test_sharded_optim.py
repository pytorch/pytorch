# Owner(s): ["oncall: distributed"]

from copy import deepcopy

import torch
import torch.optim as optim
from torch.distributed._shard import shard_parameter, sharded_tensor
from torch.distributed._shard.sharded_optim import ShardedOptimizer
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.testing._internal.common_distributed import requires_nccl, skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests

from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    with_comms,
)


class MyShardedModel(torch.nn.Module):
    def __init__(self, spec=None, group=None):
        super().__init__()
        # Use same seed.
        torch.manual_seed(0)
        self.param = torch.nn.Parameter(torch.rand(5, 10))
        if spec is not None:
            self.sharded_param = torch.nn.Parameter(
                sharded_tensor.rand(
                    spec, 20, 10, requires_grad=True, process_group=group
                )
            )
        else:
            self.sharded_param = torch.nn.Parameter(torch.rand(5, 10))

    def forward(self, input):
        if isinstance(self.sharded_param, sharded_tensor.ShardedTensor):
            return self.param + self.sharded_param.local_shards()[0].tensor + input
        else:
            return self.sharded_param + self.param + input


class MyShardedLinear(torch.nn.Module):
    def __init__(self, rank=None):
        super().__init__()
        # Use same seed.
        torch.manual_seed(0)
        self.linear1 = torch.nn.Linear(17, 12)
        self.linear2 = torch.nn.Linear(12, 29)
        self.gelu = torch.nn.GELU()

        if rank:
            self.linear1.cuda(rank)
            self.linear2.cuda(rank)

    def shard_parameter(self):
        rowwise_sharding_spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )

        colwise_sharding_spec = ChunkShardingSpec(
            dim=1,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )

        shard_parameter(self.linear1, "weight", rowwise_sharding_spec)
        shard_parameter(self.linear2, "weight", colwise_sharding_spec)

    def forward(self, inp):
        return self.linear2(self.gelu(self.linear1(inp)))


class TestShardedOptimizer(ShardedTensorTestBase):
    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_sharded_optim(self):
        rowwise_spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        local_model = MyShardedModel().cuda()
        sharded_model = MyShardedModel(spec=rowwise_spec).cuda()

        # copy the parameters from local model
        sharded_model.sharded_param.local_shards()[0].tensor = (
            local_model.sharded_param.detach().clone().requires_grad_()
        )

        local_optim = optim.SGD(local_model.parameters(), lr=0.1)
        sharded_model_params = dict(sharded_model.named_parameters())
        sharded_optim = ShardedOptimizer(sharded_model_params, optim.SGD, lr=0.1)

        local_optim.zero_grad()
        sharded_optim.zero_grad()

        before_update = deepcopy(sharded_optim.named_params)

        inp = torch.rand([5, 10]).cuda(self.rank).requires_grad_()

        # run forward
        local_output = local_model(inp)
        sharded_output = sharded_model(inp)
        # backward
        local_output.sum().backward()
        sharded_output.sum().backward()

        # optimizer update
        local_optim.step()
        sharded_optim.step()

        # make sure the parameters (including sharded param)
        # get updated by the optimizer, and the updated
        # local params are the same as the sharded params
        for key, val in before_update.items():
            new_val = sharded_optim.named_params[key]
            if isinstance(val, sharded_tensor.ShardedTensor):
                self.assertNotEqual(
                    val.local_shards()[0].tensor, new_val.local_shards()[0].tensor
                )
                self.assertEqual(
                    new_val.local_shards()[0].tensor, local_model.sharded_param
                )
            else:
                self.assertNotEqual(val, new_val)
                self.assertEqual(new_val, local_model.param)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_named_params_with_sharded_tensor(self):
        rowwise_spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        sharded_model = MyShardedModel(spec=rowwise_spec).cuda()
        sharded_model_params = dict(sharded_model.named_parameters())
        param_keys = list(sharded_model_params.keys())
        self.assertEqual(len(param_keys), 2)
        self.assertTrue("param" in param_keys)
        self.assertTrue("sharded_param" in param_keys)

        sharded_linear = MyShardedLinear(rank=self.rank).cuda()
        sharded_linear.shard_parameter()
        sharded_linear_params = dict(sharded_linear.named_parameters())
        param_keys = list(sharded_linear_params.keys())
        self.assertEqual(len(param_keys), 4)
        self.assertTrue("linear1.bias" in param_keys)
        self.assertTrue("linear2.bias" in param_keys)
        self.assertTrue("linear1.weight" in param_keys)
        self.assertTrue("linear2.weight" in param_keys)
        self.assertFalse("bias" in param_keys)


if __name__ == "__main__":
    run_tests()
