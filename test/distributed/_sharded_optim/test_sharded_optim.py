# Owner(s): ["oncall: distributed"]

import torch
import torch.optim as optim
import torch.distributed._sharded_tensor as sharded_tensor

from copy import deepcopy
from torch.distributed._sharding_spec import (
    ChunkShardingSpec,
)
from torch.distributed._sharded_optim import (
    ShardedOptimizer,
    module_named_params_with_sharded_tensor
)
from torch.testing._internal.common_distributed import (
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    run_tests,
)

from torch.testing._internal.distributed._sharded_tensor import (
    ShardedTensorTestBase,
    with_comms,
)

class MyShardedModel(torch.nn.Module):
    def __init__(self, spec=None, group=None):
        super(MyShardedModel, self).__init__()
        # Use same seed.
        torch.manual_seed(0)
        self.param = torch.nn.Parameter(torch.rand(5, 10))
        if spec is not None:
            self.sharded_param = sharded_tensor.rand(spec, 20, 10, requires_grad=True, process_group=group)
        else:
            self.sharded_param = torch.rand(5, 10)

    def forward(self, input):
        if isinstance(self.sharded_param, sharded_tensor.ShardedTensor):
            return self.param + self.sharded_param.local_shards()[0].tensor + input
        else:
            return self.sharded_param + self.param + input


class TestShardedOptimizer(ShardedTensorTestBase):

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_sharded_optim(self):
        # row-wise sharded_linear
        rowwise_spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        model = MyShardedModel(spec=rowwise_spec).cuda(self.rank)

        model_params = dict(module_named_params_with_sharded_tensor(model))

        optimizer = ShardedOptimizer(model_params, optim.SGD, lr=0.1)
        optimizer.zero_grad()

        before_update = deepcopy(optimizer.params)

        # run forward
        torch.manual_seed(self.rank)
        inp = torch.rand([5, 10]).cuda(self.rank).requires_grad_()
        sharded_output = model(inp)
        # backward
        sharded_output.sum().backward()
        # optimizer update
        optimizer.step()

        # make sure the parameters (including sharded param)
        # get updated by the optimizer
        for key, val in before_update.items():
            new_val = optimizer.params[key]
            if isinstance(val, sharded_tensor.ShardedTensor):
                self.assertNotEqual(
                    val.local_shards()[0].tensor,
                    new_val.local_shards()[0].tensor
                )
            else:
                self.assertNotEqual(val, new_val)


if __name__ == '__main__':
    run_tests()
