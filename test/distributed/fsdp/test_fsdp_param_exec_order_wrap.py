import torch
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.distributed.fsdp.wrap import ParamExecOrderWrapPolicy, always_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer0 = torch.nn.Linear(5, 6)
        self.layer1 = torch.nn.Linear(6, 6, bias=False)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(6, 3, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(3, 6, bias=False),
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # `layer0` -> `layer2` -> `layer1`
        # the forward execution order is NOT consistent with the model definition order.
        z = self.relu(self.layer0(x))
        z = self.relu(self.layer2(z))
        z = self.relu(self.layer1(z))
        return z

    def get_input(self, device: torch.device):
        return (torch.randn((8, 5)).to(device),)

    def get_loss(self, input, output):
        return output.sum()

    def run_backward(self, loss):
        loss.backward()

    @staticmethod
    def wrap(sharding_strategy: ShardingStrategy, device: torch.device, init_policy=always_wrap_policy):
        model = Model()
        wrap_policy = ParamExecOrderWrapPolicy(init_policy=init_policy)
        fsdp_model = FSDP(model, auto_wrap_policy=wrap_policy, sharding_strategy=sharding_strategy)
        return fsdp_model.to(device)


class TestFSDPExecOrder(FSDPTest):
    @property
    def device(self):
        return torch.device("cuda")

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "sharding_strategy",
        [ShardingStrategy.FULL_SHARD, ShardingStrategy.SHARD_GRAD_OP],
    )
    @parametrize("iters", [1, 3])
    def test_fsdp_flatten_params_exec_order(self, sharding_strategy: ShardingStrategy, iters: int):
        """Tests the basic APIs of FSDP with ParamExecOrderWrapPolicy"""
        fsdp_model = Model.wrap(sharding_strategy, self.device)
        for _ in range(iters):
            inp = fsdp_model.module.get_input(self.device)
            output = fsdp_model(*inp)
            loss = fsdp_model.module.get_loss(inp, output).to(self.device)
            fsdp_model.module.run_backward(loss)
        assert set(fsdp_model.flatten_named_params_exec_order()) == set(list(fsdp_model.named_parameters()))
        # Since the forward execution order is NOT consistent with the model definition order,
        # the ordering in flatten_named_params_exec_order should be different from named_parameters
        assert fsdp_model.flatten_named_params_exec_order() != list(fsdp_model.named_parameters())
        assert fsdp_model._use_param_exec_order_policy
        assert not fsdp_model._param_exec_order_prep_stage
        # After the first iteration, all _params_exec_order_hook_handle should be removed.
        for p in fsdp_model.parameters():
            assert not hasattr(p, "_params_exec_order_hook_handle")


instantiate_parametrized_tests(TestFSDPExecOrder)

if __name__ == "__main__":
    run_tests()
