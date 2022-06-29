# Owner(s): ["oncall: distributed"]

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._symbolic_trace import TracingConfig
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
from torch.distributed.fsdp.wrap import always_wrap_policy, ParamExecOrderWrapPolicy
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer0 = torch.nn.Linear(6, 6)
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
        return (torch.randn((8, 6)).to(device),)

    def get_loss(self, input, output):
        return (output - input[0]).sum()

    @staticmethod
    def wrap(
        sharding_strategy: ShardingStrategy,
        device: torch.device,
        wrap_policy=always_wrap_policy,
    ):
        model = Model()
        fsdp_model = FSDP(
            model, auto_wrap_policy=wrap_policy, sharding_strategy=sharding_strategy
        )
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
    def test_fsdp_flatten_params_exec_order(
        self,
        sharding_strategy: ShardingStrategy,
        iters: int,
    ):
        """Tests the basic APIs of FSDP with ParamExecOrderWrapPolicy"""
        wrap_policy = ParamExecOrderWrapPolicy(init_policy=always_wrap_policy)
        fsdp_model = Model.wrap(sharding_strategy, self.device, wrap_policy=wrap_policy)
        self.assertTrue(fsdp_model._is_param_exec_order_prep_stage())
        for _ in range(iters):
            input = fsdp_model.module.get_input(self.device)
            output = fsdp_model(*input)
            loss = fsdp_model.module.get_loss(input, output).to(self.device)
            loss.backward()
        params_list = list(fsdp_model.parameters())
        # Since the forward execution order is NOT consistent with
        # the model definition order, the ordering in flatten_named_params_exec_order
        # should be different from named_parameters.
        self.assertEqual(
            fsdp_model._fsdp_params_exec_order,
            [params_list[0], params_list[2], params_list[3], params_list[1]],
        )
        self.assertTrue(fsdp_model._use_param_exec_order_policy())
        self.assertTrue(not fsdp_model._is_param_exec_order_prep_stage())

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "sharding_strategy",
        [ShardingStrategy.FULL_SHARD, ShardingStrategy.SHARD_GRAD_OP],
    )
    @parametrize("iters", [0, 1])
    def test_fsdp_flatten_params_exec_order_symbolic_trace(
        self,
        sharding_strategy: ShardingStrategy,
        iters: int,
    ):
        """
        Tests ParamExecOrderWrapPolicy with symbolic tracing.
        With symbolic tracing enabled, _is_param_exec_order_prep_stage
        should always set as False.
        """
        wrap_policy = ParamExecOrderWrapPolicy(
            init_policy=always_wrap_policy, tracing_config=TracingConfig()
        )
        fsdp_model = Model.wrap(
            sharding_strategy,
            self.device,
            wrap_policy=wrap_policy,
        )
        for _ in range(iters):
            input = fsdp_model.module.get_input(self.device)
            output = fsdp_model(*input)
            loss = fsdp_model.module.get_loss(input, output).to(self.device)
            loss.backward()
        params_list = list(fsdp_model.parameters())
        # Since the forward execution order is NOT consistent with the model definition order,
        # the ordering in flatten_named_params_exec_order should be different from named_parameters
        self.assertEqual(
            fsdp_model._fsdp_params_exec_order,
            [params_list[0], params_list[2], params_list[3], params_list[1]],
        )
        self.assertTrue(fsdp_model._use_param_exec_order_policy())
        self.assertTrue(not fsdp_model._is_param_exec_order_prep_stage())


instantiate_parametrized_tests(TestFSDPExecOrder)

if __name__ == "__main__":
    run_tests()
