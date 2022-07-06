# Owner(s): ["oncall: distributed"]

import torch
import copy
from torch.optim import SGD
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
        self.layer0 = torch.nn.Linear(6, 6)
        self.layer1 = torch.nn.Linear(6, 6)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(6, 3),
            torch.nn.ReLU(),
            torch.nn.Linear(3, 6),
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
        return (torch.randn((8, 6)).to(device), )

    def get_loss(self, input, output):
        return (output - input[0]).sum()

    @staticmethod
    def wrap(
        model,
        sharding_strategy: ShardingStrategy,
        device: torch.device,
        wrap_policy=always_wrap_policy,
    ):
        fsdp_model = FSDP(
            model,
            auto_wrap_policy=wrap_policy,
            sharding_strategy=sharding_strategy
        )
        return fsdp_model.to(device)


class TestFSDPExecOrder(FSDPTest):
    @property
    def device(self):
        return torch.device("cuda")

    def get_model_param_count(self, m):
        return sum([p.numel() for p in m.parameters()])

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "sharding_strategy",
        [ShardingStrategy.FULL_SHARD, ShardingStrategy.SHARD_GRAD_OP],
    )
    @parametrize("iters", [1, 3])
    def test_fsdp_flatten_params_exec_order(self, sharding_strategy: ShardingStrategy, iters: int):
        """Tests the basic APIs of FSDP with ParamExecOrderWrapPolicy"""
        model = Model()
        policy_exec_order = ParamExecOrderWrapPolicy(
            init_policy=always_wrap_policy
        )
        fsdp_model = Model.wrap(model, sharding_strategy, self.device, policy_exec_order)
        self.assertTrue(fsdp_model._is_param_exec_order_prep_stage())
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
            [
                params_list[0],
                params_list[2],
                params_list[3],
                params_list[1]
            ]
        )
        self.assertTrue(fsdp_model._use_param_exec_order_policy())
        self.assertTrue(not fsdp_model._is_param_exec_order_prep_stage())

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "sharding_strategy",
        [ShardingStrategy.FULL_SHARD, ShardingStrategy.SHARD_GRAD_OP],
    )
    @parametrize("iters", [1, 10])
    @parametrize("group_size", [1, 2, 4])
    def test_fsdp_accuracy(
        self,
        sharding_strategy: ShardingStrategy,
        iters: int,
        group_size: int,
    ):
        """Tests the accuracy with ParamExecOrderWrapPolicy"""
        model = Model()
        model_copy = copy.deepcopy(model)
        policy_exec_order = ParamExecOrderWrapPolicy(
            init_policy=always_wrap_policy,
            group_size=group_size,
        )
        fsdp_model_1 = Model.wrap(
            model,
            sharding_strategy,
            self.device,
            policy_exec_order
        )
        fsdp_model_2 = Model.wrap(
            model_copy,
            sharding_strategy,
            self.device,
            always_wrap_policy
        )
        input = fsdp_model_1.module.get_input(self.device)
        # initialization
        output_1 = fsdp_model_1(*input)
        loss_1 = fsdp_model_1.module.get_loss(input, output_1).to(self.device)
        loss_1.backward()
        # end initialization
        optim_1 = SGD(fsdp_model_1.parameters(), lr=0.1)
        optim_2 = SGD(fsdp_model_2.parameters(), lr=0.1)
        for _ in range(iters):
            output_1 = fsdp_model_1(*input)
            loss_1 = fsdp_model_1.module.get_loss(input, output_1).to(self.device)
            loss_1.backward()
            optim_1.step()
            optim_1.zero_grad()
        for _ in range(iters):
            output_2 = fsdp_model_2(*input)
            loss_2 = fsdp_model_2.module.get_loss(input, output_2).to(self.device)
            loss_2.backward()
            optim_2.step()
            optim_2.zero_grad()
        self.assertEqual(loss_1, loss_2)

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "sharding_strategy",
        [ShardingStrategy.FULL_SHARD, ShardingStrategy.SHARD_GRAD_OP],
    )
    def test_group_fsdp_wraps(self, sharding_strategy: ShardingStrategy):
        model1, model2 = Model(), Model()
        num_params_model1 = len(list(model1.parameters()))
        num_params_model2 = len(list(model2.parameters()))
        raw_model_size = self.get_model_param_count(model1) + self.get_model_param_count(model2)

        fsdp_model1 = Model.wrap(model1, sharding_strategy, self.device, None)
        fsdp_model2 = Model.wrap(model2, sharding_strategy, self.device, None)
        out_module = FSDP.group_fsdp_modules([fsdp_model1, fsdp_model2])
        # out_module should only contain one FlatParameter
        assert len(list(out_module.parameters())) == 1

        with out_module.summon_full_params(out_module):
            self.assertEqual(len(list(out_module.parameters())), num_params_model1 + num_params_model2)
            self.assertEqual(raw_model_size, self.get_model_param_count(out_module))


instantiate_parametrized_tests(TestFSDPExecOrder)

if __name__ == "__main__":
    run_tests()