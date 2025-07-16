# Owner(s): ["oncall: distributed"]

import os
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed._composable.contract import _get_registry
from torch.distributed._composable.replicate_with_fsdp import (
    _get_managed_modules,
    replicate,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    run_subtests,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
)


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 2)
        self.fc3 = nn.Linear(2, 2)

    def forward(self, x):
        return self.fc3(self.fc2(self.fc1(x)))


class ReplicateStateDictTest(MultiProcessTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def _init_pg(self):
        dist.init_process_group(
            backend="nccl",
            rank=self.rank,
            world_size=self.world_size,
            store=dist.FileStore(self.file_name, self.world_size),
        )

    def _check_state_dict_parity(self, sd_1, sd_2):
        for k1, k2 in zip(sd_1.keys(), sd_2.keys()):
            self.assertEqual(k1, k2)

        for v1, v2 in zip(sd_1.values(), sd_2.values()):
            self.assertEqual(v1, v2)


class ReplicateTest(MultiProcessTestCase):
    @property
    def world_size(self) -> int:
        return 2

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def _init_pg(self):
        dist.init_process_group(
            backend="gloo",
            rank=self.rank,
            world_size=self.world_size,
            store=dist.FileStore(self.file_name, self.world_size),
        )

    def check_sharded_parity(
        self,
        replicated_module: nn.Module,
        sharded_module: nn.Module,
        prefixes_to_ignore: tuple[str, ...] = (),
    ):
        for (replicated_name, replicated_param), (sharded_name, sharded_param) in zip(
            replicated_module.named_parameters(), sharded_module.named_parameters()
        ):
            clean_sharded_name = sharded_name
            for prefix in prefixes_to_ignore:
                clean_sharded_name = clean_sharded_name.replace(prefix, "")
            self.assertEqual(replicated_name, clean_sharded_name)
            self.assertIsInstance(sharded_param, DTensor)
            assert isinstance(sharded_param, DTensor)  # mypy
            mesh, placements = sharded_param.device_mesh, sharded_param.placements
            if tuple(placements) == (Shard(0), Shard(0)):
                raise AssertionError(
                    "FSDP's (Shard(0), Shard(0)) layout differs from distribute_tensor(), "
                    "so we cannot check for equality using it"
                )
            sharded_ref_param = distribute_tensor(replicated_param, mesh, placements)
            self.assertEqual(sharded_param.to_local(), sharded_ref_param.to_local())
            if replicated_param.grad is None:
                self.assertIsNone(sharded_param.grad)
                continue
            self.assertIsNotNone(sharded_param.grad)
            sharded_ref_grad = distribute_tensor(
                replicated_param.grad, mesh, placements
            )
            self.assertIsInstance(sharded_param.grad, DTensor)
            assert isinstance(sharded_param.grad, DTensor)  # mypy
            self.assertEqual(sharded_param.grad.to_local(), sharded_ref_grad.to_local())

    def _compare_module(self, mod, replicate_mod):
        local_batch_size = 1
        global_batch_size = self.world_size * local_batch_size
        input = torch.randn(global_batch_size, 2)
        target = torch.randn(global_batch_size, 2)

        def step_model(model, input, target):
            model.train()
            output = model(input)
            loss = F.mse_loss(output, target.to(output.device))
            loss.backward()
            for param in model.parameters():
                with torch.no_grad():
                    param -= param.grad
                param.grad = None

        for iteration in range(2):
            step_model(mod, input, target)
            step_model(
                replicate_mod,
                input[
                    self.rank * local_batch_size : (self.rank + 1) * local_batch_size
                ],
                target[
                    self.rank * local_batch_size : (self.rank + 1) * local_batch_size
                ],
            )

            self.assertEqual(
                len(list(mod.parameters())),
                len(list(replicate_mod.parameters())),
            )
            for x, y in zip(mod.parameters(), replicate_mod.parameters()):
                self.assertEqual(x, y.to_local(), rtol=1.3e-06, atol=5e-5)

            # Shuffle the input so that DDP input is different
            torch.manual_seed(iteration)
            input = input[torch.randperm(global_batch_size)]

    def run_subtests(self, *args, **kwargs):
        return run_subtests(self, *args, **kwargs)

    @skip_if_lt_x_gpu(2)
    def test_replicate_net_module(self):
        """This tests that replicate works on a simple net module"""
        self._init_pg()
        model = Net()
        replicate_model = deepcopy(model)
        layers = [
            replicate_model.fc1,
            replicate_model.fc2,
            replicate_model.fc3,
        ]

        for i, layer in enumerate(layers):
            if i % 2 == 0:
                replicate(layer)
            else:
                fully_shard(layer)

        replicate_model = replicate(replicate_model)

        for parameter in replicate_model.fc1.parameters():
            self.assertEqual(parameter.placements, (Replicate(), Shard(dim=0)))

        for parameter in replicate_model.fc2.parameters():
            self.assertEqual(parameter.placements, (Shard(dim=0),))

    @skip_if_lt_x_gpu(2)
    def test_replicate_managed_modules(self):
        """
        This tests to ensure that if a module is not fully_sharded or replicated, it is still managed by the parent module
        """
        self._init_pg()
        model = Net()
        replicate_model = deepcopy(model)

        replicate(replicate_model.fc1)
        fully_shard(replicate_model.fc2)

        replicate_model = replicate(replicate_model)
        managed_modules = _get_managed_modules((replicate_model,))

        self.assertEqual(len(managed_modules), 2)

    @skip_if_lt_x_gpu(2)
    def test_replicate_transformer(self):
        """This tests that replicate works on a transformer model with fully_shard and replicate layers"""
        self._init_pg()
        self.run_subtests(
            {
                "sharding_strategy": ["replicate", "fully_shard"],
            },
            self._test_replicate_transformer,
        )

    def _composable_api_module_check(self, module, sharding_strategy):
        if sharding_strategy == "replicate":
            self.assertTrue("replicate" in _get_registry(module))
        else:
            self.assertTrue("fully_shard" in _get_registry(module))

    def _test_replicate_transformer(self, sharding_strategy):
        model_args = ModelArgs()
        model = Transformer(model_args)

        replicate_model = deepcopy(model)

        for i, layer in enumerate(replicate_model.layers):
            if i % 2 == 0:
                replicate(layer)
            else:
                fully_shard(layer)

        if sharding_strategy == "replicate":
            replicate_model = replicate(replicate_model)

        else:
            replicate_model = fully_shard(replicate_model)

        self._composable_api_module_check(replicate_model, sharding_strategy)

        for i, layer in enumerate(replicate_model.layers):
            if i % 2 == 0:
                self.assertTrue("replicate" in _get_registry(layer))
                for parameter in layer.parameters():
                    self.assertEqual(parameter.placements, (Replicate(), Shard(dim=0)))
            else:
                self.assertTrue("fully_shard" in _get_registry(layer))
                for parameter in layer.parameters():
                    self.assertEqual(parameter.placements, (Shard(dim=0),))

    @skip_if_lt_x_gpu(2)
    def test_replicate_device_mesh(self):
        """
        This tests that a user can pass in a device mesh to replicate a module
        """

        self._init_pg()

        model = Net()
        replicate_model = deepcopy(model)

        layers = [
            replicate_model.fc1,
            replicate_model.fc2,
            replicate_model.fc3,
        ]

        for i, layer in enumerate(layers):
            if i % 2 == 0:
                replicate(layer)
            else:
                fully_shard(layer)

        device_mesh = init_device_mesh(
            "cuda", (self.world_size, 1), mesh_dim_names=("replicate", "shard")
        )

        replicate_model = replicate(replicate_model, device_mesh=device_mesh)

        for parameter in replicate_model.fc1.parameters():
            self.assertEqual(parameter.placements, (Replicate(), Shard(dim=0)))

        for parameter in replicate_model.fc2.parameters():
            self.assertEqual(parameter.placements, (Shard(dim=0),))

    def test_train_replicate_fsdp(self):
        self._init_pg()

        model = Net()
        replicate_model = deepcopy(model)

        layers = [
            replicate_model.fc1,
            replicate_model.fc2,
            replicate_model.fc3,
        ]

        for layer in layers:
            fully_shard(layer)

        replicate_model = fully_shard(replicate_model)

        optim = torch.optim.Adam(model.parameters(), lr=0.01)
        replicate_optim = torch.optim.Adam(replicate_model.parameters(), lr=0.01)

        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn(2, 2)

        for _ in range(10):
            loss = model(inp).sum()
            loss.backward()

            for param in model.parameters():
                dist.all_reduce(param.grad)

            replicate_loss = replicate_model(inp).sum()
            replicate_loss.backward()

            optim.step()
            replicate_optim.step()

            optim.zero_grad()
            replicate_optim.zero_grad()

            self.assertEqual(replicate_loss, loss)
            self.check_sharded_parity(model, replicate_model)


if __name__ == "__main__":
    run_tests()
