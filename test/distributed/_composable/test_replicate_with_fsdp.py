# Owner(s): ["oncall: distributed"]

import os
from copy import deepcopy

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._composable.contract import _get_registry
from torch.distributed._composable.replicate_with_fsdp import (
    _get_managed_modules,
    replicate,
)
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import Replicate, Shard
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    run_subtests,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_fsdp import check_sharded_parity
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
        return 4

    def init_global_mesh(self) -> DeviceMesh:
        # Prefer to test with >=4 GPUs, but for 2 GPUs, use 2-way TP
        replicate_size = 2
        return init_device_mesh(
            "cuda",
            (replicate_size, 1, self.world_size // replicate_size),
            mesh_dim_names=("replicate", "shard", "tp"),
        )

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

    @skip_if_lt_x_gpu(2)
    def test_replicate_transformer(self):
        """
        This tests that replicate works on a transformer model with fully_shard and replicate layers
        """
        self._init_pg()
        run_subtests(
            self,
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
            if i % 3 == 0:
                replicate(layer)
            elif i % 3 == 1:
                fully_shard(layer)

        if sharding_strategy == "replicate":
            replicate_model = replicate(replicate_model)

        else:
            replicate_model = fully_shard(replicate_model)

        self._composable_api_module_check(replicate_model, sharding_strategy)

        for i, layer in enumerate(replicate_model.layers):
            if i % 3 == 0:
                self.assertTrue("replicate" in _get_registry(layer))
                for parameter in layer.parameters():
                    self.assertEqual(parameter.placements, (Replicate(), Shard(dim=0)))
            elif i % 3 == 1:
                self.assertTrue("fully_shard" in _get_registry(layer))
                for parameter in layer.parameters():
                    self.assertEqual(parameter.placements, (Shard(dim=0),))

        managed_modules = _get_managed_modules((replicate_model,))

        if sharding_strategy == "replicate":
            self.assertEqual(len(managed_modules), 7)

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

        global_mesh = self.init_global_mesh()
        replicate_mesh = global_mesh["replicate", "shard"]

        for layer in layers:
            replicate_model = replicate(layer, device_mesh=replicate_mesh)

            for parameter in layer.parameters():
                self.assertEqual(parameter.device_mesh.shape, (2, 1))
                self.assertEqual(parameter.placements, (Replicate(), Shard(dim=0)))

        # replicate_model = replicate(replicate_model, device_mesh=replicate_mesh)

    def test_train_replicate_fsdp(self):
        """
        Tests that replicate_model has the same behavior as original model when training
        """
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
            check_sharded_parity(self, model, replicate_model)


if __name__ == "__main__":
    run_tests()
