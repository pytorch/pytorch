# Owner(s): ["oncall: distributed"]

import os
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed._composable.replicate import replicate
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import run_tests


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 10, bias=False)
        self.fc2 = nn.Linear(10, 50, bias=False)
        self.fc3 = nn.Linear(50, 4, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

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

    def _check_state_dict_parity(self, sd_1, sd_2):
        for k1, k2 in zip(sd_1.keys(), sd_2.keys()):
            self.assertEqual(k1, k2)

        for v1, v2 in zip(sd_1.values(), sd_2.values()):
            self.assertEqual(v1, v2)

    def test_replicate_single_module_save_load(self):
        """
        Tests that replicate() on a single module state_dict
        matches local module state_dict.
        """
        model = Net()
        replicate_model = replicate(deepcopy(model))
        local_sd = model.state_dict()
        ddp_sd = replicate_model.state_dict()
        self._check_state_dict_parity(local_sd, ddp_sd)

    def test_replicate_non_root_multiple_save_load(self):
        """
        Tests tha replicate() on multiple submodules matches
        local module state_dict.
        """
        model = Net()
        replicate_model = deepcopy(model)
        replicate(replicate_model.fc1)
        replicate(replicate_model.fc2)
        replicate(replicate_model.fc3)

        local_sd = model.state_dict()
        ddp_sd = replicate_model.state_dict()
        self._check_state_dict_parity(local_sd, ddp_sd)

class ReplicateTest(MultiProcessTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def _compare_module(self, mod, replicate_mod):
        dist.init_process_group(
            backend="gloo",
            rank=self.rank,
            world_size=self.world_size,
            store=dist.FileStore(self.file_name, self.world_size),
        )

        local_batch_size = 1
        global_batch_size = self.world_size * local_batch_size
        input = torch.randn(global_batch_size, 2)
        target = torch.randn(global_batch_size, 4)

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
                    self.rank
                    * local_batch_size : (self.rank + 1)
                    * local_batch_size
                ],
                target[
                    self.rank
                    * local_batch_size : (self.rank + 1)
                    * local_batch_size
                ],
            )

            self.assertEqual(
                len(list(mod.parameters())),
                len(list(replicate_mod.parameters())),
            )
            for i, j in zip(mod.parameters(), replicate_mod.parameters()):
                self.assertEqual(i, j, rtol=1.3e-06, atol=5e-5)

            # Shuffle the input so that DDP input is different
            torch.manual_seed(iteration)
            input = input[torch.randperm(global_batch_size)]

    def test_replicate_single_module(self):
        model = Net()
        replicate_model = replicate(deepcopy(model))
        self._compare_module(model, replicate_model)

    def test_replicate_multi_module(self):
        model = Net()
        replicate_model = deepcopy(model)
        replicate(replicate_model.fc1)
        replicate(replicate_model.fc2)
        replicate(replicate_model.fc3)
        self._compare_module(model, replicate_model)

    def test_replicate_with_kwargs(self):
        model = Net()
        replicate_model = replicate(
            deepcopy(model), bucket_cap_mb=1, gradient_as_bucket_view=True
        )
        self._compare_module(model, replicate_model)


if __name__ == "__main__":
    run_tests()
