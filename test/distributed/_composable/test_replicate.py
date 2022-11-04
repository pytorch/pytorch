import os
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed._composable.replicate import mark_root_module, replicate
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

    def _prepare_module(self, global_batch_size):
        model = Net()
        input = torch.randn(global_batch_size, 2)
        target = torch.randn(global_batch_size, 4)
        return model, input, target

    def test_replicate(self):
        dist.init_process_group(
            backend="gloo",
            rank=self.rank,
            world_size=self.world_size,
            store=dist.FileStore(self.file_name, self.world_size),
        )

        local_batch_size = 1
        global_batch_size = self.world_size * local_batch_size
        model, input, target = self._prepare_module(global_batch_size)
        replicate_model = mark_root_module(replicate(deepcopy(model)))

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
            step_model(model, input, target)
            step_model(
                replicate_model,
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
                len(list(model.parameters())),
                len(list(replicate_model.parameters())),
            )
            for i, j in zip(model.parameters(), replicate_model.parameters()):
                self.assertEqual(i, j, rtol=1.3e-06, atol=5e-5)

            # Shuffle the input so that DDP input is different
            torch.manual_seed(iteration)
            input = input[torch.randperm(global_batch_size)]


if __name__ == "__main__":
    run_tests()
