# Owner(s): ["oncall: distributed"]

import os
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed._composable.replicate import replicate
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests


class Net(nn.Module):
    def __init__(self):
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
            backend="gloo",
            rank=self.rank,
            world_size=self.world_size,
            store=dist.FileStore(self.file_name, self.world_size),
        )

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
        self._init_pg()
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
        self._init_pg()
        model = Net()
        replicate_model = deepcopy(model)
        replicate(replicate_model.fc1)
        replicate(replicate_model.fc2)
        replicate(replicate_model.fc3)

        local_sd = model.state_dict()
        ddp_sd = replicate_model.state_dict()
        self._check_state_dict_parity(local_sd, ddp_sd)


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
            for i, j in zip(mod.parameters(), replicate_mod.parameters()):
                self.assertEqual(i, j, rtol=1.3e-06, atol=5e-5)

            # Shuffle the input so that DDP input is different
            torch.manual_seed(iteration)
            input = input[torch.randperm(global_batch_size)]

    def test_replicate_single_module(self):
        self._init_pg()
        model = Net()
        replicate_model = replicate(deepcopy(model))
        self._compare_module(model, replicate_model)

    @skip_if_lt_x_gpu(2)
    def test_replicate_move_args_kwargs_to_device(self):
        class MyNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(2, 2)

            def forward(self, inp, *, kwarg=None):
                if kwarg is not None:
                    inp = inp @ kwarg
                return self.a(inp)

        self._init_pg()
        torch.cuda.set_device(self.rank)
        model = MyNet().cuda()
        replicate(model, device_id=torch.cuda.current_device())
        # CPU input ensures replicate can move arg and kwargs to device.
        a, b = torch.randn(2, 2), torch.randn(2, 2)
        model(a, kwarg=b).sum().backward()

    @skip_if_lt_x_gpu(2)
    def test_replicate_ignore_module(self):
        self._init_pg()
        torch.cuda.set_device(self.rank)
        # Seed ensures diff input and thus different local grads across ranks.
        torch.manual_seed(self.rank)
        torch.cuda.manual_seed(self.rank)
        model = Net().cuda()
        replicate(model, ignored_modules=[model.fc1])
        # CPU input ensures that replicate can move input to GPU as DDP does.
        inp = torch.randn(5, 2, device="cuda") * (self.rank + 1)
        out = model(inp) * 10
        out.sum().backward()
        # FC1 grads should not be synchronized, FC2 and 3 should be.
        fc1_grad = model.fc1.weight.grad
        tensor_list = [torch.zeros_like(fc1_grad) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list, fc1_grad)
        grad, rest = tensor_list[0], tensor_list[1:]
        for g in rest:
            self.assertNotEqual(grad, g)

        for dp_grad in [model.fc2.weight.grad, model.fc3.weight.grad]:
            tensor_list = [
                torch.zeros_like(dp_grad) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(tensor_list, dp_grad)
            grad, rest = tensor_list[0], tensor_list[1:]
            for g in rest:
                self.assertEqual(grad, g)

    def test_replicate_multi_module(self):
        self._init_pg()
        model = Net()
        replicate_model = deepcopy(model)
        replicate(replicate_model.fc1)
        replicate(replicate_model.fc2)
        replicate(replicate_model.fc3)
        self._compare_module(model, replicate_model)

    def test_replicate_with_kwargs(self):
        self._init_pg()
        model = Net()
        replicate_model = replicate(
            deepcopy(model), bucket_cap_mb=1, gradient_as_bucket_view=True
        )
        self._compare_module(model, replicate_model)

    @skip_if_lt_x_gpu(2)
    def test_replicate_device_id(self):
        self._init_pg()
        model = Net()
        model_cuda = deepcopy(model).cuda()
        model_cuda2 = deepcopy(model_cuda)
        replicate(model, device_id=torch.device("cpu"))
        # DDP instance is attached in first pre forward
        model(torch.randn(2, 2))
        replicate_ddp_weakref = replicate.state(model)._ddp_weakref()
        # Should be None for CPU training
        self.assertEqual(None, replicate_ddp_weakref.device_ids)

        replicate(model_cuda, device_id=torch.device(torch.cuda.current_device()))
        # DDP instance is attached in first pre forward
        model_cuda(torch.randn(2, 2))
        replicate_ddp_weakref = replicate.state(model_cuda)._ddp_weakref()
        self.assertEqual([0], replicate_ddp_weakref.device_ids)
        # Pass in int as device_id
        replicate(model_cuda2, device_id=int(torch.cuda.current_device()))
        # DDP instance is attached in first pre forward
        model_cuda2(torch.randn(2, 2))
        replicate_ddp_weakref = replicate.state(model_cuda2)._ddp_weakref()
        self.assertEqual([0], replicate_ddp_weakref.device_ids)

    def test_replicate_wrong_device_id_type(self):
        self._init_pg()
        model = Net()
        with self.assertRaisesRegex(
            RuntimeError, "Expected device_id to be int or torch.device"
        ):
            replicate(model, device_id=[torch.device("cpu")])


if __name__ == "__main__":
    run_tests()
