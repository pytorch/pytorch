# Owner(s): ["oncall: distributed"]

import copy
import functools
import unittest

import torch
import torch.nn as nn
from torch.distributed._composable.offload import frozen_offload, to_gpu
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, TestCase


class MLP(nn.Module):
    def __init__(self, dim: int, device: torch.device):
        super().__init__()
        self.lin1 = nn.Linear(dim, 4 * dim, device=device)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(4 * dim, dim, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.lin2(self.relu(self.lin1(x))))


class FrozenMLP(MLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for param in self.parameters():
            param.requires_grad_(False)


class TestFrozenOffload(TestCase):
    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_parity(self):
        torch.manual_seed(0)
        lin_dim = 1024
        model = nn.Sequential(
            FrozenMLP(lin_dim, torch.device("cuda")),
            MLP(lin_dim, torch.device("cuda")),
            nn.Linear(lin_dim, lin_dim, device="cuda"),
            nn.ReLU(),
        )
        ref_model = copy.deepcopy(model)
        ref_optim = torch.optim.Adam(
            [p for p in ref_model.parameters() if p.requires_grad], lr=1e-2
        )
        frozen_offload(model[0])
        optim = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=1e-2
        )
        # For users: replace the following hook with any other backward hook
        # to run `to_gpu()` to prefetch for the next iteration
        model[2].weight.register_post_accumulate_grad_hook(
            functools.partial(to_gpu, model[0])
        )
        # Check that initialization is correct
        for (n1, p1), (n2, p2) in zip(
            model[0].named_parameters(), ref_model[0].named_parameters()
        ):
            self.assertEqual(n1, n2)
            self.assertEqual(p1.device, torch.device("cpu"))
            self.assertTrue(p2.is_cuda)
            self.assertEqual(p1.cuda(), p2)

        # Check that training is correct (e.g. stream sync)
        torch.manual_seed(42)
        inp = torch.randn((16, lin_dim), device="cuda")
        for i in range(10):
            losses = []
            for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                _optim.zero_grad()
                loss = _model(inp).sum()
                losses.append(loss)
                loss.backward()
                _optim.step()

            self.assertEqual(losses[0], losses[1])


if __name__ == "__main__":
    run_tests()
