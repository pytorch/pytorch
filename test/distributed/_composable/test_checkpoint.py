# Owner(s): ["oncall: distributed"]

from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)

import torch
import torch.nn as nn
from torch.distributed._composable import checkpoint

import unittest
from collections import deque
from copy import deepcopy


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(100, 100)
        self.seq = nn.Sequential(
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.seq(self.l1(x))


class TestCheckpoint(TestCase):
    def _get_graph_size(self, out: torch.Tensor) -> int:
        q = deque([out.grad_fn])
        num_functions = 0
        while len(q):
            fn = q.pop()
            num_functions += 1
            for next_fn, _ in fn.next_functions:
                if next_fn:
                    q.append(next_fn)

        return num_functions

    def _test_tensor_only(self, net: nn.Module, x: torch.Tensor) -> None:
        x1 = x.clone()
        x2 = x.clone()
        x1.requires_grad = True
        x2.requires_grad = True

        net1 = net
        net2 = deepcopy(net)

        # no checkpoint
        loss1 = net1(x1).sum()
        graph_size1 = self._get_graph_size(loss1)
        loss1.backward()

        # with checkpoint
        checkpoint(net2.seq)
        loss2 = net2(x2).sum()
        graph_size2 = self._get_graph_size(loss2)
        loss2.backward()

        self.assertTrue(graph_size2 < graph_size1)

        for p1, p2 in zip(net1.parameters(), net2.parameters()):
            self.assertEqual(p1.grad, p2.grad)

    def test_tensor_only_cpu(self):
        x = torch.randn(20, 100)
        net = ToyModel()
        self._test_tensor_only(net, x)

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_tensor_only_gpu(self):
        x = torch.randn(20, 100, device="cuda:0")
        net = ToyModel().to("cuda:0")
        self._test_tensor_only(net, x)


if __name__ == "__main__":
    run_tests()
