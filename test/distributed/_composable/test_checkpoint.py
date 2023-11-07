# Owner(s): ["oncall: distributed"]

import unittest
from collections import deque, OrderedDict
from contextlib import ContextDecorator
from copy import deepcopy
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributed._composable import checkpoint
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, TestCase


class MemoryDelta(ContextDecorator):
    def __init__(self, device: torch.device):
        self.device: torch.device = device
        self.active_memory_enter: int = 0
        self.active_memory_exit: int = 0

    def __enter__(self):
        self.active_memory_enter = (
            torch.cuda.memory_stats()["active_bytes.all.current"]
            if self.device.type == "cuda"
            else 0
        )
        return self

    def __exit__(self, *exc):
        self.active_memory_exit = (
            torch.cuda.memory_stats()["active_bytes.all.current"]
            if self.device.type == "cuda"
            else 0
        )

    def delta(self) -> int:
        return self.active_memory_exit - self.active_memory_enter


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


class RandomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.randn(100, 100))

    def forward(self, x):
        y = torch.matmul(self.p, torch.randn(100, 100, device=self.p.device))
        return torch.matmul(x, y)


class MultiOutputModel(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn((100, 100), device=device))
        self.w2 = nn.Parameter(torch.randn((100, 100), device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x @ self.w1
        z = nn.functional.relu(z)
        z = z @ self.w2
        return z.sin(), z.cos()


class MultiInputModel(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.w = nn.Parameter(torch.randn((100, 100), device=device))

    def forward(self, xs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        assert len(xs) == 2, f"Expects 2 args but got {len(xs)}"
        x, y = xs
        z = x + y
        z = z @ self.w
        return nn.functional.relu(z)


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

    def _test_tensor_only(
        self,
        net: nn.Module,
        x: torch.Tensor,
    ) -> None:
        x1 = x.clone()
        x2 = x.clone()
        x1.requires_grad = True
        x2.requires_grad = True

        net1 = net
        net2 = deepcopy(net)

        # no checkpoint
        with MemoryDelta(x.device) as mem1:
            loss1 = net1(x1).sum()
        graph_size1 = self._get_graph_size(loss1)
        loss1.backward()

        # with checkpoint
        checkpoint(net2.seq)
        with MemoryDelta(x.device) as mem2:
            loss2 = net2(x2).sum()
        loss2.backward()

        if x.is_cuda:
            self.assertTrue(mem2.delta() < mem1.delta())

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

    def test_random_cpu(self):
        x1 = torch.randn(20, 100, requires_grad=True)
        x2 = x1.clone()

        net1 = RandomModel()
        net2 = deepcopy(net1)

        cpu_rng_state = torch.get_rng_state()
        net1(x1).sum().backward()
        torch.set_rng_state(cpu_rng_state)
        checkpoint(net2)(x2).sum().backward()

        for p1, p2 in zip(net1.parameters(), net2.parameters()):
            self.assertEqual(p1.grad, p2.grad)

    def test_multi_args(self):
        """
        Tests checkpoint for modules with multiple output args and hence
        multiple backward function input args.
        """
        device = torch.device("cpu")
        net1 = nn.Sequential(
            MultiOutputModel(device),
            MultiInputModel(device),
            MultiOutputModel(device),
            MultiInputModel(device),
        )
        net2 = deepcopy(net1)
        checkpoint(net2[0])
        checkpoint(net2[2])
        x1 = torch.randn(20, 100, requires_grad=True)
        x2 = x1.clone()
        net1(x1).sum().backward()
        net2(x2).sum().backward()
        for p1, p2 in zip(net1.parameters(), net2.parameters()):
            self.assertEqual(p1.grad, p2.grad)

    def test_clears_state_on_error_in_forward(self):
        class MyModel(torch.nn.Module):
            def __init__(self, raise_in_recomp):
                super().__init__()
                self.fwd_count = 0
                self.raise_in_recomp = raise_in_recomp
                self.a = torch.nn.Linear(2, 2)

            def forward(self, x):
                if self.raise_in_recomp and self.fwd_count == 1:
                    raise RuntimeError("foo")
                else:
                    if not self.raise_in_recomp:
                        # raise in the first forward
                        raise RuntimeError("foo")
                    self.fwd_count += 1
                    return self.a(x)

        m = MyModel(raise_in_recomp=True)
        m_seq = torch.nn.Sequential(OrderedDict({"m": m}))
        checkpoint(m_seq.m)
        inp = torch.randn(1, 2)
        out = m_seq(inp).sum()
        # Should raise in forward recomputation
        with self.assertRaisesRegex(RuntimeError, "foo"):
            out.backward()

        # Check that _ac_generator is cleared out
        self.assertEqual(None, checkpoint.state(m)._ac_generator)

        m = MyModel(raise_in_recomp=False)
        checkpoint(m)
        inp = torch.randn(1, 2)
        # Should raise in first forward
        with self.assertRaises(RuntimeError):
            m(inp)

        self.assertEqual(None, checkpoint.state(m)._ac_generator)


if __name__ == "__main__":
    run_tests()
