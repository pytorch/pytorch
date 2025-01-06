# Owner(s): ["oncall: distributed"]

import unittest
from collections import deque, OrderedDict
from contextlib import ContextDecorator, contextmanager, nullcontext
from copy import deepcopy
from functools import partial
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributed._composable import checkpoint
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils.checkpoint import CheckpointError


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
    def __init__(self) -> None:
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
    def __init__(self) -> None:
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def test_checkpoint_kwargs(self):
        class MyModel(torch.nn.Module):
            def __init__(self, raise_exp: bool, change_shape_in_recomp: bool):
                super().__init__()
                self.fwd_count = 0
                self.raise_exp = raise_exp
                self.change_shape_in_recomp = change_shape_in_recomp
                self.a = torch.nn.Linear(2, 2)

            def forward(self, x):
                if self.raise_exp and self.fwd_count == 0:
                    raise RuntimeError("foo")
                if self.raise_exp and self.fwd_count == 1:
                    raise RuntimeError("bar")
                if self.change_shape_in_recomp and self.fwd_count == 1:
                    x.relu_()
                random_tensor = torch.randn(1, 2)
                x = self.a(x + random_tensor)
                self.fwd_count += 1
                return x

        m = MyModel(True, False)
        m0, m1, m2, m3 = (deepcopy(m) for _ in range(4))

        # composable checkpoint does not support use_reentrant=True
        with self.assertRaisesRegex(
            NotImplementedError,
            "use_reentrant=True is not supported in composable checkpoint. "
            "Please use torch.utils.checkpoint.checkpoint instead.",
        ):
            checkpoint(m, use_reentrant=True)

        # check giving an unsupported kwarg
        with self.assertRaisesRegex(ValueError, "Unexpected keyword arguments: foo"):
            checkpoint(m0, foo="bar")

        handled_fwd_exp = False
        handled_recomp_exp = False

        @contextmanager
        def fwd_ctx(mod: MyModel):
            try:
                mod.raise_exp = False
                yield
            finally:
                nonlocal handled_fwd_exp
                handled_fwd_exp = True
                mod.raise_exp = True

        @contextmanager
        def recomp_ctx(mod: MyModel):
            try:
                mod.raise_exp = False
                yield
            finally:
                nonlocal handled_recomp_exp
                handled_recomp_exp = True
                mod.raise_exp = True

        # Test different context functions
        x = torch.randn(1, 2, requires_grad=True)
        checkpoint(
            m1, context_fn=lambda: (partial(fwd_ctx, m1)(), partial(recomp_ctx, m1)())
        )
        m1(x.clone()).sum().backward()
        self.assertEqual((handled_fwd_exp, handled_recomp_exp), (True, True))

        checkpoint(m2, context_fn=lambda: (nullcontext(), partial(recomp_ctx, m2)()))
        with self.assertRaisesRegex(RuntimeError, "foo"):
            m2(x.clone())

        handled_fwd_exp = False  # Reset flag
        checkpoint(m3, context_fn=lambda: (partial(fwd_ctx, m3)(), nullcontext()))
        with self.assertRaisesRegex(RuntimeError, "bar"):
            m3(x.clone()).sum().backward()
        self.assertEqual(handled_fwd_exp, True)

        # Test determinism check failure
        m4 = MyModel(False, True)
        m5 = deepcopy(m4)
        # Determinism check should not throw an error,
        # but autograd should throw a RuntimeError
        checkpoint(m4, determinism_check="none")
        with self.assertRaises(RuntimeError):
            m4(x.clone()).sum().backward()

        # Determinism check should throw a CheckpointError
        checkpoint(m5, determinism_check="default")
        with self.assertRaises(CheckpointError):
            m5(x.clone()).sum().backward()

        # Test preserving random state
        m6 = MyModel(False, False)
        m7, m8 = (deepcopy(m6) for _ in range(2))
        checkpoint(m7, preserve_rng_state=False)
        checkpoint(m8, preserve_rng_state=True)

        for mi in (m6, m7, m8):
            torch.manual_seed(42)
            loss = mi(x.clone()).sum()
            torch.manual_seed(41)
            loss.backward()
        # check that m6 and m7 have at least one different grad
        self.assertNotEqual(
            (p1.grad for p1 in m6.parameters()), (p2.grad for p2 in m7.parameters())
        )
        # check that m6 and m8 have identical grads
        for p1, p2 in zip(m6.parameters(), m8.parameters()):
            self.assertEqual(p1.grad, p2.grad)


if __name__ == "__main__":
    run_tests()
