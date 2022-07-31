# Owner(s): ["module: cuda graphs"]

import functools
import sys

from unittest.mock import patch

import torch
from torch.testing._internal.common_utils import run_tests, TestCase

try:
    import functorch  # noqa: F401
    import torchdynamo
    from torch.cuda._dynamo_graphs import aot_autograd_cudagraphs

    TEST_DYNAMO = True
except ImportError:
    TEST_DYNAMO = False

TEST_CUDA = torch.cuda.is_available()

if not TEST_CUDA or not TEST_DYNAMO:
    print("CUDA or dynamo not available, skipping tests", file=sys.stderr)
    TestCase = object  # noqa: F811


def composed(*decs):
    def deco(f):
        for dec in reversed(decs):
            f = dec(f)
        return f

    return deco


def assert_aot_autograd_counter(ok=True):
    def deco(f):
        @functools.wraps(f)
        def wrap(self, *args, **kwargs):
            torchdynamo.utils.counters.clear()
            r = f(self, *args, **kwargs)
            c_ok = torchdynamo.utils.counters["aot_autograd"]["ok"]
            c_not_ok = torchdynamo.utils.counters["aot_autograd"]["not_ok"]
            if ok:
                self.assertGreater(c_ok, 0)
                self.assertEqual(c_not_ok, 0)
            else:
                self.assertEqual(c_ok, 0)
                self.assertGreater(c_not_ok, 0)
            return r

        return wrap

    return deco


def patch_all(ok=True):
    return composed(
        patch("torchdynamo.config.verify_correctness", True),
        assert_aot_autograd_counter(ok),
    )


N_ITERS = 5


class TestDynamoCudaGraphs(TestCase):
    @patch_all()
    def test_basic(self):
        def model(x, y):
            return (x + y) * y

        with torchdynamo.optimize(aot_autograd_cudagraphs):
            for i in range(N_ITERS):
                x = torch.randn(3, device="cuda", requires_grad=True)
                y = torch.randn(3, device="cuda")
                loss = model(x, y).sum()
                loss.backward()

    @patch_all()
    def test_dtoh(self):
        def model(x, y):
            a = x + y
            b = a.cpu() * 3
            return b

        with torchdynamo.optimize(aot_autograd_cudagraphs):
            for i in range(N_ITERS):
                x = torch.randn(3, device="cuda", requires_grad=True)
                y = torch.randn(3, device="cuda")
                loss = model(x, y).sum()
                loss.backward()

    @patch_all()
    def test_htod(self):
        def model(x, y):
            a = x + y
            return a * 3

        with torchdynamo.optimize(aot_autograd_cudagraphs):
            for i in range(N_ITERS):
                x = torch.randn(3, device="cuda", requires_grad=True)
                y = torch.randn((), device="cpu")
                loss = model(x, y).sum()
                loss.backward()

    @patch("functorch._src.config.use_functionalize", True)
    @patch_all(ok=False)  # input mutation not supported yet
    def test_mutate_input(self):
        def model(x, y):
            y.add_(3)
            return x * y

        with torchdynamo.optimize(aot_autograd_cudagraphs):
            for i in range(N_ITERS):
                with self.subTest(i):
                    x = torch.randn(3, device="cuda", requires_grad=True)
                    y = torch.randn(3, device="cuda")
                    y_orig = y.clone()
                    loss = model(x, y).sum()
                    self.assertEqual(y, y_orig + 3)
                    loss.backward()

    @patch_all()
    def test_mutate_constant(self):
        def model(x, y):
            c = torch.tensor(1)
            c.add_(2)
            return x * y * 0 + c

        with torchdynamo.optimize(aot_autograd_cudagraphs):
            for i in range(N_ITERS):
                with self.subTest(i):
                    x = torch.randn(1, device="cuda", requires_grad=True)
                    y = torch.randn(1, device="cuda")
                    loss = model(x, y).sum()
                    self.assertEqual(loss, torch.tensor(3.0, device="cuda"))
                    loss.backward()

    @patch_all()
    def test_factory(self):
        def model(y):
            x = torch.zeros(3, device="cuda:0")
            x.add_(3)
            return x * y

        with torchdynamo.optimize(aot_autograd_cudagraphs):
            for i in range(N_ITERS):
                with self.subTest(i):
                    y = torch.randn(3, device="cuda:0", requires_grad=True)
                    loss = model(y).sum()
                    loss.backward()

    @patch("functorch._src.config.use_functionalize", True)
    @patch_all()
    def test_mutated_metadata(self):
        # more tortured example at
        # https://github.com/pytorch/pytorch/issues/81385
        def model(x):
            x = x.clone()
            x.resize_(20)
            x.fill_(2)
            return x

        with torchdynamo.optimize(aot_autograd_cudagraphs):
            for i in range(N_ITERS):
                with self.subTest(i):
                    x = torch.empty(0, device="cuda:0")
                    rx = model(x)
                    self.assertEqual(rx, torch.full((20,), 2.0, device="cuda:0"))

    @patch("functorch._src.config.use_functionalize", True)
    @patch_all()
    def test_dead_fill(self):
        def model(x):
            x = x.clone()
            y = x[0:0]
            x.fill_(2)
            y.fill_(3)
            return x, y

        with torchdynamo.optimize(aot_autograd_cudagraphs):
            for i in range(N_ITERS):
                with self.subTest(i):
                    x = torch.empty(20, device="cuda:0")
                    rx, ry = model(x)
                    self.assertEqual(rx, torch.full((20,), 2.0, device="cuda:0"))
                    self.assertEqual(ry, torch.empty(0, device="cuda:0"))


if __name__ == "__main__":
    run_tests()
