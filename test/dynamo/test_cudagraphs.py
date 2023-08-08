# Owner(s): ["module: cuda graphs"]

import functools
import unittest

import torch

import torch._dynamo
import torch._dynamo.config
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.testing import same
from torch.testing._internal.common_utils import (
    skipIfRocm,
    TEST_CUDA_GRAPH,
    TEST_WITH_ROCM,
)


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
            torch._dynamo.utils.counters.clear()
            r = f(self, *args, **kwargs)
            c_ok = torch._dynamo.utils.counters["aot_autograd"]["ok"]
            c_not_ok = torch._dynamo.utils.counters["aot_autograd"]["not_ok"]
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
        unittest.skipIf(TEST_WITH_ROCM, "ROCm not supported"),
        torch._dynamo.config.patch(
            verify_correctness=True, dynamic_shapes=True, automatic_dynamic_shapes=True
        ),
        assert_aot_autograd_counter(ok),
    )


N_ITERS = 5


@unittest.skipIf(not torch.cuda.is_available(), "these tests require cuda")
class TestAotCudagraphs(torch._dynamo.test_case.TestCase):
    @patch_all()
    def test_basic(self):
        def model(x, y):
            return (x + y) * y

        @torch._dynamo.optimize("cudagraphs")
        def fn(x, y):
            for i in range(N_ITERS):
                loss = model(x, y).sum()
                loss.backward()

        x = torch.randn(3, device="cuda", requires_grad=True)
        y = torch.randn(3, device="cuda")
        fn(x, y)

    @patch_all()
    def test_dtoh(self):
        def model(x, y):
            a = x + y
            b = a.cpu() * 3
            return b

        @torch._dynamo.optimize("cudagraphs")
        def fn(x, y):
            for i in range(N_ITERS):
                loss = model(x, y).sum()
                loss.backward()

        x = torch.randn(3, device="cuda", requires_grad=True)
        y = torch.randn(3, device="cuda")
        fn(x, y)

    @patch_all()
    def test_htod(self):
        def model(x, y):
            a = x + y
            return a * 3

        @torch._dynamo.optimize("cudagraphs")
        def fn(x, y):
            for i in range(N_ITERS):
                loss = model(x, y).sum()
                loss.backward()

        x = torch.randn(3, device="cuda", requires_grad=True)
        y = torch.randn((), device="cpu")
        fn(x, y)

    @skipIfRocm
    def test_mutate_input(self):
        def model(x, y):
            y.add_(3)
            return x * y

        @torch._dynamo.optimize("cudagraphs")
        def fn(x, y):
            for i in range(N_ITERS):
                with self.subTest(i):
                    y_orig = y.clone()
                    loss = model(x, y).sum()
                    self.assertTrue(same(y, y_orig + 3))
                    loss.backward()

        x = torch.randn(3, device="cuda", requires_grad=True)
        y = torch.randn(3, device="cuda")
        fn(x, y)

    @patch_all()
    def test_mutate_constant(self):
        def model(x, y):
            c = torch.tensor(1)
            c.add_(2)
            return x * y * 0 + c

        @torch._dynamo.optimize("cudagraphs")
        def fn(x, y):
            for i in range(N_ITERS):
                with self.subTest(i):
                    loss = model(x, y).sum()
                    self.assertTrue(same(loss, torch.tensor(3.0, device="cuda")))
                    loss.backward()

        x = torch.randn(1, device="cuda", requires_grad=True)
        y = torch.randn(1, device="cuda")
        fn(x, y)

    @patch_all()
    def test_factory(self):
        def model(y):
            x = torch.zeros(3, device="cuda:0")
            x.add_(3)
            return x * y

        @torch._dynamo.optimize("cudagraphs")
        def fn(y):
            for i in range(N_ITERS):
                with self.subTest(i):
                    loss = model(y).sum()
                    loss.backward()

        y = torch.randn(3, device="cuda:0", requires_grad=True)
        fn(y)

    @patch_all()
    def test_mutated_metadata(self):
        # more tortured example at
        # https://github.com/pytorch/pytorch/issues/81385
        def model(x):
            x = x.clone()
            x.resize_(20)
            x.fill_(2)
            return x

        @torch._dynamo.optimize("cudagraphs")
        def fn(x):
            for i in range(N_ITERS):
                with self.subTest(i):
                    rx = model(x)
                    self.assertTrue(same(rx, torch.full((20,), 2.0, device="cuda:0")))

        x = torch.empty(0, device="cuda:0")
        fn(x)

    @patch_all()
    def test_dead_fill(self):
        def model(x):
            x = x.clone()
            y = x[0:0]
            x.fill_(2)
            y.fill_(3)
            return x, y

        @torch._dynamo.optimize("cudagraphs")
        def fn(x):
            for i in range(N_ITERS):
                with self.subTest(i):
                    rx, ry = model(x)
                    self.assertTrue(same(rx, torch.full((20,), 2.0, device="cuda:0")))
                    self.assertTrue(same(ry, torch.empty(0, device="cuda:0")))

        x = torch.empty(20, device="cuda:0")
        fn(x)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if not TEST_CUDA_GRAPH:
        if __name__ == "__main__":
            import sys

            sys.exit(0)
        raise unittest.SkipTest("cuda graph test is skipped")

    run_tests()
