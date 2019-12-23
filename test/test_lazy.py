import torch
from common_utils import (
    TestCase,
    run_tests,
)
import time


class TestLazy(TestCase):
    def test_bench(self):
        a = torch.rand(1, 1)
        b = torch.rand(1, 1)
        a_ = a.to_lazy()
        b_ = b.to_lazy()
        iters = int(1e2)

        t = time.time()
        for _ in range(iters):
            c = a + b
            a = c
        print(time.time() - t)

        t = time.time()
        for _ in range(iters):
            c_ = a_ + b_
            a_ = c_
        print(time.time() - t)
        c_ = c_.to_eager()
        torch.testing.assert_allclose(c_, c)

    def test_basic(self):
        a = torch.rand(10, 10)
        b = torch.rand(10, 10)
        c = a
        d = b

        a = a.to_lazy()
        b = b.to_lazy()

        k = a + b
        k = torch.abs(k)
        l = c + d
        l = torch.abs(l)

        k = k.to_eager()
        torch.testing.assert_allclose(k, l)

    def test_multireturn(self):
        a = torch.rand(10, 10, 10)
        b = torch.rand(10, 10, 10)
        c = a
        d = b

        a = a.to_lazy()
        b = b.to_lazy()

        l = c + d
        _, l = torch.std_mean(l)
        k = a + b
        _, k = torch.std_mean(k)

        k = k.to_eager()
        torch.testing.assert_allclose(k, l)

    def test_grad(self):
        def d(a, b):
            k = a + b
            k = k.abs()
            return k

        torch.manual_seed(0)

        a0 = torch.rand(10, requires_grad=True)
        b0 = torch.rand(10)

        torch.manual_seed(0)

        a1_ = torch.rand(10)
        a1 = a1_.to_lazy()
        a1.requires_grad = True
        b1 = torch.rand(10).to_lazy()

        c0 = d(a0, b0)
        c0.backward(torch.ones(10))

        c1 = d(a1, b1)
        torch.testing.assert_allclose(c0, c1.to_eager())

        c1.backward(torch.ones(10))
        k = a1.grad.to_eager()
        torch.testing.assert_allclose(a0.grad, k)


if __name__ == "__main__":
    run_tests()
