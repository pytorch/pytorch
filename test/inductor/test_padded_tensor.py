import torch
import torch.nn as nn
from torch._inductor.experimental.padded_tensor import PaddedTensor
from torch._inductor.test_case import run_tests, TestCase
from torch.nn import functional as F


class PaddedTensorFunctionalTests(TestCase):
    def setUp(self):
        super().setUp()

    def test_dynamic_inner(self):
        """
        Test that we can use padded dimension in inner dimension.
        This results in a non-symbolic dimension in the output shape.
        """

        def f(a, b, c):
            return a @ b + c

        f = torch.compile(f, fullgraph=True)
        multipliers = {0: 16, 1: 16}

        for i in range(3, 9):
            a = torch.randn([3, i])
            b = torch.randn([i, 7])
            c = torch.randn([3, 7])

            a_p = PaddedTensor.from_tensor(a, multipliers)
            b_p = PaddedTensor.from_tensor(b, multipliers)
            c_p = PaddedTensor.from_tensor(c, multipliers)

            y = f(a, b, c)
            y_p = f(a_p, b_p, c_p)

            self.assertEqual(y_p.shape, (16, 16))
            self.assertEqual(y_p.original_tensor.shape, (3, 7))
            self.assertEqual(y, y_p.unpad())

    def test_dynamic_outer(self):
        """
        Test that we can use padded dimension in outer dimension.
        This results in a symbolic dimension in the output shape.
        """

        def f(a, b, c):
            return a @ b + c

        f = torch.compile(f, fullgraph=True)
        multipliers = {0: 16, 1: 16}

        for i in range(3, 9):
            a = torch.randn([3, i])
            b = torch.randn([i, 7])
            c = torch.randn([3, 7])

            a_p = PaddedTensor.from_tensor(a, multipliers)
            b_p = PaddedTensor.from_tensor(b, multipliers)
            c_p = PaddedTensor.from_tensor(c, multipliers)

            y = f(a, b, c)
            y_p = f(a_p, b_p, c_p)

            self.assertEqual(y_p.shape, (16, 16))
            self.assertEqual(y_p.original_tensor.shape, (3, 7))
            self.assertEqual(y, y_p.unpad())

    def test_bucketing(self):
        """
        Test that we 1. compile a new graph on a shape that is larger than the original bucket.
        2. don't compile on a shape that is smaller than the original bucket.
        """

        def f(a, b, c):
            return a @ b + c

        f = torch.compile(f, fullgraph=True)
        multipliers = {0: 16, 1: 16}

        for i in range(3, 22):
            # Every multiple of 16, we allow recompilation.
            if i % 16 == 0:
                torch._dynamo.config.error_on_recompile = False
            # It takes 2 iterations to trace graph with symbolic shapes. So after 2 iterations
            # after multiples of 16, we disallow recompilation.
            if i % 16 == 2:
                torch._dynamo.config.error_on_recompile = True

            a = PaddedTensor.from_tensor(torch.randn([3, 5]), multipliers)
            b = PaddedTensor.from_tensor(torch.randn([5, i]), multipliers)
            c = PaddedTensor.from_tensor(torch.randn([3, i]), multipliers)

            y = f(a, b, c)

            self.assertEqual(y.shape, (16, 16 if i <= 16 else 32))
            self.assertEqual(y.original_tensor.shape, (3, i))

        torch._dynamo.config.error_on_recompile = False


class AtenOpTests(TestCase):
    def setUp(self):
        super().setUp()

    def test_pointwise_unary(self):
        """
        Test that original and padded tensor produce same result on pointwise unary functions.
        """

        def f(a):
            return torch.sin(a)

        f = torch.compile(f, fullgraph=True)
        multipliers = {0: 16}

        a = torch.randn([3, 5])
        a_p = PaddedTensor.from_tensor(a, multipliers)

        y = f(a)
        y_p = f(a_p)

        self.assertEqual(y, y_p.unpad())

    def test_pointwise_binary(self):
        """
        Test that original and padded tensor produce same result on pointwise binary functions.
        """

        def f(a, b):
            return torch.add(a, b)

        f = torch.compile(f, fullgraph=True)
        multipliers = {0: 16}

        a = torch.randn([3, 5])
        b = torch.randn([3, 5])
        a_p = PaddedTensor.from_tensor(a, multipliers)
        b_p = PaddedTensor.from_tensor(b, multipliers)

        y = f(a, b)
        y_p = f(a_p, b_p)

        self.assertEqual(y, y_p.unpad())

    def test_mm(self):
        """
        Test that original and padded tensor produce same result on mm function.
        """

        def f(a, b):
            return a @ b

        f = torch.compile(f, fullgraph=True)
        multipliers = {0: 16, 1: 16}

        a = torch.randn([3, 5])
        b = torch.randn([5, 7])
        a_p = PaddedTensor.from_tensor(a, multipliers)
        b_p = PaddedTensor.from_tensor(b, multipliers)

        y = f(a, b)
        y_p = f(a_p, b_p)

        self.assertEqual(y, y_p.unpad())

    def test_sum(self):
        """
        Test that original and padded tensor produce same result on sum function.
        """

        def f(a):
            return a.sum()

        f = torch.compile(f, fullgraph=True)
        multipliers = {0: 16}

        a = torch.randn([3, 5])
        a_p = PaddedTensor.from_tensor(a, multipliers)

        y = f(a)
        y_p = f(a_p)

        self.assertEqual(y, y_p.unpad())

    def test_flatten(self):
        """
        Test that original and padded tensor produce same result on flatten function.
        """

        def f(a):
            return torch.flatten(a)

        f = torch.compile(f, fullgraph=True)

        multipliers = {0: 16}
        a = torch.randn([3, 5])
        a_p = PaddedTensor.from_tensor(a, multipliers)

        y = f(a)
        y_p = f(a_p)

        self.assertEqual(y, y_p.unpad())


class NNOpTests(TestCase):
    def setUp(self):
        super().setUp()

    def test_linear_on_3d(self):
        """
        Test that original and padded tensor produce same result on linar layer on 3D inputs.
        This tests if the view-mm-view decomposition is handled properly.
        """

        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(7, 9)

            def forward(self, x):
                return self.linear(x)

        mod = TestModule()
        mod = torch.compile(mod, fullgraph=True)

        multipliers = {0: 16, 1: 16}

        for i in range(3, 9):
            x = torch.randn((i, i, 7))
            x_p = PaddedTensor.from_tensor(x, multipliers)

            y = mod(x)
            y_p = mod(x_p)

            self.assertEqual(y_p.shape, (16, 16, 9))
            self.assertEqual(y_p.original_tensor.shape, (i, i, 9))
            self.assertEqual(y, y_p.unpad())


if __name__ == "__main__":
    run_tests()
