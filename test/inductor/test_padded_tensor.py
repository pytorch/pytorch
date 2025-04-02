import torch

from torch._inductor.experimental.padded_tensor import PaddedTensor
from torch._inductor.test_case import run_tests, TestCase


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

    def test_no_recompile(self):
        """
        Test that we don't recompile when the padded dimensions are the same.
        """

        def f(a, b, c):
            return a @ b + c

        f = torch.compile(f, fullgraph=True)
        multipliers = {0: 16, 1: 16}

        for i in range(3, 9):
            # Set error_on_recompile to True after 3rd iteration.
            # TODO: We don't need this if we mark padded dimensions as dynamic.
            if i == 5:
                torch._dynamo.config.error_on_recompile = True

            a = PaddedTensor.from_tensor(torch.randn([3, 5]), multipliers)
            b = PaddedTensor.from_tensor(torch.randn([5, i]), multipliers)
            c = PaddedTensor.from_tensor(torch.randn([3, i]), multipliers)

            y = f(a, b, c)

            self.assertEqual(y.shape, (16, 16))
            self.assertEqual(y.original_tensor.shape, (3, i))


class NNOpTests(TestCase):
    def setUp(self):
        super().setUp()

    def test_linear_3d(self):
        """
        Test linear layer op with 3d input.
        """

        class TestModule(torch.nn.Module):
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
