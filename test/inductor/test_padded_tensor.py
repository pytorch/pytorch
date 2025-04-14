import torch

from torch._inductor.experimental.padded_tensor import PaddedTensor
from torch._inductor.test_case import run_tests, TestCase
from transformer_model import *

from torch.utils import _pytree as pytree


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

        torch._dynamo.config.error_on_recompile = False

    def test_bucketing(self):
        """
        Test that we compile a new graph on a shape that is larger than the original bucket.
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

    def test_sum(self):
        def f(a):
            return a.sum()

        f = torch.compile(f, fullgraph=True)
        multipliers = {1: 16}

        a = torch.randn([2, 121])
        a_p = PaddedTensor.from_tensor(a, multipliers)

        y = f(a_p)

        self.assertEqual(y.shape, ())
        self.assertEqual(y.original_tensor.shape, ())


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


class ModelTests(TestCase):
    def setUp(self):
        super().setUp()

    def test_transformer_equiv(self):
        with torch.device("cuda"), torch.no_grad():
            bsz, seqlen_max = 2, 32
            seqlen_multiple = 16

            # Set up transformer
            args = ModelArgs.from_name("mini")
            transformer = Transformer(args)
            transformer.setup_caches(bsz, seqlen_max)

            transformer = torch.compile(
                transformer, fullgraph=True, mode="reduce-overhead"
            )

            for seqlen in range(3, 15):
                print("seqlen =", seqlen)
                # Set error_on_recompile to True after 3rd iteration.
                if seqlen == 5:
                    torch._dynamo.config.error_on_recompile = True

                # Run unpadded
                inputs = (
                    torch.randint(0, 3, (bsz, seqlen)),
                    torch.arange(0, seqlen, dtype=torch.int32),
                )

                torch.compiler.cudagraph_mark_step_begin()

                out = transformer(*inputs)
                out = out.clone()

                # Run padded
                inputs_p = [
                    PaddedTensor.from_tensor(
                        inputs[0], multipliers={0: 1, 1: seqlen_multiple}
                    ),
                    PaddedTensor.from_tensor(
                        inputs[1], multipliers={0: seqlen_multiple}, neutral_element=-1
                    ),
                ]

                torch.compiler.cudagraph_mark_step_begin()

                out_p = transformer(*inputs_p)
                out_p = out_p.clone()

                # Check
                self.assertEqual(out, out_p.unpad())

        torch._dynamo.config.error_on_recompile = False

    def test_transformer_bucketing(self):
        with torch.device("cuda"), torch.no_grad():
            bsz, seqlen_max = 2, 64
            seqlen_multiple = 16

            # Set up transformer
            args = ModelArgs.from_name("mini")
            transformer = Transformer(args)
            transformer.setup_caches(bsz, seqlen_max)

            transformer = torch.compile(
                transformer, fullgraph=True, mode="reduce-overhead"
            )

            for seqlen in range(3, 60):
                print("seqlen =", seqlen)

                # Set error_on_recompile to True after 3rd or 5th iteration for each bucket.
                if seqlen % 16 > 5:
                    torch._dynamo.config.error_on_recompile = True
                else:
                    torch._dynamo.config.error_on_recompile = False

                # Run unpadded
                inputs = (
                    torch.randint(0, 3, (bsz, seqlen)),
                    torch.arange(0, seqlen, dtype=torch.int32),
                )

                # Run padded
                inputs_p = [
                    PaddedTensor.from_tensor(
                        inputs[0], multipliers={0: 1, 1: seqlen_multiple}
                    ),
                    PaddedTensor.from_tensor(
                        inputs[1], multipliers={0: seqlen_multiple}, neutral_element=-1
                    ),
                ]

                torch.compiler.cudagraph_mark_step_begin()

                out_p = transformer(*inputs_p)
                out_p = out_p.clone()

        torch._dynamo.config.error_on_recompile = False


if __name__ == "__main__":
    run_tests()
