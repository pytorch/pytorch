import torch
import torch.nn as nn
import torch.utils._pytree as pytree
from torch._inductor.runtime.benchmarking import benchmarker
from torch._inductor.test_case import run_tests, TestCase
from torch.ops import aten

from padded_tensor import *
from utils import *

from transformer_model import *


class PaddedTensorTestCase(TestCase):
    def setUp(self):
        super().setUp()

    def are_equal(self, outputs, outputs_p):
        outputs = [outputs] if not isinstance(outputs, list) else outputs
        outputs_p = [outputs_p] if not isinstance(outputs_p, list) else outputs_p

        for x, x_p in zip(outputs, outputs_p):
            x_p = x_p.tensor if isinstance(x_p, PaddedTensor) else x_p

            slice_idx = [slice(0, s) for s in x.shape]
            self.assertEqual(x, x_p[tuple(slice_idx)])

    def assert_padded_dims(self, z: PaddedTensor, padded_dim_idxs: List[int]):
        padded_dim_idxs_set = set(padded_dim_idxs)

        for dim_idx in range(len(z.shape)):
            if dim_idx in padded_dim_idxs_set:
                self.assertTrue(z.orig_shape[dim_idx].is_padded)
            else:
                self.assertFalse(z.orig_shape[dim_idx].is_padded)


class NNOpTests(PaddedTensorTestCase):
    def setUp(self):
        super().setUp()

    def test_linear_2d(self):
        for bias in [True, False]:
            lin_layer = nn.Linear(3, 3, bias=bias)

            x = torch.randn(5, 3)
            x_p = PaddedTensor(x, [6, 1])

            z = lin_layer(x_p)

            self.assertEqual(z.shape, torch.Size([6, 3]))
            self.assertEqual(z.unpad().shape, torch.Size([5, 3]))

    def test_linear_3d(self):
        for bias in [True, False]:
            lin_layer = nn.Linear(3, 3, bias=bias)

            x = torch.randn(7, 5, 3)
            x_p = PaddedTensor(x, [8, 1, 1])

            z = lin_layer(x_p)

            self.assertEqual(z.shape, torch.Size([8, 5, 3]))
            self.assertEqual(z.unpad().shape, torch.Size([7, 5, 3]))


class AtenOpTests(PaddedTensorTestCase):
    def setUp(self):
        super().setUp()

    def test_elementwise_unary(self):
        for op in [aten.tril, aten.sin, aten.rsqrt, aten.silu]:
            a = PaddedTensor(torch.randn(3, 3), [4, 4])
            z = op(a)

            self.assertEqual(z.shape, torch.Size([4, 4]))
            self.assertEqual(z.unpad().shape, torch.Size([3, 3]))
            self.assert_padded_dims(z, [0, 1])

    def test_elementwise_binary(self):
        for op in [aten.add, aten.sub, aten.mul, aten.div]:
            a = PaddedTensor(torch.randn(3, 5), [4, 6])
            b = PaddedTensor(torch.randn(3, 5), [4, 6])
            z = op(a, b)

            self.assertEqual(z.shape, torch.Size([4, 6]))
            self.assertEqual(z.unpad().shape, torch.Size([3, 5]))
            self.assert_padded_dims(z, [0, 1])

    def test_squeeze(self):
        a = PaddedTensor(torch.randn(3, 1, 5), [4, 1, 6])
        z = aten.squeeze(a, 1)

        self.assertEqual(z.shape, torch.Size([4, 6]))
        self.assertEqual(z.unpad().shape, torch.Size([3, 5]))

    def test_view_collapse(self):
        # Collapse start
        x = PaddedTensor(torch.randn(3, 5, 7), [4, 6, 1])
        z = aten.view(x, [24, 7])

        self.assertEqual(z.unpad().shape, torch.Size([15, 7]))
        self.assert_padded_dims(z, [0])

        # Collapse end
        x = PaddedTensor(torch.randn(3, 5, 7), [4, 6, 1])
        z = aten.view(x, [4, 42])

        self.assertEqual(z.unpad().shape, torch.Size([3, 35]))
        self.assert_padded_dims(z, [0, 1])

        # Collapse middle
        x = PaddedTensor(torch.randn(3, 5, 7, 9), [4, 6, 1, 1])
        z = aten.view(x, [4, 42, 9])

        self.assertEqual(z.unpad().shape, torch.Size([3, 35, 9]))
        self.assert_padded_dims(z, [0, 1])

        # Collapse multiple
        x = PaddedTensor(torch.randn(3, 5, 7, 9, 11), [4, 6, 1, 1, 1])
        z = aten.view(x, [24, 7, 99])

        self.assertEqual(z.unpad().shape, torch.Size([15, 7, 99]))
        self.assert_padded_dims(z, [0])

    def test_view_expand(self):
        # Expand start
        x = PaddedTensor(torch.randn(3, 5, 7), [1, 6, 1])
        z = aten.view(x, [18, 7])

        self.assertEqual(z.unpad().shape, torch.Size([15, 7]))
        self.assert_padded_dims(z, [0])

        # Expand end
        x = PaddedTensor(torch.randn(3, 5, 7), [1, 6, 1])
        z = aten.view(x, [3, 42])
        self.assertEqual(z.unpad().shape, torch.Size([3, 35]))
        self.assert_padded_dims(z, [1])

        # Test that unpad throws an exception, when we can't infer the dim.
        x = PaddedTensor(torch.randn(15, 7), [4, 1, 1])
        z = aten.view(x, [4, 4, 7])
        with self.assertRaisesRegex(
            Exception,
            "PaddedTensor couldn't figure out a shape, likely due to an expansion.",
        ):
            z.unpad()


class ModelTests(PaddedTensorTestCase):
    def setUp(self):
        super().setUp()

    def test_transformer_model(self):
        with torch.no_grad():
            with torch.device("cuda"):
                pad = 4
                bsz, seqlen = 4, 2 + pad

                # Set up transformer
                args = ModelArgs.from_name("stories15M")
                transformer = Transformer(args)
                transformer.setup_caches(bsz, seqlen)

                # Set up inputs
                inputs = (
                    torch.randint(0, 3, (bsz, seqlen - pad)).to(device="cuda"),
                    torch.randint(0, 3, (seqlen - pad,)).to(device="cuda"),
                )

                inputs_p = [
                    PaddedTensor(inputs[0], [bsz, seqlen], None),
                    PaddedTensor(inputs[1], [seqlen], None, -1),
                ]

                # Run
                out = transformer(*inputs)

                transformer = torch.compile(transformer)  # , mode="reduce-overhead")
                out_p = transformer(*inputs_p)
                out_p = pytree.tree_map(lambda x: x.unpad(), out_p)

                # Check
                self.are_equal(out, out_p)

                is_out_equal = pytree.tree_map(
                    lambda o, p: torch.allclose(o, p, atol=1e-5), out, out_p
                )
                self.assertTrue(is_out_equal)


class FunctionalTests(PaddedTensorTestCase):
    def setUp(self):
        super().setUp()

    def test_mem_stable(self):
        def f(a, b):
            return torch.mm(a, b)

        f = torch.compile(f, mode="reduce-overhead")

        mems = []

        torch.cuda.reset_peak_memory_stats()
        for offset in range(10):
            N = 4080 + offset
            a = torch.rand(N, N, device="cuda")
            b = torch.rand(N, N, device="cuda")

            # a = PaddedTensor(a, [4096, 4096])
            # b = PaddedTensor(b, [4096, 4096])

            out = f(a, b)

            mem = torch.cuda.max_memory_allocated(0)
            mem_in_gb = mem / 1024 / 1024 / 1024
            print(mem_in_gb)

            mems.append(mem_in_gb)

        # Check that memory allocation is stable
        self.assertTrue(all(mem < mems[0] * 2.5 for mem in mems))

    def test_no_recompile(self):
        def fn(a, b):
            return torch.mm(a, b)

        torch._dynamo.config.error_on_recompile = True
        fn = torch.compile(fn, mode="reduce-overhead")

        multipliers = [16, 16]
        fn(
            PaddedTensor(torch.randn(3, 5), multipliers),
            PaddedTensor(torch.randn(5, 7), multipliers),
        )
        fn(
            PaddedTensor(torch.randn(3, 7), multipliers),
            PaddedTensor(torch.randn(7, 11), multipliers),
        )


if __name__ == "__main__":
    run_tests()
