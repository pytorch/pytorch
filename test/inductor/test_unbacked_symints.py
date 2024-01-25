# Owner(s): ["module: inductor"]

import torch

from torch._dynamo import config as dynamo_config
from torch._inductor import config as inductor_config
from torch.testing import make_tensor

from torch.testing._internal.common_utils import IS_LINUX, TestCase as TorchTestCase
from torch.testing._internal.inductor_utils import HAS_CUDA


class TestUnbackedSymints(TorchTestCase):
    def test_expand(self):
        def fn(x, y):
            nz = torch.nonzero(x)
            # unbacked symint in nz.size
            x_exp = nz.expand([-1, 128])
            # unbacked symint in target sizes
            y_exp = y.expand([-1, nz.size(0)])
            return x_exp, y_exp

        example_inputs = (
            torch.randn((32), device="cuda"),
            torch.randn((32, 1), device="cuda"),
        )

        with dynamo_config.patch({"capture_dynamic_output_shape_ops": True}):
            actual = torch.compile(fn, fullgraph=True)(*example_inputs)
            expected = fn(*example_inputs)

        torch.testing.assert_close(actual, expected)

    def test_expand_mismatch(self):
        def fn(x):
            nz = x.nonzero()
            return nz.expand([-1, 128])

        x = make_tensor(32, 4, device="cpu", dtype=torch.float32, exclude_zero=True)
        with dynamo_config.patch({"capture_dynamic_output_shape_ops": True}):
            with self.assertRaises(torch._dynamo.exc.TorchRuntimeError):
                actual = torch.compile(fn, fullgraph=True)(x)

    def test_autotuning(self):
        def fn(x, y):
            nz = torch.nonzero(x)
            # unbacked symint in the GEMM input shape
            a = x.new_ones([nz.size(0), y.size(0)])
            return a @ y

        example_inputs = (
            torch.randn((64), device="cuda"),
            torch.randn((32, 16), device="cuda"),
        )

        with dynamo_config.patch({"capture_dynamic_output_shape_ops": True}):
            with inductor_config.patch(
                {
                    "max_autotune_gemm": True,
                    "max_autotune_gemm_backends": "TRITON",
                }
            ):
                actual = torch.compile(fn, fullgraph=True)(*example_inputs)
                expected = fn(*example_inputs)

        torch.testing.assert_close(actual, expected)

    def test_split_with_sizes(self):
        def fn(x, y):
            l = y.tolist()
            s = torch.split(x, l)
            d = l[0] + l[1] + l[2]
            return s[0].sum(), d

        example_inputs = (torch.randn((32), device="cuda"), torch.tensor((7, 16, 9)))

        with dynamo_config.patch({"capture_scalar_outputs": True}):
            actual = torch.compile(fn, fullgraph=True)(*example_inputs)
            expected = fn(*example_inputs)

        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests
    from torch._inductor.utils import is_big_gpu

    if IS_LINUX and HAS_CUDA and is_big_gpu(0):
        run_tests()
