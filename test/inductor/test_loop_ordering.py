# Owner(s): ["module: inductor"]
import torch
from torch._inductor import config as inductor_config, metrics
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.test_operators import realize
from torch.testing._internal.inductor_utils import HAS_CUDA


@inductor_config.patch(
    {
        "benchmark_kernel": True,
        "triton.unique_kernel_names": True,
    }
)
class LoopOrderingTest(TestCase):
    def test_apbt_realize(self):
        M = 1024
        N = 2048

        def f(x, y):
            """
            There will be 2 kernels being generated without loop ordering after fusion:
              https://gist.github.com/shunting314/44df83f71de2c110232c50ac6638ed69
            """
            x = realize(x * 2)
            y = realize(y * 3)
            return x + y

        x = torch.randn(M, N)
        y = torch.randn(N, M).t()

        expect = f(x, y)
        actual = torch.compile(f)(x, y)
        self.assertTrue(torch.allclose(expect, actual, atol=1e-3, rtol=1e-3))
        self.assertEqual(1, metrics.generated_kernel_count)


if __name__ == "__main__":
    if HAS_CUDA:
        torch.set_default_device("cuda")
        run_tests()
