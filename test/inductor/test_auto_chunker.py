import torch
from torch._inductor import config
from torch._inductor.test_case import TestCase
from torch._dynamo.utils import same
import os

USE_LARGE_INPUT = os.environ.get("USE_LARGE_INPUT") == "1"

@config.patch("AutoChunker.enable", True)
class AutoChunkerTest(TestCase):
    def test_matmul(self):
        M, K, N = 1024, 16, 1024

        if USE_LARGE_INPUT:
            M = 1024 * 32
            K = 256
            N = 1024 * 32

        dtype = torch.float32
        _input = torch.randn(
            M, K, dtype=dtype, requires_grad=True, device="cuda"
        )
        weight = torch.randn(
            K, N, dtype=dtype, requires_grad=True, device="cuda"
        )

        def f(_input, weight):
            out = (_input * 2) @ weight
            _sum = out.sum()
            _sum.backward()
            return _sum

        expect = (f(_input, weight), _input.grad, weight.grad)

        _input.grad = None
        weight.grad =None

        torch.cuda.reset_peak_memory_stats()
        opt_f = torch.compile(f)
        actual = (opt_f(_input, weight), _input.grad, weight.grad)

        print(f"Peak memory {torch.cuda.max_memory_allocated() / 10 ** 9 :.6f} GB")

        self.assertTrue(same(expect, actual, tol=1e-3), f"{expect=}\n{actual=}")



if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
