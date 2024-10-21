import torch
from torch._inductor import config
from torch._inductor.test_case import TestCase
from torch._dynamo.utils import same
import os

# TODO: always test with large input. Skip the test if the GPU
# does not have enough memory
USE_LARGE_INPUT = os.environ.get("USE_LARGE_INPUT", "1") == "1"

@config.patch("AutoChunker.enable", True)
class AutoChunkerTest(TestCase):
    def common_matmul_test(self, has_softmax):
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
            if has_softmax:
                out = out.softmax(dim=-1)
            _sum = out.sum()
            _sum.backward()
            return _sum

        expect = (f(_input, weight), _input.grad, weight.grad)

        _input.grad = None
        weight.grad =None

        torch.cuda.reset_peak_memory_stats()
        opt_f = torch.compile(f)
        actual = (opt_f(_input, weight), _input.grad, weight.grad)
        peak_memory = torch.cuda.max_memory_allocated()

        print(f"Peak memory {peak_memory / 10 ** 9 :.6f} GB")

        self.assertTrue(same(expect, actual, tol=1e-3), f"{expect=}\n{actual=}")

        # Only assert peak memory saving for large input. For small input, the saving can
        # be largely distorted by other memory allocation such as the tensor used to clear L2
        # cache by triton perf benchmarking API.
        if USE_LARGE_INPUT:
            expected_bound = M * N * dtype.itemsize
            self.assertTrue(peak_memory < expected_bound, f"Actual peak_memory {peak_memory}, expected bound {expected_bound}")

    def test_matmul_trivial(self):
        self.common_matmul_test(has_softmax=False)

    def test_matmul_softmax(self):
        self.common_matmul_test(has_softmax=True)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
