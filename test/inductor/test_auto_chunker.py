import torch
from torch._inductor import config
from torch._inductor.test_case import TestCase


@config.patch("AutoChunker.enable", True)
class AutoChunkerTest(TestCase):
    def test_matmul(self):
        M = 1024
        K = 16
        N = 1024

        _input = torch.randn(
            M, K, dtype=torch.bfloat16, requires_grad=True, device="cuda"
        )
        weight = torch.randn(
            K, N, dtype=torch.bfloat16, requires_grad=True, device="cuda"
        )

        def f(_input, weight):
            out = (_input * 2) @ weight
            out.sum().backward()

        opt_f = torch.compile(f)
        opt_f(_input, weight)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
