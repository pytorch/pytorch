# Owner(s): ["module: inductor"]
import torch
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.inductor_utils import HAS_CUDA


class B2BGEMMTest(TestCase):
    @torch._inductor.config.patch(b2b_gemm_pass=True)
    def test_b2b_gemm_supported_size(self):
        """sizes match the supported (hardcoded) value"""

        def f(m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
            return torch.matmul(torch.matmul(m1, m2), m3)

        f_opt = torch.compile(f)
        A = torch.randn((512, 32), device="cuda", dtype=torch.float16)
        B = torch.randn((32, 256), device="cuda", dtype=torch.float16)
        C = torch.randn((256, 32), device="cuda", dtype=torch.float16)
        res, (code,) = run_and_get_code(f_opt, A, B, C)
        self.assertEqual(torch.allclose(f(A, B, C), res, atol=0.5, rtol=0.01), True)
        self.assertEqual("B2B_GEMM_TRITON_ENTRANCE" in code, True)

    @torch._inductor.config.patch(b2b_gemm_pass=True)
    def test_b2b_gemm_unsupported_size(self):
        """sizes don't match the supported (hardcoded) value"""

        def f(m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
            return torch.matmul(torch.matmul(m1, m2), m3)

        f_opt = torch.compile(f)
        A = torch.randn((512, 64), device="cuda", dtype=torch.float16)
        B = torch.randn((64, 256), device="cuda", dtype=torch.float16)
        C = torch.randn((256, 64), device="cuda", dtype=torch.float16)
        res, (code,) = run_and_get_code(f_opt, A, B, C)
        self.assertEqual(torch.allclose(f(A, B, C), res, atol=0.5, rtol=0.01), True)
        self.assertEqual("B2B_GEMM_TRITON_ENTRANCE" in code, False)


if __name__ == "__main__":
    if HAS_CUDA:
        run_tests()
