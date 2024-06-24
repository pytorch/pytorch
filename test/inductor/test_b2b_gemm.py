import torch

from torch._inductor.test_case import run_tests, TestCase

class B2BGEMMTest(TestCase):

    def test_b2b_gemm_1(self):
        """ sizes match the supported (hardcoded) value """
        f = lambda m1, m2, m3: torch.matmul(torch.matmul(m1, m2), m3)
        f_opt = torch.compile(f)
        A = torch.randn((512, 32), device="cuda", dtype=torch.float16)
        B = torch.randn((32, 256), device="cuda", dtype=torch.float16)
        C = torch.randn((256, 32), device="cuda", dtype=torch.float16)
        self.assertEqual(torch.allclose(f(A, B, C), f_opt(A, B, C), atol=0.5, rtol=0.01), True)

    def test_b2b_gemm_2(self):
        """ sizes don't match the supported (hardcoded) value """
        f = lambda m1, m2, m3: torch.matmul(torch.matmul(m1, m2), m3)
        f_opt = torch.compile(f)
        A = torch.randn((512, 64), device="cuda", dtype=torch.float16)
        B = torch.randn((64, 256), device="cuda", dtype=torch.float16)
        C = torch.randn((256, 64), device="cuda", dtype=torch.float16)
        self.assertEqual(torch.allclose(f(A, B, C), f_opt(A, B, C), atol=0.5, rtol=0.01), True)

if __name__ == "__main__":
    run_tests()
