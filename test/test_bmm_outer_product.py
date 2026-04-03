# Owner(s): ["module: nn"]

import unittest

import torch
from torch._native.ops.bmm_outer_product.triton_impl import _is_outer_product
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.inductor_utils import HAS_GPU


@unittest.skipIf(not HAS_GPU, "requires GPU")
class TestBmmOuterProduct(TestCase):
    def _check_bmm(self, a, b, **kwargs):
        self.assertEqual(torch.bmm(a, b), a @ b, **kwargs)

    def test_shapes(self):
        shapes = [
            (4, 8, 16),
            (32, 8, 256),
            (16, 128, 512),
            (1, 64, 128),
            (8, 1, 1),
            (64, 256, 512),
            (256, 8, 2048),
        ]
        for B, M, N in shapes:
            with self.subTest(B=B, M=M, N=N):
                a = torch.randn(B, M, 1, device="cuda")
                b = torch.randn(B, 1, N, device="cuda")
                self._check_bmm(a, b)

    def test_basic_dtypes(self):
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                a = torch.randn(4, 8, 1, device="cuda", dtype=dtype)
                b = torch.randn(4, 1, 16, device="cuda", dtype=dtype)
                self.assertEqual(torch.bmm(a, b), a @ b)

    def test_large_shapes(self):
        a = torch.randn(512, 256, 1, device="cuda")
        b = torch.randn(512, 1, 4096, device="cuda")
        self.assertEqual(torch.bmm(a, b), a @ b)

    def test_permuted_inputs(self):
        B, M, N = 4, 8, 16
        cases = [
            (
                torch.randn(M, B, 1, device="cuda").permute(1, 0, 2),
                torch.randn(B, 1, N, device="cuda"),
            ),
            (
                torch.randn(B, M, 1, device="cuda"),
                torch.randn(N, B, 1, device="cuda").permute(1, 2, 0),
            ),
            (
                torch.randn(M, B, 1, device="cuda").permute(1, 0, 2),
                torch.randn(N, B, 1, device="cuda").permute(1, 2, 0),
            ),
        ]
        for a, b in cases:
            self.assertEqual(torch.bmm(a, b), a @ b)

    def test_fallback_non_outer_product(self):
        a = torch.randn(4, 8, 16, device="cuda")
        b = torch.randn(4, 16, 32, device="cuda")
        self.assertEqual(torch.bmm(a, b), a @ b, atol=1e-5, rtol=1.3e-6)

    def test_batch_one(self):
        a = torch.randn(1, 64, 1, device="cuda")
        b = torch.randn(1, 1, 128, device="cuda")
        self.assertEqual(torch.bmm(a, b), a @ b)

    def test_m_one_n_one(self):
        a = torch.randn(8, 1, 1, device="cuda")
        b = torch.randn(8, 1, 1, device="cuda")
        self.assertEqual(torch.bmm(a, b), a @ b)

    def test_gradient_flow(self):
        a = torch.randn(4, 8, 1, device="cuda", requires_grad=True)
        b = torch.randn(4, 1, 16, device="cuda", requires_grad=True)
        result = torch.bmm(a, b)
        result.sum().backward()
        self.assertIsNotNone(a.grad)
        self.assertIsNotNone(b.grad)
        self.assertEqual(a.grad.shape, a.shape)
        self.assertEqual(b.grad.shape, b.shape)


class TestOuterProductDetection(TestCase):
    def test_is_outer_product(self):
        self.assertTrue(_is_outer_product(torch.empty(4, 8, 1), torch.empty(4, 1, 16)))
        self.assertTrue(_is_outer_product(torch.empty(4, 8, 1), torch.empty(4, 1, 1)))
        self.assertFalse(
            _is_outer_product(torch.empty(4, 8, 16), torch.empty(4, 16, 32))
        )
        self.assertFalse(_is_outer_product(torch.empty(8, 1), torch.empty(1, 16)))
        self.assertFalse(_is_outer_product(torch.empty(4, 8, 1), torch.empty(4, 2, 16)))
        self.assertFalse(_is_outer_product(torch.empty(4, 8, 3), torch.empty(4, 1, 16)))
        self.assertFalse(
            _is_outer_product(
                torch.empty(4, 8, 1, dtype=torch.complex64),
                torch.empty(4, 1, 16, dtype=torch.complex64),
            )
        )


if __name__ == "__main__":
    run_tests()
