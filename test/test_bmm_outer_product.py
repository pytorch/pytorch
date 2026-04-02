# Owner(s): ["module: nn"]

import unittest
from unittest import mock

import torch
from torch._native.ops.bmm_outer_product.triton_impl import (
    _get_bmm_outer_product_backend,
    _is_outer_product,
)
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.inductor_utils import HAS_GPU


@unittest.skipIf(not HAS_GPU, "requires GPU")
class TestBmmOuterProduct(TestCase):
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

    def test_cutedsl_backend(self):
        with mock.patch.dict(
            "os.environ",
            {"TORCH_BMM_OUTER_PRODUCT_BACKEND": "cutedsl"},
            clear=False,
        ):
            for batch, m_dim, n_dim in ((4, 128, 64), (32, 8, 256), (7, 10, 70)):
                a = torch.randn(batch, m_dim, 1, device="cuda", dtype=torch.float16)
                b = torch.randn(batch, 1, n_dim, device="cuda", dtype=torch.float16)
                with self.subTest(batch=batch, m_dim=m_dim, n_dim=n_dim):
                    self.assertEqual(torch.bmm(a, b), a @ b)


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

    def test_backend_selection(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(_get_bmm_outer_product_backend(), "triton")
        with mock.patch.dict(
            "os.environ",
            {"TORCH_BMM_OUTER_PRODUCT_BACKEND": "cutedsl"},
            clear=True,
        ):
            self.assertEqual(_get_bmm_outer_product_backend(), "cutedsl")
        with mock.patch.dict(
            "os.environ",
            {"TORCH_BMM_OUTER_PRODUCT_BACKEND": "unknown"},
            clear=True,
        ):
            self.assertEqual(_get_bmm_outer_product_backend(), "triton")


if __name__ == "__main__":
    run_tests()
