# Owner(s): ["module: nn"]

import torch
from torch._native.ops.bmm_outer_product.triton_impl import (
    _bmm_outer_product_cond,
    _is_outer_product,
)
from torch.testing._internal.common_device_type import (
    deviceCountAtLeast,
    instantiate_device_type_tests,
    onlyAccelerator,
    skipXPUIf,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class TestBmmOuterProductDevice(TestCase):
    def _check_bmm(self, a, b, **kwargs):
        self.assertEqual(torch.bmm(a, b), a @ b, **kwargs)

    @onlyAccelerator
    @skipXPUIf(True, "https://github.com/pytorch/pytorch/issues/180318")
    def test_shapes(self, device):
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
                a = torch.randn(B, M, 1, device=device)
                b = torch.randn(B, 1, N, device=device)
                self._check_bmm(a, b)

    @onlyAccelerator
    def test_basic_dtypes(self, device):
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                a = torch.randn(4, 8, 1, device=device, dtype=dtype)
                b = torch.randn(4, 1, 16, device=device, dtype=dtype)
                self.assertEqual(torch.bmm(a, b), a @ b)

    @onlyAccelerator
    def test_permuted_inputs(self, device):
        B, M, N = 4, 8, 16
        cases = [
            (
                torch.randn(M, B, 1, device=device).permute(1, 0, 2),
                torch.randn(B, 1, N, device=device),
            ),
            (
                torch.randn(B, M, 1, device=device),
                torch.randn(N, B, 1, device=device).permute(1, 2, 0),
            ),
            (
                torch.randn(M, B, 1, device=device).permute(1, 0, 2),
                torch.randn(N, B, 1, device=device).permute(1, 2, 0),
            ),
        ]
        for a, b in cases:
            self.assertEqual(torch.bmm(a, b), a @ b)

    @onlyAccelerator
    def test_fallback_non_outer_product(self, device):
        a = torch.randn(4, 8, 16, device=device)
        b = torch.randn(4, 16, 32, device=device)
        self.assertEqual(torch.bmm(a, b), a @ b, atol=1e-5, rtol=1.3e-6)

    @onlyAccelerator
    def test_batch_one(self, device):
        a = torch.randn(1, 64, 1, device=device)
        b = torch.randn(1, 1, 128, device=device)
        self.assertEqual(torch.bmm(a, b), a @ b)

    @onlyAccelerator
    def test_m_one_n_one(self, device):
        a = torch.randn(8, 1, 1, device=device)
        b = torch.randn(8, 1, 1, device=device)
        self.assertEqual(torch.bmm(a, b), a @ b)

    @onlyAccelerator
    def test_gradient_flow(self, device):
        a = torch.randn(4, 8, 1, device=device, requires_grad=True)
        b = torch.randn(4, 1, 16, device=device, requires_grad=True)
        result = torch.bmm(a, b)
        result.sum().backward()
        self.assertIsNotNone(a.grad)
        self.assertIsNotNone(b.grad)
        self.assertEqual(a.grad.shape, a.shape)
        self.assertEqual(b.grad.shape, b.shape)

    @onlyAccelerator
    def test_mixed_device_outer_product_fallback(self, device):
        a = torch.randn(4, 8, 1)
        b = torch.randn(4, 1, 16, device=device)
        with self.assertRaises(RuntimeError):
            torch.bmm(a, b)

    @onlyAccelerator
    @deviceCountAtLeast(2)
    def test_non_current_device_outer_product(self, devices):
        old_device = torch.accelerator.current_device_index()
        try:
            torch.accelerator.set_device_index(devices[0])
            a = torch.randn(4, 8, 1, device=devices[1])
            b = torch.randn(4, 1, 16, device=devices[1])

            out = torch.bmm(a, b)

            self.assertEqual(torch.accelerator.current_device_index(), devices[0])
            self.assertEqual(out.device, torch.device(devices[1]))
            self.assertEqual(out, a * b)

            mismatched_a = torch.randn(4, 8, 1, device=devices[0])
            with self.assertRaisesRegex(RuntimeError, "same device|different"):
                torch.bmm(mismatched_a, b)
        finally:
            torch.accelerator.set_device_index(old_device)

    @onlyAccelerator
    def test_cow_inputs_accepted_by_override(self, device):
        # The override accepts copy-on-write inputs (it wraps its read-only
        # inputs in ConstTensorWrapper and reads through const_data_ptr()).
        # Assert the cond fires on COW inputs -- it previously excluded them --
        # so a regression back to declining COW would be caught here.
        a = torch.randn(4, 64, 1, device=device)
        b = torch.randn(4, 1, 48, device=device)
        a_cow = a._lazy_clone()
        b_cow = b._lazy_clone()
        self.assertTrue(torch._C._is_cow_tensor(a_cow))
        self.assertTrue(torch._C._is_cow_tensor(b_cow))

        self.assertTrue(_bmm_outer_product_cond(a_cow, b_cow))

    @onlyAccelerator
    def test_cow_inputs_not_materialized(self, device):
        # A copy-on-write input routed through the override is read via
        # const_data_ptr() and not materialized. Verify the result is correct
        # and the inputs stay COW across the call.
        a = torch.randn(4, 64, 1, device=device)
        b = torch.randn(4, 1, 48, device=device)
        a_cow = a._lazy_clone()
        b_cow = b._lazy_clone()
        self.assertTrue(torch._C._is_cow_tensor(a_cow))
        self.assertTrue(torch._C._is_cow_tensor(b_cow))

        out = torch.bmm(a_cow, b_cow)

        self.assertEqual(out, a @ b)
        self.assertTrue(torch._C._is_cow_tensor(a_cow))
        self.assertTrue(torch._C._is_cow_tensor(b_cow))


class TestBmmOuterProduct(TestCase):
    def test_cpu_outer_product_fallback(self):
        a = torch.randn(4, 8, 1)
        b = torch.randn(4, 1, 16)
        self.assertTrue(a.device.type == "cpu")
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


instantiate_device_type_tests(TestBmmOuterProductDevice, globals(), allow_xpu=True)

if __name__ == "__main__":
    run_tests()
