# Owner(s): ["module: PrivateUse1"]

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestAutocast(TestCase):
    def test_autocast_with_unsupported_type(self):
        """Test autocast with unsupported dtype (float32)"""
        with self.assertWarnsRegex(
            UserWarning,
            "In openreg autocast, but the target dtype is not supported. Disabling autocast.\n"
            "openreg Autocast only supports dtypes of torch.float16, torch.bfloat16 currently.",
        ):
            with torch.autocast(device_type="openreg", dtype=torch.float32):
                _ = torch.ones(10)

    def test_autocast_operator_not_supported(self):
        """Test that binary_cross_entropy is not supported in autocast"""
        with self.assertRaisesRegex(
            RuntimeError,
            "torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast.",
        ):
            x = torch.randn(2, 3, device="openreg")
            y = torch.randn(2, 3, device="openreg")
            with torch.autocast(device_type="openreg", dtype=torch.float16):
                _ = torch.nn.functional.binary_cross_entropy(x, y)

    def test_autocast_low_precision(self):
        """Test low precision operations (mm) in autocast"""
        with torch.amp.autocast(device_type="openreg", dtype=torch.float16):
            x = torch.randn(2, 3, device="openreg")
            y = torch.randn(3, 3, device="openreg")
            result = torch.mm(x, y)
            self.assertEqual(result.dtype, torch.float16)

    def test_autocast_fp32(self):
        """Test fp32 operations (asin) in autocast"""
        with torch.amp.autocast(device_type="openreg"):
            x = torch.randn(2, device="openreg", dtype=torch.float16)
            result = torch.asin(x)
            self.assertEqual(result.dtype, torch.float32)

    def test_autocast_default_dtype(self):
        """Test default autocast dtype"""
        openreg_fast_dtype = torch.get_autocast_dtype(device_type="openreg")
        self.assertEqual(openreg_fast_dtype, torch.half)

    def test_autocast_set_dtype(self):
        """Test setting autocast dtype"""
        for dtype in [torch.float16, torch.bfloat16]:
            torch.set_autocast_dtype("openreg", dtype)
            self.assertEqual(torch.get_autocast_dtype("openreg"), dtype)

    def test_autocast_bfloat16(self):
        """Test autocast with bfloat16 dtype"""
        with torch.amp.autocast(device_type="openreg", dtype=torch.bfloat16):
            x = torch.randn(2, 3, device="openreg", dtype=torch.float32)
            y = torch.randn(3, 3, device="openreg", dtype=torch.float32)
            result = torch.mm(x, y)
            self.assertEqual(result.dtype, torch.bfloat16)

    def test_autocast_low_precision_bfloat16(self):
        """Test low precision operations with bfloat16"""
        with torch.amp.autocast(device_type="openreg", dtype=torch.bfloat16):
            x = torch.randn(2, 3, device="openreg")
            y = torch.randn(3, 3, device="openreg")
            result = torch.mm(x, y)
            self.assertEqual(result.dtype, torch.bfloat16)

    def test_autocast_fp32_with_bfloat16(self):
        """Test fp32 operations with bfloat16 autocast"""
        with torch.amp.autocast(device_type="openreg", dtype=torch.bfloat16):
            x = torch.randn(2, device="openreg", dtype=torch.bfloat16)
            result = torch.asin(x)
            self.assertEqual(result.dtype, torch.float32)

    def test_autocast_nested_context(self):
        """Test nested autocast contexts"""
        with torch.amp.autocast(device_type="openreg", dtype=torch.float16):
            x = torch.randn(2, 3, device="openreg")
            y = torch.randn(3, 3, device="openreg")
            result1 = torch.mm(x, y)
            self.assertEqual(result1.dtype, torch.float16)

            # Nested autocast context with bfloat16
            with torch.amp.autocast(device_type="openreg", dtype=torch.bfloat16):
                result2 = torch.mm(x, y)
                self.assertEqual(result2.dtype, torch.bfloat16)

            # After exiting nested context, should restore to float16
            result3 = torch.mm(x, y)
            self.assertEqual(result3.dtype, torch.float16)

    def test_autocast_fallthrough_operation(self):
        """Test fallthrough operations (operations not specially registered)"""
        with torch.amp.autocast(device_type="openreg", dtype=torch.float16):
            x = torch.randn(2, 3, device="openreg", dtype=torch.float32)
            # add operation is not specially registered, should fallthrough
            result = torch.add(x, x)
            # fallthrough operations should preserve input type or use default behavior
            self.assertEqual(result.dtype, torch.float32)

    def test_autocast_with_requires_grad(self):
        """Test autocast interaction with requires_grad"""
        with torch.amp.autocast(device_type="openreg", dtype=torch.float16):
            x = torch.randn(2, 3, device="openreg", requires_grad=True)
            y = torch.randn(3, 3, device="openreg", requires_grad=True)
            result = torch.mm(x, y)
            self.assertEqual(result.dtype, torch.float16)
            self.assertTrue(result.requires_grad)

            # Test backward propagation
            loss = result.sum()
            loss.backward()
            self.assertIsNotNone(x.grad)
            self.assertIsNotNone(y.grad)

    def test_autocast_mixed_input_dtypes(self):
        """Test combinations of different input dtypes"""
        with torch.amp.autocast(device_type="openreg", dtype=torch.float16):
            x = torch.randn(2, 3, device="openreg", dtype=torch.float32)
            y = torch.randn(3, 3, device="openreg", dtype=torch.float16)
            # mm operation should convert inputs to low precision
            result = torch.mm(x, y)
            self.assertEqual(result.dtype, torch.float16)

    def test_autocast_already_target_dtype(self):
        """Test when inputs are already in target dtype"""
        with torch.amp.autocast(device_type="openreg", dtype=torch.float16):
            x = torch.randn(2, 3, device="openreg", dtype=torch.float16)
            y = torch.randn(3, 3, device="openreg", dtype=torch.float16)
            result = torch.mm(x, y)
            self.assertEqual(result.dtype, torch.float16)

    def test_autocast_combination_operations(self):
        """Test multiple operations combination under autocast"""
        with torch.amp.autocast(device_type="openreg", dtype=torch.float16):
            x = torch.randn(2, 3, device="openreg")
            y = torch.randn(3, 3, device="openreg")
            z = torch.randn(2, device="openreg")

            # Low precision operation
            result1 = torch.mm(x, y)
            self.assertEqual(result1.dtype, torch.float16)

            # fp32 operation
            result2 = torch.asin(z)
            self.assertEqual(result2.dtype, torch.float32)

            # Combined operations
            result3 = torch.mm(result1, y)
            self.assertEqual(result3.dtype, torch.float16)

    def test_autocast_disable(self):
        """Test disabling autocast"""
        with torch.amp.autocast(
            device_type="openreg", dtype=torch.float16, enabled=False
        ):
            x = torch.randn(2, 3, device="openreg", dtype=torch.float32)
            y = torch.randn(3, 3, device="openreg", dtype=torch.float32)
            result = torch.mm(x, y)
            # When autocast is disabled, should preserve original dtype
            self.assertEqual(result.dtype, torch.float32)

    def test_autocast_cache_enabled(self):
        """Test autocast caching"""
        with torch.amp.autocast(
            device_type="openreg", dtype=torch.float16, cache_enabled=True
        ):
            x = torch.randn(2, 3, device="openreg")
            y = torch.randn(3, 3, device="openreg")
            result1 = torch.mm(x, y)
            result2 = torch.mm(x, y)
            self.assertEqual(result1.dtype, torch.float16)
            self.assertEqual(result2.dtype, torch.float16)

    def test_autocast_fp32_operation_with_float16_input(self):
        """Test fp32 operations receiving float16 input"""
        with torch.amp.autocast(device_type="openreg", dtype=torch.float16):
            x = torch.randn(2, device="openreg", dtype=torch.float16)
            result = torch.asin(x)
            # asin should output float32
            self.assertEqual(result.dtype, torch.float32)

    def test_autocast_fp32_operation_with_float32_input(self):
        """Test fp32 operations receiving float32 input"""
        with torch.amp.autocast(device_type="openreg", dtype=torch.float16):
            x = torch.randn(2, device="openreg", dtype=torch.float32)
            result = torch.asin(x)
            # asin should output float32
            self.assertEqual(result.dtype, torch.float32)


if __name__ == "__main__":
    run_tests()
