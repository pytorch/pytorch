# Owner(s): ["module: tests"]

"""
Test file for Issue #149002: isinf should work for Float8 types

Float8 types that cannot represent infinity (e4m3fn, e4m3fnuz, e5m2fnuz, e8m0fnu)
should return all False from isinf() instead of throwing a RuntimeError.

Only Float8_e5m2 can represent infinity.

This test verifies the fix in aten/src/ATen/native/TensorCompare.cpp
"""

import unittest
import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCPU,
    onlyCUDA,
)


class TestIsinfFloat8(TestCase):
    """Test cases for isinf with Float8 types."""

    def test_isinf_float8_e4m3fn_returns_all_false(self):
        """Float8_e4m3fn cannot represent infinity, so isinf should return all False."""
        if not hasattr(torch, 'float8_e4m3fn'):
            self.skipTest("float8_e4m3fn not available")
        
        # Create tensor with random values
        x = torch.randn(10).to(torch.float8_e4m3fn)
        result = torch.isinf(x)
        
        # All should be False since e4m3fn cannot represent infinity
        self.assertEqual(result.dtype, torch.bool)
        self.assertFalse(result.any().item())
        self.assertEqual(result.shape, x.shape)

    def test_isinf_float8_e4m3fnuz_returns_all_false(self):
        """Float8_e4m3fnuz cannot represent infinity, so isinf should return all False."""
        if not hasattr(torch, 'float8_e4m3fnuz'):
            self.skipTest("float8_e4m3fnuz not available")
        
        x = torch.randn(10).to(torch.float8_e4m3fnuz)
        result = torch.isinf(x)
        
        self.assertEqual(result.dtype, torch.bool)
        self.assertFalse(result.any().item())
        self.assertEqual(result.shape, x.shape)

    def test_isinf_float8_e5m2fnuz_returns_all_false(self):
        """Float8_e5m2fnuz cannot represent infinity, so isinf should return all False."""
        if not hasattr(torch, 'float8_e5m2fnuz'):
            self.skipTest("float8_e5m2fnuz not available")
        
        x = torch.randn(10).to(torch.float8_e5m2fnuz)
        result = torch.isinf(x)
        
        self.assertEqual(result.dtype, torch.bool)
        self.assertFalse(result.any().item())
        self.assertEqual(result.shape, x.shape)

    def test_isinf_float8_e8m0fnu_returns_all_false(self):
        """Float8_e8m0fnu cannot represent infinity, so isinf should return all False."""
        if not hasattr(torch, 'float8_e8m0fnu'):
            self.skipTest("float8_e8m0fnu not available")
        
        # e8m0fnu is exponent-only, create valid values
        x = torch.tensor([1.0, 2.0, 4.0, 8.0]).to(torch.float8_e8m0fnu)
        result = torch.isinf(x)
        
        self.assertEqual(result.dtype, torch.bool)
        self.assertFalse(result.any().item())
        self.assertEqual(result.shape, x.shape)

    def test_isinf_float8_e5m2_can_have_infinity(self):
        """Float8_e5m2 CAN represent infinity, so isinf should detect it."""
        if not hasattr(torch, 'float8_e5m2'):
            self.skipTest("float8_e5m2 not available")
        
        # Create a tensor with infinity
        x_float = torch.tensor([1.0, float('inf'), 2.0, float('-inf'), 3.0])
        x = x_float.to(torch.float8_e5m2)
        result = torch.isinf(x)
        
        self.assertEqual(result.dtype, torch.bool)
        # e5m2 should detect the infinities
        expected = torch.tensor([False, True, False, True, False])
        self.assertEqual(result, expected)

    def test_isinf_float8_empty_tensor(self):
        """isinf should work with empty Float8 tensors."""
        if not hasattr(torch, 'float8_e4m3fn'):
            self.skipTest("float8_e4m3fn not available")
        
        x = torch.empty(0).to(torch.float8_e4m3fn)
        result = torch.isinf(x)
        
        self.assertEqual(result.dtype, torch.bool)
        self.assertEqual(result.shape, torch.Size([0]))

    def test_isinf_float8_multidimensional(self):
        """isinf should preserve shape for multidimensional Float8 tensors."""
        if not hasattr(torch, 'float8_e4m3fn'):
            self.skipTest("float8_e4m3fn not available")
        
        x = torch.randn(3, 4, 5).to(torch.float8_e4m3fn)
        result = torch.isinf(x)
        
        self.assertEqual(result.dtype, torch.bool)
        self.assertEqual(result.shape, torch.Size([3, 4, 5]))
        self.assertFalse(result.any().item())

    def test_isinf_float8_non_contiguous(self):
        """isinf should work with non-contiguous Float8 tensors."""
        if not hasattr(torch, 'float8_e4m3fn'):
            self.skipTest("float8_e4m3fn not available")
        
        x = torch.randn(10, 10).to(torch.float8_e4m3fn)
        x_noncontig = x[::2, ::2]  # Non-contiguous view
        
        self.assertFalse(x_noncontig.is_contiguous())
        
        result = torch.isinf(x_noncontig)
        
        self.assertEqual(result.dtype, torch.bool)
        self.assertEqual(result.shape, x_noncontig.shape)
        self.assertFalse(result.any().item())


class TestIsnanFloat8(TestCase):
    """Test cases for isnan with Float8 types (for completeness)."""

    def test_isnan_float8_e4m3fn(self):
        """isnan should work with Float8_e4m3fn."""
        if not hasattr(torch, 'float8_e4m3fn'):
            self.skipTest("float8_e4m3fn not available")
        
        x = torch.randn(10).to(torch.float8_e4m3fn)
        # This should not raise an error
        result = torch.isnan(x)
        
        self.assertEqual(result.dtype, torch.bool)
        self.assertEqual(result.shape, x.shape)


class TestInductorNanAsserts(TestCase):
    """
    Test cases for torch.compile with nan_asserts=True and Float8 types.
    
    This is the original issue #149002: when config.nan_asserts is True,
    torch.compile generates code that calls isinf() on input tensors,
    which would fail for Float8_e4m3fn before the fix.
    """

    def test_compile_with_nan_asserts_float8_e4m3fn(self):
        """torch.compile with nan_asserts should work with Float8_e4m3fn."""
        if not hasattr(torch, 'float8_e4m3fn'):
            self.skipTest("float8_e4m3fn not available")
        
        import torch._inductor.config as config
        
        original_nan_asserts = config.nan_asserts
        try:
            config.nan_asserts = True
            
            class Model(torch.nn.Module):
                def forward(self, x):
                    return x.half() + 1
            
            model = Model()
            x = torch.randn(10).to(torch.float8_e4m3fn)
            
            # This should not raise RuntimeError anymore
            compiled_model = torch.compile(model, fullgraph=True)
            result = compiled_model(x)
            
            # Verify the result is correct
            expected = x.half() + 1
            self.assertEqual(result, expected)
            
        finally:
            config.nan_asserts = original_nan_asserts


# Instantiate for both CPU and CUDA if available
instantiate_device_type_tests(TestIsinfFloat8, globals())


if __name__ == "__main__":
    run_tests()
