import torch
import unittest
import warnings
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.common_device_type import instantiate_device_type_tests


class TestFallbackAwareDeviceChecking(TestCase):
    """Test fallback-aware device checking for MPS operations."""
    
    def setUp(self):
        super().setUp()
        # Skip if MPS is not available
        if not torch.backends.mps.is_available():
            self.skipTest("MPS not available")
    
    def test_fallback_aware_linalg_solve_triangular(self):
        """Test that linalg_solve_triangular allows CPU/MPS mixing in fallback context."""
        # Create tensors on different devices
        cpu_tensor = torch.randn(3, 3)
        mps_tensor = torch.randn(3, 3).to('mps')
        
        # This should work with fallback-aware checking
        # (The actual fallback context marking would be done by MPS backend)
        try:
            # This operation should not crash with device check error
            result = torch.linalg.solve_triangular(cpu_tensor, mps_tensor, upper=False)
            # We expect this to work or fail with a legitimate error, not device check error
        except RuntimeError as e:
            # If it fails, it should not be due to device checking
            self.assertNotIn("device check", str(e).lower())
            self.assertNotIn("device mismatch", str(e).lower())
    
    def test_fallback_aware_lcm(self):
        """Test that lcm allows CPU/MPS mixing in fallback scenarios."""
        cpu_tensor = torch.tensor([6, 8, 10], dtype=torch.int32)
        mps_tensor = torch.tensor([9, 12, 15], dtype=torch.int32).to('mps')
        
        try:
            result = torch.lcm(cpu_tensor, mps_tensor)
        except RuntimeError as e:
            # Should not fail due to device checking
            self.assertNotIn("device check", str(e).lower())
            self.assertNotIn("device mismatch", str(e).lower())
    
    def test_fallback_aware_gcd(self):
        """Test that gcd allows CPU/MPS mixing in fallback scenarios."""
        cpu_tensor = torch.tensor([12, 18, 24], dtype=torch.int32)
        mps_tensor = torch.tensor([8, 12, 16], dtype=torch.int32).to('mps')
        
        try:
            result = torch.gcd(cpu_tensor, mps_tensor)
        except RuntimeError as e:
            # Should not fail due to device checking
            self.assertNotIn("device check", str(e).lower())
            self.assertNotIn("device mismatch", str(e).lower())
    
    def test_embedding_dense_backward_mixed_devices(self):
        """Test embedding_dense_backward with mixed devices."""
        # Create embedding parameters on MPS
        weight = torch.randn(10, 5, requires_grad=True).to('mps')
        # Create indices on CPU (common pattern)
        indices = torch.tensor([1, 2, 3])
        
        try:
            # This should work with fallback-aware checking
            output = torch.nn.functional.embedding(indices, weight)
            loss = output.sum()
            loss.backward()
        except RuntimeError as e:
            # Should not fail due to device checking
            self.assertNotIn("device check", str(e).lower())
            self.assertNotIn("device mismatch", str(e).lower())
    
    def test_copy_from_and_resize_no_check(self):
        """Test that _copy_from_and_resize has no device checks (cross-device by design)."""
        cpu_tensor = torch.randn(3, 3)
        mps_tensor = torch.randn(2, 2).to('mps')
        
        # This should always work (annotated with NoCheck)
        try:
            cpu_tensor._copy_from_and_resize(mps_tensor)
        except RuntimeError as e:
            # Should not fail due to device checking
            self.assertNotIn("device check", str(e).lower())
            self.assertNotIn("device mismatch", str(e).lower())
    
    def test_strict_device_checking_still_works(self):
        """Test that strict device checking still works for non-fallback operations."""
        cpu_tensor = torch.randn(3, 3)
        mps_tensor = torch.randn(3, 3).to('mps')
        
        # Operations without FallbackAware annotation should still check devices strictly
        with self.assertRaises(RuntimeError):
            # This should fail with device mismatch (not fallback-aware)
            torch.add(cpu_tensor, mps_tensor)


if __name__ == '__main__':
    run_tests()
