"""
Test suite for fallback-aware device checking
"""

import torch
import unittest
import os
from torch.testing._internal.common_utils import TestCase, run_tests

class TestFallbackDeviceChecks(TestCase):
    
    def setUp(self):
        # Ensure MPS fallback is disabled by default for testing
        if "PYTORCH_ENABLE_MPS_FALLBACK" in os.environ:
            del os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]
    
    def test_strict_device_check_default_behavior(self):
        """Test that strict device checks work by default"""
        if not torch.backends.mps.is_available():
            self.skipTest("MPS not available")
        
        A_cpu = torch.randn(3, 3).triu()
        B_mps = torch.randn(3, 2, device='mps')
        
        # Should fail with clear error message
        with self.assertRaisesRegex(RuntimeError, 
                                  r"linalg_solve_triangular.*same device.*cpu.*mps"):
            torch.linalg.solve_triangular(A_cpu, B_mps)
    
    def test_fallback_enabled_allows_mixing(self):
        """Test that enabling fallback allows CPU/MPS mixing for compatible ops"""
        if not torch.backends.mps.is_available():
            self.skipTest("MPS not available")
        
        # Enable fallback
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
        try:
            # Operations that support fallback should work
            a = torch.tensor([6], device='mps')
            b = torch.tensor([9], device='mps')
            result = torch.lcm(a, b)  # May fallback to CPU internally
            self.assertEqual(result.item(), 18)
            
        finally:
            # Cleanup
            del os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]
    
    def test_cross_device_copy_always_works(self):
        """Test that cross-device copy operations work regardless of checks"""
        if not torch.backends.mps.is_available():
            self.skipTest("MPS not available")
        
        cpu_tensor = torch.randn(3, 4)
        mps_tensor = torch.empty(3, 4, device='mps')
        
        # Should work even without fallback enabled
        mps_tensor.copy_(cpu_tensor)
        
        # Verify copy worked
        torch.testing.assert_close(mps_tensor.cpu(), cpu_tensor)
    
    def test_embedding_backward_compatibility(self):
        """Test that embedding operations work with proper device placement"""
        if not torch.backends.mps.is_available():
            self.skipTest("MPS not available")
        
        embedding = torch.nn.Embedding(10, 3).to('mps')
        indices = torch.tensor([0, 1, 2], device='mps', dtype=torch.long)
        
        # Should work when all tensors are on same device
        output = embedding(indices)
        output.sum().backward()
        
        self.assertIsNotNone(embedding.weight.grad)
        self.assertEqual(embedding.weight.grad.device.type, 'mps')
    
    def test_scalar_tensor_device_checks(self):
        """Test device checking behavior with scalar tensors"""
        if not torch.backends.mps.is_available():
            self.skipTest("MPS not available")
        
        # Scalar tensors should follow same device rules
        scalar_cpu = torch.tensor(1.0)
        tensor_mps = torch.randn(3, 3, device='mps')
        
        with self.assertRaisesRegex(RuntimeError, "same device"):
            torch.add(scalar_cpu, tensor_mps)
    
    def test_error_message_quality(self):
        """Test that error messages are helpful and mention fallback"""
        if not torch.backends.mps.is_available():
            self.skipTest("MPS not available")
        
        A_cpu = torch.randn(3, 3).triu()
        B_mps = torch.randn(3, 2, device='mps')
        
        try:
            torch.linalg.solve_triangular(A_cpu, B_mps)
            self.fail("Should have raised error")
        except RuntimeError as e:
            error_msg = str(e)
            # Check that error mentions fallback option
            self.assertIn("PYTORCH_ENABLE_MPS_FALLBACK", error_msg)
            self.assertIn(".to()", error_msg)
            self.assertIn("linalg_solve_triangular", error_msg)
    
    def test_same_device_operations_still_work(self):
        """Test that same-device operations continue working"""
        devices = ['cpu']
        if torch.cuda.is_available():
            devices.append('cuda')
        if torch.backends.mps.is_available():
            devices.append('mps')
        
        for device in devices:
            with self.subTest(device=device):
                A = torch.randn(4, 4, device=device).triu()
                A += torch.eye(4, device=device) * 0.1
                B = torch.randn(4, 3, device=device)
                
                # Should work without any issues
                result = torch.linalg.solve_triangular(A, B, upper=True)
                self.assertEqual(result.device.type, device)
                
                # Verify mathematical correctness
                torch.testing.assert_close(A @ result, B, rtol=1e-4, atol=1e-5)

if __name__ == '__main__':
    run_tests()
