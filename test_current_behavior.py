#!/usr/bin/env python3
"""
Reproduction script for torch.linalg.solve_triangular device mismatch segfault

This script demonstrates the segmentation fault that occurs in CURRENT PyTorch
when calling solve_triangular with tensors on different devices.

After the fix is merged, this same script should show proper error messages
instead of segfaulting.

Issue: https://github.com/pytorch/pytorch/issues/142048
"""
import torch
import sys

def test_device_mismatch_current():
    """Demonstrate current behavior - segfault or improper error"""
    print("=" * 70)
    print("REPRODUCTION OF DEVICE MISMATCH SEGFAULT")
    print("=" * 70)
    print(f"PyTorch version: {torch.__version__}")
    
    if not torch.backends.mps.is_available():
        print("❌ MPS not available - this test requires macOS with Metal support")
        print("This test demonstrates the CPU vs MPS device mismatch issue")
        return False
    
    print("✅ MPS available")
    print("\nThis test will likely SEGFAULT with current PyTorch...")
    print("After the fix, it should show a proper RuntimeError instead.\n")
    
    # This will segfault in current PyTorch, but should show proper error after fix
    print("🧪 Creating tensors on different devices:")
    A = torch.randn(3, 3, device='cpu')
    A = torch.triu(A) + torch.eye(3)  # Make upper triangular and non-singular
    B = torch.randn(3, 2, device='mps')
    
    print(f"  A: {A.shape} on {A.device}")
    print(f"  B: {B.shape} on {B.device}")
    print("\n💥 Calling solve_triangular (this will likely segfault)...")
    
    try:
        result = torch.linalg.solve_triangular(A, B, upper=True)
        print(f"❌ UNEXPECTED: Operation succeeded: {result.shape} on {result.device}")
        print("   This suggests the fix may already be applied or there's another issue")
        return True
        
    except RuntimeError as e:
        if "Expected all tensors to be on the same device" in str(e):
            print(f"✅ FIXED BEHAVIOR: Proper device error: {e}")
            print("   The fix has been successfully applied!")
            return True
        else:
            print(f"⚠️  Different RuntimeError: {e}")
            print("   This is not the expected device mismatch error")
            return False
            
    except Exception as e:
        print(f"❌ UNEXPECTED EXCEPTION: {type(e).__name__}: {e}")
        print("   This might indicate a segfault was caught or other issue")
        return False
    
    # This script demonstrates the problem - it will likely segfault here
    # After the fix is applied, this should not happen
    return True

if __name__ == "__main__":
    print("DEVICE MISMATCH SEGFAULT REPRODUCTION")
    print("=====================================")
    print("This script reproduces the segfault bug.")
    print("After the fix is applied, you should see proper error messages.\n")
    
    success = test_device_mismatch_current()
    
    if success:
        print("\n✅ Test completed - device mismatch behavior observed")
    else:
        print("\n❌ Test failed - could not reproduce or test the issue")
    
    print("\nExpected behavior AFTER fix:")
    print("RuntimeError: linalg.solve_triangular: Expected all tensors to be on the same device, but found at least two devices, cpu and mps!")
    
    sys.exit(0 if success else 1)
