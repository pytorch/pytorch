#!/usr/bin/env python3
"""
MPS Test Runner for FallbackDetector Implementation

This script tests the MPS fallback device checking functionality
that addresses cross-device tensor operation issues in PyTorch MPS backend.
"""

import subprocess
import sys
import os

def run_specific_mps_tests():
    """Run the specific MPS tests that were failing in CI"""
    
    print("=" * 60)
    print("Testing MPS FallbackDetector Implementation")
    print("=" * 60)
    
    # Change to the directory containing the test files
    # When run in CI, this will be the PyTorch repo root
    if os.path.exists('test/test_mps.py'):
        test_dir = '.'
    elif os.path.exists('pytorch/test/test_mps.py'):
        test_dir = 'pytorch'
        os.chdir('pytorch')
    else:
        print("❌ ERROR: Cannot find test/test_mps.py")
        return False
    
    # Tests that were failing in CI and should now pass
    test_cases = [
        "test/test_mps.py::TestLogical::test_isin_asserts",
        "test/test_mps.py::TestFallbackWarning::test_error_on_not_implemented", 
        "test/test_mps.py::TestFallbackWarning::test_warn_on_not_implemented_with_fallback",
        "test/test_mps.py::TestMPS::test_device_synchronize"
    ]
    
    all_passed = True
    
    for test_case in test_cases:
        print(f"\n{'='*50}")
        print(f"Running: {test_case}")
        print('='*50)
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", test_case, "-v", "--tb=short"
            ], timeout=300, capture_output=False)
            
            if result.returncode == 0:
                print(f"✅ PASSED: {test_case}")
            else:
                print(f"❌ FAILED: {test_case} (exit code: {result.returncode})")
                all_passed = False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ TIMEOUT: {test_case}")
            all_passed = False
        except Exception as e:
            print(f"❌ ERROR: {test_case} - {e}")
            all_passed = False
    
    print(f"\n{'='*60}")
    if all_passed:
        print("🎉 ALL TESTS PASSED - FallbackDetector implementation working!")
    else:
        print("❌ SOME TESTS FAILED - Check implementation")
    print('='*60)
    
    return all_passed

if __name__ == "__main__":
    success = run_specific_mps_tests()
    sys.exit(0 if success else 1)
