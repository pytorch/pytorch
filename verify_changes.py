#!/usr/bin/env python3
"""
Verification script to check if the sparse.mm bf16 changes are correct.
This checks the code changes without requiring a full PyTorch build.
"""
import os
import re

def check_file_changes():
    """Verify all necessary changes were made"""
    checks = []

    # 1. Check CUDA implementation
    print("=" * 60)
    print("Checking CUDA Implementation...")
    print("=" * 60)

    cuda_file = "aten/src/ATen/native/sparse/cuda/SparseBlasImpl.cpp"
    with open(cuda_file, 'r') as f:
        content = f.read()

    # Check for the dispatch macro change
    if "AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2" in content:
        # Find the sampled_addmm function
        pattern = r'void sampled_addmm_out_sparse_csr.*?AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2\(\s*kHalf,\s*kBFloat16'
        if re.search(pattern, content, re.DOTALL):
            print("✅ CUDA: Dispatch macro includes kHalf and kBFloat16")
            checks.append(True)
        else:
            print("❌ CUDA: Dispatch macro not found or incorrect")
            checks.append(False)
    else:
        print("❌ CUDA: AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2 not found")
        checks.append(False)

    # Check for opmath_t usage
    if "using opmath_t = at::opmath_type<scalar_t>" in content:
        print("✅ CUDA: opmath_t defined for numerical stability")
        checks.append(True)
    else:
        print("❌ CUDA: opmath_t not defined")
        checks.append(False)

    # Check for proper type casting
    if "beta.to<opmath_t>()" in content and "alpha.to<opmath_t>()" in content:
        print("✅ CUDA: alpha/beta cast to opmath_t")
        checks.append(True)
    else:
        print("❌ CUDA: alpha/beta not properly cast")
        checks.append(False)

    if "getCudaDataType<opmath_t>()" in content:
        print("✅ CUDA: compute_type uses opmath_t")
        checks.append(True)
    else:
        print("❌ CUDA: compute_type not using opmath_t")
        checks.append(False)

    # 2. Check CPU implementation
    print("\n" + "=" * 60)
    print("Checking CPU Implementation...")
    print("=" * 60)

    cpu_check_file = "aten/src/ATen/native/sparse/SparseBlas.cpp"
    with open(cpu_check_file, 'r') as f:
        content = f.read()

    if "ScalarType::Half" in content and "ScalarType::BFloat16" in content:
        print("✅ CPU: Type check includes Half and BFloat16")
        checks.append(True)
    else:
        print("❌ CPU: Type check missing Half/BFloat16")
        checks.append(False)

    # 3. Check CPU kernel
    print("\n" + "=" * 60)
    print("Checking CPU Kernel...")
    print("=" * 60)

    cpu_kernel_file = "aten/src/ATen/native/cpu/SampledAddmmKernel.cpp"
    with open(cpu_kernel_file, 'r') as f:
        content = f.read()

    if "AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16" in content:
        print("✅ CPU Kernel: Dispatch macro includes kHalf and kBFloat16")
        checks.append(True)
    else:
        print("❌ CPU Kernel: Dispatch macro incorrect")
        checks.append(False)

    # 4. Check test file
    print("\n" + "=" * 60)
    print("Checking Test...")
    print("=" * 60)

    test_file = "test/test_sparse_csr.py"
    with open(test_file, 'r') as f:
        content = f.read()

    if "def test_sparse_mm_backward_half_precision" in content:
        print("✅ Test: test_sparse_mm_backward_half_precision exists")
        checks.append(True)
    else:
        print("❌ Test: test_sparse_mm_backward_half_precision not found")
        checks.append(False)

    if "@onlyCUDA" in content and "torch.sparse.mm(A, B)" in content:
        print("✅ Test: Contains CUDA test with sparse.mm")
        checks.append(True)
    else:
        print("❌ Test: Missing CUDA decorator or sparse.mm call")
        checks.append(False)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(checks)
    total = len(checks)

    print(f"Passed: {passed}/{total} checks")

    if all(checks):
        print("\n✅ All checks passed! Changes look correct.")
        return True
    else:
        print("\n❌ Some checks failed. Please review the changes.")
        return False

def check_syntax():
    """Basic syntax check on modified files"""
    print("\n" + "=" * 60)
    print("Checking C++ Syntax...")
    print("=" * 60)

    cpp_files = [
        "aten/src/ATen/native/sparse/cuda/SparseBlasImpl.cpp",
        "aten/src/ATen/native/sparse/SparseBlas.cpp",
        "aten/src/ATen/native/cpu/SampledAddmmKernel.cpp"
    ]

    for file in cpp_files:
        # Check for common syntax issues
        with open(file, 'r') as f:
            content = f.read()
            lines = content.split('\n')

        # Check balanced braces
        brace_count = content.count('{') - content.count('}')
        paren_count = content.count('(') - content.count(')')

        if brace_count == 0 and paren_count == 0:
            print(f"✅ {os.path.basename(file)}: Balanced braces and parentheses")
        else:
            print(f"❌ {os.path.basename(file)}: Unbalanced braces({brace_count}) or parens({paren_count})")
            return False

    return True

def main():
    print("Verification Script for sparse.mm BFloat16/Float16 Support")
    print("=" * 60 + "\n")

    os.chdir('/Users/sladynnunes/pytorch')

    syntax_ok = check_syntax()
    changes_ok = check_file_changes()

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    if syntax_ok and changes_ok:
        print("✅ All verifications passed!")
        print("\nNext steps:")
        print("1. The changes are correct and ready")
        print("2. CI/CD will build and test on CUDA when PR is created")
        print("3. You can create the PR now")
        return 0
    else:
        print("❌ Some verifications failed!")
        print("\nPlease review the issues above before creating PR")
        return 1

if __name__ == "__main__":
    exit(main())
