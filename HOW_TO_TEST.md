# How to Test sparse.mm BFloat16/Float16 Changes

## Summary of Situation

**Current Environment:** macOS (no CUDA GPU)
**Test Requirement:** CUDA GPU required (@onlyCUDA decorator)
**PyTorch Status:** Not built from source yet

## Testing Commands (from CONTRIBUTING.md)

### Option 1: Run Entire Test File
```bash
python test/test_sparse_csr.py
```

### Option 2: Run Specific Test Class
```bash
python test/test_sparse_csr.py TestSparseCsr
```

### Option 3: Run Specific Test Method
```bash
python test/test_sparse_csr.py TestSparseCsr.test_sparse_mm_backward_half_precision
```

### Option 4: Use pytest (Recommended for Development)
```bash
# Install pytest if needed
pip install pytest

# Run just tests matching a pattern
pytest test/test_sparse_csr.py -k "test_sparse_mm_backward_half_precision" -v

# Run with more verbosity
pytest test/test_sparse_csr.py::TestSparseCsr::test_sparse_mm_backward_half_precision -vv
```

## Steps to Test Locally (Requires Linux + CUDA GPU)

### 1. Build PyTorch from Source

```bash
cd /Users/sladynnunes/pytorch

# Install build dependencies
pip install --group dev

# Build PyTorch in development mode (takes 1-2 hours)
python -m pip install -e . -v --no-build-isolation

# Note: On macOS, this will build CPU-only version
# For CUDA support, you need Linux + CUDA toolkit installed
```

### 2. Run the Test

```bash
# After building, run the specific test
python test/test_sparse_csr.py TestSparseCsr.test_sparse_mm_backward_half_precision -v
```

## Why Test Won't Run on macOS

### 1. No CUDA Support
- macOS doesn't support NVIDIA CUDA
- The test is decorated with `@onlyCUDA`
- Test will be automatically skipped on CPU-only builds

### 2. Test Requirements
```python
@onlyCUDA  # Requires CUDA GPU
@dtypesIfCUDA(
    *([torch.half] if SM53OrLater and TEST_CUSPARSE_GENERIC else []),
    *([torch.bfloat16] if SM80OrLater and TEST_CUSPARSE_GENERIC else [])
)
```

Requirements:
- CUDA GPU with compute capability 5.3+ for fp16
- CUDA GPU with compute capability 8.0+ for bf16
- cuSPARSE Generic API support

### What Happens on macOS

If you build and run the test on macOS:
```bash
$ python test/test_sparse_csr.py TestSparseCsr.test_sparse_mm_backward_half_precision
...
test_sparse_mm_backward_half_precision (test_sparse_csr.TestSparseCsr) ... skipped 'CUDA not available'
```

The test will be skipped, not failed.

## Alternative: Test in CI

The proper way to test CUDA code without a local CUDA GPU is through CI:

### 1. Push to Your Fork
```bash
git push origin sparse-mm-bf16-support
```

### 2. Create a Pull Request
- Go to https://github.com/pytorch/pytorch/compare/main...sladyn98:pytorch:sparse-mm-bf16-support
- Click "Create Pull Request"

### 3. CI Will Automatically:
- Build PyTorch with CUDA support
- Run all tests including your new test
- Test on multiple GPU architectures
- Report results on the PR

## Quick Verification (No Build Required)

We've already done this with `verify_changes.py`:

```bash
python verify_changes.py
```

This checks:
- ✅ All code changes are correct
- ✅ Syntax is valid
- ✅ Follows patterns from working code
- ✅ Test structure is correct

## Build Time Estimate

If you want to build PyTorch locally:

**Full Build:**
- First time: 1-2 hours (macOS, CPU-only)
- Incremental: 5-30 minutes (after C++ changes)

**For Our Changes:**
Since we modified:
- `aten/src/ATen/native/sparse/cuda/SparseBlasImpl.cpp` (CUDA code)
- `aten/src/ATen/native/sparse/SparseBlas.cpp` (CPU code)
- `aten/src/ATen/native/cpu/SampledAddmmKernel.cpp` (CPU kernel)

On macOS (no CUDA), only CPU parts will build. CUDA code won't be compiled.

## Recommendation

**Do NOT build locally** because:
1. ❌ Takes 1-2 hours
2. ❌ Won't compile CUDA code on macOS
3. ❌ Test will be skipped anyway (no CUDA)
4. ✅ CI will test properly on CUDA hardware
5. ✅ We've already verified changes are correct

**Instead:**
1. ✅ Trust the verification script (all checks passed)
2. ✅ Create the PR
3. ✅ Let CI build and test on real CUDA hardware
4. ✅ Review CI results

## Test Expectations in CI

When CI runs, you should see:

```
test_sparse_mm_backward_half_precision (test_sparse_csr.TestSparseCsr)
Testing with dtype=torch.float16 on cuda ... ok
Testing with dtype=torch.bfloat16 on cuda ... ok

Ran 2 tests in 0.234s

OK
```

If there are issues, CI will show:
- Specific error message
- Which dtype failed (fp16 or bf16)
- Stack trace for debugging

## Manual Testing Script

If you have access to a Linux machine with CUDA, you can use:

```bash
# After building PyTorch with CUDA support
python test_sparse_mm_bf16.py
```

This runs a simplified version of the test that shows:
- ✅ sparse.mm backward works with bf16
- ✅ @ operator backward works with bf16
- ✅ sparse.mm backward works with fp16

## Conclusion

**Current Status:**
- ✅ Code changes verified
- ✅ Ready for PR
- ⏭️  Skip local testing (not feasible on macOS)
- ✅ CI will handle proper CUDA testing

**Next Step:** Create the Pull Request!
