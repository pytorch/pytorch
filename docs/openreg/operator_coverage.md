# Operator Coverage Guide

## Overview

Operator coverage describes which operations are implemented and fully functional on a backend. This guide explains what "coverage" means, how it's staged over time, and how maintainers evaluate backend maturity using coverage signals.

---

## What is Operator Coverage?

**Operator Coverage** is the set of PyTorch operations that are fully implemented and tested on a device backend. An operation is considered "covered" when:

1. ✅ A kernel is implemented and registered for the device
2. ✅ All major data types are supported (float32, float64, int32, int64, etc.)
3. ✅ Autograd/gradient computation works (if the op is differentiable)
4. ✅ Edge cases and error conditions are handled
5. ✅ Tests exist and pass reliably

**Operators NOT in coverage:**

- ❌ Not yet implemented
- ❌ Partially implemented (only some dtypes supported)
- ❌ Broken or failing tests
- ❌ Intentionally unsupported (hardware limitations, etc.)

---

## Backend Maturity Stages

Backends evolve in stages. Knowing which stage you're targeting helps set realistic goals.

### Stage 1: Minimal (Proof of Concept)

**Criteria:**
- 5-15 core operations working
- Only float32 dtype
- Basic memory/device management
- No autograd support

**Example Operations:** `zeros`, `ones`, `empty`, `add`, `mul`, `reshape`

**Test Coverage:** Basic functional tests only

**Duration:** 1-4 weeks (prototype phase)

**Example:** Initial OpenReg simulator

---

### Stage 2: Core (MVP - Minimum Viable Product)

**Criteria:**
- 30-50 essential operations
- float32 + float64 support
- Basic autograd (backward pass)
- Memory management (allocation, transfer)
- Error handling for common cases

**Example Operations:**
- Math: `add`, `sub`, `mul`, `div`, `matmul`, `sum`, `mean`
- Creation: `zeros`, `ones`, `randn`, `arange`
- Manipulation: `reshape`, `transpose`, `cat`, `stack`
- Autograd-ready operations with backward registered

**Test Coverage:** 
- Parametrized tests for device/dtype
- Numerical correctness tests
- Error handling tests
- Basic gradient tests

**Duration:** 4-12 weeks

**Example:** Early-stage custom backend

---

### Stage 3: Extended (Production Ready)

**Criteria:**
- 100-200 operations
- Multiple dtypes: float32, float64, int32, int64, complex64, complex128, etc.
- Full autograd support (forward and backward)
- Advanced features: in-place operations, out= arguments, batched operations
- Comprehensive error messages
- Performance optimizations

**Example Operations:** All of Stage 2, plus:
- Advanced: `softmax`, `layer_norm`, `attention`, `conv2d`, `linear`
- Broadcasting and reductions: `broadcast_to`, `squeeze`, `unsqueeze`
- Type conversions and dtypes: `to`, `as_tensor`, `cast`
- Sorting, indexing: `sort`, `argsort`, `nonzero`, `gather`

**Test Coverage:**
- All operations with parametrization
- Device/dtype/autograd combinations
- Edge cases and corner cases
- Performance benchmarks
- Numerical stability tests

**Duration:** 3-6 months

**Example:** Mature custom backend

---

### Stage 4: Advanced (Specialized)

**Criteria:**
- 300+ operations
- All PyTorch dtypes supported
- Specialized features: quantization, sparse tensors, reduced precision (bfloat16, float16)
- Hardware-specific optimizations
- Distributed training support (if applicable)
- Custom kernels for performance

**Example Operations:** Everything, plus:
- Quantized operations: `quantize`, `dequantize`, `qint8` dtypes
- Sparse operations: `sparse_coo_tensor`, `sparse_matmul`
- Reduced precision: `bfloat16`, `float16` with careful rounding
- Custom operators: domain-specific operations

**Test Coverage:**
- Exhaustive testing across all configurations
- Stress tests and load tests
- Integration tests with PyTorch ecosystem
- Regression tests for performance

**Duration:** 6+ months (ongoing)

**Example:** CUDA, CPU backends in PyTorch

---

## How to Track Coverage

### Tracking File

Create a `COVERAGE.md` or `OPERATOR_SUPPORT.md` in your backend documentation:

```markdown
# OpenReg Operator Coverage

## Stage: 2 (MVP)

### Fully Supported (50 ops)
- Math: add, sub, mul, div, sqrt, exp, log, sin, cos
- Creation: zeros, ones, empty, randn, full, arange
- Indexing: getitem, setitem, index_put
- Reduction: sum, mean, max, min, all, any
- Autograd: backward, requires_grad, grad

### Partially Supported (5 ops)
- matmul: float32 only (no float64 yet)
- conv2d: only 1D convolution
- linear: autograd not yet implemented

### Not Yet Supported (100+ ops)
- CUDA fused operations: flashy_attention, fused_scaled_dot_product_attention
- Sparse: sparse_matmul, sparse_add
- Quantized: qint8_add, qint8_matmul
```

### Automated Tracking

Use a test script to count passing tests per device:

```python
import subprocess
import re

def count_passing_tests(device_type):
    result = subprocess.run(
        ["python", "-m", "pytest", "test/openreg/tests/", "-q", f"--tb=no"],
        capture_output=True,
        text=True,
    )
    # Parse output: "50 passed, 10 failed, 5 skipped"
    match = re.search(r"(\d+) passed", result.stdout)
    return int(match.group(1)) if match else 0

print(f"CPU tests passing: {count_passing_tests('cpu')}")
print(f"OpenReg tests passing: {count_passing_tests('openreg')}")
```

---

## Coverage Goals for OpenReg

As an official reference framework, OpenReg should target **Stage 3 (Production Ready)** with at least:

- **100+ operations** covering all common use cases
- **Multiple dtypes:** float32, float64, int32, int64, int8, uint8
- **Autograd support:** Forward and backward passes
- **Comprehensive tests:** Parametrized across device/dtype/autograd combinations

**Initial Release (v1.0):**
- 50 core operations (Stage 2)
- float32 + float64
- Basic autograd

**Mature Release (v2.0):**
- 100+ operations (Stage 3)
- All common dtypes
- Full autograd + specialized features

---

## How Maintainers Evaluate Backend Readiness

### Checklist for Readiness

**Minimal (Can merge):**
- [ ] Implementation plan documented (this file)
- [ ] Basic test infrastructure in place
- [ ] 5+ core operations implemented and tested
- [ ] Device/memory management working
- [ ] Clear error messages for unimplemented operations

**MVP (Ready for general use):**
- [ ] 30+ operations implemented
- [ ] float32 + float64 supported
- [ ] Basic autograd working
- [ ] Parametrized tests for all operations
- [ ] Performance benchmarks established
- [ ] Documentation complete

**Production (Ready for integration):**
- [ ] 100+ operations
- [ ] All common dtypes
- [ ] Full autograd + specialized features
- [ ] 95%+ tests passing
- [ ] Performance regression tests
- [ ] Security audits completed

### Test Dashboards

Use CI/CD to track coverage trends:

```yaml
# Example GitHub Actions workflow
coverage:
  runs-on: ubuntu-latest
  steps:
    - name: Run OpenReg tests
      run: |
        python -m pytest test/openreg/tests/ -q --tb=no
        # Outputs: "50 passed, 10 failed"
        
    - name: Track in dashboard
      run: |
        # Post results to internal tracking system
        curl -X POST https://internal-dashboard/coverage \
          -d '{"device": "openreg", "passed": 50, "failed": 10, "date": "2024-01-13"}'
```

---

## Common Coverage Gaps

### ❌ Premature Optimization

**Don't:** Spend weeks optimizing a kernel that's not yet tested.

**Do:** Implement + test first, optimize later.

### ❌ Incomplete Autograd

**Don't:** Implement forward but not backward.

**Do:** Implement both or skip autograd tests with a note.

### ❌ Missing Dtype Support

**Don't:** Only support float32 and claim coverage.

**Do:** Add float64 tests early; expanding dtypes is straightforward.

### ❌ Ignoring Edge Cases

**Don't:** Test only happy-path cases.

**Do:** Include edge cases (empty tensors, scalars, size-1 dimensions, etc.).

---

## Examples from PyTorch Ecosystems

### CPU Backend

- **Operations:** 2,000+
- **Supported dtypes:** float32, float64, float16, bfloat16, int32, int64, int8, uint8, complex64, complex128, bool
- **Autograd:** Full support
- **Stage:** Advanced (Stage 4)
- **Test count:** 10,000+ tests across all combinations

### CUDA Backend

- **Operations:** 2,000+
- **Supported dtypes:** Same as CPU + additional optimizations
- **Autograd:** Full support
- **Stage:** Advanced (Stage 4)
- **Special features:** Fused kernels, mixed precision, async operations

### OpenReg (Target)

- **Operations:** 100+ (Stage 3 target)
- **Supported dtypes:** float32, float64, int32, int64 (Stage 2+)
- **Autograd:** Basic to full (depends on stage)
- **Stage:** MVP → Production Ready
- **Special features:** Reference implementation for new backends

---

## Next Steps

1. **Document your current stage** in a COVERAGE.md file
2. **Set a target stage** (MVP or Production Ready)
3. **List operations** you plan to implement
4. **Create a timeline** (4 weeks per stage on average)
5. **Track progress** using test counts and CI dashboards
6. **Review coverage regularly** and adjust goals as needed

See [adding_tests.md](adding_tests.md) for how to add tests for each new operation, and [failure_interpretation.md](failure_interpretation.md) for debugging test failures.
