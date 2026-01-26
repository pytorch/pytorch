---
name: metal-kernel
description: Write Metal/MPS kernels for PyTorch operators. Use when adding MPS device support to operators, implementing Metal shaders, or porting CUDA kernels to Apple Silicon. Covers native_functions.yaml dispatch, host-side operators, and Metal kernel implementation.
---

# Metal Kernel Writing Guide

This skill guides you through implementing Metal kernels for PyTorch operators on Apple Silicon.

**Important:** The goal of this skill is to use native Metal capabilities via the `c10/metal/` infrastructure, NOT MPSGraph. Native Metal kernels provide better control, performance, and maintainability.

## Overview

There are two workflows covered by this skill:

1. **Adding new MPS support** - Implementing a new operator from scratch
2. **Migrating from MPSGraph** - Converting existing MPSGraph-based operators to native Metal

Both workflows involve:
1. **Update dispatch** in `aten/src/ATen/native/native_functions.yaml`
2. **Write Metal kernel** in `aten/src/ATen/native/mps/kernels/`
3. **Implement host-side stub** in `aten/src/ATen/native/mps/operations/`

## Step 1: Update native_functions.yaml

**Location:** `aten/src/ATen/native/native_functions.yaml`

### For New Operators

Find the operator entry and add MPS dispatch:

```yaml
# Simple MPS-specific implementation
- func: my_op(Tensor self) -> Tensor
  dispatch:
    CPU: my_op_cpu
    CUDA: my_op_cuda
    MPS: my_op_mps

# Shared implementation across devices (preferred for structured kernels)
- func: my_op.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
  dispatch:
    CPU, CUDA, MPS: my_op_out

# Structured kernel (preferred for new ops)
- func: my_op.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
  structured: True
  structured_inherits: TensorIteratorBase
  dispatch:
    CPU, CUDA, MPS: my_op_out
```

### For Migrating from MPSGraph

When migrating an existing operator from MPSGraph to native Metal, **consolidate the dispatch entry**:

```yaml
# BEFORE (MPSGraph-based, separate dispatch)
- func: atan2.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
  structured: True
  structured_inherits: TensorIteratorBase
  dispatch:
    CPU, CUDA: atan2_out
    MPS: atan2_out_mps  # Separate MPS implementation

# AFTER (native Metal, shared dispatch via stub)
- func: atan2.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
  structured: True
  structured_inherits: TensorIteratorBase
  dispatch:
    CPU, CUDA, MPS: atan2_out  # MPS now uses the same stub mechanism
```

**Key change:** Replace `MPS: my_op_out_mps` with adding `MPS` to the shared dispatch line (e.g., `CPU, CUDA, MPS: my_op_out`).

**Dispatch naming conventions:**
- `MPS: function_name_mps` - MPS-specific implementation (old MPSGraph pattern)
- `CPU, CUDA, MPS: function_name` - Shared stub implementation (native Metal pattern)

## Step 2: Implement Metal Kernel

**Location:** `aten/src/ATen/native/mps/kernels/`

### Unary Kernel Pattern

```metal
// MyKernel.metal
#include <c10/metal/indexing.h>
#include <c10/metal/utils.h>
#include <metal_stdlib>

using namespace metal;
using namespace c10::metal;

// Define operation functor
struct my_op_functor {
  template <typename T>
  inline T operator()(const T x) {
    return /* your operation */;
  }
};

// Register for supported types
REGISTER_UNARY_OP(my_op, float, float);
REGISTER_UNARY_OP(my_op, half, half);
REGISTER_UNARY_OP(my_op, bfloat, bfloat);
```

### Binary Kernel Pattern

```metal
struct my_binary_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return /* your operation */;
  }
};

REGISTER_BINARY_OP(my_binary, float, float);
REGISTER_BINARY_OP(my_binary, half, half);
```

### Binary Kernel Type Registration Macros

For binary operations, use the convenience macros defined in `BinaryKernel.metal`:

```metal
// Floating-point types only (float, half, bfloat)
REGISTER_FLOAT_BINARY_OP(my_op);

// Integral types with float output (for math ops like atan2, copysign)
// Registers: long->float, int->float, short->float, uchar->float, char->float, bool->float
REGISTER_INT2FLOAT_BINARY_OP(my_op);

// Integral types with same-type output (for bitwise/logical ops)
// Registers: long, int, short, uchar, char, bool
REGISTER_INTEGER_BINARY_OP(my_op);

// Floating-point with opmath precision (for ops needing higher precision)
REGISTER_OPMATH_FLOAT_BINARY_OP(my_op);
```

**Common patterns:**
- Math functions (atan2, copysign, logaddexp): Use both `REGISTER_FLOAT_BINARY_OP` and `REGISTER_INT2FLOAT_BINARY_OP`
- Comparison/logical ops (maximum, minimum): Use both `REGISTER_FLOAT_BINARY_OP` and `REGISTER_INTEGER_BINARY_OP`
- Arithmetic ops (add, sub, mul): Use both `REGISTER_FLOAT_BINARY_OP` and `REGISTER_INTEGER_BINARY_OP`

**Example for atan2 (supports both float and int inputs):**
```metal
struct atan2_functor {
  template <typename T, enable_if_t<is_floating_point_v<T>, bool> = true>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(precise::atan2(float(a), float(b)));
  }
  template <typename T, enable_if_t<is_integral_v<T>, bool> = true>
  inline float operator()(const T a, const T b) {
    return precise::atan2(float(a), float(b));
  }
};

REGISTER_FLOAT_BINARY_OP(atan2);
REGISTER_INT2FLOAT_BINARY_OP(atan2);
```

### With Scalar Parameter

```metal
struct my_alpha_functor {
  template <typename T>
  inline T operator()(const T a, const T b, const T alpha) {
    return a + c10::metal::mul(alpha, b);
  }
};

REGISTER_UNARY_ALPHA_OP(my_alpha, float, float, float);
REGISTER_UNARY_ALPHA_OP(my_alpha, half, half, half);
```

### Type-Specialized Functor

```metal
struct special_functor {
  // Floating point types
  template <typename T, enable_if_t<is_scalar_floating_point_v<T>, bool> = true>
  inline T operator()(const T x) {
    return precise::exp(x);  // Use precise math
  }

  // Integral types
  template <typename T, enable_if_t<is_scalar_integral_v<T>, bool> = true>
  inline float operator()(const T x) {
    return precise::exp(float(x));
  }

  // Complex types (float2 for cfloat, half2 for chalf)
  template <typename T, enable_if_t<is_complex_v<T>, bool> = true>
  inline T operator()(const T x) {
    // x.x = real, x.y = imaginary
    return T(/* real */, /* imag */);
  }
};
```

**Note on complex types:** Complex numbers in Metal are represented as vector types:
- `c10::complex<float>` maps to `float2` (x = real, y = imaginary)
- `c10::complex<half>` maps to `half2`

Use `is_complex_v<T>` to specialize for complex types in functors.

### Available c10/metal Utilities

**utils.h:**
- `opmath_t<T>` - Operation math type (half->float)
- `accum_t<T>` - Accumulation type for reductions
- `max()`, `min()` with NaN propagation

**special_math.h:**
- `precise::exp()`, `precise::log()`, `precise::sqrt()`
- `precise::sin()`, `precise::cos()`, `precise::tan()`
- `erf()`, `erfc()`, `erfinv()`

**indexing.h:**
- `REGISTER_UNARY_OP(name, in_type, out_type)`
- `REGISTER_BINARY_OP(name, in_type, out_type)`
- `REGISTER_UNARY_ALPHA_OP(name, in_type, alpha_type, out_type)`

## Step 3: Implement Host-Side Stub

**Location:** `aten/src/ATen/native/mps/operations/`

Choose or create an appropriate file based on operation type:
- `UnaryKernel.mm` - Single input operations via stub dispatch
- `BinaryKernel.mm` - Two input operations via stub dispatch
- `UnaryOps.mm` / `BinaryOps.mm` - Legacy MPSGraph implementations (for reference)
- `ReduceOps.mm` - Reductions (sum, mean, max, etc.)
- Create new file for distinct operation categories

### Stub Registration Pattern (Preferred for Native Metal)

For structured kernels that use the TensorIterator pattern:

```objc
// In BinaryKernel.mm (or appropriate file)

static void my_op_mps_kernel(TensorIteratorBase& iter) {
  lib.exec_binary_kernel(iter, "my_op");  // "my_op" matches the functor name in .metal
}

// Register the MPS stub - this connects to the dispatch system
REGISTER_DISPATCH(my_op_stub, &my_op_mps_kernel)
```

**For unary operations:**
```objc
static void my_unary_mps_kernel(TensorIteratorBase& iter) {
  lib.exec_unary_kernel(iter, "my_unary");
}

REGISTER_DISPATCH(my_unary_stub, &my_unary_mps_kernel)
```

### Migration: Removing Old MPSGraph Implementation

When migrating from MPSGraph, also remove the old implementation:

1. **Remove from BinaryOps.mm (or UnaryOps.mm):**
   - Delete the `TORCH_IMPL_FUNC(my_op_out_mps)` implementation
   - Remove the corresponding `#include <ATen/ops/my_op_native.h>` header

2. **Add to BinaryKernel.mm (or UnaryKernel.mm):**
   - Add the static kernel function
   - Add the `REGISTER_DISPATCH` call

## Step 4: Compile

After making changes, compile to verify everything builds correctly:

```bash
cd build && ninja torch_cpu
```

## Testing

Basic operator support is already tested by `test_output_match` in `test/test_mps.py`. After implementing an operator, enable testing by removing expected failures:

### 1. Remove from common_mps.py

**Location:** `torch/testing/_internal/common_mps.py`

Find and remove the operator from skip/xfail lists:

```python
# Remove entries like:
MPS_XFAILLIST = {
    "my_op": ...,  # Remove this line
}

MPS_SKIPLIST = {
    "my_op": ...,  # Remove this line
}
```

### 2. Remove from OpInfo decorators

**Location:** `torch/testing/_internal/common_methods_invocations.py` (or related files)

Remove MPS-specific decorators from the OpInfo:

```python
OpInfo(
    "my_op",
    # Remove decorators like:
    # decorators=[skipMPS, expectedFailureMPS("reason")],
    ...
)
```

### 3. Run tests to verify

```bash
# Run the specific operator test
python test/test_mps.py -k test_output_match_my_op

# Or run full MPS test suite
python test/test_mps.py
```

## Checklist

- [ ] Added MPS dispatch to `native_functions.yaml`
- [ ] Implemented Metal kernel in `kernels/`
- [ ] Implemented host-side operator in `operations/`
- [ ] Handles empty tensors
- [ ] Handles non-contiguous tensors
- [ ] Supports required dtypes (float32, float16, bfloat16, and often complex types via float2/half2)
- [ ] Removed expected failures from `torch/testing/_internal/common_mps.py`
- [ ] Removed skip/xfail decorators from OpInfo (if applicable)
