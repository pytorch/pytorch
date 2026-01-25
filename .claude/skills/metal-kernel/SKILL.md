---
name: metal-kernel
description: Write Metal/MPS kernels for PyTorch operators. Use when adding MPS device support to operators, implementing Metal shaders, or porting CUDA kernels to Apple Silicon. Covers native_functions.yaml dispatch, host-side operators, and Metal kernel implementation.
---

# Metal Kernel Writing Guide

This skill guides you through implementing Metal kernels for PyTorch operators on Apple Silicon.

**Important:** The goal of this skill is to use native Metal capabilities via the `c10/metal/` infrastructure, NOT MPSGraph. Native Metal kernels provide better control, performance, and maintainability.

## Overview

Adding MPS support to a PyTorch operator involves three steps:
1. **Register dispatch** in `aten/src/ATen/native/native_functions.yaml`
2. **Write Metal kernel** in `aten/src/ATen/native/mps/kernels/`
3. **Implement host-side operator** in `aten/src/ATen/native/mps/operations/`

## Step 1: Add MPS Dispatch to native_functions.yaml

**Location:** `aten/src/ATen/native/native_functions.yaml`

Find the operator entry and add MPS dispatch:

```yaml
# Simple MPS-specific implementation
- func: my_op(Tensor self) -> Tensor
  dispatch:
    CPU: my_op_cpu
    CUDA: my_op_cuda
    MPS: my_op_mps

# Shared implementation across devices
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

**Dispatch naming conventions:**
- `MPS: function_name_mps` - MPS-specific implementation
- `CPU, CUDA, MPS: function_name` - Shared implementation
- `SparseMPS: function_name_sparse_mps` - Sparse tensor support

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
- `c10::complex<float>` → `float2` (x = real, y = imaginary)
- `c10::complex<half>` → `half2`

Use `is_complex_v<T>` to specialize for complex types in functors.

### Available c10/metal Utilities

**utils.h:**
- `opmath_t<T>` - Operation math type (half→float)
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

## Step 3: Implement Host-Side Operator

**Location:** `aten/src/ATen/native/mps/operations/`

Choose or create an appropriate file based on operation type:
- `UnaryOps.mm` - Single input operations (relu, sigmoid, exp, etc.)
- `BinaryOps.mm` - Two input operations (add, mul, etc.)
- `ReduceOps.mm` - Reductions (sum, mean, max, etc.)
- Create new file for distinct operation categories

The host-side code dispatches to your Metal kernel. Look at existing implementations in the operations directory for patterns on how to launch Metal kernels using the `c10/metal` infrastructure.

## Step 4: Compile

After making changes, compile to verify everything builds correctly:

```bash
cd build && ninja torch_cpu
```

Fix any compilation errors before proceeding to testing.

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
