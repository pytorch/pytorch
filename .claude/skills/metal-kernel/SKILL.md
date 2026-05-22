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

**Update every overload.** A single op typically has several
`native_functions.yaml` entries — functional / inplace / `.out`, plus
`Tensor` and `Scalar` variants. Each entry has its own `dispatch:` block, and
each one must be moved over. Any entry left pointing at `MPS: my_op_mps`
still routes that overload to the MPSGraph code, so callers can silently
land on the old path depending on which overload they hit. Before declaring
the migration done, grep the legacy function name and confirm no entry still
references it.

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

## Debugging Metal Kernels with `torch.mps.compile_shader`

Use `torch.mps.compile_shader` to JIT-compile and test individual Metal kernels in isolation. This is invaluable for debugging multi-kernel pipelines where you need to verify each stage independently.

### Basic Usage

```python
import torch

source = '''
#include <metal_stdlib>
using namespace metal;

kernel void my_kernel(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
  output[tid] = input[tid] * 2.0;
}
'''

lib = torch.mps.compile_shader(source)

inp = torch.tensor([1.0, 2.0, 3.0], device='mps')
out = torch.zeros(3, device='mps')
lib.my_kernel(inp, out, threads=[3, 1, 1], group_size=[3, 1, 1])
torch.mps.synchronize()
print(out)  # tensor([2., 4., 6.], device='mps:0')
```

### Dispatch Semantics

`compile_shader` uses **`dispatchThreads`** semantics (same as `mtl_dispatch1DJob` in PyTorch):
- `threads=[N, 1, 1]` — total number of threads (NOT threadgroups)
- `group_size=[G, 1, 1]` — threads per threadgroup

This differs from the `dispatchThreadgroups` API used by some host-side code. To match `dispatchThreadgroups:MTLSizeMake(num_tgs, num_slices, 1) threadsPerThreadgroup:MTLSizeMake(TG_SIZE, 1, 1)`:

```python
# Equivalent compile_shader call:
lib.kernel(args...,
    threads=[num_tgs * TG_SIZE, num_slices, 1],
    group_size=[TG_SIZE, 1, 1])
```

### Constant Buffer Parameters

Pass scalar constants as single-element tensors:

```python
slice_size = torch.tensor([1024], dtype=torch.int32, device='mps')
lib.my_kernel(data, output, slice_size, threads=[1024, 1, 1], group_size=[256, 1, 1])
```

### Debugging Strategy for Multi-Kernel Pipelines

When a pipeline of kernels (e.g., histogram → prefix_sum → scatter) produces wrong results, test each kernel individually and verify its output against a Python/NumPy reference:

```python
# 1. Run GPU kernel
lib.histogram(keys, hist, ..., threads=[N, 1, 1], group_size=[256, 1, 1])
torch.mps.synchronize()

# 2. Compute reference in Python
ref_hist = compute_histogram_cpu(keys.cpu().numpy(), ...)

# 3. Compare
assert np.array_equal(hist.cpu().numpy(), ref_hist), "Histogram mismatch!"
```

This isolates which kernel in the pipeline is broken, rather than debugging the entire pipeline at once.

### Common Pitfalls

- **Wrong `threads` count** — `threads` is total threads, not threadgroups. For 5 threadgroups of 256, use `threads=[1280, 1, 1]`.
- **Threadgroup memory** — `compile_shader` doesn't support `[[threadgroup(N)]]` parameters directly. If your kernel needs threadgroup memory, restructure to use `threadgroup` arrays declared inside the kernel body instead.

## Working with TensorIterators

`REGISTER_UNARY_OP` / `REGISTER_BINARY_OP` hide the iterator plumbing. Kernels
with extra params or non-elementwise layouts have to drive `TensorIterator`
directly. The non-obvious rules:

- **Pass `TensorIteratorBase&`, not `Tensor&`.** `Tensor&` loses the
  offset/shape info `with_32bit_indexing()` produces. When the stub hands you
  a `const TensorBase&` (e.g. `bernoulli_scalar_stub`), build the iter inline
  with `at::TensorIterator::borrowing_nullary_op(self)` rather than
  const-casting.

- **`iter.tensor(0)` returns the *whole* tensor for sub-iters.** After
  `with_32bit_indexing()`, sub-iters still reference the original storage, so
  binding `iter.tensor(0)` naively makes every sub-iter overwrite the same
  prefix and leaves the tail uninitialized. Use `bind_iter_tensors`, which
  computes the per-chunk offset and binds buffer 0 to the slice. Dispatch
  with `iter.numel()`, never `iter.tensor(0).numel()`:
  ```cpp
  bind_iter_tensors(computeEncoder, iter, /*ntensors=*/1);
  mtl_setArgs<1>(computeEncoder, params, ..., numel);
  mtl_dispatch1DJob(computeEncoder, pso, threads);
  ```

- **Prefer `mtl_setArgs<N>` over chained `mtl_setBytes`.**
  `mtl_setArgs<1>(encoder, a, b, c)` binds `a`/`b`/`c` at slots 1/2/3 with the
  same overload resolution as the macros (`std::array<long,2>` →
  `constant long2&`).

- **Use `ceil_div`.** Host: `at::ceil_div` from `<ATen/ceil_div.h>`. Metal:
  `c10::metal::ceil_div` from `<c10/metal/common.h>` (unqualified after
  `using namespace c10::metal;`). Both wrap `(a + b - 1) / b`.

- **`IF_CONSTEXPR` for Metal 3/4 portability.** Metal 4 has `if constexpr`;
  Metal 3 doesn't. Use the macro from `<c10/metal/common.h>`, e.g.
  `if IF_CONSTEXPR (sizeof(T) == 8) { ... }`.

- **Share the CPU/CUDA stub via `REGISTER_MPS_DISPATCH`.** When the op has a
  `DECLARE_DISPATCH` stub upstream (distributions, fused ops), wire MPS in
  with `REGISTER_MPS_DISPATCH(stub_name, &fn)` instead of an MPS-specific
  `native_functions.yaml` entry. The stub already takes `TensorIteratorBase&`.

## Working with Large Tensors

Most Metal kernels take `numel` as `uint32_t` and index in 32 bits, so anything
past `INT32_MAX` needs host-side splitting. Drive this through `TensorIterator`
rather than slicing manually.

**Decompose via `iter.with_32bit_indexing()`:**

```cpp
if (!iter.can_use_32bit_indexing()) {
  for (auto&& sub_iter : iter.with_32bit_indexing()) {
    my_kernel_impl(sub_iter, ...);
  }
  return;
}
```

Every yielded sub-iter satisfies `can_use_32bit_indexing()`, so recursion
terminates after one level. The threshold is `INT32_MAX` (TensorIterator uses
signed 32-bit offsets), not `UINT32_MAX` — a test just past `INT32_MAX`
exercises the split but not the `uint32_t` cast; the cast itself only matters
for `numel > 2^32`.

**Use a checked cast when narrowing:**

```cpp
const uint32_t numel = c10::checked_convert<uint32_t>(iter.numel(), "uint32_t");
```

Use `c10::checked_convert` (`<c10/util/TypeCast.h>`) so wraparound becomes a
`TORCH_CHECK` failure instead of a corrupt output.

## Error Reporting from Kernels

**NEVER copy results to CPU for the purposes of error checking.** Any
`.item()`, `.cpu()`, or other host read on a GPU tensor forces a full
GPU->CPU sync that drains every in-flight op on the stream — not just the
reduction you're inspecting. In a realistic pipeline this stalls the
whole queue, and the cost dwarfs the actual check. Guarding the sync
behind `is_mps()` doesn't help; the stall happens every time the op runs.
Validate on-device instead and report errors through the mechanism below.

GPU code can't throw, but kernels can write into a shared error buffer that the
host raises as `c10::AcceleratorError` on the next sync. Use
`TORCH_REPORT_ERROR(error_buf, ...)` from `<c10/metal/error.h>`. Variadic
arguments are concatenated; integers are formatted in base 10. Delivery is
asynchronous — the failing thread keeps running, and the error surfaces
whenever `MPSStream::checkLastError()` next runs (after a `synchronize()` or
the next op that drains the stream), so don't rely on it for control flow
inside the same dispatch. Crucially, this adds *no* forced sync: the error
piggybacks on whatever sync the user's code already does.

**Kernel side:** take a `device ErrorMessages* error_buf` argument and call
`TORCH_REPORT_ERROR` on the bad path. Skip the offending element afterwards so
you don't also corrupt memory:

```metal
#include <c10/metal/error.h>

kernel void index_set_1d(
    device float* self,
    constant float* values,
    constant long* indices,
    constant uint& self_numel,
    device ::c10::metal::ErrorMessages* error_buf,
    uint tid [[thread_position_in_grid]]) {
  long idx = indices[tid];
  if (idx < 0 || idx >= long(self_numel)) {
    TORCH_REPORT_ERROR(
        error_buf, "index ", idx, " out of bounds for size ", long(self_numel));
    return;
  }
  self[idx] = values[tid];
}
```

**Host side:** bind the stream's error buffer as the corresponding argument.
`mtl_setArgs` accepts it directly:

```objc
auto* stream = getCurrentMPSStream();
mtl_setArgs(encoder, self, values, indices, uint32_t(self.numel()),
            stream->getErrorBuffer());
```

The buffer is owned by `MPSStream`, sized for 30 messages, and reset after
each `checkLastError()` drain. Only the first message is reported; later ones
are kept for debugging but the `AcceleratorError` carries `msg[0]`.

## Checklist

- [ ] Added MPS dispatch to `native_functions.yaml`
- [ ] Implemented Metal kernel in `kernels/`
- [ ] Implemented host-side operator in `operations/`
- [ ] Handles empty tensors
- [ ] Handles non-contiguous tensors
- [ ] Supports required dtypes (float32, float16, bfloat16, and often complex types via float2/half2)
- [ ] Removed expected failures from `torch/testing/_internal/common_mps.py`
- [ ] Removed skip/xfail decorators from OpInfo (if applicable)
