---
name: cuda-index-width
description: Choose 32-bit vs 64-bit index math in PyTorch CUDA kernels. Use when fixing large-tensor indexing overflows, deciding whether to use int64_t, canUse32BitIndexMath, CUDA_KERNEL_LOOP_TYPE, or AT_DISPATCH_INDEX_TYPES, and when considering binary-size or performance impact of index-type templating.
---

# CUDA Index Width in PyTorch

Use this skill when a CUDA kernel overflows `int` indexing, fails near `2^31` elements, or needs a review of `int` vs `int64_t` index math.

## Start with the overflow site

Do not blindly convert all index variables to `int64_t`. Find the expression that can exceed 32 bits and classify where it runs:

- **One-time setup per CTA or per output element group**: prefer one local 64-bit cast at the first multiply.
- **Hot per-element linear indexing**: template the kernel on `index_t` and dispatch `int` vs `int64_t`.
- **Unsupported algorithm with 32-bit-only assumptions**: add a clear `TORCH_CHECK(canUse32BitIndexMath(...))` instead of silently overflowing.
- **Grid dimension overflow**: changing arithmetic type is not enough; add tiling/striding over that dimension.

## Canonical utilities

Include/use the existing PyTorch utilities instead of ad hoc checks:

```cpp
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/cuda/detail/KernelUtils.h>
```

- `at::native::canUse32BitIndexMath(tensor, INT_MAX)` checks both `numel` and maximum storage offset.
- For CUDA files that already include cuda detail wrappers, `at::cuda::detail::canUse32BitIndexMath` may be available as an alias.
- `AT_DISPATCH_INDEX_TYPES(cond ? ScalarType::Int : ScalarType::Long, "name", [&] { ... });` provides `index_t`.
- `CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t)` keeps grid-stride loops correct for either width.

Check every tensor whose offsets are computed with the selected index type, not just the output tensor.

## Fix patterns

### 1. Localized base-offset overflow

If only a base pointer offset can overflow and it is computed outside the hot loop, keep the kernel otherwise unchanged:

```cpp
int64_t plane = blockIdx.x;
input = input + plane * strideD;
output = output + plane * osizeH * osizeW;
```

This avoids doubling kernel instantiations and keeps inner-loop arithmetic 32-bit. Use this when dimensions inside the tile still fit in `int`.

### 2. Grid-stride linear kernels

If the loop index, modulo/division decomposition, or final `data[index]` access can exceed 32 bits, template the kernel:

```cpp
template <typename scalar_t, typename index_t>
__global__ void kernel(index_t n, const scalar_t* in, scalar_t* out) {
  CUDA_KERNEL_LOOP_TYPE(index, n, index_t) {
    out[index] = in[index];
  }
}

AT_DISPATCH_INDEX_TYPES(
    canUse32BitIndexMath(out, INT_MAX) && canUse32BitIndexMath(in, INT_MAX)
        ? ScalarType::Int
        : ScalarType::Long,
    "kernel_index_type",
    [&] {
      kernel<scalar_t, index_t><<<blocks, threads, 0, stream>>>(n, in, out);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
```

Prefer this over unconditionally changing the loop index to `int64_t`, because 64-bit division/modulo in a hot CUDA loop can be measurable.

### 3. TensorInfo/accessor or strided kernels

When offsets are computed from sizes/strides, dispatch on an index type only if all participating tensors pass `canUse32BitIndexMath` for that type. Remember that a small `numel()` tensor can still need 64-bit offsets if it is a large strided view.

### 4. 32-bit-only kernels

If supporting 64-bit indexing would require a larger algorithm rewrite or would exceed CUDA launch limits, fail early:

```cpp
TORCH_CHECK(
    canUse32BitIndexMath(input) && canUse32BitIndexMath(output),
    "op_name: tensors must fit into 32-bit index math");
```

Only use this when the operator already has a documented or accepted size limitation; do not turn a reported correctness bug into an unnecessary limitation.

## Binary-size and performance tradeoffs

Templating on `index_t` duplicates each affected kernel for every scalar dtype and memory-format specialization. Before adding index dispatch to several kernels, ask whether the overflow is in a hot path or only in one setup expression.

A/B candidate fixes when the choice is not obvious:

1. Build each candidate from a clean diff using the same build environment.
2. Record changed CUDA object and library sizes:
   ```bash
   stat -c '%s %n' build/aten/src/ATen/CMakeFiles/torch_cuda.dir/native/cuda/<file>.cu.o torch/lib/libtorch_cuda.so
   ```
3. Check symbol multiplication for the kernel name:
   ```bash
   nm -S --size-sort -C torch/lib/libtorch_cuda.so | rg '<kernel_name>|index_t|long|int'
   ```
4. If hot-loop arithmetic changed, benchmark representative small and large tensors; do not report performance from sanitizer runs.

Default decision:

- **One or two 64-bit setup multiplies**: prefer the local cast.
- **Per-element indexing may exceed 32 bits**: prefer `index_t` dispatch.
- **Existing kernel family already dispatches `index_t`**: extend the existing pattern.
- **Changing many scalar-specialized kernels**: consider binary size before templating all of them.

## Tests for large-index fixes

Add a regression that crosses the exact boundary that failed:

- Use the smallest dtype and output size that still exercises the overflow.
- Assert `tensor.numel() > torch.iinfo(torch.int32).max` or assert the specific offset boundary.
- Sample values from both below and above the boundary; avoid full-tensor CPU comparisons for huge tensors.
- Run the original repro under compute-sanitizer for memory bugs:
  ```bash
  CUDA_LAUNCH_BLOCKING=1 PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool memcheck --error-exitcode=99 <python> repro.py
  ```

For PyTorch tests, prefer adding the regression near related pooling/indexing tests and guard expensive cases with `@largeTensorTest` and the relevant device decorator.

## Review checklist

- [ ] The exact overflowing expression is identified.
- [ ] 64-bit math is limited to the expressions that need it, or the kernel is templated when the hot index needs it.
- [ ] `canUse32BitIndexMath` considers every tensor whose offsets use the selected type.
- [ ] CUDA launch dimensions still fit hardware limits.
- [ ] The test fails before the fix and passes after it, or the report explains why pre-fix failure was not rerun.
- [ ] Binary-size/performance impact is mentioned if new `index_t` dispatch duplicates kernels.
