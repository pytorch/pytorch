#pragma once

#include <ATen/jit_macros.h>

#if AT_USE_JITERATOR()

#include <ATen/cuda/CUDAConfig.h>

#include <ATen/OpMathType.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/TensorIteratorDynamicCasting.h>

#include <ATen/native/cuda/MemoryAccess.cuh>

#if !AT_ROCM_ENABLED()
#include <ATen/native/cuda/CUDAJitLoops.cuh>
#else
#error Jiterator not supported on ROCm
#endif

namespace at {
namespace native {

/* Note [Jiterator]
The "jiterator" simply just-in-time compiles the same kernels that
Loops.cuh (and CUDALoops.cuh) usually build. This reduces build time,
build size, and initial CUDA context size.

By default on non-Windows systems, it also caches compiled kernels in ~/.cache/torch/kernels.
This behavior is controlled with two environment variables:
  - USE_PYTORCH_KERNEL_CACHE, if set to zero then this will disable all cache use
  - PYTORCH_KERNEL_CACHE_PATH, if set specifies the folder to use for cached kernels

The jiterator currently has some limitations, however. It cannot:
  - handle math on complex datatypes
  - handle kernels with scalar parameters

These improvements will likely come soon.

For examples of how to use the jiterator see the i1 and gcd kernel
implementations, which pass jittable strings implementing their
operations instead of the typical CUDA functors.

To pass a runtime argument (similar to lambda captures in non-JIT kernels),
we need to pass to additional arguments to `jitted_gpu_kernel` by value.
Currently only primitive C++ types used for computation are valid.
The order of these extra arguments should be same as the order they appear
in kernel's function signature. (look at polygamma for example)

NOTE: One big restriction being that these arguments should be after the
arguments provided by TensorIterator. Eg. While capturing `n`, where
`scalar_t x` and `scalar_t y` are provided by TensorIterator,
* foo(scalar_t x, scalar_t y, int n) works!
* foo(int n, scalar_t x, scalar_y) doesn't work
* foo(scalar_t x, int n, scalar_y) doesn't work

*/

// Entrypoint for jitted GPU kernels.
// Only handles elementwise unary and binary kernels with a
//   common dtype and a single output.
// NOTE: this assumes the op's iterator has a common_dtype.
// NOTE: We use std::tuple instead of parameter pack
//  for `extra_args` due to following
// bug on older versions of clang
// https://bugs.llvm.org/show_bug.cgi?id=23029
template <
    char const* name,
    typename return_type,
    typename f_inputs_type,
    int arity,
    typename... Args>
void jitted_gpu_kernel(
    TensorIteratorBase& iter,
    const std::string& f,
    at::cuda::jit::BinaryFuncVariant scalar_pos =
        at::cuda::jit::BinaryFuncVariant::NoScalar,
    at::opmath_type<f_inputs_type> scalar_val = 0,
    std::tuple<Args...> extra_args = std::make_tuple()) {
  // TODO: much of preamble is common to both jitted_gpu_kernel and gpu_kernel
  //   Maybe it could be refactored?
  for (int arg = 0; arg < iter.ntensors(); arg++) {
    TORCH_INTERNAL_ASSERT(
      iter.device(arg).is_cuda(),
      "argument ", arg, ": expected a CUDA device but found ", iter.device(arg));
  }

  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      jitted_gpu_kernel<name, return_type, f_inputs_type, arity>(
          sub_iter, f, scalar_pos, scalar_val, extra_args);
    }

    return;
  }

  // Computes if dynamic casting is needed
  // Dynamic casting is needed if an input's dtype differs from the common dtype
  //   or if the result dtype differs from the output's dtype
  // Note: this is intentionally divergent from calling needs_dynamic_casting,
  //   which is more general and inspects a lambda to determine if dynamic
  //   casting is needed.
  bool needs_dynamic_casting = false;

  // Checks output
  const ScalarType return_scalar_type = c10::CppTypeToScalarType<return_type>::value;
  const auto dtype0 = iter.dtype(0);
  if (dtype0 != return_scalar_type) {
    needs_dynamic_casting = true;
  }

  // Checks input(s)
  const ScalarType inputs_scalar_type = c10::CppTypeToScalarType<f_inputs_type>::value;
  for (auto i = decltype(arity){1}; i < (arity + 1); ++i) {
    const auto dtypei = iter.dtype(i);
    if (dtypei != inputs_scalar_type) {
      needs_dynamic_casting = true;
      break;
    }
  }
  if (scalar_pos == at::cuda::jit::BinaryFuncVariant::NoScalar) {
    // NOTE: With `scalar_pos=NoScalar`,`scalar_val` is not used
    // for computation in the generated code and hence we pass a dummy
    // value of `0`.
    jitted_gpu_kernel_impl<
        /*name*/ name,
        /*return_type=*/return_type,
        /*f_inputs_type=*/f_inputs_type,
        arity,
        at::cuda::jit::BinaryFuncVariant::NoScalar>(
        iter, f, needs_dynamic_casting, /*scalar_val=*/0, extra_args);
  } else if (scalar_pos == at::cuda::jit::BinaryFuncVariant::RhsScalar) {
    jitted_gpu_kernel_impl<
        /*name*/ name,
        /*return_type=*/return_type,
        /*f_inputs_type=*/f_inputs_type,
        arity,
        at::cuda::jit::BinaryFuncVariant::RhsScalar>(
        iter,
        f,
        needs_dynamic_casting,
        scalar_val,
        extra_args);

  } else {
    jitted_gpu_kernel_impl<
        /*name*/ name,
        /*return_type=*/return_type,
        /*f_inputs_type=*/f_inputs_type,
        arity,
        at::cuda::jit::BinaryFuncVariant::LhsScalar>(
        iter,
        f,
        needs_dynamic_casting,
        scalar_val,
        extra_args);
  }
}

// TODO: support runtime state capture similar to `jitted_gpu_kernel`.
template <char const *name, typename return_type, typename f_inputs_type>
void opmath_jitted_gpu_kernel_with_scalars(TensorIteratorBase& iter, const std::string& f) {
  TORCH_INTERNAL_ASSERT(iter.ntensors() == 3);
  //currently jiterator only handles binary functions where both inputs are of the same type (f_inputs_type)
  using opmath_t = at::opmath_type<f_inputs_type>;
  if (iter.is_cpu_scalar(1)) {
    auto scalar_val = iter.scalar_value<opmath_t>(1);
    iter.remove_operand(1);
    // TODO: When all kernels that use gpu_kernel_with_scalars are
    // ported to structured, this device guard can be deleted.  This
    // works around incorrect device guard generation for pre-structured
    // kernels device guards, but structured kernels do it right and
    // we can assume the device is already set correctly
    const OptionalDeviceGuard device_guard(iter.device(1));
    jitted_gpu_kernel<name, return_type, f_inputs_type, 1>(iter, f, at::cuda::jit::BinaryFuncVariant::LhsScalar, scalar_val);
  } else if (iter.is_cpu_scalar(2)) {
    auto scalar_val = iter.scalar_value<opmath_t>(2);
    iter.remove_operand(2);
    jitted_gpu_kernel<name, return_type, f_inputs_type, 1>(iter, f, at::cuda::jit::BinaryFuncVariant::RhsScalar, scalar_val);
  } else {
    jitted_gpu_kernel<name, return_type, f_inputs_type, 2>(iter, f);
  }
}

}}  // at::native

#endif // AT_USE_JITERATOR()
