#define TORCH_ASSERT_NO_OPERATORS
#include <limits>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/cuda/jit_utils.h>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Math.cuh>

namespace at::native {

#if AT_USE_JITERATOR()
constexpr char log_name[] = "log_kernel";
#endif

void log_kernel_cuda(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (at::isComplexType(common_dtype)) {
#if AT_USE_JITERATOR()
    static const auto log_string = jiterator_stringify(
        template <typename T> T log_kernel(T x) { return ::log(x); });
    AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, common_dtype, "log_cuda", [&]() {
      jitted_gpu_kernel<
          /*name=*/log_name,
          /*return_dtype=*/scalar_t,
          /*common_dtype=*/scalar_t,
          /*arity=*/1>(iter, log_string);
    });
#else
    AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, iter.common_dtype(), "log_cuda", [&]() {
      gpu_kernel(
          iter, [] GPU_LAMBDA(scalar_t a) -> scalar_t {
            using opmath_t = at::opmath_type<scalar_t>;
            return ::log(static_cast<opmath_t>(a));
          });
    });
#endif
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.common_dtype(), "log_cuda", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
        return ::log(a);
      });
    });
  }
}

constexpr char log10_name[] = "log10_kernel";
void log10_kernel_cuda(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (at::isComplexType(common_dtype)) {
#if AT_USE_JITERATOR()
    static const auto log10_string = jiterator_stringify(
        template <typename T> T log10_kernel(T x) { return ::log10(x); });
    AT_DISPATCH_COMPLEX_TYPES(common_dtype, "log10_cuda", [&]() {
      jitted_gpu_kernel<
          /*name=*/log10_name,
          /*return_dtype=*/scalar_t,
          /*common_dtype=*/scalar_t,
          /*arity=*/1>(iter, log10_string);
    });
#else
    AT_DISPATCH_COMPLEX_TYPES(iter.common_dtype(), "log10_cuda", [&]() {
      gpu_kernel(
          iter, [] GPU_LAMBDA(scalar_t a) -> scalar_t { return ::log10(a); });
    });
#endif
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.common_dtype(), "log10_cuda", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
        return ::log10(a);
      });
    });
  }
}

void log1p_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.common_dtype(), "log1p_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return ::log1p(a);
    });
  });
}

constexpr char log2_name[] = "log2_kernel";
void log2_kernel_cuda(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (at::isComplexType(common_dtype)) {
#if AT_USE_JITERATOR()
    static const auto log2_string = jiterator_stringify(
        template <typename T> T log2_kernel(T x) { return ::log2(x); });
    AT_DISPATCH_COMPLEX_TYPES(common_dtype, "log2_cuda", [&]() {
      jitted_gpu_kernel<
          /*name=*/log2_name,
          /*return_dtype=*/scalar_t,
          /*common_dtype=*/scalar_t,
          /*arity=*/1>(iter, log2_string);
    });
#else
    AT_DISPATCH_COMPLEX_TYPES(iter.common_dtype(), "log2_cuda", [&]() {
      gpu_kernel(
          iter, [] GPU_LAMBDA(scalar_t a) -> scalar_t { return ::log2(a); });
    });
#endif
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.common_dtype(), "log2_cuda", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
        return ::log2(a);
      });
    });
  }
}

REGISTER_DISPATCH(log_stub, &log_kernel_cuda)
REGISTER_DISPATCH(log10_stub, &log10_kernel_cuda)
REGISTER_DISPATCH(log2_stub, &log2_kernel_cuda)
REGISTER_DISPATCH(log1p_stub, &log1p_kernel_cuda)

} // namespace at::native
