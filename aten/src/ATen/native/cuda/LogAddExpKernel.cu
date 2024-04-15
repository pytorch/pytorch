#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/OpMathType.h>
#include <c10/util/MathConstants.h>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at::native {

void logaddexp_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::BFloat16, ScalarType::Half,
      iter.dtype(), "logaddexp_cuda",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        gpu_kernel(iter, [] GPU_LAMBDA (scalar_t a_, scalar_t b_) -> scalar_t {
          const auto a = static_cast<opmath_t>(a_);
          const auto b = static_cast<opmath_t>(b_);
          if (::isinf(a) && a == b) {
            return a;
          } else {
            const auto m = ::max(a, b);
            return m + ::log1p(::exp(-::abs(a - b)));
          }
        });
      });
}

void logaddexp2_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::BFloat16, ScalarType::Half,
      iter.dtype(), "logaddexp2_cuda",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        const auto inv_log_2 = static_cast<opmath_t>(1.0 / c10::ln_2<double>);
        gpu_kernel(iter, [inv_log_2] GPU_LAMBDA (scalar_t a_, scalar_t b_) -> scalar_t {
          const auto a = static_cast<opmath_t>(a_);
          const auto b = static_cast<opmath_t>(b_);
          if (::isinf(a) && a == b) {
            return a;
          } else {
            const auto m = ::max(a, b);
            return m + ::log1p(::exp2(-::abs(a - b))) * inv_log_2;
          }
        });
      });
}

REGISTER_DISPATCH(logaddexp_stub, &logaddexp_kernel_cuda);
REGISTER_DISPATCH(logaddexp2_stub, &logaddexp2_kernel_cuda);

} // namespace at::native
