#define TORCH_ASSERT_NO_OPERATORS
#define _USE_MATH_DEFINES

#include <ATen/native/Activation.h>

#include <cmath>

#include <thrust/tuple.h>

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/core/TensorBase.h>
#include <c10/core/Scalar.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <ATen/cuda/ApplyGridUtils.cuh>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/native/cuda/Loops.cuh>

namespace at::native {
namespace {

void softplus_kernel(
    TensorIteratorBase& iter,
    const Scalar& beta_,
    const Scalar& threshold_) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "softplus_cuda",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        auto beta = beta_.to<opmath_t>();
        auto threshold = threshold_.to<opmath_t>();
        gpu_kernel(iter, [beta, threshold] GPU_LAMBDA(scalar_t a) -> scalar_t {
          opmath_t aop = static_cast<opmath_t>(a);
          opmath_t y = aop * beta;
          if (y > threshold) {
            return aop;
          }
          // Use numerically stable formula: exp(-abs(y)) never overflows
          // For y > 0: log1p(exp(-abs(y))) / beta + a
          // For y <= 0: log1p(exp(-abs(y))) / beta + 0
          opmath_t z = std::log1p(std::exp(-std::abs(y))) / beta;
          return (y > opmath_t(0) ? aop : opmath_t(0)) + z;
        });
      });
}

void softplus_backward_kernel(
    TensorIteratorBase& iter,
    const Scalar& beta_,
    const Scalar& threshold_) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "softplus_backward_cuda",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        auto beta = beta_.to<opmath_t>();
        auto threshold = threshold_.to<opmath_t>();
        gpu_kernel(
            iter,
            [beta, threshold] GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
              opmath_t aop = static_cast<opmath_t>(a);
              opmath_t bop = static_cast<opmath_t>(b);
              opmath_t y = bop * beta;
              if (y > threshold) {
                // sigmoid approaches 1
                return aop;
              }
              // Use numerically stable formula: grad / (1 + exp(-y))
              // exp(-y) won't overflow for y >= 0, and for y < 0 this is still stable
              return aop / (opmath_t(1.) + std::exp(-y));
            });
      });
}

} // namespace

REGISTER_DISPATCH(softplus_stub, &softplus_kernel)
REGISTER_DISPATCH(softplus_backward_stub, &softplus_backward_kernel)

} // namespace at::native
