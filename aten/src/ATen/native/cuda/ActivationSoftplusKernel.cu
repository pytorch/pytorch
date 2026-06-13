#define TORCH_ASSERT_NO_OPERATORS
#define _USE_MATH_DEFINES

#include <ATen/native/Activation.h>

#include <cmath>
#include <limits>

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
          if ((aop * beta) > threshold) {
            return a; 
          }
          opmath_t result = (::log1p(std::exp(aop * beta))) / beta;
          if constexpr (!std::is_same_v<scalar_t, opmath_t>) {
            opmath_t max_val = static_cast<opmath_t>(std::numeric_limits<scalar_t>::max());
            // Clamp to max_val to prevent silent overflow on downcast.
            // Note: If result is NaN, IEEE 754 rules dictate that (NaN > max_val) 
            // evaluates to false. The ternary safely falls through and returns NaN.
            return static_cast<scalar_t>(result > max_val ? max_val : result);
          } else {
            // For full-precision types, opmath_t == scalar_t. 
            // We return result directly to avoid a cast.
            return result;
          }
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
              opmath_t z = std::exp(bop * beta);
              return (bop * beta) > threshold ? aop
                                              : aop * z / (z + opmath_t(1.));
            });
      });
}

} // namespace

REGISTER_DISPATCH(softplus_stub, &softplus_kernel)
REGISTER_DISPATCH(softplus_backward_stub, &softplus_backward_kernel)

} // namespace at::native
