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
          return (aop * beta) > threshold
              ? aop
              : (::log1p(std::exp(aop * beta))) / beta;
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

REGISTER_DISPATCH(softplus_stub, &softplus_kernel);
REGISTER_DISPATCH(softplus_backward_stub, &softplus_backward_kernel);

} // namespace at::native
