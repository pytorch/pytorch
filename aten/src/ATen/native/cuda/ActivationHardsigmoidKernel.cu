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

void hardsigmoid_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "hardsigmoid_cuda",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        const opmath_t zero(0.0f);
        const opmath_t one_sixth(1.0f / 6.0f);
        const opmath_t three(3.0f);
        const opmath_t six(6.0f);
        gpu_kernel(
            iter,
            [zero, one_sixth, three, six] GPU_LAMBDA(
                scalar_t self_val) -> scalar_t {
              opmath_t x = static_cast<opmath_t>(self_val);
              return std::min(std::max(x + three, zero), six) * one_sixth;
            });
      });
}

void hardsigmoid_backward_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "hardsigmoid_backward_cuda",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        const opmath_t zero(0.0f);
        const opmath_t three(3.0f);
        const opmath_t neg_three(-3.0f);
        const opmath_t one_sixth(1.0f / 6.0f);
        gpu_kernel(
            iter,
            [zero, three, neg_three, one_sixth] GPU_LAMBDA(
                scalar_t grad_val_, scalar_t self_val_) -> scalar_t {
              opmath_t grad_val = static_cast<opmath_t>(grad_val_);
              opmath_t self_val = static_cast<opmath_t>(self_val_);
              return (self_val > neg_three && self_val < three)
                  ? grad_val * one_sixth
                  : zero;
            });
      });
}

} // namespace

REGISTER_DISPATCH(hardsigmoid_stub, &hardsigmoid_kernel)
REGISTER_DISPATCH(hardsigmoid_backward_stub, &hardsigmoid_backward_kernel)

} // namespace at::native
