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

void hardswish_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "hardswish_cuda", [&]() {
    using opmath_t = at::opmath_type<scalar_t>;
    const opmath_t zero(0.0f);
    const opmath_t one_sixth(1.0f / 6.0f);
    const opmath_t three(3.0f);
    const opmath_t six(6.0f);
    gpu_kernel(iter, [zero, one_sixth, three, six]GPU_LAMBDA(scalar_t self_val) -> scalar_t {
      opmath_t x = static_cast<opmath_t>(self_val);
      return x * std::min(std::max(x + three, zero), six) * one_sixth;
    });
  });
}

void hardswish_backward_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "hardswish_backward_cuda", [&]() {
    using opmath_t = at::opmath_type<scalar_t>;
    const opmath_t zero(0.0f);
    const opmath_t three(3.0f);
    const opmath_t neg_three(-3.0f);
    const opmath_t one_half(0.5f);
    gpu_kernel(
      iter,
      [zero, three, neg_three, one_half]GPU_LAMBDA(scalar_t grad_val_, scalar_t self_val_) -> scalar_t {
        opmath_t grad_val = static_cast<opmath_t>(grad_val_);
        opmath_t self_val = static_cast<opmath_t>(self_val_);
        if (self_val < neg_three) {
          return zero;
        } else if (self_val <= three) {
          return grad_val * ((self_val / three) + one_half);
        } else {
          return grad_val;
        }
    });
  });
}
} // namespace

REGISTER_DISPATCH(hardswish_stub, &hardswish_kernel);
REGISTER_DISPATCH(hardswish_backward_stub, &hardswish_backward_kernel);

} // namespace at::native
