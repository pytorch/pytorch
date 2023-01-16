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

void mish_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "mish_cuda",
      [&]() {
        gpu_kernel(iter, [] GPU_LAMBDA(scalar_t x) -> scalar_t {
          using opmath_t = at::opmath_type<scalar_t>;
          const opmath_t x_acc = static_cast<opmath_t>(x);
          return x_acc *
              c10::cuda::compat::tanh(
                     c10::cuda::compat::log1p(c10::cuda::compat::exp(x_acc)));
        });
      });
}

void mish_backward_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "mish_backward_cuda",
      [&]() {
        gpu_kernel(iter, [] GPU_LAMBDA(scalar_t dy, scalar_t x) -> scalar_t {
          using opmath_t = at::opmath_type<scalar_t>;
          const opmath_t dy_acc = static_cast<opmath_t>(dy);
          const opmath_t x_acc = static_cast<opmath_t>(x);
          const opmath_t s_acc =
              opmath_t(1) / (opmath_t(1) + c10::cuda::compat::exp(-x_acc));
          const opmath_t t_acc = c10::cuda::compat::tanh(
              c10::cuda::compat::log1p(c10::cuda::compat::exp(x_acc)));
          return dy_acc *
              (t_acc + x_acc * s_acc * (opmath_t(1) - t_acc * t_acc));
        });
      });
}
} // namespace

REGISTER_DISPATCH(mish_stub, &mish_kernel);
REGISTER_DISPATCH(mish_backward_stub, &mish_backward_kernel);

} // namespace at::native
