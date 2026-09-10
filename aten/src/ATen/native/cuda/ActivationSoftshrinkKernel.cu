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
#include <ATen/NumericUtils.h>
#include <ATen/cuda/ApplyGridUtils.cuh>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/native/cuda/Loops.cuh>

namespace at::native {
namespace {

void softshrink_kernel(TensorIteratorBase& iter, const Scalar& value) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "softshrink_cuda",
      [&]() {
        auto lambd = value.to<scalar_t>();
        gpu_kernel(iter, [lambd] GPU_LAMBDA(scalar_t a) -> scalar_t {
          return at::_isnan(a) ? a : (a > lambd ? a - lambd : (a < -lambd ? a + lambd : scalar_t(0)));
        });
      });
}

void shrink_backward_kernel(TensorIteratorBase& iter, const Scalar& value) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "shrink_backward_cuda",
      [&]() {
        auto lambd = value.to<scalar_t>();
        gpu_kernel(
            iter,
            [lambd] GPU_LAMBDA(
                scalar_t grad_val, scalar_t self_val) -> scalar_t {
              return (self_val >= -lambd && self_val <= lambd) ? scalar_t(0)
                                                               : grad_val;
            });
      });
}
} // namespace

REGISTER_DISPATCH(softshrink_stub, &softshrink_kernel)
REGISTER_DISPATCH(shrink_backward_stub, &shrink_backward_kernel)

} // namespace at::native
