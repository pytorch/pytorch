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

// -----------------------------------
// log_sigmoid forward
// -----------------------------------

void launch_log_sigmoid_forward_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, iter.common_dtype(), "log_sigmoid_forward_cuda", [&] {
        using opmath_t = at::opmath_type<scalar_t>;

        gpu_kernel(iter, [] GPU_LAMBDA(scalar_t in_) -> scalar_t {
          const opmath_t in = in_;
          const auto min = std::min(opmath_t(0), in);
          const auto z = std::exp(-std::abs(in));
          return min - std::log1p(z);
        });
      });
}

namespace {
// -----------------------------------
// log_sigmoid backward
// -----------------------------------
void log_sigmoid_backward_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, iter.common_dtype(), "log_sigmoid_backward_cuda", [&] {
        using opmath_t = at::opmath_type<scalar_t>;
        gpu_kernel(
            iter, [] GPU_LAMBDA(scalar_t in_, scalar_t grad_out_) -> scalar_t {
              const opmath_t in = in_;
              const opmath_t grad_out = grad_out_;

              auto in_negative = in < opmath_t(0);
              auto max_deriv = in_negative ? opmath_t(1) : opmath_t(0);
              auto sign = in_negative ? opmath_t(1) : -opmath_t(1);
              const auto z = std::exp(-std::abs(in));
              return grad_out * (max_deriv - sign * (z / (opmath_t(1) + z)));
            });
      });
}
} // namespace

REGISTER_DISPATCH(log_sigmoid_backward_stub, &log_sigmoid_backward_kernel);

} // namespace at::native
