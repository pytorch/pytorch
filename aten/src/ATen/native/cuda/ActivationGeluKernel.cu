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

void GeluCUDAKernelImpl(TensorIteratorBase& it, GeluType approximate) {
  if (approximate == GeluType::Tanh) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, it.dtype(), "GeluCUDAKernelImpl", [&]() {
      gpu_kernel(it, [] GPU_LAMBDA(scalar_t x) -> scalar_t {
        using opmath_t = at::opmath_type<scalar_t>;
        constexpr opmath_t pointfive = opmath_t(0.5);
        constexpr opmath_t one = opmath_t(1);
        constexpr opmath_t kBeta = M_SQRT2 * M_2_SQRTPI * opmath_t(0.5);
        constexpr opmath_t kKappa = 0.044715;
        auto x_cube = static_cast<opmath_t>(x) * static_cast<opmath_t>(x) * static_cast<opmath_t>(x);
        auto inner = kBeta * (static_cast<opmath_t>(x) + kKappa * x_cube);
        return pointfive * static_cast<opmath_t>(x) * (one + c10::cuda::compat::tanh(inner));
      });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, it.dtype(), "GeluCUDAKernelImpl", [&]() {
      gpu_kernel(it, [] GPU_LAMBDA(scalar_t x) -> scalar_t {
        using opmath_t = at::opmath_type<scalar_t>;
        constexpr opmath_t kAlpha = M_SQRT1_2;
        constexpr opmath_t pointfive = opmath_t(0.5);
        constexpr opmath_t one = opmath_t(1);
        return static_cast<opmath_t>(x) * pointfive * (one + ::erf(static_cast<opmath_t>(x) * kAlpha));
      });
    });
  }
}

void GeluBackwardCUDAKernelImpl(TensorIteratorBase& it, GeluType approximate) {
  if (approximate == GeluType::Tanh) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        it.dtype(), "GeluBackwardCUDAKernelImpl", [&]() {
          gpu_kernel(it, [] GPU_LAMBDA(scalar_t dy, scalar_t x) -> scalar_t {
            using opmath_t = at::opmath_type<scalar_t>;
            constexpr opmath_t pointfive = opmath_t(0.5);
            constexpr opmath_t one = opmath_t(1);
            constexpr opmath_t three = opmath_t(3);
            constexpr opmath_t kBeta = M_SQRT2 * M_2_SQRTPI * pointfive;
            constexpr opmath_t kKappa = 0.044715;
            auto x_sq = static_cast<opmath_t>(x) * static_cast<opmath_t>(x);
            auto x_cube = x_sq * static_cast<opmath_t>(x);
            auto inner = kBeta * (static_cast<opmath_t>(x) + kKappa * x_cube);
            auto tanh_inner = c10::cuda::compat::tanh(inner);

            auto left = pointfive * static_cast<opmath_t>(x);
            auto right = one + tanh_inner;

            auto left_derivative = pointfive * right;

            auto tanh_derivative = one - tanh_inner * tanh_inner;
            auto inner_derivative = kBeta * (one + three * kKappa * x_sq);
            auto right_derivative = left * tanh_derivative * inner_derivative;

            return static_cast<opmath_t>(dy) * (left_derivative + right_derivative);
        });
      });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        it.dtype(), "GeluBackwardCUDAKernelImpl", [&]() {
          gpu_kernel(it, [] GPU_LAMBDA(scalar_t dy, scalar_t x) -> scalar_t {
            using opmath_t = at::opmath_type<scalar_t>;
            constexpr opmath_t minuspointfive = opmath_t(-0.5);
            constexpr opmath_t pointfive = opmath_t(0.5);
            constexpr opmath_t one = opmath_t(1);
            constexpr opmath_t kBeta = M_2_SQRTPI * M_SQRT1_2 * pointfive;
            constexpr opmath_t kAlpha = M_SQRT1_2;
            const opmath_t cdf =
                pointfive * (one + ::erf(static_cast<opmath_t>(x) * kAlpha));
            const opmath_t pdf =
                c10::cuda::compat::exp(
                    minuspointfive * static_cast<opmath_t>(x) * static_cast<opmath_t>(x)) *
                kBeta;
            return static_cast<opmath_t>(dy) * (cdf + static_cast<opmath_t>(x) * pdf);
          });
        });
  }
}

} // namespace at::native
