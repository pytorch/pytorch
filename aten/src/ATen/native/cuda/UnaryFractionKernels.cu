#define TORCH_ASSERT_NO_OPERATORS
#include <limits>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Math.cuh>

namespace at::native {

// We manually overload ceil because std::ceil does not work with std::complex types.
template <typename scalar_t>
__host__ __device__ static inline scalar_t ceil_wrapper(scalar_t a) {
  return std::ceil(a);
}

template<typename T>
__host__ __device__ static inline std::complex<T> ceil_wrapper(std::complex<T> v) {
  return std::complex<T>(std::ceil(v.real()), std::ceil(v.imag()));
}

void ceil_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      iter.dtype(), "ceil_cuda",
      [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
          return ceil_wrapper(a);
        });
      });
}

void frac_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      iter.dtype(), "frac_cuda",
      [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
          return a - ::trunc(a);
        });
      });
}

// We manually overload floor because std::floor does not work with std::complex types.
template <typename scalar_t>
__host__ __device__ static inline scalar_t floor_wrapper(scalar_t a) {
  return std::floor(a);
}

template<typename T>
__host__ __device__ static inline std::complex<T> floor_wrapper(std::complex<T> v) {
  return std::complex<T>(std::floor(v.real()), std::floor(v.imag()));
}

void floor_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      iter.dtype(), "floor_cuda",
      [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
          return floor_wrapper(a);
        });
      });
}

template <typename scalar_t>
__host__ __device__ static inline scalar_t reciprocal_wrapper(scalar_t a) {
  return static_cast<scalar_t>(1)/a;
}

template<typename T>
__host__ __device__ static inline c10::complex<T> reciprocal_wrapper(c10::complex<T> v) {
  // Handle extreme cases for numpy compatibility
  auto both_inf = [](T real, T imag) {
    return (::isinf(real) && ::isinf(imag));
  };

  auto either_inf = [](T real, T imag) {
    return ::isinf(real) || ::isinf(imag);
  };

  auto either_nan = [](T real, T imag) {
    return ::isnan(real) || ::isnan(imag);
  };

  if (either_nan(v.real(), v.imag()) || both_inf(v.real(), v.imag())) {
    // If either is Nan or both are infinite, return {nan, nan}
    return {std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN()};
  } else if (either_inf(v.real(), v.imag())) {
    // If either is Inf, return {0, 0}
    return {0, 0};
  }
  const c10::complex<T> one = c10::complex<T>(1.0, 0);
  return one/v;
}

void reciprocal_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      iter.common_dtype(), "reciprocal_cuda",
      [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
          return reciprocal_wrapper(a);
        });
      });
}

// We manually overload nearbyint because std::nearbyint does not work with std::complex types and ROCm.
template <typename scalar_t>
__host__ __device__ static inline scalar_t nearbyint_wrapper(scalar_t a) {
  return static_cast<scalar_t>(::nearbyintf(static_cast<float>(a)));
}

__host__ __device__ static inline double nearbyint_wrapper(double a) {
  return ::nearbyint(a);
}

#pragma push
#pragma nv_diag_suppress 177   // Function was declared but never referenced
__host__ __device__ static inline c10::complex<float> nearbyint_wrapper(c10::complex<float> a) {
  return c10::complex<float>(::nearbyintf(static_cast<float>(a.real())), ::nearbyintf(static_cast<float>(a.imag())));
}

__host__ __device__ static inline c10::complex<double> nearbyint_wrapper(c10::complex<double> a) {
  return c10::complex<double>(::nearbyint(static_cast<double>(a.real())), ::nearbyint(static_cast<double>(a.imag())));
}
#pragma pop

void round_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      iter.dtype(), "round_cuda",
      [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
          // We do not use std::round because we would like to round midway numbers to the nearest even integer.
          return nearbyint_wrapper(a);
        });
      });
}

void round_decimals_kernel_cuda(TensorIteratorBase& iter, int64_t decimals) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      iter.dtype(), "round_cuda",
      [&]() {
        bool neg_flag = false;
        scalar_t ten_pow_decimals;
        if (decimals < 0) {
          decimals = -decimals;
          neg_flag = true;
        }
        ten_pow_decimals = static_cast<scalar_t>(std::pow(10, decimals));
        gpu_kernel(iter, [ten_pow_decimals, neg_flag]GPU_LAMBDA(scalar_t a) -> scalar_t {
          return neg_flag ? std::nearbyint(a / ten_pow_decimals) * ten_pow_decimals
                          : std::nearbyint(a * ten_pow_decimals) / ten_pow_decimals;
        });
      });
}

// We manually overload trunc because std::trunc does not work with std::complex types and ROCm.
template <typename scalar_t>
__host__ __device__ static inline scalar_t trunc_wrapper(scalar_t a) {
  return static_cast<scalar_t>(::truncf(static_cast<float>(a)));
}

__host__ __device__ static inline double trunc_wrapper(double a) {
  return ::trunc(a);
}

__host__ __device__ static inline c10::complex<float> trunc_wrapper(c10::complex<float> a) {
  return c10::complex<float>(::truncf(static_cast<float>(a.real())), ::truncf(static_cast<float>(a.imag())));
}

__host__ __device__ static inline c10::complex<double> trunc_wrapper(c10::complex<double> a) {
  return c10::complex<double>(::trunc(static_cast<double>(a.real())), ::trunc(static_cast<double>(a.imag())));
}

void trunc_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      iter.dtype(), "trunc_cuda",
      [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
          return trunc_wrapper(a);
        });
      });
}

REGISTER_DISPATCH(ceil_stub, &ceil_kernel_cuda);
REGISTER_DISPATCH(frac_stub, &frac_kernel_cuda);
REGISTER_DISPATCH(floor_stub, &floor_kernel_cuda);
REGISTER_DISPATCH(reciprocal_stub, &reciprocal_kernel_cuda);
REGISTER_DISPATCH(round_stub, &round_kernel_cuda);
REGISTER_DISPATCH(round_decimals_stub, &round_decimals_kernel_cuda);
REGISTER_DISPATCH(trunc_stub, &trunc_kernel_cuda);

} // namespace at::native
