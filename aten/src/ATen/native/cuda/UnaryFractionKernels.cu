#include <limits>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/AccumulateType.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Math.cuh>
#include <ATen/native/cuda/zmath.cuh>

namespace at { namespace native {

// We manually overload ceil because std::ceil does not work with std::complex types.
template <typename scalar_t>
__host__ __device__ static inline scalar_t ceil_wrapper(scalar_t a) {
  return std::ceil(a);
}

template<typename T>
__host__ __device__ static inline std::complex<T> ceil_wrapper(std::complex<T> v) {
  return std::complex<T>(std::ceil(v.real()), std::ceil(v.imag()));
}

void ceil_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(ScalarType::Half, iter.dtype(), "ceil_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return ceil_wrapper(a);
    });
  });
}

void frac_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "frac_cuda", [&]() {
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

void floor_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(ScalarType::Half, iter.dtype(), "floor_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return floor_wrapper(a);
    });
  });
}

void reciprocal_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "reciprocal_cuda", [&]() {
    using acc_t = acc_type<scalar_t, /*is_cuda=*/true>;
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return static_cast<acc_t>(1) / a;
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

__host__ __device__ static inline std::complex<float> nearbyint_wrapper(std::complex<float> a) {
  return std::complex<float>(::nearbyintf(static_cast<float>(a.real())), ::nearbyintf(static_cast<float>(a.imag())));
}

__host__ __device__ static inline std::complex<double> nearbyint_wrapper(std::complex<double> a) {
  return std::complex<double>(::nearbyint(static_cast<double>(a.real())), ::nearbyint(static_cast<double>(a.imag())));
}

void round_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(ScalarType::Half, iter.dtype(), "round_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      // We do not use std::round because we would like to round midway numbers to the nearest even integer.
      return nearbyint_wrapper(a);
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

__host__ __device__ static inline std::complex<float> trunc_wrapper(std::complex<float> a) {
  return std::complex<float>(::truncf(static_cast<float>(a.real())), ::truncf(static_cast<float>(a.imag())));
}

__host__ __device__ static inline std::complex<double> trunc_wrapper(std::complex<double> a) {
  return std::complex<double>(::trunc(static_cast<double>(a.real())), ::trunc(static_cast<double>(a.imag())));
}

void trunc_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(ScalarType::Half, iter.dtype(), "trunc_cuda", [&]() {
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
REGISTER_DISPATCH(trunc_stub, &trunc_kernel_cuda);

}} // namespace at::native
