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

// We manually overload log because std::log does not work with thrust::complex types.
template<typename scalar_t>
__host__ __device__ static inline scalar_t log_wrapper(scalar_t v) {
  return ::log(v);
}

template<typename T>
__host__ __device__ static inline thrust::complex<T> log_wrapper(thrust::complex<T> v) {
  return thrust::log(v);
}

void log_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "log_cuda", [&]() {
    AT_SKIP_BFLOAT16_IF_NOT_ROCM(scalar_t, "log_cuda", [&] {
      using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
      gpu_kernel(iter, []GPU_LAMBDA(thrust_t a) -> thrust_t {
        return log_wrapper(a);
      });
    });
  });
}

// We manually overload log10 because std::log10 does not work with thrust::complex types.
template<typename scalar_t>
__host__ __device__ static inline scalar_t log10_wrapper(scalar_t v) {
  return ::log10(v);
}

template<typename T>
__host__ __device__ static inline thrust::complex<T> log10_wrapper(thrust::complex<T> v) {
  return thrust::log10(v);
}

void log10_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "log10_cuda", [&]() {
    AT_SKIP_BFLOAT16_IF_NOT_ROCM(scalar_t, "log10_cuda", [&] {
      using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
      gpu_kernel(iter, []GPU_LAMBDA(thrust_t a) -> thrust_t {
        return log10_wrapper(a);
      });
    });
  });
}

void log1p_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "log1p_cuda", [&]() {
    AT_SKIP_BFLOAT16_IF_NOT_ROCM(scalar_t, "log1p_cuda", [&] {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
        return ::log1p(a);
      });
    });
  });
}

// We manually overload log2 because std::log2 does not work with thrust::complex types.
template<typename scalar_t>
__host__ __device__ static inline scalar_t log2_wrapper(scalar_t v) {
  return ::log2(v);
}

template<typename T>
__host__ __device__ static inline thrust::complex<T> log2_wrapper(thrust::complex<T> v) {
  const thrust::complex<T> log2 = thrust::complex<T>(::log(2.0), 0.0);
  return thrust::log(v)/log2;
}

void log2_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "log2_cuda", [&]() {
    AT_SKIP_BFLOAT16_IF_NOT_ROCM(scalar_t, "log2_cuda", [&] {
      using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
      gpu_kernel(iter, []GPU_LAMBDA(thrust_t a) -> thrust_t {
        return log2_wrapper(a);
      });
    });
  });
}

REGISTER_DISPATCH(log_stub, &log_kernel_cuda);
REGISTER_DISPATCH(log10_stub, &log10_kernel_cuda);
REGISTER_DISPATCH(log2_stub, &log2_kernel_cuda);
REGISTER_DISPATCH(log1p_stub, &log1p_kernel_cuda);

}} // namespace at::native
