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

// We manually overload angle because std::angle does not work with std::thrust::complex types.
template<typename scalar_t>
__host__ __device__ static inline scalar_t angle_wrapper(scalar_t v) {
  return 0;
}

template<typename T>
__host__ __device__ static inline thrust::complex<T> angle_wrapper(thrust::complex<T> v) {
  return thrust::arg(v);
}

void angle_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(iter.dtype(), "angle_cuda", [&]() {
    using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
    gpu_kernel(iter, []GPU_LAMBDA(thrust_t a) -> thrust_t {
      return angle_wrapper(a);
    });
  });
}

// We manually overload real because std::real does not work with std::thrust::complex types.
template<typename scalar_t>
__host__ __device__ static inline scalar_t real_wrapper(scalar_t v) {
  return v;
}

template<typename T>
__host__ __device__ static inline thrust::complex<T> real_wrapper(thrust::complex<T> v) {
  return v.real();
}

void real_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(iter.dtype(), "real_cuda", [&]() {
    using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
    gpu_kernel(iter, []GPU_LAMBDA(thrust_t a) -> thrust_t {
      return real_wrapper(a);
    });
  });
}

// We manually overload imag because std::imag does not work with std::thrust::complex types.
template<typename scalar_t>
__host__ __device__ static inline scalar_t imag_wrapper(scalar_t v) {
  return 0;
}

template<typename T>
__host__ __device__ static inline thrust::complex<T> imag_wrapper(thrust::complex<T> v) {
  return v.imag();
}

void imag_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(iter.dtype(), "imag_cuda", [&]() {
    using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
    gpu_kernel(iter, []GPU_LAMBDA(thrust_t a) -> thrust_t {
      return imag_wrapper(a);
    });
  });
}

// We manually overload conj because std::conj does not work with std::thrust::complex types.
template<typename scalar_t>
__host__ __device__ static inline scalar_t conj_wrapper(scalar_t v) {
  return v;
}

template<typename T>
__host__ __device__ static inline thrust::complex<T> conj_wrapper(thrust::complex<T> v) {
  return thrust::conj(v);
}

void conj_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(iter.dtype(), "conj_cuda", [&]() {
    using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
    gpu_kernel(iter, []GPU_LAMBDA(thrust_t a) -> thrust_t {
      return conj_wrapper(a);
    });
  });
}

REGISTER_DISPATCH(angle_stub, &angle_kernel_cuda);
REGISTER_DISPATCH(real_stub, &real_kernel_cuda);
REGISTER_DISPATCH(imag_stub, &imag_kernel_cuda);
REGISTER_DISPATCH(conj_stub, &conj_kernel_cuda);

}} // namespace at::native
