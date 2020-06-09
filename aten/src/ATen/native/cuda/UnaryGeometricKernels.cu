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

// We manually overload acos because std::acos does not work with thrust::complex types.
template<typename scalar_t>
__host__ __device__ static inline scalar_t acos_wrapper(scalar_t v) {
  return ::acos(v);
}

template<typename T>
__host__ __device__ static inline thrust::complex<T> acos_wrapper(thrust::complex<T> v) {
  return thrust::acos(v);
}

void acos_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(ScalarType::Half, iter.dtype(), "acos_cuda", [&]() {
    using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
    gpu_kernel(iter, []GPU_LAMBDA(thrust_t a) -> thrust_t {
      return acos_wrapper(a);
    });
  });
}

// We manually overload asin because std::asin does not work with thrust::complex types.
template<typename scalar_t>
__host__ __device__ static inline scalar_t asin_wrapper(scalar_t v) {
  return ::asin(v);
}

template<typename T>
__host__ __device__ static inline thrust::complex<T> asin_wrapper(thrust::complex<T> v) {
  return thrust::asin(v);
}

void asin_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(ScalarType::Half, iter.dtype(), "asin_cuda", [&]() {
    using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
    gpu_kernel(iter, []GPU_LAMBDA(thrust_t a) -> thrust_t {
      return asin_wrapper(a);
    });
  });
}

// We manually overload asin because std::atan does not work with thrust::complex types.
template<typename scalar_t>
__host__ __device__ static inline scalar_t atan_wrapper(scalar_t v) {
  return ::atan(v);
}

template<typename T>
__host__ __device__ static inline thrust::complex<T> atan_wrapper(thrust::complex<T> v) {
  return thrust::atan(v);
}

void atan_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(ScalarType::Half, iter.dtype(), "atan_cuda", [&]() {
    using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
    gpu_kernel(iter, []GPU_LAMBDA(thrust_t a) -> thrust_t {
      return atan_wrapper(a);
    });
  });
}


// We manually overload sin because std::sin does not work with thrust::complex types.
template<typename scalar_t>
__host__ __device__ static inline scalar_t sin_wrapper(scalar_t v) {
  return ::sin(v);
}

template<typename T>
__host__ __device__ static inline thrust::complex<T> sin_wrapper(thrust::complex<T> v) {
  return thrust::sin(v);
}

void sin_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(ScalarType::Half, iter.dtype(), "sin_cuda", [&]() {
    using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
    gpu_kernel(iter, []GPU_LAMBDA(thrust_t a) -> thrust_t {
      return sin_wrapper(a);
    });
  });
}

template<typename scalar_t>
__host__ __device__ static inline scalar_t cos_wrapper(scalar_t v) {
  return ::cos(v);
}

template<typename T>
__host__ __device__ static inline thrust::complex<T> cos_wrapper(thrust::complex<T> v) {
  return thrust::cos(v);
}

void cos_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "cos_cuda", [&]() {
    using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
    gpu_kernel(iter, []GPU_LAMBDA(thrust_t a) -> thrust_t {
      return cos_wrapper(a);
    });
  });
}

// We manually overload sinh because std::sinh does not work with thrust::complex types.
template<typename scalar_t>
__host__ __device__ static inline scalar_t sinh_wrapper(scalar_t v) {
  return ::sinh(v);
}

template<typename T>
__host__ __device__ static inline thrust::complex<T> sinh_wrapper(thrust::complex<T> v) {
  return thrust::sinh(v);
}

void sinh_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(ScalarType::Half, iter.dtype(), "sinh_cuda", [&]() {
    using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
    gpu_kernel(iter, []GPU_LAMBDA(thrust_t a) -> thrust_t {
      return sinh_wrapper(a);
    });
  });
}

// We manually overload cosh because std::cosh does not work with thrust::complex types.
template<typename scalar_t>
__host__ __device__ static inline scalar_t cosh_wrapper(scalar_t v) {
  return ::cosh(v);
}

template<typename T>
__host__ __device__ static inline thrust::complex<T> cosh_wrapper(thrust::complex<T> v) {
  return thrust::cosh(v);
}

void cosh_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(ScalarType::Half, iter.dtype(), "cosh_cuda", [&]() {
    using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
    gpu_kernel(iter, []GPU_LAMBDA(thrust_t a) -> thrust_t {
      return cosh_wrapper(a);
    });
  });
}

template<typename scalar_t>
__host__ __device__ static inline scalar_t tanh_wrapper(scalar_t v) {
  return ::tanh(v);
}

template<typename T>
__host__ __device__ static inline thrust::complex<T> tanh_wrapper(thrust::complex<T> v) {
  return thrust::tanh(v);
}

void tanh_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "tanh_cuda", [&]() {
    AT_SKIP_BFLOAT16_IF_NOT_ROCM(scalar_t, "tanh_cuda", [&] {
      using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
      gpu_kernel(iter, []GPU_LAMBDA(thrust_t a) -> thrust_t {
        return tanh_wrapper(a);
      });
    });
  });
}

void acosh_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "acosh_cuda", [&]() {
    AT_SKIP_BFLOAT16_IF_NOT_ROCM(scalar_t, "acosh_cuda", [&] {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
              return ::acosh(a);
      });
    });
  });
}

void asinh_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "asinh_cuda", [&]() {
    AT_SKIP_BFLOAT16_IF_NOT_ROCM(scalar_t, "asinh_cuda", [&] {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
              return ::asinh(a);
      });
    });
  });
}

void atanh_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "atanh_cuda", [&]() {
    AT_SKIP_BFLOAT16_IF_NOT_ROCM(scalar_t, "atanh_cuda", [&] {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
              return ::atanh(a);
      });
    });
  });
}

template<typename scalar_t>
__host__ __device__ static inline scalar_t tan_wrapper(scalar_t v) {
  return ::tan(v);
}

template<typename T>
__host__ __device__ static inline thrust::complex<T> tan_wrapper(thrust::complex<T> v) {
  return thrust::tan(v);
}

void tan_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(ScalarType::Half, iter.dtype(), "tan_cuda", [&]() {
    using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
    gpu_kernel(iter, []GPU_LAMBDA(thrust_t a) -> thrust_t {
      return tan_wrapper(a);
    });
  });
}

REGISTER_DISPATCH(acos_stub, &acos_kernel_cuda);
REGISTER_DISPATCH(acosh_stub, &acosh_kernel_cuda);
REGISTER_DISPATCH(asinh_stub, &asinh_kernel_cuda);
REGISTER_DISPATCH(atanh_stub, &atanh_kernel_cuda);
REGISTER_DISPATCH(asin_stub, &asin_kernel_cuda);
REGISTER_DISPATCH(atan_stub, &atan_kernel_cuda);
REGISTER_DISPATCH(sin_stub, &sin_kernel_cuda);
REGISTER_DISPATCH(cos_stub, &cos_kernel_cuda);
REGISTER_DISPATCH(sinh_stub, &sinh_kernel_cuda);
REGISTER_DISPATCH(cosh_stub, &cosh_kernel_cuda);
REGISTER_DISPATCH(tanh_stub, &tanh_kernel_cuda);
REGISTER_DISPATCH(tan_stub, &tan_kernel_cuda);

}} // namespace at::native
