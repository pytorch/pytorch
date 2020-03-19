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

// We manually overload abs because std::abs does not work with thrust::complex types and ROCm.
template<typename scalar_t>
__host__ __device__ static inline scalar_t abs_wrapper(scalar_t v) {
  return ::abs(v);
}

template<typename T>
__host__ __device__ static inline thrust::complex<T> abs_wrapper(thrust::complex<T> v) {
  return thrust::abs(v);
}

__host__ __device__ static inline uint8_t abs_wrapper(uint8_t v) {
  return v;
}

__host__ __device__ static inline bool abs_wrapper(bool v) {
  return v;
}

void abs_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(ScalarType::Half, ScalarType::Bool, iter.dtype(), "abs_cuda", [&]() {
    using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
    gpu_kernel(iter, []GPU_LAMBDA(thrust_t a) -> thrust_t {
      return abs_wrapper(a);
    });
  });
}

void logical_not_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND2(kBool, kHalf, iter.dtype(1), "logical_not_cuda", [&]() {
    using self_t = scalar_t;
    AT_DISPATCH_ALL_TYPES_AND2(kBool, kHalf, iter.dtype(0), "logical_not_cuda", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(self_t a) -> scalar_t { return static_cast<scalar_t>(!a); });
    });
  });
}

void neg_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "neg_cuda", [&]() {
    AT_SKIP_BFLOAT16_IF_NOT_ROCM(scalar_t, "neg_cuda", [&] {
      using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
      gpu_kernel(iter, []GPU_LAMBDA(thrust_t a) -> thrust_t {
        return -a;
      });
    });
  });
}

void sign_kernel_cuda(TensorIterator& iter){
  if (iter.dtype() == ScalarType::Bool) {
    gpu_kernel(iter, []GPU_LAMBDA(bool a){
      return a;
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND(ScalarType::Half, iter.dtype(), "sign_cuda", [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
            scalar_t zero = scalar_t(0);
            return (zero < a) - (a < zero);
        });
    });
  }
}

REGISTER_DISPATCH(abs_stub, &abs_kernel_cuda);
REGISTER_DISPATCH(logical_not_stub, &logical_not_kernel_cuda);
REGISTER_DISPATCH(neg_stub, &neg_kernel_cuda);
REGISTER_DISPATCH(sign_stub, &sign_kernel_cuda);

}} // namespace at::native
