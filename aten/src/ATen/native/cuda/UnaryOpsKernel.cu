#include <limits>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>

namespace at { namespace native {

void bitwise_not_kernel_cuda(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    gpu_kernel(iter, []GPU_LAMBDA(bool a) {
      return !a;
    });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "bitwise_not_cuda", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
        return ~a;
      });
    });
  }
}

void logical_not_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND2(kBool, kHalf, iter.dtype(1), "logical_not_cuda", [&]() {
    using self_t = scalar_t;
    AT_DISPATCH_ALL_TYPES_AND2(kBool, kHalf, iter.dtype(0), "logical_not_cuda", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(self_t a) -> scalar_t { return static_cast<scalar_t>(!a); });
    });
  });
}

void ceil_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "ceil_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return std::ceil(a);
    });
  });
}

void floor_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "floor_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return std::floor(a);
    });
  });
}

void neg_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND(ScalarType::Half, iter.dtype(), "neg_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return -a;
    });
  });
}

void round_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "round_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      // We do not use std::round because we would like to round midway numbers to the nearest even integer.
      return std::nearbyint(a);
    });
  });
}

void trunc_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "trunc_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return std::trunc(a);
    });
  });
}

// overloading rsqrt and rsqrtf
template <typename scalar_t>
__host__ __device__ static inline scalar_t rsqrt_wrapper(scalar_t a) {
  return static_cast<scalar_t>(::rsqrtf(static_cast<float>(a)));
}

__host__ __device__ static inline double rsqrt_wrapper(double a) {
  return ::rsqrt(a);
}

void rsqrt_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "rsqrt_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return rsqrt_wrapper(a);
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

void erfinv_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "erfinv_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return erfinvf(a);
    });
  });
}

REGISTER_DISPATCH(bitwise_not_stub, &bitwise_not_kernel_cuda);
REGISTER_DISPATCH(logical_not_stub, &logical_not_kernel_cuda);
REGISTER_DISPATCH(ceil_stub, &ceil_kernel_cuda);
REGISTER_DISPATCH(floor_stub, &floor_kernel_cuda);
REGISTER_DISPATCH(neg_stub, &neg_kernel_cuda);
REGISTER_DISPATCH(round_stub, &round_kernel_cuda);
REGISTER_DISPATCH(rsqrt_stub, &rsqrt_kernel_cuda);
REGISTER_DISPATCH(sign_stub, &sign_kernel_cuda);
REGISTER_DISPATCH(trunc_stub, &trunc_kernel_cuda);
REGISTER_DISPATCH(erfinv_stub, &erfinv_kernel_cuda);
}}
