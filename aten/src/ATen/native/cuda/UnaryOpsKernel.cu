#include <limits>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Math.cuh>

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

void expm1_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "expm1_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return ::expm1(a);
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

void log_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "log_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return std::log(a);
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

// We manually overload nearbyint because std::nearbyint does not work with ROCm.
template <typename scalar_t>
__host__ __device__ static inline scalar_t nearbyint_wrapper(scalar_t a) {
  return static_cast<scalar_t>(::nearbyintf(static_cast<float>(a)));
}

__host__ __device__ static inline double nearbyint_wrapper(double a) {
  return ::nearbyint(a);
}

void round_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "round_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      // We do not use std::round because we would like to round midway numbers to the nearest even integer.
      return nearbyint_wrapper(a);
    });
  });
}

// We manually overload trunc because std::trunc does not work with ROCm.
template <typename scalar_t>
__host__ __device__ static inline scalar_t trunc_wrapper(scalar_t a) {
  return static_cast<scalar_t>(::truncf(static_cast<float>(a)));
}

__host__ __device__ static inline double trunc_wrapper(double a) {
  return ::trunc(a);
}

void trunc_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "trunc_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return trunc_wrapper(a);
    });
  });
}

void rsqrt_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "rsqrt_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      // In CUDA, ::rsqrt is overloaded for float and at::Half here is implicitly cast to float.
      return ::rsqrt(a);
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
      return ::erfinv(a);
    });
  });
}

void digamma_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "digamma_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return calc_digamma(a);
    });
  });
}

void trigamma_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "trigamma_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return calc_trigamma(a);
    });
  });
}

void polygamma_kernel_cuda(TensorIterator& iter, int64_t n) {
  switch (n) {
    case 0: digamma_kernel_cuda(iter); break;
    case 1: trigamma_kernel_cuda(iter); break;
    default: TORCH_CHECK(false, "polygamma(n,x) is not implemented for n>=2, but was ", n);
  }
}

void lgamma_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "lgamma_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return ::lgamma(a);
    });
  });
}

REGISTER_DISPATCH(bitwise_not_stub, &bitwise_not_kernel_cuda);
REGISTER_DISPATCH(logical_not_stub, &logical_not_kernel_cuda);
REGISTER_DISPATCH(ceil_stub, &ceil_kernel_cuda);
REGISTER_DISPATCH(expm1_stub, &expm1_kernel_cuda);
REGISTER_DISPATCH(floor_stub, &floor_kernel_cuda);
REGISTER_DISPATCH(log_stub, &log_kernel_cuda);
REGISTER_DISPATCH(neg_stub, &neg_kernel_cuda);
REGISTER_DISPATCH(round_stub, &round_kernel_cuda);
REGISTER_DISPATCH(rsqrt_stub, &rsqrt_kernel_cuda);
REGISTER_DISPATCH(sign_stub, &sign_kernel_cuda);
REGISTER_DISPATCH(trunc_stub, &trunc_kernel_cuda);
REGISTER_DISPATCH(erfinv_stub, &erfinv_kernel_cuda);
REGISTER_DISPATCH(digamma_stub, &digamma_kernel_cuda);
REGISTER_DISPATCH(polygamma_stub, &polygamma_kernel_cuda);
REGISTER_DISPATCH(lgamma_stub, &lgamma_kernel_cuda);
}}
