#include <limits>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/AccumulateType.h>
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

void exp_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "exp_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return ::exp(a);
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

// We manually overload rsqrt because std::rsqrt does not work with complex types.
template<typename scalar_t>
__host__ __device__ static inline scalar_t rsqrt_wrapper(scalar_t v) {
  return ::rsqrt(v);
}

template<typename T>
__host__ __device__ static inline c10::complex<T> rsqrt_wrapper(c10::complex<T> v) {
  const c10::complex<T> one = c10::complex<T>(1.0, 0);
  // std::sqrt for c10::complex is overloaded in c10/util/complex_math.h
  return one / ::sqrt(v);
}

void rsqrt_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(ScalarType::Half, iter.dtype(), "rsqrt_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      // In CUDA, ::rsqrt is overloaded for float and at::Half here is implicitly cast to float.
      return rsqrt_wrapper(a);
    });
  });
}

void sqrt_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "sqrt_cuda", [&]() {
    AT_SKIP_BFLOAT16_IF_NOT_ROCM(scalar_t, "sqrt_cuda", [&] {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
        return ::sqrt(a);
      });
    });
  });
}

void sigmoid_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "sigmoid_cuda", [&]() {
    AT_SKIP_BFLOAT16_IF_NOT_ROCM(scalar_t, "sigmoid_cuda", [&] {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
        scalar_t one = scalar_t(1);
        return  one / (one + std::exp(- a));
      });
    });
  });
}

void erf_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "erf_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return ::erf(a);
    });
  });
}

void erfc_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "erfc_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return ::erfc(a);
    });
  });
}

void erfinv_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "erfinv_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return ::erfinv(a);
    });
  });
}

void clamp_kernel_cuda(TensorIterator& iter, Scalar min_value, Scalar max_value) {
  AT_DISPATCH_ALL_TYPES_AND(kHalf, iter.dtype(), "clamp_cuda", [&]() {
    auto lower = min_value.to<scalar_t>();
    auto upper = max_value.to<scalar_t>();
    gpu_kernel(iter, [=]GPU_LAMBDA(scalar_t v) -> scalar_t {
      return (v < lower) ? lower : (v > upper ? upper : v);
    });
  });
}

void clamp_min_kernel_cuda(TensorIterator& iter, Scalar min_value) {
  AT_DISPATCH_ALL_TYPES_AND(kHalf, iter.dtype(), "clamp_min_cuda", [&]() {
    auto lower = min_value.to<scalar_t>();
    gpu_kernel(iter, [=]GPU_LAMBDA(scalar_t v) -> scalar_t {
      return v < lower ? lower : v;
    });
  });
}

void clamp_max_kernel_cuda(TensorIterator& iter, Scalar max_value) {
  AT_DISPATCH_ALL_TYPES_AND(kHalf, iter.dtype(), "clamp_max_cuda", [&]() {
    auto upper = max_value.to<scalar_t>();
    gpu_kernel(iter, [=]GPU_LAMBDA(scalar_t v) -> scalar_t {
      return v > upper ? upper : v;
    });
  });
}

REGISTER_DISPATCH(bitwise_not_stub, &bitwise_not_kernel_cuda);
REGISTER_DISPATCH(exp_stub, &exp_kernel_cuda);
REGISTER_DISPATCH(expm1_stub, &expm1_kernel_cuda);
REGISTER_DISPATCH(rsqrt_stub, &rsqrt_kernel_cuda);
REGISTER_DISPATCH(sqrt_stub, &sqrt_kernel_cuda);
REGISTER_DISPATCH(sigmoid_stub, &sigmoid_kernel_cuda);
REGISTER_DISPATCH(erf_stub, &erf_kernel_cuda);
REGISTER_DISPATCH(erfc_stub, &erfc_kernel_cuda);
REGISTER_DISPATCH(erfinv_stub, &erfinv_kernel_cuda);
REGISTER_DISPATCH(clamp_stub, &clamp_kernel_cuda);
REGISTER_DISPATCH(clamp_min_stub, &clamp_min_kernel_cuda);
REGISTER_DISPATCH(clamp_max_stub, &clamp_max_kernel_cuda);
}}
