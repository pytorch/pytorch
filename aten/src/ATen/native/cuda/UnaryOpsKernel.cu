#include <ATen/native/UnaryOps.h>

#include <limits>

#include <ATen/AccumulateType.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Math.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/Math.cuh>
#include <ATen/NumericUtils.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <ATen/NumericUtils.h>
#include <c10/util/complex.h>

namespace at {
namespace native {

void bitwise_not_kernel_cuda(TensorIteratorBase& iter) {
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

void exp_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.common_dtype(), "exp_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return std::exp(a);
    });
  });
}

void expm1_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::BFloat16, ScalarType::Half,
      iter.common_dtype(), "expm1_cuda",
      [&]() {
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

void rsqrt_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      ScalarType::BFloat16, ScalarType::Half,
      iter.common_dtype(), "rsqrt_cuda",
      [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
          // In CUDA, ::rsqrt is overloaded for float and at::Half here is implicitly cast to float.
          return rsqrt_wrapper(a);
        });
      });
}

void sqrt_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.common_dtype(), "sqrt_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return ::sqrt(a);
    });
  });
}

void clamp_kernel_cuda(TensorIteratorBase& iter, const Scalar& min_value, const Scalar& max_value) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "clamp_cuda", [&]() {
    auto lower = min_value.to<scalar_t>();
    auto upper = max_value.to<scalar_t>();
    gpu_kernel(iter, [=]GPU_LAMBDA(scalar_t v) -> scalar_t {
      // Propagate nan, which doesn't propagate automatically for ROCm
      if (_isnan(v)) {
        return v;
      } else {
        return ::min(::max(v, lower), upper);
      }
    });
  });
}

void clamp_min_kernel_cuda(TensorIteratorBase& iter, const Scalar& min_value) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "clamp_min_cuda", [&]() {
    auto lower = min_value.to<scalar_t>();
    gpu_kernel(iter, [=]GPU_LAMBDA(scalar_t v) -> scalar_t {
      // Propagate nan, which doesn't propagate automatically for ROCm
      if (_isnan(v)) {
        return v;
      } else {
        return ::max(v, lower);
      }
    });
  });
}

void clamp_max_kernel_cuda(TensorIteratorBase& iter, const Scalar& max_value) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "clamp_max_cuda", [&]() {
    auto upper = max_value.to<scalar_t>();
    gpu_kernel(iter, [=]GPU_LAMBDA(scalar_t v) -> scalar_t {
      // Propagate nan, which doesn't propagate automatically for ROCm
      if (_isnan(v)) {
        return v;
      } else {
        return ::min(v, upper);
      }
    });
  });
}

void nan_to_num_kernel_cuda(
    TensorIteratorBase& iter,
    c10::optional<double> nan,
    c10::optional<double> pos_inf,
    c10::optional<double> neg_inf) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "nan_to_num_cuda", [&]() {
    scalar_t nan_replacement = static_cast<scalar_t>(nan.value_or(0.));
    scalar_t pos_inf_replacement = pos_inf.has_value()
        ? static_cast<scalar_t>(pos_inf.value())
        : std::numeric_limits<scalar_t>::max();
    scalar_t neg_inf_replacement = neg_inf.has_value()
        ? static_cast<scalar_t>(neg_inf.value())
        : std::numeric_limits<scalar_t>::lowest();
    gpu_kernel(iter, [=] GPU_LAMBDA(scalar_t a) -> scalar_t {
      return (
          at::_isnan(a)
              ? nan_replacement
              : (a == std::numeric_limits<scalar_t>::infinity()
                     ? pos_inf_replacement
                     : (a == -std::numeric_limits<scalar_t>::infinity()
                            ? neg_inf_replacement
                            : a)));
    });
  });
}

void frexp_kernel_cuda(TensorIteratorBase& iter) {
#ifdef __HIP_PLATFORM_HCC__
  // Reference: https://rocmdocs.amd.com/en/latest/ROCm_API_References/HIP-MATH.html
  //            https://github.com/ROCm-Developer-Tools/HIP/issues/2169
  // ROCm does not support frexp function yet
  TORCH_CHECK(false, "torch.frexp() is not implemented on ROCm platform.");
#else
  AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::Half,
    // The iter.dtype() here is the dtype of mantissa output.
    // It's a floating point type and must be the same as the input's dtype.
    iter.dtype(),
    "frexp_cuda", [&]() {
      gpu_kernel_multiple_outputs(iter, [=] GPU_LAMBDA (scalar_t a) -> thrust::tuple<scalar_t, int32_t> {
        int32_t exponent;
        scalar_t mantissa = std::frexp(a, &exponent);
        return {mantissa, exponent};
      });
  });
#endif
}

REGISTER_DISPATCH(bitwise_not_stub, &bitwise_not_kernel_cuda);
REGISTER_DISPATCH(exp_stub, &exp_kernel_cuda);
REGISTER_DISPATCH(expm1_stub, &expm1_kernel_cuda);
REGISTER_DISPATCH(rsqrt_stub, &rsqrt_kernel_cuda);
REGISTER_DISPATCH(sqrt_stub, &sqrt_kernel_cuda);
REGISTER_DISPATCH(nan_to_num_stub, &nan_to_num_kernel_cuda);
REGISTER_DISPATCH(frexp_stub, &frexp_kernel_cuda);

} // namespace native
} // namespace at
