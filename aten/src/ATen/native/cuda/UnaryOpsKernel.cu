#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/UnaryOps.h>

#include <limits>

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Math.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/jit_utils.h>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/Math.cuh>
#include <ATen/NumericUtils.h>
#include <ATen/OpMathType.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/core/Scalar.h>
#include <c10/util/complex.h>

namespace at::native {

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

CONSTEXPR_EXCEPT_WIN_CUDA char exp_name[] = "exp_kernel";
void exp_kernel_cuda(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (at::isComplexType(common_dtype)) {
    #if AT_USE_JITERATOR()
      static const auto exp_string = jiterator_stringify(
          template <typename T>
          T exp_kernel(T x) {
            return std::exp(x);
      }); // exp_string
      AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, common_dtype, "exp_cuda", [&]() {
          jitted_gpu_kernel<
              /*name=*/exp_name,
              /*return_dtype=*/scalar_t,
              /*common_dtype=*/scalar_t,
              /*arity=*/1>(iter, exp_string);
      });
    #else
      AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, common_dtype, "exp_cuda", [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
          using opmath_t = at::opmath_type<scalar_t>;
          return std::exp(static_cast<opmath_t>(a));
        });
      });
    #endif
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, common_dtype, "exp_cuda", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
        return std::exp(a);
      });
    });
  }
}

void expm1_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
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
C10_HOST_DEVICE static inline scalar_t rsqrt_wrapper(scalar_t v) {
  return ::rsqrt(v);
}

template<typename T>
C10_HOST_DEVICE static inline c10::complex<T> rsqrt_wrapper(c10::complex<T> v) {
  const c10::complex<T> one = c10::complex<T>(1.0, 0);
  // std::sqrt for c10::complex is overloaded in c10/util/complex_math.h
  return one / ::sqrt(v);
}

CONSTEXPR_EXCEPT_WIN_CUDA char rsqrt_name[] = "rsqrt_kernel";
void rsqrt_kernel_cuda(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (at::isComplexType(common_dtype)) {
    #if AT_USE_JITERATOR()
      static const auto rsqrt_string = jiterator_stringify(
          template <typename T>
          T rsqrt_kernel(T x) {
            const T one = T{1};
            return one / std::sqrt(x);
      }); // rsqrt_string
      AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, common_dtype, "rsqrt_cuda", [&]() {
          jitted_gpu_kernel<
              /*name=*/rsqrt_name,
              /*return_dtype=*/scalar_t,
              /*common_dtype=*/scalar_t,
              /*arity=*/1>(iter, rsqrt_string);
      });
    #else
      AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, common_dtype, "rsqrt_cuda", [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
          using opmath_t = at::opmath_type<scalar_t>;
          return rsqrt_wrapper(static_cast<opmath_t>(a));
        });
      });
    #endif
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::BFloat16, ScalarType::Half,
      iter.common_dtype(), "rsqrt_cuda",
      [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
          // In CUDA, ::rsqrt is overloaded for float and at::Half here is implicitly cast to float.
          return rsqrt_wrapper(a);
        });
      });
  }
}

CONSTEXPR_EXCEPT_WIN_CUDA char sqrt_name[] = "sqrt_kernel";
void sqrt_kernel_cuda(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (at::isComplexType(common_dtype)) {
    #if AT_USE_JITERATOR()
      static const auto sqrt_string = jiterator_stringify(
          template <typename T>
          T sqrt_kernel(T x) {
            return std::sqrt(x);
      }); // sqrt_string
      AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, common_dtype, "sqrt_cuda", [&]() {
          jitted_gpu_kernel<
              /*name=*/sqrt_name,
              /*return_dtype=*/scalar_t,
              /*common_dtype=*/scalar_t,
              /*arity=*/1>(iter, sqrt_string);
      });
    #else
      AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, common_dtype, "sqrt_cuda", [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
          using opmath_t = at::opmath_type<scalar_t>;
          return ::sqrt(static_cast<opmath_t>(a));
        });
      });
    #endif
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, common_dtype, "sqrt_cuda", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
        return std::sqrt(a);
      });
    });
  }
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
}

REGISTER_DISPATCH(bitwise_not_stub, &bitwise_not_kernel_cuda);
REGISTER_DISPATCH(exp_stub, &exp_kernel_cuda);
REGISTER_DISPATCH(expm1_stub, &expm1_kernel_cuda);
REGISTER_DISPATCH(rsqrt_stub, &rsqrt_kernel_cuda);
REGISTER_DISPATCH(sqrt_stub, &sqrt_kernel_cuda);
REGISTER_DISPATCH(nan_to_num_stub, &nan_to_num_kernel_cuda);
REGISTER_DISPATCH(frexp_stub, &frexp_kernel_cuda);

} // namespace at::native
