#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/JitLoops.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/util/TypeSafeSignMath.h>

#include <type_traits>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at { namespace native {

template<typename scalar_t>
struct DivFunctor {
  __device__ scalar_t operator() (scalar_t a, scalar_t b) const {
    return a / b;
  }
};

template<typename T>
struct MulFunctor {
  __device__ T operator() (T a, T b) const {
    return a * b;
  }
};

// Workaround for the error: '*' in boolean context, suggest '&&' instead [-Werror=int-in-bool-context]
template<>
struct MulFunctor<bool> {
  __device__ bool operator() (bool a, bool b) const {
    return a && b;
  }
};


void div_true_kernel_cuda(TensorIteratorBase& iter) {
  if (iter.is_cpu_scalar(2)) {
    // optimization for floating-point types: if the second operand is a CPU
    // scalar, compute a * reciprocal(b). Note that this may lose one bit of
    // precision compared to computing the division.
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, iter.common_dtype(), "div_true_cuda", [&]() {
      using opmath_t = at::opmath_type<scalar_t>;
      auto inv_b = opmath_t(1.0) / iter.scalar_value<opmath_t>(2);
      iter.remove_operand(2);
      gpu_kernel(iter, BUnaryFunctor<scalar_t, scalar_t, scalar_t, MulFunctor<opmath_t>>(
        MulFunctor<opmath_t>(), inv_b));
    });
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, iter.common_dtype(), "div_true_cuda", [&]() {
      DivFunctor<scalar_t> f;
      gpu_kernel_with_scalars(iter, f);
    });
  }
}

void div_trunc_kernel_cuda(TensorIteratorBase& iter) {
  auto dtype = iter.common_dtype();
  if (isIntegralType(dtype, /*includeBool*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(dtype, "div_trunc_cuda", [&]() {
      gpu_kernel_with_scalars(iter, [] GPU_LAMBDA (scalar_t a, scalar_t b) -> scalar_t {
        return a / b;
      });
    });
  } else if (iter.is_cpu_scalar(2)) {
    // optimization for floating-point types: if the second operand is a CPU
    // scalar, compute a * reciprocal(b). Note that this may lose one bit of
    // precision compared to computing the division.
    AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, dtype, "div_trunc_cuda", [&]() {
      using accscalar_t = at::acc_type<scalar_t, true>;
      auto inv_b = accscalar_t(1.0) / iter.scalar_value<accscalar_t>(2);
      iter.remove_operand(2);
      gpu_kernel(iter, [inv_b] GPU_LAMBDA (scalar_t a) -> scalar_t {
        return std::trunc(a * inv_b);
      });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, dtype, "div_trunc_cuda", [&]() {
      gpu_kernel_with_scalars(iter, [] GPU_LAMBDA (scalar_t a, scalar_t b) -> scalar_t {
        return std::trunc(a / b);
      });
    });
  }
}

void div_floor_kernel_cuda(TensorIteratorBase& iter) {
  // See NOTE: [Floor Division in Python]
  const auto dtype = iter.common_dtype();
  if (dtype == kByte) {
    // In the special case of unsigned integer division, floor division is
    // equivalent to truncation division (since the signs of the divisor and
    // dividend are always the same)
    return div_trunc_kernel_cuda(iter);
  } else if (isIntegralType(dtype, /*includeBool*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(dtype, "div_floor_cuda", [&]() {
      gpu_kernel_with_scalars(iter, [] GPU_LAMBDA (scalar_t a, scalar_t b) -> scalar_t {
        if (c10::signs_differ(a, b)) {
          // Subtracts one from the results of truncation division if the
          // divisor and dividend have different sign(bit)s and the remainder of
          // the division is nonzero
          const auto quot = a / b;
          const auto rem = a % b;
          return rem ? quot - 1 : quot;
        }

        return a / b;
      });
    });
  } else if (iter.is_cpu_scalar(2)) {
    // optimization for floating-point types: if the second operand is a CPU
    // scalar, compute a * reciprocal(b). Note that this may lose one bit of
    // precision compared to computing the division.
    AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, dtype, "div_floor_cuda", [&]() {
      using accscalar_t = at::acc_type<scalar_t, true>;
      auto b = iter.scalar_value<accscalar_t>(2);
      if (C10_UNLIKELY(b == 0)) {
        return div_true_kernel_cuda(iter);
      }

      auto inv_b = accscalar_t(1.0) / b;
      iter.remove_operand(2);
      gpu_kernel(iter, [b, inv_b] GPU_LAMBDA (scalar_t a) -> scalar_t {
        auto mod = std::fmod(a, b);
        auto div = (a - mod) * inv_b;
        if ((mod != 0) && (b < 0) != (mod < 0)) {
          div -= scalar_t(1);
        }

        scalar_t floordiv;
        if (div != 0) {
          floordiv = std::floor(div);
          if (div - floordiv > scalar_t(0.5)) {
            floordiv += scalar_t(1.0);
          }
        } else {
          floordiv = c10::cuda::compat::copysign(scalar_t(0), a * inv_b);
        }
        return floordiv;
      });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, dtype, "div_floor_cuda", [&]() {
      gpu_kernel_with_scalars(iter, [] GPU_LAMBDA (scalar_t a, scalar_t b) -> scalar_t {
        if (C10_UNLIKELY(b == 0)) {
          return a / b;
        }

        auto mod = std::fmod(a, b);
        auto div = (a - mod) / b;
        if ((mod != 0) && (b < 0) != (mod < 0)) {
          div -= scalar_t(1);
        }

        scalar_t floordiv;
        if (div != 0) {
          floordiv = std::floor(div);
          if (div - floordiv > scalar_t(0.5)) {
            floordiv += scalar_t(1.0);
          }
        } else {
          floordiv = c10::cuda::compat::copysign(scalar_t(0), a / b);
        }
        return floordiv;
      });
    });
  }
}

const char mul_name[] = "mul_kernel";
void mul_kernel_cuda(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (common_dtype == kComplexHalf) {
    using scalar_t = c10::complex<at::Half>;
    #if AT_USE_JITERATOR()
      static const auto mul_string = jiterator_stringify(
        template <typename T>
        T mul_kernel(T a, T b) {
          return a * b;
        }
      );
      jitted_gpu_kernel<mul_name, scalar_t, scalar_t, 2>(iter, mul_string);
    #else
      using opmath_t = at::opmath_type<scalar_t>;
      opmath_gpu_kernel_with_scalars<scalar_t>(iter, MulFunctor<opmath_t>());
    #endif
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kHalf, kBFloat16, kBool, iter.common_dtype(), "mul_cuda", [&]() {
      using opmath_t = at::opmath_type<scalar_t>;
      opmath_gpu_kernel_with_scalars<scalar_t>(iter, MulFunctor<opmath_t>());
    });
  }
}

REGISTER_DISPATCH(div_true_stub, &div_true_kernel_cuda);
REGISTER_DISPATCH(div_trunc_stub, &div_trunc_kernel_cuda);
REGISTER_DISPATCH(div_floor_stub, &div_floor_kernel_cuda);
REGISTER_DISPATCH(mul_stub, &mul_kernel_cuda);

}} // namespace at::native
