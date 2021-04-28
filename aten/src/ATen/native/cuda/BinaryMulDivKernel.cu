#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <ATen/native/cuda/Loops.cuh>

#include <type_traits>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at { namespace native {

template<typename scalar_t, typename accscalar_t>
struct MulScalarFunctor {
    MulScalarFunctor(accscalar_t b_): b(b_) {}
    __device__ scalar_t operator() (scalar_t a) const {
      return a * b;
    }
  private:
    accscalar_t b;
};

template<typename scalar_t>
struct DivFunctor {
  __device__ scalar_t operator() (scalar_t a, scalar_t b) const {
    return a / b;
  }
};

template<typename scalar_t>
struct MulFunctor {
  __device__ scalar_t operator() (scalar_t a, scalar_t b) const {
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
      using accscalar_t = at::acc_type<scalar_t, true>;
      auto inv_b = accscalar_t(1.0) / iter.scalar_value<accscalar_t>(2);
      iter.remove_operand(2);
      MulScalarFunctor<scalar_t, decltype(inv_b)> f(inv_b);
      gpu_kernel(iter, f);
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
        if (!std::is_unsigned<scalar_t>::value && (a < 0) != (b < 0)) {
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

void mul_kernel_cuda(TensorIteratorBase& iter) {
  if (!isIntegralType(iter.common_dtype(), /*includeBool*/ true) &&
    (iter.is_cpu_scalar(1) || iter.is_cpu_scalar(2))) {
    //if common dtype is half the scalar constant can overflow in half precision, and yet the result can
    //still be representable in the half dtype. Cast scalar to acc_type to have better accuracy
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, iter.common_dtype(), "mul_cuda", [&]() {
      using accscalar_t = at::acc_type<scalar_t, true>;
      int scalar_arg = iter.is_cpu_scalar(1) ? 1 : 2;
      auto b = iter.scalar_value<accscalar_t>(scalar_arg);
      iter.remove_operand(scalar_arg);
      const cuda::OptionalCUDAGuard device_guard(device_of(iter.tensor(1)));
      MulScalarFunctor<scalar_t, decltype(b)> f(b);
      gpu_kernel(iter, f);
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kHalf, kBFloat16, kBool, iter.common_dtype(), "mul_cuda", [&]() {
      MulFunctor<scalar_t> f;
      gpu_kernel_with_scalars(iter, f);
    });
  }
}

REGISTER_DISPATCH(div_true_stub, &div_true_kernel_cuda);
REGISTER_DISPATCH(div_trunc_stub, &div_trunc_kernel_cuda);
REGISTER_DISPATCH(div_floor_stub, &div_floor_kernel_cuda);
REGISTER_DISPATCH(mul_stub, &mul_kernel_cuda);

}} // namespace at::native
