#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/zmath.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <c10/macros/Macros.h>


// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at { namespace native {

void add_kernel_cuda(TensorIterator& iter, Scalar alpha_scalar) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kHalf, kBool, kBFloat16, iter.common_dtype(), "add_cuda/sub_cuda", [&]() {
    using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
    auto alpha = thrust_t(alpha_scalar.to<scalar_t>());
    gpu_kernel_with_scalars(iter, [alpha]GPU_LAMBDA(thrust_t a, thrust_t b) -> thrust_t {
      return a + alpha * b;
    });
  });
}

static void sub_kernel_cuda(TensorIterator& iter, Scalar alpha_scalar) {
  add_kernel_cuda(iter, -alpha_scalar);
}

void div_kernel_cuda(TensorIterator& iter) {
  if (!isIntegralType(iter.common_dtype(), /*includeBool*/ false) && iter.is_cpu_scalar(2)) {
    // optimization for floating-point types: if the second operand is a CPU
    // scalar, compute a * reciprocal(b). Note that this may lose one bit of
    // precision compared to computing the division.
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, iter.common_dtype(), "div_cuda", [&]() {
      using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
      auto inv_b = thrust_t(1.0) / thrust_t(iter.scalar_value<scalar_t>(2));
      iter.remove_operand(2);
      gpu_kernel(iter, [inv_b]GPU_LAMBDA(thrust_t a) -> thrust_t {
        return a * inv_b;
      });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kHalf, kBFloat16, iter.common_dtype(), "div_cuda", [&]() {
      using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
      gpu_kernel_with_scalars(iter, []GPU_LAMBDA(thrust_t a, thrust_t b) -> thrust_t {
        return a / b;
      });
    });
  }
}

void mul_kernel_cuda(TensorIterator& iter) {
  if (iter.common_dtype() == ScalarType::Bool) {
    // Workaround for the error: '*' in boolean context, suggest '&&' instead [-Werror=int-in-bool-context]
    gpu_kernel_with_scalars(iter, []GPU_LAMBDA(bool a, bool b) -> bool {
      return a && b;
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kHalf, kBFloat16, iter.common_dtype(), "mul_cuda", [&]() {
      using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
      gpu_kernel_with_scalars(iter, []GPU_LAMBDA(thrust_t a, thrust_t b) -> thrust_t {
        return a * b;
      });
    });
  }
}

void remainder_kernel_cuda(TensorIterator& iter) {
  if (isIntegralType(iter.dtype(), /*includeBool*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "remainder_cuda", [&]() {
      using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
      gpu_kernel_with_scalars(iter, []GPU_LAMBDA(thrust_t a, thrust_t b) -> thrust_t {
        CUDA_KERNEL_ASSERT(b != 0);
        thrust_t r = a % b;
        if ((r != 0) && ((r < 0) != (b < 0))) {
          r += b;
        }
        return r;
      });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "remainder_cuda", [&]() {
      using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
      gpu_kernel_with_scalars(iter,
        []GPU_LAMBDA(thrust_t a, thrust_t b) __ubsan_ignore_float_divide_by_zero__ -> thrust_t {
          return a - b * static_cast<thrust_t>(std::floor(a / b));
        });
    });
  }
}

REGISTER_DISPATCH(add_stub, &add_kernel_cuda);
REGISTER_DISPATCH(sub_stub, &sub_kernel_cuda);
REGISTER_DISPATCH(div_stub, &div_kernel_cuda);
REGISTER_DISPATCH(mul_stub, &mul_kernel_cuda);
REGISTER_DISPATCH(remainder_stub, &remainder_kernel_cuda);

}} // namespace at::native
