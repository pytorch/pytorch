#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/BinaryInternal.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/util/TypeSafeSignMath.h>
#include <c10/util/generic_math.h>
#include <ATen/native/cuda/BinaryInternal.h>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/native/cuda/Loops.cuh>

#include <type_traits>

namespace at::native {
namespace binary_internal {

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
      gpu_kernel_with_scalars(
          iter, [] GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
            return c10::div_floor_integer(a, b);
      });
    });
  } else if (iter.is_cpu_scalar(2)) {
    // optimization for floating-point types: if the second operand is a CPU
    // scalar, compute a * reciprocal(b). Note that this may lose one bit of
    // precision compared to computing the division.
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, dtype, "div_floor_cuda", [&]() {
          using accscalar_t = at::acc_type<scalar_t, true>;
          auto b = iter.scalar_value<accscalar_t>(2);
          if (C10_UNLIKELY(b == 0)) {
            return div_true_kernel_cuda(iter);
          }

          auto inv_b = accscalar_t(1.0) / b;
          iter.remove_operand(2);
          gpu_kernel(iter, [b, inv_b] GPU_LAMBDA(scalar_t a) -> scalar_t {
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
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, dtype, "div_floor_cuda", [&]() {
          gpu_kernel_with_scalars(
              iter, [] GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
                return c10::div_floor_floating(a, b);
              });
        });
  }
}
} // namespace binary_internal

REGISTER_DISPATCH(div_floor_stub, &binary_internal::div_floor_kernel_cuda)

} // namespace at::native
