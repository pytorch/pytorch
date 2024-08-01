#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/util/TypeSafeSignMath.h>
#include <ATen/native/cuda/BinaryInternal.h>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/native/cuda/Loops.cuh>

#include <type_traits>

namespace at::native {
namespace binary_internal {

CONSTEXPR_EXCEPT_WIN_CUDA char div_name[] = "div_kernel";
void div_true_kernel_cuda(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (iter.common_dtype() == kComplexHalf) {
    using scalar_t = c10::complex<at::Half>;
#if AT_USE_JITERATOR()
    static const auto div_string = jiterator_stringify(
        template <typename T> T div_kernel(T a, T b) { return a / b; });
    opmath_jitted_gpu_kernel_with_scalars<div_name, scalar_t, scalar_t>(
        iter, div_string);
#else
    using opmath_t = at::opmath_type<scalar_t>;
    opmath_gpu_kernel_with_scalars<scalar_t>(iter, DivFunctor<opmath_t>());
#endif
    return;
  }
  if (iter.is_cpu_scalar(2)) {
    // optimization for floating-point types: if the second operand is a CPU
    // scalar, compute a * reciprocal(b). Note that this may lose one bit of
    // precision compared to computing the division.
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        kHalf, kBFloat16, common_dtype, "div_true_cuda", [&]() {
          using opmath_t = at::opmath_type<scalar_t>;
          auto inv_b = opmath_t(1.0) / iter.scalar_value<opmath_t>(2);
          iter.remove_operand(2);
          gpu_kernel(
              iter,
              BUnaryFunctor<scalar_t, scalar_t, scalar_t, MulFunctor<opmath_t>>(
                  MulFunctor<opmath_t>(), inv_b));
        });
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        kHalf, kBFloat16, common_dtype, "div_true_cuda", [&]() {
          DivFunctor<scalar_t> f;
          gpu_kernel_with_scalars(iter, f);
        });
  }
}
} // namespace binary_internal

REGISTER_DISPATCH(div_true_stub, &binary_internal::div_true_kernel_cuda);

} // namespace at::native
