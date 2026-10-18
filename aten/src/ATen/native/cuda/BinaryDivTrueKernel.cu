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

constexpr char div_name[] = "div_kernel";

bool is_power_of_two(double b) {
  int exponent = 0;
  return std::abs(std::frexp(b, &exponent)) == 0.5;
}

bool is_power_of_two(c10::complex<double>) {
  return false;
}

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
    // precision compared to computing the division. If 1/b overflows or is
    // subnormal in opmath (tiny or huge b), divide by b instead.
    //
    // float32 keeps the reciprocal only when b is a power of two, where it is
    // exact, so a / b is correctly rounded like CPU and tensor/tensor division;
    // the division costs nothing measurable for bandwidth-bound kernels. Half and
    // bfloat16 results round from float opmath, so the extra bit rarely matters,
    // and float64 division is much slower than multiplication on some GPUs.
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        kHalf, kBFloat16, common_dtype, "div_true_cuda", [&]() {
          using opmath_t = at::opmath_type<scalar_t>;
          using high_prec_t = std::conditional_t<
              c10::is_complex<scalar_t>::value,
              c10::complex<double>,
              double>;
          const auto b = iter.scalar_value<high_prec_t>(2);
          const auto inv_b = static_cast<opmath_t>(high_prec_t(1.0) / b);
          iter.remove_operand(2);
          const bool use_reciprocal = is_usable_reciprocal(inv_b) &&
              (!std::is_same_v<scalar_t, float> || is_power_of_two(b));
          if (use_reciprocal) {
            gpu_kernel(
                iter,
                BUnaryFunctor<scalar_t, scalar_t, scalar_t, MulFunctor<opmath_t>>(
                    MulFunctor<opmath_t>(), inv_b));
          } else {
            gpu_kernel(
                iter,
                BUnaryFunctor<scalar_t, scalar_t, scalar_t, DivFunctor<opmath_t>>(
                    DivFunctor<opmath_t>(), static_cast<opmath_t>(b)));
          }
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

REGISTER_DISPATCH(div_true_stub, &binary_internal::div_true_kernel_cuda)

} // namespace at::native
