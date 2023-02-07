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
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/native/cuda/Loops.cuh>

#include <type_traits>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at::native {

CONSTEXPR_EXCEPT_WIN_CUDA char mul_name[] = "mul_kernel";
void mul_kernel_cuda(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (common_dtype == kComplexHalf) {
    using scalar_t = c10::complex<at::Half>;
#if AT_USE_JITERATOR()
    static const auto mul_string = jiterator_stringify(
        template <typename T> T mul_kernel(T a, T b) { return a * b; });
    opmath_jitted_gpu_kernel_with_scalars<mul_name, scalar_t, scalar_t>(
        iter, mul_string);
#else
    using opmath_t = at::opmath_type<scalar_t>;
    opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
        iter, binary_internal::MulFunctor<opmath_t>());
#endif
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        kHalf, kBFloat16, kBool, iter.common_dtype(), "mul_cuda", [&]() {
          using opmath_t = at::opmath_type<scalar_t>;
          opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
              iter, binary_internal::MulFunctor<opmath_t>());
        });
  }
}

REGISTER_DISPATCH(mul_stub, &mul_kernel_cuda);

} // namespace at::native
