#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/Math.cuh>
#include <limits>

namespace at::native {

#if AT_USE_JITERATOR()
CONSTEXPR_EXCEPT_WIN_CUDA char sinh_name[] = "sinh_impl";
#endif

void sinh_kernel_cuda(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (at::isComplexType(common_dtype)) {
#if AT_USE_JITERATOR()
    static const auto sinh_string = jiterator_stringify(
        template <typename T> T sinh_impl(T a) { return std::sinh(a); });
    AT_DISPATCH_COMPLEX_TYPES_AND(
        kComplexHalf, common_dtype, "sinh_name", [&]() {
          jitted_gpu_kernel<
              /*name=*/sinh_name,
              /*return_dtype=*/scalar_t,
              /*common_dtype=*/scalar_t,
              /*arity=*/1>(iter, sinh_string);
        });
#else
    AT_DISPATCH_COMPLEX_TYPES_AND(
        kComplexHalf, common_dtype, "sinh_name", [&]() {
          gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a) -> scalar_t {
            using opmath_t = at::opmath_type<scalar_t>;
            return ::sinh(static_cast<opmath_t>(a));
          });
        });
#endif
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        ScalarType::Half,
        ScalarType::BFloat16,
        common_dtype,
        "sinh_cuda",
        [&]() {
          gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a) -> scalar_t {
            return ::sinh(a);
          });
        });
  }
}

REGISTER_DISPATCH(sinh_stub, &sinh_kernel_cuda);

} // namespace at::native
