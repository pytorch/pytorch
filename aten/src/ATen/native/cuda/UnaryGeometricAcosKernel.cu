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

#if 0 && AT_USE_JITERATOR()
CONSTEXPR_EXCEPT_WIN_CUDA char acos_name[] = "acos_impl";
#endif
void acos_kernel_cuda(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (at::isComplexType(common_dtype)) {
    // Disabled due to accuracy issues
#if 0 && AT_USE_JITERATOR()
    static const auto acos_string = jiterator_stringify(
        template <typename T> T acos_impl(T a) { return std::acos(a); });
    AT_DISPATCH_COMPLEX_TYPES_AND(
        kComplexHalf, common_dtype, "acos_name", [&]() {
          jitted_gpu_kernel<
              /*name=*/acos_name,
              /*return_dtype=*/scalar_t,
              /*common_dtype=*/scalar_t,
              /*arity=*/1>(iter, acos_string);
        });
#else
    AT_DISPATCH_COMPLEX_TYPES_AND(
        kComplexHalf, common_dtype, "acos_name", [&]() {
          gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a) -> scalar_t {
            using opmath_t = at::opmath_type<scalar_t>;
            return ::acos(static_cast<opmath_t>(a));
          });
        });
#endif
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        ScalarType::Half,
        ScalarType::BFloat16,
        common_dtype,
        "acos_cuda",
        [&]() {
          gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a) -> scalar_t {
            return ::acos(a);
          });
        });
  }
}

REGISTER_DISPATCH(acos_stub, &acos_kernel_cuda);

} // namespace at::native
