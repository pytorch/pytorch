#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>

namespace at::native {

template<typename scalar_t>
struct AbsFunctor {
  __device__ __forceinline__ scalar_t operator() (const scalar_t a) const {
    return std::abs(a);
  }
};

CONSTEXPR_EXCEPT_WIN_CUDA char abs_name[] = "abs_kernel";
void abs_kernel_cuda(TensorIteratorBase& iter) {
  auto dtype = iter.dtype();
  if (at::isComplexType(dtype)) {
#if AT_USE_JITERATOR()
    static const auto abs_string = jiterator_stringify(
        template <typename T> T abs_kernel(T x) { return std::abs(x); });
    AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, dtype, "abs_cuda", [&]() {
      jitted_gpu_kernel<
          /*name=*/abs_name,
          /*return_dtype=*/scalar_t,
          /*common_dtype=*/scalar_t,
          /*arity=*/1>(iter, abs_string);
    });
#else
    AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, dtype, "abs_cuda", [&]() {
      using opmath_t = at::opmath_type<scalar_t>;
      gpu_kernel(iter, AbsFunctor<opmath_t>());
    });
#endif
  } else {
    AT_DISPATCH_ALL_TYPES_AND3(
        ScalarType::Half,
        ScalarType::BFloat16,
        ScalarType::Bool,
        iter.dtype(),
        "abs_cuda",
        [&]() { gpu_kernel(iter, AbsFunctor<scalar_t>()); });
  }
}

  REGISTER_DISPATCH(abs_stub, &abs_kernel_cuda);

} // namespace at::native
