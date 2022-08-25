#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/native/special.h>
#include <ATen/native/special/modified_bessel_k.h>

#include <limits>

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/cuda/jit_utils.h>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/Math.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Math.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/NumericUtils.h>
#include <c10/core/Scalar.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/util/complex.h>

namespace at {
namespace native {
namespace {
const auto modified_bessel_k_string = jiterator_stringify(
  template<typename T1>
  T1 modified_bessel_k(T1 v, T1 z) {
    return v;
  } // T1 modified_bessel_k(T1 v, T1 z)
); // modified_bessel_k_string

const char modified_bessel_k_name[] = "modified_bessel_k";

void modified_bessel_k_kernel_cuda(TensorIteratorBase& iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_k_cuda", [&]() {
      opmath_jitted_gpu_kernel_with_scalars<modified_bessel_k_name, scalar_t, scalar_t>(iterator, modified_bessel_k_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_k_cuda", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t v, scalar_t z) -> scalar_t {
      return at::native::special::modified_bessel_k(v, z);
    });
  });
#endif
} // void modified_bessel_k_kernel_cuda(TensorIteratorBase& iterator)
} // namespace (anonymous)

REGISTER_DISPATCH(modified_bessel_k_stub, &modified_bessel_k_kernel_cuda);
} // namespace native
} // namespace at
