#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/native/special.h>
#include <ATen/native/special/bessel_j.h>

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
const auto bessel_j_string = jiterator_stringify(
  template<typename T1>
  T1 bessel_j(T1 v, T1 z) {
    return v;
  } // T1 bessel_j(T1 v, T1 z)
); // bessel_j_string

const char bessel_j_name[] = "bessel_j";

void bessel_j_kernel_cuda(TensorIteratorBase& iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_j_cuda", [&]() {
      opmath_jitted_gpu_kernel_with_scalars<bessel_j_name, scalar_t, scalar_t>(iterator, bessel_j_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_j_cuda", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t v, scalar_t z) -> scalar_t {
      return at::native::special::bessel_j(v, z);
    });
  });
#endif
} // void bessel_j_kernel_cuda(TensorIteratorBase& iterator)
} // namespace (anonymous)

REGISTER_DISPATCH(bessel_j_stub, &bessel_j_kernel_cuda);
} // namespace native
} // namespace at
