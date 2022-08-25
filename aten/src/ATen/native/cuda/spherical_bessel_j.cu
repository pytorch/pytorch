#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/native/special.h>
#include <ATen/native/special/spherical_bessel_j.h>

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
const auto spherical_bessel_j_string = jiterator_stringify(
  template<typename T1>
  T1 spherical_bessel_j(T1 n, T1 z) {
    return n;
  } // T1 spherical_bessel_j(T1 n, T1 z)
); // spherical_bessel_j_string

const char spherical_bessel_j_name[] = "spherical_bessel_j";

void spherical_bessel_j_kernel_cuda(TensorIteratorBase& iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_bessel_j_cuda", [&]() {
      opmath_jitted_gpu_kernel_with_scalars<spherical_bessel_j_name, scalar_t, scalar_t>(iterator, spherical_bessel_j_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_bessel_j_cuda", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t n, scalar_t z) -> scalar_t {
      return at::native::special::spherical_bessel_j(n, z);
    });
  });
#endif
} // void spherical_bessel_j_kernel_cuda(TensorIteratorBase& iterator)
} // namespace (anonymous)

REGISTER_DISPATCH(spherical_bessel_j_stub, &spherical_bessel_j_kernel_cuda);
} // namespace native
} // namespace at
