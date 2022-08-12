#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/native/UnaryOps.h>

#include <limits>

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Math.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/Math.cuh>
#include <ATen/native/cuda/jit_utils.h>
#include <ATen/NumericUtils.h>
#include <c10/core/Scalar.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/util/complex.h>

namespace at {
namespace native {
namespace {
const auto spherical_bessel_j_0_string = jiterator_stringify(
  template<typename T>
  T spherical_bessel_j_0(T x) {
    return x;
  } // spherical_bessel_j_0(T x)
); // spherical_bessel_j_0_string

const char spherical_bessel_j_0_name[] = "spherical_bessel_j_0";

void spherical_bessel_j_0_kernel_cuda(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_bessel_j_0_cuda", [&]() {
    jitted_gpu_kernel<spherical_bessel_j_0_name, scalar_t, scalar_t, 1>(iterator, spherical_bessel_j_0_string);
  });
  #else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_bessel_j_0_cuda", [&]() {
    gpu_kernel(iterator, []GPU_LAMBDA(scalar_t x) -> scalar_t {
      return x;
    });
  });
#endif // AT_USE_JITERATOR()
} // void spherical_bessel_j_0_kernel_cuda(TensorIteratorBase &iterator)
} // namespace (anonymous)
REGISTER_DISPATCH(special_spherical_bessel_j_0_stub, &spherical_bessel_j_0_kernel_cuda);
} // namespace native
} // namespace at
