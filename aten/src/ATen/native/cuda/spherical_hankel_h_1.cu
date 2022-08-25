#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/native/special.h>
#include <ATen/native/special/spherical_hankel_h_1.h>

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
const auto spherical_hankel_h_1_string = jiterator_stringify(
  template<typename T1>
  T1 spherical_hankel_h_1(T1 n, T1 z) {
    return n;
  } // T1 spherical_hankel_h_1(T1 n, T1 z)
); // spherical_hankel_h_1_string

const char spherical_hankel_h_1_name[] = "spherical_hankel_h_1";

void spherical_hankel_h_1_kernel_cuda(TensorIteratorBase& iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_hankel_h_1_cuda", [&]() {
      opmath_jitted_gpu_kernel_with_scalars<spherical_hankel_h_1_name, scalar_t, scalar_t>(iterator, spherical_hankel_h_1_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_hankel_h_1_cuda", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t n, scalar_t z) -> scalar_t {
      return at::native::special::spherical_hankel_h_1(n, z);
    });
  });
#endif
} // void spherical_hankel_h_1_kernel_cuda(TensorIteratorBase& iterator)
} // namespace (anonymous)

REGISTER_DISPATCH(spherical_hankel_h_1_stub, &spherical_hankel_h_1_kernel_cuda);
} // namespace native
} // namespace at
