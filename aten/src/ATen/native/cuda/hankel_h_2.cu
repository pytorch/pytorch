#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/native/special.h>
#include <ATen/native/special/hankel_h_2.h>

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
const auto hankel_h_2_string = jiterator_stringify(
  template<typename T1>
  T1 hankel_h_2(T1 v, T1 z) {
    return v;
  } // T1 hankel_h_2(T1 v, T1 z)
); // hankel_h_2_string

const char hankel_h_2_name[] = "hankel_h_2";

void hankel_h_2_kernel_cuda(TensorIteratorBase& iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hankel_h_2_cuda", [&]() {
      opmath_jitted_gpu_kernel_with_scalars<hankel_h_2_name, scalar_t, scalar_t>(iterator, hankel_h_2_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hankel_h_2_cuda", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t v, scalar_t z) -> scalar_t {
      return at::native::special::hankel_h_2(v, z);
    });
  });
#endif
} // void hankel_h_2_kernel_cuda(TensorIteratorBase& iterator)
} // namespace (anonymous)

REGISTER_DISPATCH(hankel_h_2_stub, &hankel_h_2_kernel_cuda);
} // namespace native
} // namespace at
