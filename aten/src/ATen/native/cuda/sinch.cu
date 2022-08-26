#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/native/special.h>
#include <ATen/native/special/sinch.h>

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
const auto sinch_string = jiterator_stringify(
  template<typename T1>
  T1 sinch(T1 z) {
    return z;
  } // T1 sinch(T1 a)
); // sinch_string

const char sinch_name[] = "sinch";

void special_sinch_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iterator.common_dtype(), "sinch_cuda_kernel", [&]() {
    jitted_gpu_kernel<sinch_name, scalar_t, scalar_t, 1>(iterator, sinch_string);
  });
#else
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iterator.common_dtype(), "sinch_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t z) -> scalar_t {
      return at::native::special_functions::sinch(z);
    });
  });
#endif
} // void special_sinch_cuda_kernel(TensorIteratorBase &iterator)
} // namespace (anonymous)

REGISTER_DISPATCH(special_sinch_stub, &special_sinch_cuda_kernel);
} // namespace native
} // namespace at
