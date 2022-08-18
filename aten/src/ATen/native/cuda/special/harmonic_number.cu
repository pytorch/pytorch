#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/native/special.h>
#include <ATen/native/special/harmonic_number.h>

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
const auto harmonic_number_string = jiterator_stringify(
  template<typename T1>
  T1 harmonic_number(T1 n) {
    return n;
  } // T1 harmonic_number(T1 n)
); // harmonic_number_string

const char harmonic_number_name[] = "harmonic_number";

void special_harmonic_number_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "harmonic_number_cuda_kernel", [&]() {
    jitted_gpu_kernel<harmonic_number_name, scalar_t, scalar_t, 1>(iterator, harmonic_number_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "harmonic_number_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t n) -> scalar_t {
      return at::native::special::harmonic_number<scalar_t>(n);
    });
  });
#endif
} // void special_harmonic_number_cuda_kernel(TensorIteratorBase &iterator)
} // namespace (anonymous)

REGISTER_DISPATCH(special_harmonic_number_stub, &special_harmonic_number_cuda_kernel);
} // namespace native
} // namespace at
