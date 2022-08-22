#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/native/special.h>
#include <ATen/native/special/tan_pi.h>

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
const auto tan_pi_string = jiterator_stringify(
  template<typename T1>
  T1 tan_pi(T1 z) {
    return z;
  } // T1 tan_pi(T1 z)
); // tan_pi_string

const char tan_pi_name[] = "tan_pi";

void special_tan_pi_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iterator.common_dtype(), "tan_pi_cuda_kernel", [&]() {
    jitted_gpu_kernel<tan_pi_name, scalar_t, scalar_t, 1>(iterator, tan_pi_string);
  });
#else
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iterator.common_dtype(), "tan_pi_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t z) -> scalar_t {
      return at::native::special::tan_pi(z);
    });
  });
#endif
} // void special_tan_pi_cuda_kernel(TensorIteratorBase &iterator)
} // namespace (anonymous)

REGISTER_DISPATCH(special_tan_pi_stub, &special_tan_pi_cuda_kernel);
} // namespace native
} // namespace at
