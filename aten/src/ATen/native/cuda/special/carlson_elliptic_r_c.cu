#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/native/special.h>
#include <ATen/native/special/carlson_elliptic_r_c.h>

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
const auto carlson_elliptic_r_c_string = jiterator_stringify(
  template<typename T1>
  T1 carlson_elliptic_r_c(T1 x, T1 y) {
    return x;
  } // T1 carlson_elliptic_r_c(T1 x, T1 y)
); // carlson_elliptic_r_c_string

const char carlson_elliptic_r_c_name[] = "carlson_elliptic_r_c";

void carlson_elliptic_r_c_kernel_cuda(TensorIteratorBase& iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "carlson_elliptic_r_c_cuda", [&]() {
      opmath_jitted_gpu_kernel_with_scalars<carlson_elliptic_r_c_name, scalar_t, scalar_t>(iterator, carlson_elliptic_r_c_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "carlson_elliptic_r_c_cuda", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t x, scalar_t y) -> scalar_t {
      return at::native::special::carlson_elliptic_r_c(x, y);
    });
  });
#endif
} // void carlson_elliptic_r_c_kernel_cuda(TensorIteratorBase& iterator)
} // namespace (anonymous)

REGISTER_DISPATCH(carlson_elliptic_r_c_stub, &carlson_elliptic_r_c_kernel_cuda);
} // namespace native
} // namespace at
