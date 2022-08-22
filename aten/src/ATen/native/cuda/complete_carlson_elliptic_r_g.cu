#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/native/special.h>
#include <ATen/native/special/complete_carlson_elliptic_r_g.h>

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
const auto complete_carlson_elliptic_r_g_string = jiterator_stringify(
  template<typename T1>
  T1 complete_carlson_elliptic_r_g(T1 x, T1 y) {
    return x;
  } // T1 complete_carlson_elliptic_r_g(T1 x, T1 y)
); // complete_carlson_elliptic_r_g_string

const char complete_carlson_elliptic_r_g_name[] = "complete_carlson_elliptic_r_g";

void complete_carlson_elliptic_r_g_kernel_cuda(TensorIteratorBase& iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "complete_carlson_elliptic_r_g_cuda", [&]() {
      opmath_jitted_gpu_kernel_with_scalars<complete_carlson_elliptic_r_g_name, scalar_t, scalar_t>(iterator, complete_carlson_elliptic_r_g_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "complete_carlson_elliptic_r_g_cuda", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t x, scalar_t y) -> scalar_t {
      return at::native::special::complete_carlson_elliptic_r_g(x, y);
    });
  });
#endif
} // void complete_carlson_elliptic_r_g_kernel_cuda(TensorIteratorBase& iterator)
} // namespace (anonymous)

REGISTER_DISPATCH(complete_carlson_elliptic_r_g_stub, &complete_carlson_elliptic_r_g_kernel_cuda);
} // namespace native
} // namespace at
