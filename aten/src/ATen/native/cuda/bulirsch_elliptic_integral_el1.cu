#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/native/special.h>
#include <ATen/native/special/bulirsch_elliptic_integral_el1.h>

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
const auto bulirsch_elliptic_integral_el1_string = jiterator_stringify(
  template<typename T1>
  T1 bulirsch_elliptic_integral_el1(T1 x, T1 k_c) {
    return x;
  } // T1 bulirsch_elliptic_integral_el1(T1 x, T1 k_c)
); // bulirsch_elliptic_integral_el1_string

const char bulirsch_elliptic_integral_el1_name[] = "bulirsch_elliptic_integral_el1";

void bulirsch_elliptic_integral_el1_kernel_cuda(TensorIteratorBase& iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bulirsch_elliptic_integral_el1_cuda", [&]() {
      opmath_jitted_gpu_kernel_with_scalars<bulirsch_elliptic_integral_el1_name, scalar_t, scalar_t>(iterator, bulirsch_elliptic_integral_el1_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bulirsch_elliptic_integral_el1_cuda", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t x, scalar_t k_c) -> scalar_t {
      return at::native::special::bulirsch_elliptic_integral_el1(x, k_c);
    });
  });
#endif
} // void bulirsch_elliptic_integral_el1_kernel_cuda(TensorIteratorBase& iterator)
} // namespace (anonymous)

REGISTER_DISPATCH(bulirsch_elliptic_integral_el1_stub, &bulirsch_elliptic_integral_el1_kernel_cuda);
} // namespace native
} // namespace at
