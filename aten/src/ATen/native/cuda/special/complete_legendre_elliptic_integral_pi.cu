#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/native/special.h>
#include <ATen/native/special/complete_legendre_elliptic_integral_pi.h>

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
const auto complete_legendre_elliptic_integral_pi_string = jiterator_stringify(
  template<typename T1>
  T1 complete_legendre_elliptic_integral_pi(T1 n, T1 k) {
    return n;
  } // T1 complete_legendre_elliptic_integral_pi(T1 n, T1 k)
); // complete_legendre_elliptic_integral_pi_string

const char complete_legendre_elliptic_integral_pi_name[] = "complete_legendre_elliptic_integral_pi";

void complete_legendre_elliptic_integral_pi_kernel_cuda(TensorIteratorBase& iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "complete_legendre_elliptic_integral_pi_cuda", [&]() {
      opmath_jitted_gpu_kernel_with_scalars<complete_legendre_elliptic_integral_pi_name, scalar_t, scalar_t>(iterator, complete_legendre_elliptic_integral_pi_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "complete_legendre_elliptic_integral_pi_cuda", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t n, scalar_t k) -> scalar_t {
      return at::native::special::complete_legendre_elliptic_integral_pi(n, k);
    });
  });
#endif
} // void complete_legendre_elliptic_integral_pi_kernel_cuda(TensorIteratorBase& iterator)
} // namespace (anonymous)

REGISTER_DISPATCH(complete_legendre_elliptic_integral_pi_stub, &complete_legendre_elliptic_integral_pi_kernel_cuda);
} // namespace native
} // namespace at
