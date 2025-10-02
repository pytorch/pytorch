#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/Math.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/cuda/jit_utils.h>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at::native {

// See note [Jiterator]
constexpr char gcd_name[] = "gcd";
void gcd_kernel_cuda(TensorIteratorBase& iter) {
  #if AT_USE_JITERATOR()
    AT_DISPATCH_INTEGRAL_TYPES(iter.common_dtype(), "gcd_cuda", [&]() {
      jitted_gpu_kernel</*name=*/gcd_name,
                        /*return_dtype=*/ scalar_t,
                        /*common_dtype=*/ scalar_t,
                        /*arity=*/ 2>(iter, gcd_string);
    });
  #else
    AT_DISPATCH_INTEGRAL_TYPES(iter.common_dtype(), "gcd_cuda", [&]() {
      gpu_kernel(iter, [] GPU_LAMBDA (scalar_t a, scalar_t b) -> scalar_t {
        return calc_gcd(a, b);
      });
    });
  #endif // AT_USE_JITERATOR()
}

// See note [Jiterator]
constexpr char lcm_name[] = "lcm";
void lcm_kernel_cuda(TensorIteratorBase& iter) {
  #if AT_USE_JITERATOR()
    AT_DISPATCH_INTEGRAL_TYPES(iter.common_dtype(), "lcm_cuda", [&]() {
      jitted_gpu_kernel</*name=*/lcm_name,
                        /*return_dtype=*/ scalar_t,
                        /*common_dtype=*/ scalar_t,
                        /*arity=*/ 2>(iter, lcm_string);
    });
  #else
    AT_DISPATCH_INTEGRAL_TYPES(iter.common_dtype(), "lcm_cuda", [&]() {
      gpu_kernel(iter, [] GPU_LAMBDA (scalar_t a, scalar_t b) -> scalar_t {
        scalar_t g = calc_gcd(a, b);
        return (g == 0) ? 0 : ::abs(a / g * b);
      });
    });
  #endif // AT_USE_JITERATOR()
}

REGISTER_DISPATCH(gcd_stub, &gcd_kernel_cuda)
REGISTER_DISPATCH(lcm_stub, &lcm_kernel_cuda)

} // namespace at::native
