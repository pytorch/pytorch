#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/Math.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at { namespace native {

void gcd_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "gcd_cuda", [&]() {
    gpu_kernel(iter, [] GPU_LAMBDA (scalar_t a, scalar_t b) -> scalar_t {
      return calc_gcd(a, b);
    });
  });
}

void lcm_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "lcm_cuda", [&]() {
    gpu_kernel(iter, [] GPU_LAMBDA (scalar_t a, scalar_t b) -> scalar_t {
      scalar_t g = calc_gcd(a, b);
      return (g == 0) ? 0 : ::abs(a / g * b);
    });
  });
}

REGISTER_DISPATCH(gcd_stub, &gcd_kernel_cuda);
REGISTER_DISPATCH(lcm_stub, &lcm_kernel_cuda);

}} // namespace at::native
