#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at { namespace native {

void logaddexp_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "logaddexp_cuda", [&]() {
    gpu_kernel(iter, [] GPU_LAMBDA (scalar_t a, scalar_t b) -> scalar_t {
      if (::isinf(a) && a == b) {
        return a;
      }
      else {
        scalar_t m = ::max(a, b);
        return m + ::log((scalar_t)(1.0) + ::exp(-::abs(a - b)));
      }
    });
  });
}

void logaddexp2_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "logaddexp2_cuda", [&]() {
    gpu_kernel(iter, [] GPU_LAMBDA (scalar_t a, scalar_t b) -> scalar_t {
      if (::isinf(a) && a == b) {
        return a;
      }
      else {
        scalar_t m = ::max(a, b);
        return m + ::log2((scalar_t)(1.0) + ::pow((scalar_t)(2.0), -::abs(a - b)));
      }
    });
  });
}

REGISTER_DISPATCH(logaddexp_stub, &logaddexp_kernel_cuda);
REGISTER_DISPATCH(logaddexp2_stub, &logaddexp2_kernel_cuda);

}} // namespace at::native
