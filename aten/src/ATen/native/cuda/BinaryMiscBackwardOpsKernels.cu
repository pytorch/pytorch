#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at { namespace native {

void sigmoid_backward_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "sigmoid_backward_cuda", [&]() {
    AT_SKIP_BFLOAT16_IF_NOT_ROCM(scalar_t, "sigmoid_backward_cuda", [&] {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return a * (scalar_t(1.) - b) * b;
      });
    });
  });
}

void tanh_backward_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "tanh_backward_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a * (scalar_t(1.) - b * b);
    });
  });
}

REGISTER_DISPATCH(sigmoid_backward_stub, &sigmoid_backward_kernel_cuda);
REGISTER_DISPATCH(tanh_backward_stub, &tanh_backward_kernel_cuda);

}} // namespace at::native
