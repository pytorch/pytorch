#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>

#if defined(__CUDACC__)
#include <cuda.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAMathCompat.h>
#elif defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <c10/hip/HIPMathCompat.h>
#endif

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at::native {

namespace {

C10_HOST_DEVICE inline c10::Half copysign_half(c10::Half a, c10::Half b) {
  return c10::Half(
      static_cast<uint16_t>((a.x & 0x7fff) | (b.x & 0x8000)),
      c10::Half::from_bits());
}

C10_HOST_DEVICE inline c10::BFloat16 copysign_bf16(c10::BFloat16 a, c10::BFloat16 b) {
  return c10::BFloat16(
      static_cast<uint16_t>((a.x & 0x7fff) | (b.x & 0x8000)),
      c10::BFloat16::from_bits());
}

} // namespace

void copysign_kernel_cuda(TensorIteratorBase& iter) {
  auto dtype = iter.common_dtype();
  if (dtype == kHalf) {
    gpu_kernel_with_scalars(iter, []GPU_LAMBDA(c10::Half a, c10::Half b) -> c10::Half {
      return copysign_half(a, b);
    });
  } else if (dtype == kBFloat16) {
    gpu_kernel_with_scalars(iter, []GPU_LAMBDA(c10::BFloat16 a, c10::BFloat16 b) -> c10::BFloat16 {
      return copysign_bf16(a, b);
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES(dtype, "copysign_cuda", [&]() {
      gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return c10::cuda::compat::copysign(a, b);
      });
    });
  }
}

REGISTER_DISPATCH(copysign_stub, &copysign_kernel_cuda)

} // namespace at::native
