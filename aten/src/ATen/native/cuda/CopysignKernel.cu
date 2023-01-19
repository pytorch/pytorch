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

void copysign_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "copysign_cuda", [&]() {
    gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return c10::cuda::compat::copysign(a, b);
    });
  });
}

REGISTER_DISPATCH(copysign_stub, &copysign_kernel_cuda);

} // namespace at::native
