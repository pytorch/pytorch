#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/Math.cuh>
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

namespace at { namespace native {

void nextafter_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "nextafter_cuda", [&]() {
    gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return ::nextafter(a, b);
    });
  });
}

void heaviside_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBool, kBFloat16, iter.dtype(), "heaviside_cuda", [&]() {
    gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a == 0 ? b : static_cast<scalar_t>(a > 0);
    });
  });
}

REGISTER_DISPATCH(nextafter_stub, &nextafter_kernel_cuda);
REGISTER_DISPATCH(heaviside_stub, &heaviside_kernel_cuda);

}} // namespace at::native
