#pragma once

#include <ATen/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/core/StreamGuard.h>
#include <c10/cuda/CUDAGuard.h>

namespace at {
namespace cuda {
namespace philox {

// We can't write a __device__ function in CUDAGeneratorImpl.h, because it's in ATen.
// Also, whatever call unpacks PhiloxCudaState in consumer kernels must be inlineable.
// Easiest thing that comes to mind is, define a free function here, in ATen/cuda.
// Any cuda consumer can include this header.
__device__ __forceinline__ std::tuple<uint64_t, uint64_t>
unpack(at::PhiloxCudaState arg) {
  if (arg.captured_) {
    return std::make_tuple(arg.seed_, *(arg.offset_.ptr) + arg.offset_intragraph);
  } else {
    return std::make_tuple(arg.seed_, arg.offset_.val);
  }
}

} // namespace philox

inline cudaStreamCaptureStatus currentStreamCaptureStatus() {
  #if defined(CUDA_VERSION) && CUDA_VERSION > 11000
  cudaStreamCaptureStatus is_capturing;
  AT_CUDA_CHECK(cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream(),
                                      &is_capturing));
  return is_capturing;
  #else
  return 0;
  #endif
}

inline void assertNotCapturing(std::string attempt) {
  TORCH_CHECK(currentStreamCaptureStatus() == 0,
              attempt,
              " during graph capture. ",
              "Current cudaStreamCaptureStatus: ",
              is_capturing);
}

} // namespace cuda
} // namespace at
