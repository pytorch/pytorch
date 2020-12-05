#pragma once

#include <ATen/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/detail/CUDAHooksInterface.h>
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
    return std::make_tuple(arg.seed_, *(arg.offset_.ptr) + arg.offset_intragraph_);
  } else {
    return std::make_tuple(arg.seed_, arg.offset_.val);
  }
}

} // namespace philox

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
// Protects against enum cudaStreamCaptureStatus implementation changes.
// Some compilers seem not to like static_assert without the messages.
static_assert(int(cudaStreamCaptureStatus::cudaStreamCaptureStatusNone) == 0,
              "unexpected int(cudaStreamCaptureStatusNone) value");
static_assert(int(cudaStreamCaptureStatus::cudaStreamCaptureStatusActive) == 1,
              "unexpected int(cudaStreamCaptureStatusActive) value");
static_assert(int(cudaStreamCaptureStatus::cudaStreamCaptureStatusInvalidated) == 2,
              "unexpected int(cudaStreamCaptureStatusInvalidated) value");
#endif

enum class CaptureStatus: int {
  #if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  None = int(cudaStreamCaptureStatus::cudaStreamCaptureStatusNone),
  Active = int(cudaStreamCaptureStatus::cudaStreamCaptureStatusActive),
  Invalidated = int(cudaStreamCaptureStatus::cudaStreamCaptureStatusInvalidated)
  #else
  None = 0
  #endif
};

inline std::ostream& operator<<(std::ostream& os, CaptureStatus status) {
  switch(status) {
    case CaptureStatus::None:
      os << "cudaStreamCaptureStatusNone";
      break;
    #if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
    case CaptureStatus::Active:
      os << "cudaStreamCaptureStatusActive";
      break;
    case CaptureStatus::Invalidated:
      os << "cudaStreamCaptureStatusInvalidated";
      break;
    #endif
    default:
      TORCH_INTERNAL_ASSERT(false,
                            "Unknown CUDA graph CaptureStatus",
                            int(status));
  }
  return os;
}

inline CaptureStatus currentStreamCaptureStatus() {
  #if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  // don't create a context if we don't have to
  if (at::detail::getCUDAHooks().hasPrimaryContext(c10::cuda::current_device())) {
    cudaStreamCaptureStatus is_capturing;
    AT_CUDA_CHECK(cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream(),
                                        &is_capturing));
    return CaptureStatus(is_capturing);
  } else {
    return CaptureStatus::None;
  }
  #else
  return CaptureStatus::None;
  #endif
}

inline void assertNotCapturing(std::string attempt) {
  auto status = currentStreamCaptureStatus();
  TORCH_CHECK(status == CaptureStatus::None,
              attempt,
              " during CUDA graph capture. If you need this call to be captured, "
              "please file an issue. "
              "Current cudaStreamCaptureStatus: ",
              status);
}

} // namespace cuda
} // namespace at
