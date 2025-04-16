#pragma once

#include <ATen/core/CachingHostAllocator.h>
#include <c10/core/Allocator.h>
#include <c10/cuda/CUDAStream.h>

namespace at::cuda {

//
// A caching allocator for CUDA host allocations (pinned memory).
//
// This provides a drop-in replacement for THCudaHostAllocator, which re-uses
// freed pinned (page-locked) memory allocations. This avoids device
// synchronizations due to cudaFreeHost calls.
//
// To ensure correct behavior, THCCachingHostAllocator_recordEvent must be
// called anytime a pointer from this allocator is used in a cudaMemcpyAsync
// call between host and device, and passed the corresponding context from the
// allocation. This is currently invoked by at::native::copy_kernel_cuda.
//
inline TORCH_CUDA_CPP_API at::HostAllocator* getCachingHostAllocator() {
  return at::getHostAllocator(at::kCUDA);
}

// Records an event in the specified stream. The allocation corresponding to the
// input `ptr`/`ctx` will not be re-used until the event has occurred.
inline TORCH_CUDA_CPP_API bool CachingHostAllocator_recordEvent(
    void* ptr,
    void* ctx,
    c10::cuda::CUDAStream stream) {
  return getCachingHostAllocator()->record_event(ptr, ctx, stream.unwrap());
}

// Releases cached pinned memory allocations via cudaHostFree
inline TORCH_CUDA_CPP_API void CachingHostAllocator_emptyCache() {
  getCachingHostAllocator()->empty_cache();
}

inline TORCH_CUDA_CPP_API at::DataPtr HostAlloc(size_t size) {
  return getCachingHostAllocator()->allocate(size);
}

inline TORCH_CUDA_CPP_API at::HostStats CachingHostAllocator_getStats() {
  return getCachingHostAllocator()->get_stats();
}

inline TORCH_CUDA_CPP_API void CachingHostAllocator_resetAccumulatedStats() {
  getCachingHostAllocator()->reset_accumulated_stats();
}

inline TORCH_CUDA_CPP_API void CachingHostAllocator_resetPeakStats() {
  getCachingHostAllocator()->reset_peak_stats();
}

} // namespace at::cuda
