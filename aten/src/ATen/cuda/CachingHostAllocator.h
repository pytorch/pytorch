#pragma once

#include <ATen/core/CachingHostAllocator.h>
#include <c10/core/Allocator.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Deprecated.h>

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
C10_DEPRECATED_MESSAGE(
  "at::cuda::getCachingHostAllocator() is deprecated. Please use at::getHostAllocator(at::kCUDA) instead.")
inline TORCH_CUDA_CPP_API at::HostAllocator* getCachingHostAllocator() {
  return at::getHostAllocator(at::kCUDA);
}

// Records an event in the specified stream. The allocation corresponding to the
// input `ptr`/`ctx` will not be re-used until the event has occurred.
C10_DEPRECATED_MESSAGE(
  "at::cuda::CachingHostAllocator_recordEvent(...) is deprecated. Please use at::getHostAllocator(at::kCUDA)->record_event(...) instead.")
inline TORCH_CUDA_CPP_API bool CachingHostAllocator_recordEvent(
    void* ptr,
    void* ctx,
    c10::cuda::CUDAStream stream) {
  return getHostAllocator(at::kCUDA)->record_event(ptr, ctx, stream.unwrap());
}

// Releases cached pinned memory allocations via cudaHostFree
C10_DEPRECATED_MESSAGE(
  "at::cuda::CachingHostAllocator_emptyCache() is deprecated. Please use at::getHostAllocator(at::kCUDA)->empty_cache() instead.")
inline TORCH_CUDA_CPP_API void CachingHostAllocator_emptyCache() {
  getHostAllocator(at::kCUDA)->empty_cache();
}

C10_DEPRECATED_MESSAGE(
  "at::cuda::HostAlloc(...) is deprecated. Please use at::getHostAllocator(at::kCUDA)->allocate(...) instead.")
inline TORCH_CUDA_CPP_API at::DataPtr HostAlloc(size_t size) {
  return getHostAllocator(at::kCUDA)->allocate(size);
}

C10_DEPRECATED_MESSAGE(
  "at::cuda::CachingHostAllocator_getStats() is deprecated. Please use at::getHostAllocator(at::kCUDA)->get_stats() instead.")
inline TORCH_CUDA_CPP_API at::HostStats CachingHostAllocator_getStats() {
  return getHostAllocator(at::kCUDA)->get_stats();
}

C10_DEPRECATED_MESSAGE(
  "at::cuda::CachingHostAllocator_resetAccumulatedStats() is deprecated. Please use at::getHostAllocator(at::kCUDA)->reset_accumulated_stats() instead.")
inline TORCH_CUDA_CPP_API void CachingHostAllocator_resetAccumulatedStats() {
  getHostAllocator(at::kCUDA)->reset_accumulated_stats();
}

C10_DEPRECATED_MESSAGE(
  "at::cuda::CachingHostAllocator_resetPeakStats() is deprecated. Please use at::getHostAllocator(at::kCUDA)->reset_peak_stats() instead.")
inline TORCH_CUDA_CPP_API void CachingHostAllocator_resetPeakStats() {
  getHostAllocator(at::kCUDA)->reset_peak_stats();
}

} // namespace at::cuda
