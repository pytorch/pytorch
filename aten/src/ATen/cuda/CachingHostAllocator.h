#pragma once

#include <c10/cuda/CUDAStream.h>
#include <c10/core/Allocator.h>

namespace at {
namespace cuda {

//
// A caching allocator for CUDA host allocations (pinned memory).
//
// This provides a drop-in replacement for THCudaHostAllocator, which re-uses
// freed pinned (page-locked) memory allocations. This avoids device
// synchronizations due to cudaFreeHost calls.
//
// To ensure correct behavior, THCCachingHostAllocator_recordEvent must be
// called anytime a pointer from this allocator is used in a cudaMemcpyAsync
// call between host and device. We implement this for storages and tensors in
// copy_from_cpu_async_ and copy_to_cpu_async_.
//
// Note that this allocator does not split larger allocations into smaller
// blocks, unlike the caching device allocator.
//

// Basic interface for the cuda host allocator to follow
// There are four public methods: allocate, free, record_event and empty cache.
class ICachingHostAllocator {
 public:
  virtual ~ICachingHostAllocator() = default;

  virtual cudaError_t malloc(void** ptr, size_t size) = 0;
  virtual cudaError_t free(void* ctx) = 0;
  virtual cudaError_t recordEvent(void *ctx, CUDAStream stream)  = 0;
  virtual void emptyCache() = 0;
};
TORCH_CUDA_CPP_API void setICachingHostAllocator(ICachingHostAllocator* pAllocator);

// The API to use if you want to use the caching host allocator anywhere
TORCH_CUDA_CPP_API at::Allocator* getCachingHostAllocator();

// Records an event in the specified stream. The allocation 'ptr' will not be
// re-used until the event has occurred.
TORCH_CUDA_CPP_API cudaError_t
CachingHostAllocator_recordEvent(void* ptr, c10::cuda::CUDAStream stream);

// Releases cached pinned memory allocations via cudaHostFree
TORCH_CUDA_CPP_API void CachingHostAllocator_emptyCache();

inline TORCH_CUDA_CPP_API at::DataPtr HostAlloc(size_t size) {
  return getCachingHostAllocator()->allocate(size);
}


} // namespace cuda
} // namespace at
