#ifndef THC_CACHING_HOST_ALLOCATOR_INC
#define THC_CACHING_HOST_ALLOCATOR_INC

#include <THC/THCGeneral.h>


#include <c10/cuda/CUDAStream.h>

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
THC_API c10::Allocator* getTHCCachingHostAllocator(void);

// Records an event in the specified stream. The allocation 'ptr' will not be
// re-used until the event has occurred.
THC_API cudaError_t THCCachingHostAllocator_recordEvent(void *ptr, at::cuda::CUDAStream stream);

// Releases cached pinned memory allocations via cudaHostFree
THC_API void THCCachingHostAllocator_emptyCache(void);

#endif
