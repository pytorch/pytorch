#ifndef THC_DEVICE_ALLOCATOR_INC
#define THC_DEVICE_ALLOCATOR_INC

#include <c10/cuda/CUDAStream.h>
#include <c10/core/Allocator.h>
#include <c10/cuda/CUDAMacros.h>

#include <mutex>

namespace c10 {
namespace cuda {

// TODO: Name this something nicer

C10_CUDA_API Allocator* THCCachingAllocator_get(void);
C10_CUDA_API void THCCachingAllocator_emptyCache(void);
C10_CUDA_API void THCCachingAllocator_cacheInfo(int dev_id, size_t* cachedAndFree, size_t* largestBlock);
C10_CUDA_API void* THCCachingAllocator_getBaseAllocation(void *ptr, size_t *size);
C10_CUDA_API void THCCachingAllocator_recordStream(void *ptr, at::cuda::CUDAStream stream);
C10_CUDA_API uint64_t THCCachingAllocator_currentMemoryAllocated(int device);
C10_CUDA_API uint64_t THCCachingAllocator_maxMemoryAllocated(int device);
C10_CUDA_API void     THCCachingAllocator_resetMaxMemoryAllocated(int device);
C10_CUDA_API uint64_t THCCachingAllocator_currentMemoryCached(int device);
C10_CUDA_API uint64_t THCCachingAllocator_maxMemoryCached(int device);
C10_CUDA_API void     THCCachingAllocator_resetMaxMemoryCached(int device);

C10_CUDA_API std::mutex* THCCachingAllocator_getCudaFreeMutex();

C10_CUDA_API std::shared_ptr<void> THCCaching_CUDAIpcDevptr(std::string handle);

}} // namespace c10::cuda

#endif
