#ifndef THC_DEVICE_ALLOCATOR_INC
#define THC_DEVICE_ALLOCATOR_INC

#ifdef __cplusplus
#include <ATen/cuda/CUDAStream.h>
#endif

#if (__cplusplus >= 201103L) || (defined(_MSC_VER) && defined(__cplusplus))
#include <mutex>
#endif

#include "THCGeneral.h"

THC_API THCDeviceAllocator* THCCachingAllocator_get(void);
THC_API void THCCachingAllocator_emptyCache(void);
THC_API void THCCachingAllocator_cacheInfo(int dev_id, size_t* cachedAndFree, size_t* largestBlock);
THC_API void* THCCachingAllocator_getBaseAllocation(void *ptr, size_t *size);
#ifdef __cplusplus
THC_API void THCCachingAllocator_recordStream(void *ptr, at::cuda::CUDAStream stream);
#endif
THC_API uint64_t THCCachingAllocator_currentMemoryAllocated(int device);
THC_API uint64_t THCCachingAllocator_maxMemoryAllocated(int device);
THC_API uint64_t THCCachingAllocator_currentMemoryCached(int device);
THC_API uint64_t THCCachingAllocator_maxMemoryCached(int device);

#if (__cplusplus >= 201103L) || (defined(_MSC_VER) && defined(__cplusplus))
THC_API std::mutex* THCCachingAllocator_getCudaFreeMutex();
#endif

AT_CUDA_API std::shared_ptr<void> THCCaching_CUDAIpcDevptr(std::string handle);
#endif
