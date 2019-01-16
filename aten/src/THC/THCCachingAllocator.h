#ifndef THC_DEVICE_ALLOCATOR_INC
#define THC_DEVICE_ALLOCATOR_INC

#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/ATenCUDAGeneral.h>

#include <mutex>

#include <THC/THCGeneral.h>

THC_API THCDeviceAllocator* THCCachingAllocator_get(void);
THC_API void THCCachingAllocator_emptyCache(void);
THC_API void THCCachingAllocator_cacheInfo(int dev_id, size_t* cachedAndFree, size_t* largestBlock);
THC_API void* THCCachingAllocator_getBaseAllocation(void *ptr, size_t *size);
THC_API void THCCachingAllocator_recordStream(void *ptr, at::cuda::CUDAStream stream);
THC_API uint64_t THCCachingAllocator_currentMemoryAllocated(int device);
THC_API uint64_t THCCachingAllocator_maxMemoryAllocated(int device);
THC_API void     THCCachingAllocator_resetMaxMemoryAllocated(int device);
THC_API uint64_t THCCachingAllocator_currentMemoryCached(int device);
THC_API uint64_t THCCachingAllocator_maxMemoryCached(int device);
THC_API void     THCCachingAllocator_resetMaxMemoryCached(int device);

THC_API std::mutex* THCCachingAllocator_getCudaFreeMutex();

AT_CUDA_API std::shared_ptr<void> THCCaching_CUDAIpcDevptr(std::string handle);
#endif
