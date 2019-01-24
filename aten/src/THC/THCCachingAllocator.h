#ifndef THC_DEVICE_ALLOCATOR_INC
#define THC_DEVICE_ALLOCATOR_INC

#include <c10/cuda/CUDAStream.h>
#include <c10/core/Allocator.h>
#include <ATen/cuda/ATenCUDAGeneral.h>

#include <mutex>

namespace at {
namespace cuda {

AT_CUDA_API Allocator* THCCachingAllocator_get(void);
AT_CUDA_API void THCCachingAllocator_emptyCache(void);
AT_CUDA_API void THCCachingAllocator_cacheInfo(int dev_id, size_t* cachedAndFree, size_t* largestBlock);
AT_CUDA_API void* THCCachingAllocator_getBaseAllocation(void *ptr, size_t *size);
AT_CUDA_API void THCCachingAllocator_recordStream(void *ptr, at::cuda::CUDAStream stream);
AT_CUDA_API uint64_t THCCachingAllocator_currentMemoryAllocated(int device);
AT_CUDA_API uint64_t THCCachingAllocator_maxMemoryAllocated(int device);
AT_CUDA_API void     THCCachingAllocator_resetMaxMemoryAllocated(int device);
AT_CUDA_API uint64_t THCCachingAllocator_currentMemoryCached(int device);
AT_CUDA_API uint64_t THCCachingAllocator_maxMemoryCached(int device);
AT_CUDA_API void     THCCachingAllocator_resetMaxMemoryCached(int device);

AT_CUDA_API std::mutex* THCCachingAllocator_getCudaFreeMutex();

AT_CUDA_API std::shared_ptr<void> THCCaching_CUDAIpcDevptr(std::string handle);

}} // namespace at::cuda

#endif
