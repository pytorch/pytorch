#ifndef THC_DEVICE_ALLOCATOR_INC
#define THC_DEVICE_ALLOCATOR_INC

#if __cplusplus >= 201103L
#include <mutex>
#endif

#include "THCGeneral.h"

THC_API THCDeviceAllocator* THCCachingAllocator_get(void);
THC_API void* THCCachingAllocator_getBaseAllocation(void *ptr, size_t *size);

#if __cplusplus >= 201103L
THC_API std::mutex* THCCachingAllocator_getCudaFreeMutex();
#endif

#endif
