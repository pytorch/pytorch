#ifndef THC_DEVICE_ALLOCATOR_INC
#define THC_DEVICE_ALLOCATOR_INC

#include "THCGeneral.h"

THC_API THCDeviceAllocator* THCCachingAllocator_get(void);
THC_API void* THCCachingAllocator_getBaseAllocation(void *ptr, size_t *size);

#endif
