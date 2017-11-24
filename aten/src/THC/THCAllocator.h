#ifndef THC_ALLOCATOR_INC
#define THC_ALLOCATOR_INC

#include "THCGeneral.h"

extern THAllocator THCudaHostAllocator;
extern THAllocator THCUVAAllocator;
THC_API THCDeviceAllocator THCIpcAllocator;

#endif
