#ifndef THC_ALLOCATOR_INC
#define THC_ALLOCATOR_INC

#include "THCGeneral.h"

THC_API THAllocator* getTHCudaHostAllocator();
THC_API THAllocator* getTHCUVAAllocator();
THC_API THCDeviceAllocator* getTHCIpcAllocator();

#endif
