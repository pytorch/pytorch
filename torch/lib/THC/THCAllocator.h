#ifndef THC_ALLOCATOR_INC
#define THC_ALLOCATOR_INC

#include "THCGeneral.h"

THC_API void THCAllocator_init(THCState *state);
THC_API void THCAllocator_shutdown(THCState *state);

#endif
