#ifndef THC_SPIN_INC
#define THC_SPIN_INC

#include "THCGeneral.h"
#include <time.h>

// enqueues a kernel that spins for the specified number of cycles
THC_API void THC_sleep(THCState* state, long long cycles);

#endif
