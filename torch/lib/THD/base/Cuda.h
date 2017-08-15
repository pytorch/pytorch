#pragma once

#ifdef WITH_CUDA
#include "../THD.h"

#include <THC/THC.h>

THD_API void THDSetCudaStatePtr(THCState **state);
THD_API void THDRegisterCudaStream(hipStream_t stream);
#endif

