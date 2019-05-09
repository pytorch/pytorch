#pragma once

#ifdef USE_CUDA
#include <THD/THD.h>

#include <THC/THC.h>

THD_API void THDSetCudaStatePtr(THCState** state);
THD_API void THDRegisterCudaStream(cudaStream_t stream);
#endif
