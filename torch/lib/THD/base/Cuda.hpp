#pragma once

#ifdef WITH_CUDA
#include <THC/THC.h>
#include "Cuda.h"

extern THCState** _THDCudaState;

inline THCState* THDGetCudaState() {
  return *_THDCudaState;
}

int THDGetStreamId(hipStream_t stream);
#endif
