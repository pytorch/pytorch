#pragma once

#ifdef USE_CUDA
#include <THC/THC.h>
#include <THD/base/Cuda.h>

extern THCState** _THDCudaState;

inline THCState* THDGetCudaState() {
  return *_THDCudaState;
}

int THDGetStreamId(cudaStream_t stream);
#endif
