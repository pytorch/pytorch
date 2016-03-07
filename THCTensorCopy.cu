#include "THCApply.cuh"
#include "THCHalf.h"

inline int curGPU() {
  int curDev;
  THCudaCheck(cudaGetDevice(&curDev));
  return curDev;
}

#include "generic/THCTensorCopy.cu"
#include "THCGenerateAllTypes.h"
