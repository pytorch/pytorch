#include "THCTensorMathReduce.cuh"
#include "THCTensor.hpp"

THC_API int
THCudaByteTensor_logicalAndAll(THCState *state, THCudaByteTensor *self) {
  THCAssertSameGPU(THCudaByteTensor_checkGPU(state, 1, self));
  unsigned char result;
  if (!THC_reduceAll(state, self,
                     thrust::identity<unsigned char>(),
                     LogicalAll(),
                     LogicalAll(),
                     (unsigned char) 1, &result, 0)) {
    THArgCheck(false, 1, CUTORCH_DIM_WARNING);
  }

  return (int) result;
}

THC_API int
THCudaByteTensor_logicalAnyAll(THCState *state, THCudaByteTensor *self) {
  THCAssertSameGPU(THCudaByteTensor_checkGPU(state, 1, self));
  unsigned char result;
  if (!THC_reduceAll(state, self,
                     thrust::identity<unsigned char>(),
                     LogicalAny(),
                     LogicalAny(),
                     (unsigned char) 0, &result, 0)) {
    THArgCheck(false, 1, CUTORCH_DIM_WARNING);
  }

  return (int) result;
}

THC_API void
THCudaByteTensor_logicalAnd(THCState* state, THCudaByteTensor *self, THCudaByteTensor *src, int dimension, int keepdim) {
  THCAssertSameGPU(THCudaByteTensor_checkGPU(state, 2, self, src));
  if (!THC_reduceDim(state, self, src,
                     thrust::identity<unsigned char>(),
                     LogicalAll(),
                     LogicalAll(),
                     (unsigned char) 1,
                     dimension,
                     keepdim)) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

THC_API void
THCudaByteTensor_logicalAny(THCState* state, THCudaByteTensor *self, THCudaByteTensor *src, int dimension, int keepdim) {
  THCAssertSameGPU(THCudaByteTensor_checkGPU(state, 2, self, src));
  if (!THC_reduceDim(state, self, src,
                     thrust::identity<unsigned char>(),
                     LogicalAny(),
                     LogicalAny(),
                     (unsigned char) 0,
                     dimension,
                     keepdim)) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}
