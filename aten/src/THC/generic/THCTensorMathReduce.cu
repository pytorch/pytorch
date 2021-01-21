#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMathReduce.cu"
#else

#include <c10/cuda/CUDAException.h>

#if !defined(THC_REAL_IS_BOOL)

void THCTensor_(prod)(THCState* state, THCTensor *self, THCTensor *src, int dimension, int keepdim) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self, src));
  if (!THC_reduceDim<scalar_t>(state, self, src,
                           thrust::identity<accreal>{},
                           ReduceMultiply<accreal>{},
                           thrust::identity<accreal>{},
                           scalar_cast<accreal>(1),
                           dimension,
                           keepdim)) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

#endif

#endif
