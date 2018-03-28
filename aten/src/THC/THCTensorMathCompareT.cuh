#ifndef THC_TENSORMATH_COMPARET_CUH
#define THC_TENSORMATH_COMPARET_CUH

#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCTensorCopy.h"
#include "THCApply.cuh"
#include "THCNumerics.cuh"
#include "THCReduce.cuh"

template <typename T, typename TOut>
struct TensorEQOp {
  __device__ inline void operator()(TOut* out, T* a, T* b) {
    *out = ScalarConvert<bool, TOut>::to(THCNumerics<T>::eq(*a, *b));
  }
};

#endif // THC_TENSORMATH_COMPARET_CUH
