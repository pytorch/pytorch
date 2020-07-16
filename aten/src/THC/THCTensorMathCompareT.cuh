#ifndef THC_TENSORMATH_COMPARET_CUH
#define THC_TENSORMATH_COMPARET_CUH

#include <THC/THCTensorMath.h>
#include <THC/THCGeneral.h>
#include <THC/THCTensorCopy.h>
#include <THC/THCApply.cuh>
#include <THC/THCNumerics.cuh>
#include <THC/THCReduce.cuh>

template <typename T, typename TOut>
struct TensorEQOp {
  __device__ inline void operator()(TOut* out, T* a, T* b) {
    *out = ScalarConvert<bool, TOut>::to(THCNumerics<T>::eq(*a, *b));
  }
};

#endif // THC_TENSORMATH_COMPARET_CUH
