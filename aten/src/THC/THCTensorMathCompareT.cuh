#ifndef THC_TENSORMATH_COMPARET_CUH
#define THC_TENSORMATH_COMPARET_CUH

#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCTensorCopy.h"
#include "THCApply.cuh"
#include "THCNumerics.cuh"
#include "THCReduce.cuh"

template <typename T, typename TOut>
struct TensorLTOp {
  __device__ inline void operator()(TOut* out, T* a, T* b) {
    *out = ScalarConvert<bool, TOut>::to(THCNumerics<T>::lt(*a, *b));
  }
};

template <typename T, typename TOut>
struct TensorGTOp {
  __device__ inline void operator()(TOut* out, T* a, T* b) {
    *out = ScalarConvert<bool, TOut>::to(THCNumerics<T>::gt(*a, *b));
  }
};

template <typename T, typename TOut>
struct TensorLEOp {
  __device__ inline void operator()(TOut* out, T* a, T* b) {
    *out = ScalarConvert<bool, TOut>::to(THCNumerics<T>::le(*a, *b));
  }
};

template <typename T, typename TOut>
struct TensorGEOp {
  __device__ inline void operator()(TOut* out, T* a, T* b) {
    *out = ScalarConvert<bool, TOut>::to(THCNumerics<T>::ge(*a, *b));
  }
};

template <typename T, typename TOut>
struct TensorEQOp {
  __device__ inline void operator()(TOut* out, T* a, T* b) {
    *out = ScalarConvert<bool, TOut>::to(THCNumerics<T>::eq(*a, *b));
  }
};

template <typename T, typename TOut>
struct TensorNEOp {
  __device__ inline void operator()(TOut* out, T* a, T* b) {
    *out = ScalarConvert<bool, TOut>::to(THCNumerics<T>::ne(*a, *b));
  }
};

template<typename TensorType, typename TensorTypeOut, typename Op>
void THC_logicalTensor(THCState *state,
                       TensorTypeOut *self_,
                       TensorType *src1,
                       TensorType *src2,
                       Op op) {
  THLongStorage* st = TensorUtils<TensorType>::newSizeOf(state, src1);
  TensorUtils<TensorTypeOut>::resize(state, self_, st, NULL);
  THLongStorage_free(st);

  THArgCheck(TensorUtils<TensorType>::getNumElements(state, src1) ==
             TensorUtils<TensorType>::getNumElements(state, src2), 3,
             "sizes do not match");

  if (!THC_pointwiseApply3(state, self_, src1, src2, op)) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

#endif // THC_TENSORMATH_COMPARET_CUH
