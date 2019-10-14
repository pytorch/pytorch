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

template<typename ScalarTypeOut, typename ScalarType, typename TensorTypeOut, typename TensorType, typename Op>
void THC_logicalTensor(THCState *state,
                       TensorTypeOut *self_,
                       TensorType *src1,
                       TensorType *src2,
                       Op op) {
  THCTensor_resize(state, self_, src1->sizes(), {});

  THArgCheck(THCTensor_nElement(state, src1) ==
             THCTensor_nElement(state, src2), 3,
             "sizes do not match");

  if (!THC_pointwiseApply3<ScalarTypeOut, ScalarType, ScalarType>(state, self_, src1, src2, op)) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

#endif // THC_TENSORMATH_COMPARET_CUH
