#ifndef THC_TENSORMATH_COMPARE_CUH
#define THC_TENSORMATH_COMPARE_CUH

#include <THC/THCTensorMath.h>
#include <THC/THCGeneral.h>
#include <THC/THCTensorCopy.h>
#include <THC/THCApply.cuh>
#include <THC/THCNumerics.cuh>

template<typename ScalarTypeOut, typename ScalarType, typename TensorTypeOut, typename TensorType, class Op>
void THC_logicalValue(THCState *state,
                      TensorTypeOut *self_,
                      TensorType *src,
                      Op op) {
  THCTensor_resize(state, self_, src->sizes(), {});

  if (!THC_pointwiseApply2<ScalarTypeOut, ScalarType>(state, self_, src, op)) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

#endif // THC_TENSORMATH_COMPARE_CUH
