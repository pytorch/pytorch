#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMathPairwise.cu"
#else

#include <ATen/NamedTensorUtils.h>

#if !defined(THC_REAL_IS_BOOL)

void THCTensor_(mul)(THCState *state, THCTensor *self_, THCTensor *src_, scalar_t value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src_));
  if (self_ == src_) {
    if (!THC_pointwiseApply1<scalar_t>(state, self_, TensorMulConstantOp<scalar_t>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src_);

    if (!THC_pointwiseApply2<scalar_t, scalar_t>(state, self_, src_, TensorMulConstantOp<scalar_t>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

void THCTensor_(fmod)(THCState *state, THCTensor *self_, THCTensor *src_, scalar_t value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src_));
  if (self_ == src_) {
    if (!THC_pointwiseApply1<scalar_t>(state, self_, TensorFmodOp<scalar_t>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src_);

    if (!THC_pointwiseApply2<scalar_t, scalar_t>(state, self_, src_, TensorFmodOp<scalar_t>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

#endif

#endif
