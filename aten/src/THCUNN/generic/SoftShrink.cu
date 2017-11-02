#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/SoftShrink.cu"
#else

#include "../common.h"

void THNN_(SoftShrink_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           accreal lambda_)
{
  real lambda = ScalarConvert<accreal, real>::to(lambda_);
  THCUNN_assertSameGPU(state, 2, input, output);
  THCTensor_(resizeAs)(state, output, input);
  THC_pointwiseApply2(state, output, input, SoftShrinkUpdateOutput<real>(lambda));
  THCudaCheck(cudaGetLastError());
}

void THNN_(SoftShrink_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           accreal lambda_)
{
  real lambda = ScalarConvert<accreal, real>::to(lambda_);
  THCUNN_check_nElement(state, input, gradOutput);
  THCUNN_assertSameGPU(state, 3, input, gradOutput, gradInput);
  THCTensor_(resizeAs)(state, gradInput, input);
  THC_pointwiseApply3(state, gradInput, input, gradOutput, SoftShrinkUpdateGradInput<real>(lambda));
  THCudaCheck(cudaGetLastError());
}

#endif
