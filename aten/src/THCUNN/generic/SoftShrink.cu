#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THCUNN/generic/SoftShrink.cu"
#else

#include <THCUNN/common.h>

void THNN_(SoftShrink_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           accreal lambda_)
{
  scalar_t lambda = ScalarConvert<accreal, scalar_t>::to(lambda_);
  THCUNN_assertSameGPU(state, 2, input, output);
  THCTensor_(resizeAs)(state, output, input);
  THC_pointwiseApply2<scalar_t, scalar_t>(state, output, input, SoftShrinkUpdateOutput<scalar_t>(lambda));
  THCudaCheck(cudaGetLastError());
}

void THNN_(SoftShrink_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           accreal lambda_)
{
  scalar_t lambda = ScalarConvert<accreal, scalar_t>::to(lambda_);
  THCUNN_check_nElement(state, input, gradOutput);
  THCUNN_assertSameGPU(state, 3, input, gradOutput, gradInput);
  THCTensor_(resizeAs)(state, gradInput, input);
  THC_pointwiseApply3<scalar_t, scalar_t, scalar_t>(state, gradInput, input, gradOutput, SoftShrinkUpdateGradInput<scalar_t>(lambda));
  THCudaCheck(cudaGetLastError());
}

#endif
