#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THCUNN/generic/LeakyReLU.cu"
#else

#include <THCUNN/common.h>

void THNN_(LeakyReLU_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           accreal negval_,
           bool inplace)
{
  scalar_t negval = ScalarConvert<accreal, scalar_t>::to(negval_);

  THCUNN_assertSameGPU(state, 2, input, output);

  if (inplace)
  {
    THC_pointwiseApply1<scalar_t>(state, input, LeakyReLUUpdateOutputIP<scalar_t>(negval));
    THCTensor_(set)(state, output, input);
  }
  else
  {
    THCTensor_(resizeAs)(state, output, input);
    THC_pointwiseApply2<scalar_t, scalar_t>(state, output, input, LeakyReLUUpdateOutput<scalar_t>(negval));
  }

  THCudaCheck(cudaGetLastError());
}

void THNN_(LeakyReLU_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           accreal negval_,
           bool inplace)
{
  scalar_t negval = ScalarConvert<accreal, scalar_t>::to(negval_);

  THCUNN_check_nElement(state, input, gradOutput);
  THCUNN_assertSameGPU(state, 3, input, gradInput, gradOutput);

  if (inplace)
  {
    THC_pointwiseApply2<scalar_t, scalar_t>(state, gradOutput, input, LeakyReLUUpdateGradInputIP<scalar_t>(negval));
    THCTensor_(set)(state, gradInput, gradOutput);
  }
  else
  {
    THCTensor_(resizeAs)(state, gradInput, input);
    THC_pointwiseApply3<scalar_t, scalar_t, scalar_t>(state, gradInput, input, gradOutput, LeakyReLUUpdateGradInput<scalar_t>(negval));
  }

  THCudaCheck(cudaGetLastError());
}

#endif
