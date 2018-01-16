#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/Threshold.cu"
#else

#include "../common.h"

void THNN_(Threshold_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           accreal threshold_,
           accreal val_,
           bool inplace)
{
  real threshold = ScalarConvert<accreal, real>::to(threshold_);
  real val = ScalarConvert<accreal, real>::to(val_);
  THCUNN_assertSameGPU(state, 2, input, output);

  if (inplace)
  {
    THC_pointwiseApply1(state, input,
      ThresholdUpdateOutputIP<real>(threshold, val)
    );
    THCTensor_(set)(state, output, input);
  }
  else
  {
    THCTensor_(resizeAs)(state, output, input);
    THC_pointwiseApply2(state, output, input,
      ThresholdUpdateOutput<real>(threshold, val)
    );
  }

  THCudaCheck(cudaGetLastError());
}

void THNN_(Threshold_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           accreal threshold_,
           accreal val_,
           bool inplace)
{
  real threshold = ScalarConvert<accreal, real>::to(threshold_);
  real val = ScalarConvert<accreal, real>::to(val_);
  (void) val;
  THCUNN_check_nElement(state, input, gradOutput);
  THCUNN_assertSameGPU(state, 3, input, gradInput, gradOutput);

  if (inplace)
  {
    THC_pointwiseApply2(state, gradOutput, input,
      ThresholdUpdateGradInputIP<real>(threshold)
    );
    THCTensor_(set)(state, gradInput, gradOutput);
  }
  else
  {
    THCTensor_(resizeAs)(state, gradInput, input);
    THC_pointwiseApply3(state, gradInput, input, gradOutput,
       ThresholdUpdateGradInput<real>(threshold)
    );
  }

  THCudaCheck(cudaGetLastError());
}

#endif
