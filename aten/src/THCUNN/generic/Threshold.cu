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
  scalar_t threshold = ScalarConvert<accreal, scalar_t>::to(threshold_);
  scalar_t val = ScalarConvert<accreal, scalar_t>::to(val_);
  THCUNN_assertSameGPU(state, 2, input, output);

  if (inplace)
  {
    THC_pointwiseApply1<scalar_t>(state, input,
      ThresholdUpdateOutputIP<scalar_t>(threshold, val)
    );
    THCTensor_(set)(state, output, input);
  }
  else
  {
    THCTensor_(resizeAs)(state, output, input);
    THC_pointwiseApply2<scalar_t, scalar_t>(state, output, input,
      ThresholdUpdateOutput<scalar_t>(threshold, val)
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
  scalar_t threshold = ScalarConvert<accreal, scalar_t>::to(threshold_);
  scalar_t val = ScalarConvert<accreal, scalar_t>::to(val_);
  (void) val;
  THCUNN_check_nElement(state, input, gradOutput);
  THCUNN_assertSameGPU(state, 3, input, gradInput, gradOutput);

  if (inplace)
  {
    THC_pointwiseApply2<scalar_t, scalar_t>(state, gradOutput, input,
      ThresholdUpdateGradInputIP<scalar_t>(threshold)
    );
    THCTensor_(set)(state, gradInput, gradOutput);
  }
  else
  {
    THCTensor_(resizeAs)(state, gradInput, input);
    THC_pointwiseApply3<scalar_t, scalar_t, scalar_t>(state, gradInput, input, gradOutput,
       ThresholdUpdateGradInput<scalar_t>(threshold)
    );
  }

  THCudaCheck(cudaGetLastError());
}

#endif
