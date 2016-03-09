#include "THCUNN.h"
#include "common.h"

struct sigmoidupdateOutput_functor
{
  __device__ void operator()(float *output, const float *input) const
  {
    *output = 1./(1.+ exp(-*input));
  }
};

void THNN_CudaSigmoid_updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *output)
{
  THCUNN_assertSameGPU(state, 2, input, output);
  THCudaTensor_resizeAs(state, output, input);
  THCudaTensor_pointwiseApply2(state, output, input, sigmoidupdateOutput_functor());
}

struct sigmoidupdateGradInput_functor
{
  __device__ void operator()(float *gradInput, const float *output, const float *gradOutput) const
  {
    *gradInput = *gradOutput * (1.-*output) * (*output);
  }
};

void THNN_CudaSigmoid_updateGradInput(THCState *state, THCudaTensor *input, THCudaTensor *gradOutput, THCudaTensor *gradInput, THCudaTensor *output)
{
  THCUNN_assertSameGPU(state, 3, output, gradOutput, gradInput);
  THCudaTensor_resizeAs(state, gradInput, output);
  THCudaTensor_pointwiseApply3(state, gradInput, output, gradOutput, sigmoidupdateGradInput_functor());
}
