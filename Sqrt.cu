#include "THCUNN.h"
#include "common.h"

struct sqrtupdateOutput_functor
{
  const float bias;

  sqrtupdateOutput_functor(float bias_)
    : bias(bias_)
  {}

  __device__ void operator()(float *output, const float *input) const
  {
    *output = sqrt(*input + bias);
  }
};

void THNN_CudaSqrt_updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *output, float eps)
{
  THCUNN_assertSameGPU(state, 2, input, output);
  THCudaTensor_resizeAs(state, output, input);
  THC_pointwiseApply2(state, output, input, sqrtupdateOutput_functor(eps));
}

struct sqrtupdateGradInput_functor
{
  sqrtupdateGradInput_functor() {}

  __device__ void operator()(float *gradInput, const float *output, const float *gradOutput) const
  {
    *gradInput = (*output == 0.0f) ? 0.0f : ((0.5f * *gradOutput) / *output);
  }
};

void THNN_CudaSqrt_updateGradInput(THCState *state, THCudaTensor *input, THCudaTensor *gradOutput, THCudaTensor *gradInput, THCudaTensor *output)
{
  THCUNN_assertSameGPU(state, 3, output, gradOutput, gradInput);
  THCudaTensor_resizeAs(state, gradInput, output);
  THC_pointwiseApply3(state, gradInput, output, gradOutput, sqrtupdateGradInput_functor());
}
