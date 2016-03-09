#include "THCUNN.h"
#include "common.h"

struct logSigmoid_updateOutput_functor
{
  __device__ void operator()(float *output, const float *input) const
  {
    float z = exp(-*input);
    *output = -log(1. + z);
  }
};

void THNN_CudaLogSigmoid_updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *output, THCudaTensor *buffer)
{
  THCUNN_assertSameGPU(state, 2, input, output);
  THCudaTensor_resizeAs(state, output, input);
  THCudaTensor_pointwiseApply2(state, output, input, logSigmoid_updateOutput_functor());
}

struct logSigmoid_updateGradInput_functor
{
  __device__ void operator()(float *gradInput, const float *input, const float *gradOutput) const
  {
    float z = exp(-*input);
    *gradInput = *gradOutput * z / (1. + z);
  }
};

void THNN_CudaLogSigmoid_updateGradInput(THCState *state, THCudaTensor *input, THCudaTensor *gradOutput,
  THCudaTensor *gradInput , THCudaTensor *buffer)
{
  THCUNN_assertSameGPU(state, 3, input, gradOutput, gradInput);
  THCudaTensor_resizeAs(state, gradInput, input);
  THCudaTensor_pointwiseApply3(state, gradInput, input, gradOutput, logSigmoid_updateGradInput_functor());
}
