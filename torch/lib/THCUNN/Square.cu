#include "THCUNN.h"
#include "common.h"

struct squareupdateOutput_functor
{
  __device__ void operator()(float* output, const float* input) const
  {
    *output = (*input) * (*input);
  }
};

void THNN_CudaSquare_updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *output)
{
  THCUNN_assertSameGPU(state, 2, input, output);
  THCudaTensor_resizeAs(state, output, input);
  THC_pointwiseApply2(state, output, input, squareupdateOutput_functor());
}

struct squareupdateGradInput_functor
{
  __device__ void operator()(float* gradInput, const float* input, const float* gradOutput) const
  {
    *gradInput = 2.0 * (*gradOutput) * (*input);
  }
};

void THNN_CudaSquare_updateGradInput(THCState *state, THCudaTensor *input, THCudaTensor *gradOutput, THCudaTensor *gradInput)
{
  THCUNN_assertSameGPU(state, 3, input, gradOutput, gradInput);
  THCudaTensor_resizeAs(state, gradInput, input);
  THC_pointwiseApply3(state, gradInput, input, gradOutput, squareupdateGradInput_functor());
}
