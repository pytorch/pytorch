#include "THCUNN.h"
#include "common.h"

struct ELUupdateOutput_functor
{
  const float alpha_;

  ELUupdateOutput_functor(float alpha)
    : alpha_(alpha)
  {}

  __device__ void operator()(float *output, const float *input) const
  {
    *output = *input <= 0 ? (exp(*input) - 1) * alpha_ : *input;
  }
};

void THNN_CudaELU_updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *output, float alpha)
{
  THCUNN_assertSameGPU(state, 2, input, output);
  THCudaTensor_resizeAs(state, output, input);
  THCudaTensor_pointwiseApply2(state, output, input, ELUupdateOutput_functor(alpha));
}

struct ELUupdateGradInput_functor
{
  const float alpha_;

  ELUupdateGradInput_functor(float alpha)
    : alpha_(alpha)
  {}

  __device__ void operator()(float *gradInput, const float *output, const float *gradOutput) const
  {
    *gradInput = (*output) <= 0 ? (*gradOutput * (*output + alpha_)) : (*gradOutput);
  }
};

void THNN_CudaELU_updateGradInput(THCState *state, THCudaTensor *input, THCudaTensor *gradOutput,
  THCudaTensor *gradInput, THCudaTensor *output, float alpha)
{
  THCUNN_assertSameGPU(state, 3, output, gradOutput, gradInput);
  THCudaTensor_resizeAs(state, gradInput, output);
  THCudaTensor_pointwiseApply3(state, gradInput, output, gradOutput, ELUupdateGradInput_functor(alpha));
}
