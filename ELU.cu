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

// in-place variant
struct ELUupdateOutputIP_functor
{
  const float alpha_;

  ELUupdateOutputIP_functor(float alpha)
    : alpha_(alpha)
  {}

  __device__ void operator()(float *x) const
  {
    *x = *x <= 0 ? (exp(*x) - 1) * alpha_ : *x;
  }
};

void THNN_CudaELU_updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *output,
  float alpha, bool inplace)
{
  THCUNN_assertSameGPU(state, 2, input, output);

  if (inplace)
  {
    THCudaTensor_pointwiseApply1(state, input, ELUupdateOutputIP_functor(alpha));
    THCudaTensor_set(state, output, input);
  }
  else
  {
    THCudaTensor_resizeAs(state, output, input);
    THCudaTensor_pointwiseApply2(state, output, input, ELUupdateOutput_functor(alpha));
  }
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

struct ELUupdateGradInputIP_functor
{
  const float alpha_;

  ELUupdateGradInputIP_functor(float alpha)
    : alpha_(alpha)
  {}

  __device__ void operator()(float *gradOutput, const float *output) const
  {
    *gradOutput = (*output) <= 0 ? (*gradOutput * (*output + alpha_)) : (*gradOutput);
  }
};

void THNN_CudaELU_updateGradInput(THCState *state, THCudaTensor *input, THCudaTensor *gradOutput,
  THCudaTensor *gradInput, THCudaTensor *output, float alpha, bool inplace)
{
  THCUNN_assertSameGPU(state, 3, output, gradOutput, gradInput);

  if (inplace)
  {
    THCudaTensor_pointwiseApply2(state, gradOutput, output, ELUupdateGradInputIP_functor(alpha));
    THCudaTensor_set(state, gradInput, gradOutput);
  }
  else
  {
    THCudaTensor_resizeAs(state, gradInput, output);
    THCudaTensor_pointwiseApply3(state, gradInput, output, gradOutput, ELUupdateGradInput_functor(alpha));
  }
}
