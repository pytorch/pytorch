#include "THCUNN.h"
#include "common.h"

struct softPlusupdateOutput_functor
{
  const float threshold;
  const float beta;

  softPlusupdateOutput_functor(float threshold_, float beta_)
    : threshold(threshold_)
    , beta(beta_)
  {}

  __device__ void operator()(float *output, const float *input) const
  {
    float betain = beta * (*input);
    *output = ((betain) > threshold) ? *input : (1/beta) * log1p(exp(betain));
  }
};

void THNN_CudaSoftPlus_updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *output, float beta, float threshold)
{
  THCUNN_assertSameGPU(state, 2, input, output);
  THCudaTensor_resizeAs(state, output, input);
  THC_pointwiseApply2(state, output, input, softPlusupdateOutput_functor(threshold, beta));
}

struct softPlusupdateGradInput_functor
{
  const float threshold;
  const float beta;

  softPlusupdateGradInput_functor(float threshold_, float beta_)
    : threshold(threshold_)
    , beta(beta_)
  {}

  __device__ void operator()(float *gradInput, const float *output, const float *gradOutput) const
  {
    float betaout = beta * (*output);
    float exp_bo = exp(betaout);
    *gradInput = ((betaout) > threshold) ? *gradOutput : *gradOutput * (exp_bo - 1) / exp_bo;
  }
};

void THNN_CudaSoftPlus_updateGradInput(THCState *state, THCudaTensor *input, THCudaTensor *gradOutput, THCudaTensor *gradInput,
  THCudaTensor *output, float beta, float threshold)
{
  THCUNN_assertSameGPU(state, 4, input, output, gradOutput, gradInput);
  THCudaTensor_resizeAs(state, gradInput, output);
  THC_pointwiseApply3(state, gradInput, output, gradOutput, softPlusupdateGradInput_functor(threshold, beta));
}
