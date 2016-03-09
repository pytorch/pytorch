#include "THCUNN.h"
#include "common.h"

struct hardtanhupdateOutput_functor
{
  const float max_val_;
  const float min_val_;

  hardtanhupdateOutput_functor(float min_val, float max_val)
    : min_val_(min_val)
    , max_val_(max_val)
  {}

  __device__ void operator()(float *output, const float *input) const
  {
    if (*input < min_val_)
      *output = min_val_;
    else if (*input <= max_val_)
      *output = *input;
    else
      *output = max_val_;
  }
};

void THNN_CudaHardTanh_updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *output, float min_val, float max_val)
{
  THCUNN_assertSameGPU(state, 2, input, output);
  THCudaTensor_resizeAs(state, output, input);
  THCudaTensor_pointwiseApply2(state, output, input,
                               hardtanhupdateOutput_functor(min_val, max_val));
}

struct hardtanhupdateGradInput_functor
{
  const float max_val_;
  const float min_val_;

  hardtanhupdateGradInput_functor(float min_val, float max_val)
    : min_val_(min_val)
    , max_val_(max_val)
  {}

  __device__ void operator()(float *gradInput, const float *input, const float *gradOutput) const
  {
    if (*input < min_val_ || *input > max_val_)
      *gradInput = 0;
    else
      *gradInput = *gradOutput;
  }
};

void THNN_CudaHardTanh_updateGradInput(THCState *state, THCudaTensor *input, THCudaTensor *gradOutput, THCudaTensor *gradInput, float min_val, float max_val)
{
  THCUNN_assertSameGPU(state, 3, input, gradOutput, gradInput);

  THCudaTensor_resizeAs(state, gradInput, input);
  THCudaTensor_pointwiseApply3(state, gradInput, input, gradOutput,
                               hardtanhupdateGradInput_functor(min_val, max_val));
}
