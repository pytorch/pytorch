#include "THCUNN.h"
#include "common.h"

struct ThresholdUpdateOutput
{
  const float threshold_;
  const float val_;

  ThresholdUpdateOutput(float threshold, float val)
    : threshold_(threshold)
    , val_(val)
  {}

  __device__ __forceinline__ void operator()(float *out, float *in)
  {
    float x = *in;
    *out = (x > threshold_) ? x : val_;
  }
};

// in-place variant
struct ThresholdUpdateOutputIP
{
  const float threshold_;
  const float val_;

  ThresholdUpdateOutputIP(float threshold, float val)
    : threshold_(threshold)
    , val_(val)
  {}

  __device__ __forceinline__ void operator()(float *x)
  {
    *x = (*x > threshold_) ? *x : val_;
  }
};

void THNN_CudaThreshold_updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *output,
  double threshold, double val, bool inplace)
{
  THCUNN_assertSameGPU(state, 2, input, output);

  if (inplace)
  {
    THC_pointwiseApply1(state, input,
      ThresholdUpdateOutputIP(threshold, val)
    );
    THCudaTensor_set(state, output, input);
  }
  else
  {
    THCudaTensor_resizeAs(state, output, input);
    THC_pointwiseApply2(state, output, input,
      ThresholdUpdateOutput(threshold, val)
    );
  }

  THCudaCheck(cudaGetLastError());
}

struct ThresholdUpdateGradInput
{
  const float threshold_;

  ThresholdUpdateGradInput(float threshold)
    : threshold_(threshold)
  {}

  __device__ __forceinline__ void operator()(
    float *gradInput, float *input, float *gradOutput) const
  {
    *gradInput = (*input > threshold_) ? *gradOutput : 0;
  }
};

struct ThresholdUpdateGradInputIP
{
  const float threshold_;

  ThresholdUpdateGradInputIP(float threshold)
    : threshold_(threshold)
  {}

  __device__ __forceinline__ void operator()(
    float *gradOutput, float *input) const
  {
    *gradOutput = (*input > threshold_) ? *gradOutput : 0;
  }
};

void THNN_CudaThreshold_updateGradInput(THCState *state, THCudaTensor *input, THCudaTensor *gradOutput,
  THCudaTensor *gradInput, double threshold, bool inplace)
{
  THCUNN_assertSameGPU(state, 3, input, gradInput, gradOutput);

  if (inplace)
  {
    THC_pointwiseApply2(state, gradOutput, input,
      ThresholdUpdateGradInputIP(threshold)
    );
    THCudaTensor_set(state, gradInput, gradOutput);
  }
  else
  {
    THCudaTensor_resizeAs(state, gradInput, input);
    THC_pointwiseApply3(state, gradInput, input, gradOutput,
       ThresholdUpdateGradInput(threshold)
    );
  }

  THCudaCheck(cudaGetLastError());
}
