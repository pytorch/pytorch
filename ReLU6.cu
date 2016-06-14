#include "THCUNN.h"
#include "common.h"

struct ReLU6UpdateOutput
{
  ReLU6UpdateOutput() {}

  __device__ __forceinline__ void operator()(float *out, float *in)
  {
    float x = *in;
    *out = (x > 0) ? ((x < 6) ? x : 6) : 0;
  }
};

// in-place variant
struct ReLU6UpdateOutputIP
{
  ReLU6UpdateOutputIP() {}

  __device__ __forceinline__ void operator()(float *x)
  {
    *x = (*x > 0) ? ((*x < 6) ? *x : 6) : 0;
  }
};

void THNN_CudaReLU6_updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *output,
  bool inplace)
{
  THCUNN_assertSameGPU(state, 2, input, output);

  if (inplace)
  {
    THC_pointwiseApply1(state, input,
      ReLU6UpdateOutputIP()
    );
    THCudaTensor_set(state, output, input);
  }
  else
  {
    THCudaTensor_resizeAs(state, output, input);
    THC_pointwiseApply2(state, output, input,
      ReLU6UpdateOutput()
    );
  }

  THCudaCheck(cudaGetLastError());
}

struct ReLU6UpdateGradInput
{
  ReLU6UpdateGradInput() {}

  __device__ __forceinline__ void operator()(
    float *gradInput, float *input, float *gradOutput) const
  {
    *gradInput = (*input > 0 && *input < 6) ? *gradOutput : 0;
  }
};

struct ReLU6UpdateGradInputIP
{
  ReLU6UpdateGradInputIP() {}

  __device__ __forceinline__ void operator()(
    float *gradOutput, float *input) const
  {
    *gradOutput = (*input > 0 && *input < 6) ? *gradOutput : 0;
  }
};

void THNN_CudaReLU6_updateGradInput(THCState *state, THCudaTensor *input, THCudaTensor *gradOutput,
  THCudaTensor *gradInput, bool inplace)
{
  THCUNN_assertSameGPU(state, 3, input, gradInput, gradOutput);

  if (inplace)
  {
    THC_pointwiseApply2(state, gradOutput, input,
      ReLU6UpdateGradInputIP()
    );
    THCudaTensor_set(state, gradInput, gradOutput);
  }
  else
  {
    THCudaTensor_resizeAs(state, gradInput, input);
    THC_pointwiseApply3(state, gradInput, input, gradOutput,
       ReLU6UpdateGradInput()
    );
  }

  THCudaCheck(cudaGetLastError());
}
