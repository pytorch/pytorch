#include "THCUNN.h"
#include "common.h"

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

struct l1cost_functor
{
  __host__ __device__ float operator()(float x, float y) const
  {
    return abs(x) + abs(y);
  }
};

void THNN_CudaL1Cost_updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *output)
{
  THCUNN_assertSameGPU(state, 1, input);
  float sum;
  long size = THCudaTensor_nElement(state, input);
  input = THCudaTensor_newContiguous(state, input);
  thrust::device_ptr<float> input_data(THCudaTensor_data(state, input));
  sum = thrust::reduce(input_data, input_data+size, (float) 0, l1cost_functor());

  THCudaTensor_free(state, input);

  THCudaTensor_set1d(state, output, 0, sum);
}

struct l1cost_updateGradInput_functor
{
  __host__ __device__ float operator()(float x) const
  {
    if (x > 0)
      return 1;
    else if (x < 0)
      return -1;
    else
      return 0;
  }
};

void THNN_CudaL1Cost_updateGradInput(THCState *state, THCudaTensor *input, THCudaTensor *gradOutput, THCudaTensor *gradInput)
{
  THCUNN_assertSameGPU(state, 2, input, gradInput);
  long size = THCudaTensor_nElement(state, input);

  input = THCudaTensor_newContiguous(state, input);
  THCudaTensor_resizeAs(state, gradInput, input);

  thrust::device_ptr<float> input_data(THCudaTensor_data(state, input));
  thrust::device_ptr<float> gradInput_data(THCudaTensor_data(state, gradInput));

  thrust::transform(input_data, input_data+size, gradInput_data, l1cost_updateGradInput_functor());

  THCudaTensor_free(state, input);
}
