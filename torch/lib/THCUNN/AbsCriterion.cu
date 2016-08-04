#include "THCUNN.h"
#include "common.h"

#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>

struct abs_functor
{
  __host__ __device__ float operator()(const float& x, const float& y) const
  {
    float z = x-y;
    return z >= 0 ? z : -z;
  }
};

void THNN_CudaAbsCriterion_updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *target, THCudaTensor *output, bool sizeAverage)
{
  THCUNN_assertSameGPU(state, 2, input, target);

  long size = THCudaTensor_nElement(state, input);

  input = THCudaTensor_newContiguous(state, input);
  target = THCudaTensor_newContiguous(state, target);

  thrust::device_ptr<float> input_data(THCudaTensor_data(state, input));
  thrust::device_ptr<float> target_data(THCudaTensor_data(state, target));
  float sum = thrust::inner_product(input_data, input_data+size, target_data, (float) 0, thrust::plus<float>(), abs_functor());

  if (sizeAverage)
    sum /= size;

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, target);

  THCudaTensor_set1d(state, output, 0, sum);
}

struct abs_updateGradInput_functor
{
  const float norm;

  abs_updateGradInput_functor(float norm_)
    : norm(norm_)
  {}

  __host__ __device__ float operator()(const float& x, const float& y) const
  {
    return (x - y) >= 0 ? norm : -norm;
  }
};

void THNN_CudaAbsCriterion_updateGradInput(THCState *state, THCudaTensor *input, THCudaTensor *target, THCudaTensor *gradInput, bool sizeAverage)
{
  THCUNN_assertSameGPU(state, 3, input, target, gradInput);

  long size = THCudaTensor_nElement(state, input);
  float norm = (sizeAverage ? 1./size : 1.);

  input = THCudaTensor_newContiguous(state, input);
  target = THCudaTensor_newContiguous(state, target);

  THCudaTensor_resizeAs(state, gradInput, input);

  thrust::device_ptr<float> input_data(THCudaTensor_data(state, input));
  thrust::device_ptr<float> target_data(THCudaTensor_data(state, target));
  thrust::device_ptr<float> gradInput_data(THCudaTensor_data(state, gradInput));

  thrust::transform(input_data, input_data+size, target_data, gradInput_data, abs_updateGradInput_functor(norm));

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, target);
}
