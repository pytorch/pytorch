#include "THCUNN.h"
#include "common.h"

#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

const float eps = 1e-12f;

struct bce_functor
{
  template <class Tuple>
  __host__ __device__
  float operator()(Tuple x)
  {
    float o = thrust::get<0>(x);
    float t = thrust::get<1>(x);
    return - (t * logf(o + eps) + (1.f - t) * logf(1.f - o + eps));
  }
};

struct bce_functor_weights
{
  template <class Tuple>
  __host__ __device__
  float operator()(Tuple x)
  {
    float o = thrust::get<0>(x);
    float t = thrust::get<1>(x);
    float w = thrust::get<2>(x);
    return - w * (t * logf(o + eps) + (1.f - t) * logf(1.f - o + eps));
  }
};

void THNN_CudaBCECriterion_updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *target, THCudaTensor *output, bool sizeAverage, THCudaTensor *weights)
{
  THCUNN_assertSameGPU(state, 3, input, target, weights);

  long size = THCudaTensor_nElement(state, input);

  input = THCudaTensor_newContiguous(state, input);
  target = THCudaTensor_newContiguous(state, target);

  thrust::device_ptr<float> input_data(THCudaTensor_data(state, input));
  thrust::device_ptr<float> target_data(THCudaTensor_data(state, target));

  float sum;
  if (weights) {
    weights = THCudaTensor_newContiguous(state, weights);
    thrust::device_ptr<float> weights_data(THCudaTensor_data(state, weights));
    sum = thrust::transform_reduce(
      thrust::make_zip_iterator(thrust::make_tuple(input_data, target_data, weights_data)),
      thrust::make_zip_iterator(thrust::make_tuple(input_data+size, target_data+size, weights_data+size)),
      bce_functor_weights(),
      (float) 0.f,
      thrust::plus<float>()
    );
    THCudaTensor_free(state, weights);
  } else {
    sum = thrust::transform_reduce(
      thrust::make_zip_iterator(thrust::make_tuple(input_data, target_data)),
      thrust::make_zip_iterator(thrust::make_tuple(input_data+size, target_data+size)),
      bce_functor(),
      (float) 0.f,
      thrust::plus<float>()
    );
  }

  if (sizeAverage)
    sum /= size;

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, target);

  THCudaTensor_set1d(state, output, 0, sum);
}

struct bce_updateGradInput_functor
{
  const float norm;

  bce_updateGradInput_functor(float norm_)
    : norm(norm_)
  {}

  template <class Tuple>
  __host__ __device__
  float operator()(Tuple x)
  {
    float o = thrust::get<0>(x);
    float t = thrust::get<1>(x);
    return - (t - o) / ((1 - o + eps) * (o + eps)) * norm;
  }
};

struct bce_updateGradInput_functor_weights
{
  const float norm;

  bce_updateGradInput_functor_weights(float norm_)
    : norm(norm_)
  {}

  template <class Tuple>
  __host__ __device__
  float operator()(Tuple x)
  {
    float o = thrust::get<0>(x);
    float t = thrust::get<1>(x);
    float w = thrust::get<2>(x);
    return - (t - o) / ((1 - o + eps) * (o + eps)) * norm * w;
  }
};

void THNN_CudaBCECriterion_updateGradInput(THCState *state, THCudaTensor *input, THCudaTensor *target, THCudaTensor *gradInput, bool sizeAverage, THCudaTensor *weights)
{
  THCUNN_assertSameGPU(state, 4, input, target, gradInput, weights);

  long size = THCudaTensor_nElement(state, input);
  float norm = (sizeAverage ? 1./size : 1.);

  input = THCudaTensor_newContiguous(state, input);
  target = THCudaTensor_newContiguous(state, target);

  THCudaTensor_resizeAs(state, gradInput, input);

  thrust::device_ptr<float> input_data(THCudaTensor_data(state, input));
  thrust::device_ptr<float> target_data(THCudaTensor_data(state, target));
  thrust::device_ptr<float> gradInput_data(THCudaTensor_data(state, gradInput));

  if (weights) {
    weights = THCudaTensor_newContiguous(state, weights);
    thrust::device_ptr<float> weights_data(THCudaTensor_data(state, weights));
    thrust::transform(
      thrust::make_zip_iterator(thrust::make_tuple(input_data, target_data, weights_data)),
      thrust::make_zip_iterator(thrust::make_tuple(input_data+size, target_data+size, weights_data+size)),
      gradInput_data,
      bce_updateGradInput_functor_weights(norm)
    );
    THCudaTensor_free(state, weights);
  } else {
    thrust::transform(
      thrust::make_zip_iterator(thrust::make_tuple(input_data, target_data)),
      thrust::make_zip_iterator(thrust::make_tuple(input_data+size, target_data+size)),
      gradInput_data,
      bce_updateGradInput_functor(norm)
    );
  }

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, target);
}
