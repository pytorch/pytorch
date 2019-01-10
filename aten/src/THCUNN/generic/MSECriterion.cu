#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/MSECriterion.cu"
#else

#include "THCApply.cuh"

void THNN_(MSECriterion_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *target,
           THCTensor *output,
           bool sizeAverage,
           bool reduce)
{
  THCUNN_check_shape(state, input, target);
  THCUNN_assertSameGPU(state, 3, input, target, output);

  if (reduce) {
    THCTensor_(resize1d)(state, output, 1);

    ptrdiff_t size = THCTensor_(nElement)(state, input);

    input = THCTensor_(newContiguous)(state, input);
    target = THCTensor_(newContiguous)(state, target);

    THCThrustAllocator thrustAlloc(state);
    thrust::device_ptr<real> input_data(THCTensor_(data)(state, input));
    thrust::device_ptr<real> target_data(THCTensor_(data)(state, target));
    accreal sum = thrust::inner_product(
#if CUDA_VERSION >= 7000
      thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
#endif
      input_data, input_data+size, target_data, (accreal) 0,
      thrust::plus<accreal>(), mse_functor<real, accreal>());

    if (sizeAverage)
      sum /= size;

    THCTensor_(free)(state, input);
    THCTensor_(free)(state, target);

    THCTensor_(set1d)(state, output, 0, ScalarConvert<accreal, real>::to(sum));
    return;
  }

  THCTensor_(resizeAs)(state, output, input);
  THC_pointwiseApply3(
      state,
      input,
      target,
      output,
      mse_updateOutput_functor<real>());
}

void THNN_(MSECriterion_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *target,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           bool sizeAverage,
           bool reduce)
{
  THCUNN_check_shape(state, input, target);
  THCUNN_assertSameGPU(state, 4, input, target, gradInput, gradOutput);

  if (reduce) {
    ptrdiff_t size = THCTensor_(nElement)(state, input);

    THCUNN_check_dim_size(state, gradOutput, 1, 0, 1);
    accreal norm = sizeAverage ? (accreal)(2)/size : (accreal)(2);
    norm *= ScalarConvert<real, accreal>::to(THCTensor_(get1d)(state, gradOutput, 0));

    input = THCTensor_(newContiguous)(state, input);
    target = THCTensor_(newContiguous)(state, target);

    THCTensor_(resizeAs)(state, gradInput, input);

    THCThrustAllocator thrustAlloc(state);
    thrust::device_ptr<real> input_data(THCTensor_(data)(state, input));
    thrust::device_ptr<real> target_data(THCTensor_(data)(state, target));
    thrust::device_ptr<real> gradInput_data(THCTensor_(data)(state, gradInput));

    thrust::transform(
#if CUDA_VERSION >= 7000
      thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
#endif
      input_data, input_data+size, target_data, gradInput_data,
      mse_updateGradInput_functor<real, accreal>(norm));

    THCTensor_(free)(state, input);
    THCTensor_(free)(state, target);
    return;
  }

  THCUNN_check_shape(state, input, gradOutput);
  ptrdiff_t size = THCTensor_(nElement)(state, input);

  input = THCTensor_(newContiguous)(state, input);
  target = THCTensor_(newContiguous)(state, target);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);

  THCTensor_(resizeAs)(state, gradInput, input);

  THCThrustAllocator thrustAlloc(state);
  thrust::device_ptr<real> input_data(THCTensor_(data)(state, input));
  thrust::device_ptr<real> target_data(THCTensor_(data)(state, target));
  thrust::device_ptr<real> gradOutput_data(THCTensor_(data)(state, gradOutput));
  thrust::device_ptr<real> gradInput_data(THCTensor_(data)(state, gradInput));

  thrust::transform(
#if CUDA_VERSION >= 7000
    thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
#endif
    input_data, input_data+size, target_data, gradInput_data,
    mse_updateGradInput_functor<real, accreal>(2));

  thrust::transform(
#if CUDA_VERSION >= 7000
    thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
#endif
    gradInput_data, gradInput_data+size, gradOutput_data, gradInput_data,
    thrust::multiplies<real>());

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, target);
  THCTensor_(free)(state, gradOutput);
}

#endif
