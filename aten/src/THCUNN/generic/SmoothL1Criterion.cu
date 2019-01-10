#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/SmoothL1Criterion.cu"
#else

#include "THCApply.cuh"

void THNN_(SmoothL1Criterion_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *target,
           THCTensor *output,
           bool sizeAverage,
           bool reduce)
{
  THCUNN_check_shape(state, input, target);
  THCUNN_assertSameGPU(state, 3, input, target, output);
  THArgCheck(
    THCTensor_(nElement)(state, input) == THCTensor_(nElement)(state, target), 2,
    "input and target need to have the same number of elements"
  );

  if (!reduce) {
    THCTensor_(resizeAs)(state, output, input);
    THC_pointwiseApply3(state, input, target, output,
                        smoothl1_updateOutput_no_reduce_functor<real>());
    return;
  }

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
    thrust::plus<accreal>(), smoothl1_functor<real, accreal>()
  );

  if (sizeAverage)
    sum /= size;

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, target);

  THCTensor_(set1d)(state, output, 0, ScalarConvert<accreal, real>::to(sum));
}

void THNN_(SmoothL1Criterion_updateGradInput)(
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
  THArgCheck(
    THCTensor_(nElement)(state, input) == THCTensor_(nElement)(state, target), 2,
    "input and target need to have the same number of elements"
  );

  THCTensor_(resizeAs)(state, gradInput, input);

  if (!reduce) {
    THCUNN_check_shape(state, gradOutput, input);
    THC_pointwiseApply3(state, input, target, gradInput,
                        smoothl1_updateGradInput_no_reduce_functor<real>());
    THCTensor_(cmul)(state, gradInput, gradInput, gradOutput);
    return;
  }

  THCUNN_check_dim_size(state, gradOutput, 1, 0, 1);

  ptrdiff_t size = THCTensor_(nElement)(state, input);
  real norm = ScalarConvert<accreal, real>::to(sizeAverage ? accreal(1)/size : accreal(1));

  input = THCTensor_(newContiguous)(state, input);
  target = THCTensor_(newContiguous)(state, target);

  THCThrustAllocator thrustAlloc(state);
  thrust::device_ptr<real> input_data(THCTensor_(data)(state, input));
  thrust::device_ptr<real> target_data(THCTensor_(data)(state, target));
  thrust::device_ptr<real> gradInput_data(THCTensor_(data)(state, gradInput));

  thrust::transform(
#if CUDA_VERSION >= 7000
    thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
#endif
    input_data, input_data+size, target_data, gradInput_data,
    smoothl1_updateGradInput_functor<real>(norm, THCTensor_(get1d)(state, gradOutput, 0))
  );

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, target);
}

#endif
