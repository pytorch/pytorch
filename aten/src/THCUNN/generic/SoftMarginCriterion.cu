#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/SoftMarginCriterion.cu"
#else

#include "THCApply.cuh"

void THNN_(SoftMarginCriterion_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *target,
           THCTensor *output,
           bool sizeAverage,
           bool reduce)
{
  THCUNN_check_shape(state, input, target);
  THCUNN_assertSameGPU(state, 3, input, target, output);

  if (!reduce) {
    THCTensor_(resizeAs)(state, output, input);
    THC_pointwiseApply3(state, input, target, output,
        softmargin_no_reduce_functor<real, accreal>());
    return;
  }

  accreal sum;
  ptrdiff_t size = THCTensor_(nElement)(state, input);

  input = THCTensor_(newContiguous)(state, input);
  target = THCTensor_(newContiguous)(state, target);
  THCTensor_(resize1d)(state, output, 1);

  thrust::device_ptr<real> input_data(THCTensor_(data)(state, input));
  thrust::device_ptr<real> target_data(THCTensor_(data)(state, target));
  sum = thrust::inner_product(input_data, input_data+size, target_data, (accreal) 0, thrust::plus<accreal>(), softmargin_functor<real, accreal>());

  if(sizeAverage)
    sum /= size;

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, target);

  THCTensor_(set1d)(state, output, 0, ScalarConvert<accreal, real>::to(sum));
}

void THNN_(SoftMarginCriterion_updateGradInput)(
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

  THCTensor_(resizeAs)(state, gradInput, input);

  if (!reduce) {
    THCUNN_check_shape(state, gradOutput, input);
    THC_pointwiseApply3(state, input, target, gradInput,
        softmargin_updateGradInput_no_reduce_functor<real, accreal>());
    THCTensor_(cmul)(state, gradInput, gradInput, gradOutput);
    return;
  }

  ptrdiff_t size = THCTensor_(nElement)(state, input);
  accreal norm = (sizeAverage ? 1./size : 1.);

  input = THCTensor_(newContiguous)(state, input);
  target = THCTensor_(newContiguous)(state, target);


  thrust::device_ptr<real> input_data(THCTensor_(data)(state, input));
  thrust::device_ptr<real> target_data(THCTensor_(data)(state, target));
  thrust::device_ptr<real> gradInput_data(THCTensor_(data)(state, gradInput));

  thrust::transform(input_data, input_data+size, target_data, gradInput_data,
                    softmargin_updateGradInput_functor<real, accreal>(norm, THCTensor_(get1d)(state, gradOutput, 0)));

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, target);
}

#endif
