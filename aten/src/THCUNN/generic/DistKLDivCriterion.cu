#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/DistKLDivCriterion.cu"
#else

#include "THCApply.cuh"

void THNN_(DistKLDivCriterion_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *target,
           THCTensor *output,
           bool sizeAverage,
           bool reduce)
{
  THCUNN_check_shape(state, input, target);
  THCUNN_assertSameGPU(state, 2, input, target);

  THArgCheck(THCTensor_(nElement)(state, input) == THCTensor_(nElement)(state, target), 2,
             "input and target need to have the same number of elements");

  if (!reduce) {
    THCTensor_(resizeAs)(state, output, input);
    THC_pointwiseApply3(state, input, target, output,
                        kl_updateOutput_no_reduce_functor<real>());
    return;
  }

  THCTensor_(resize1d)(state, output, 1);

  accreal sum;

  ptrdiff_t size = THCTensor_(nElement)(state, input);

  input = THCTensor_(newContiguous)(state, input);
  target = THCTensor_(newContiguous)(state, target);

  thrust::device_ptr<real> input_data(THCTensor_(data)(state, input));
  thrust::device_ptr<real> target_data(THCTensor_(data)(state, target));
  sum = thrust::inner_product(input_data, input_data+size, target_data, (accreal) 0, thrust::plus<accreal>(), kl_functor<real, accreal>());

  if (sizeAverage)
    sum /= size;

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, target);

  THCTensor_(set1d)(state, output, 0, ScalarConvert<accreal, real>::to(sum));
}

void THNN_(DistKLDivCriterion_updateGradInput)(
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

  THArgCheck(THCTensor_(nElement)(state, input) == THCTensor_(nElement)(state, target), 2,
             "input and target need to have the same number of elements");

  THCTensor_(resizeAs)(state, gradInput, input);

  if (!reduce) {
    THCUNN_check_shape(state, gradOutput, input);
    THC_pointwiseApply3(state, target, gradOutput, gradInput,
                        kl_updateGradInput_no_reduce_functor<real>());
    return;
  }

  THCUNN_check_dim_size(state, gradOutput, 1, 0, 1);

  ptrdiff_t size = THCTensor_(nElement)(state, input);
  real norm = (sizeAverage ? ScalarConvert<accreal, real>::to(accreal(1)/size) : ScalarConvert<int, real>::to(1));

  input = THCTensor_(newContiguous)(state, input);
  target = THCTensor_(newContiguous)(state, target);

  thrust::device_ptr<real> input_data(THCTensor_(data)(state, input));
  thrust::device_ptr<real> target_data(THCTensor_(data)(state, target));
  thrust::device_ptr<real> gradInput_data(THCTensor_(data)(state, gradInput));

  thrust::transform(input_data, input_data+size, target_data, gradInput_data,
                    kl_updateGradInput_functor<real>(norm, THCTensor_(get1d)(state, gradOutput, 0)));

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, target);
}

#endif
