#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/AbsCriterion.cu"
#else

void THNN_(AbsCriterion_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *target,
           THCTensor *output,
           int64_t reduction)
{
  THCUNN_check_shape(state, input, target);
  THCUNN_assertSameGPU(state, 3, input, target, output);

  if (reduction == Reduction::None) {
    THCTensor_(resizeAs)(state, output, input);
    THC_pointwiseApply3<scalar_t, scalar_t, scalar_t>(state, input, target, output,
                        abs_updateOutput_no_reduce_functor<scalar_t>());
    return;
  }

  THCTensor_(resize1d)(state, output, 1);

  ptrdiff_t size = THCTensor_(nElement)(state, input);

  input = THCTensor_(newContiguous)(state, input);
  target = THCTensor_(newContiguous)(state, target);

  thrust::device_ptr<scalar_t> input_data(THCTensor_(data)(state, input));
  thrust::device_ptr<scalar_t> target_data(THCTensor_(data)(state, target));
  accreal sum = thrust::inner_product(input_data, input_data+size, target_data, (accreal)0, thrust::plus<accreal>(), abs_functor<scalar_t, accreal>());

  if (reduction == Reduction::Mean)
    sum /= size;

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, target);

  THCTensor_(set1d)(state, output, 0, ScalarConvert<accreal, scalar_t>::to(sum));
}

void THNN_(AbsCriterion_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *target,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           int64_t reduction)
{
  THCUNN_check_shape(state, input, target);
  THCUNN_assertSameGPU(state, 4, input, target, gradOutput, gradInput);

  THCTensor_(resizeAs)(state, gradInput, input);

  if (reduction == Reduction::None) {
    THCUNN_check_shape(state, gradOutput, input);
    THC_pointwiseApply3<scalar_t, scalar_t, scalar_t>(state, input, target, gradInput,
                        abs_updateGradInput_no_reduce_functor<scalar_t>());
    THCTensor_(cmul)(state, gradInput, gradInput, gradOutput);
    return;
  }

  THCUNN_check_dim_size(state, gradOutput, 1, 0, 1);

  ptrdiff_t size = THCTensor_(nElement)(state, input);
  scalar_t norm = ScalarConvert<double, scalar_t>::to(reduction == Reduction::Mean ? 1./size : 1.);

  input = THCTensor_(newContiguous)(state, input);
  target = THCTensor_(newContiguous)(state, target);

  thrust::device_ptr<scalar_t> input_data(THCTensor_(data)(state, input));
  thrust::device_ptr<scalar_t> target_data(THCTensor_(data)(state, target));
  thrust::device_ptr<scalar_t> gradInput_data(THCTensor_(data)(state, gradInput));

  thrust::transform(input_data, input_data+size, target_data, gradInput_data,
                    abs_updateGradInput_functor<scalar_t>(norm, THCTensor_(get1d)(state, gradOutput, 0)));

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, target);
}

#endif
