#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/BCECriterion.cu"
#else

void THNN_(BCECriterion_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *target,
           THCTensor *output,
           bool sizeAverage,
           THCTensor *weights)
{
  THCUNN_check_nElement(state, input, target);
  THCUNN_check_nElement(state, input, weights);
  THCTensor_(resize1d)(state, output, 1);
  THCUNN_assertSameGPU(state, 3, input, target, weights);

  ptrdiff_t size = THCTensor_(nElement)(state, input);

  input = THCTensor_(newContiguous)(state, input);
  target = THCTensor_(newContiguous)(state, target);

  thrust::device_ptr<real> input_data(THCTensor_(data)(state, input));
  thrust::device_ptr<real> target_data(THCTensor_(data)(state, target));

  accreal sum;
  if (weights) {
    weights = THCTensor_(newContiguous)(state, weights);
    thrust::device_ptr<real> weights_data(THCTensor_(data)(state, weights));
    sum = thrust::transform_reduce(
      thrust::make_zip_iterator(thrust::make_tuple(input_data, target_data, weights_data)),
      thrust::make_zip_iterator(thrust::make_tuple(input_data+size, target_data+size, weights_data+size)),
      bce_functor_weights<real, accreal>(),
      (accreal) 0,
      thrust::plus<accreal>()
    );
    THCTensor_(free)(state, weights);
  } else {
    sum = thrust::transform_reduce(
      thrust::make_zip_iterator(thrust::make_tuple(input_data, target_data)),
      thrust::make_zip_iterator(thrust::make_tuple(input_data+size, target_data+size)),
      bce_functor<real, accreal>(),
      (accreal) 0,
      thrust::plus<accreal>()
    );
  }

  if (sizeAverage)
    sum /= size;

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, target);

  THCTensor_(set1d)(state, output, 0, ScalarConvert<accreal, real>::to(sum));
}

void THNN_(BCECriterion_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *target,
           THCTensor *gradInput,
           bool sizeAverage,
           THCTensor *weights)
{
  THCUNN_check_nElement(state, input, target);
  THCUNN_check_nElement(state, input, weights);
  THCUNN_assertSameGPU(state, 4, input, target, gradInput, weights);

  ptrdiff_t size = THCTensor_(nElement)(state, input);
  real norm = ScalarConvert<accreal, real>::to(sizeAverage ? accreal(1)/size : accreal(1));

  input = THCTensor_(newContiguous)(state, input);
  target = THCTensor_(newContiguous)(state, target);

  THCTensor_(resizeAs)(state, gradInput, input);

  thrust::device_ptr<real> input_data(THCTensor_(data)(state, input));
  thrust::device_ptr<real> target_data(THCTensor_(data)(state, target));
  thrust::device_ptr<real> gradInput_data(THCTensor_(data)(state, gradInput));

  if (weights) {
    weights = THCTensor_(newContiguous)(state, weights);
    thrust::device_ptr<real> weights_data(THCTensor_(data)(state, weights));
    thrust::transform(
      thrust::make_zip_iterator(thrust::make_tuple(input_data, target_data, weights_data)),
      thrust::make_zip_iterator(thrust::make_tuple(input_data+size, target_data+size, weights_data+size)),
      gradInput_data,
      bce_updateGradInput_functor_weights<real, accreal>(norm)
    );
    THCTensor_(free)(state, weights);
  } else {
    thrust::transform(
      thrust::make_zip_iterator(thrust::make_tuple(input_data, target_data)),
      thrust::make_zip_iterator(thrust::make_tuple(input_data+size, target_data+size)),
      gradInput_data,
      bce_updateGradInput_functor<real, accreal>(norm)
    );
  }

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, target);
}

#endif
