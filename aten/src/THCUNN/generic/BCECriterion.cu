#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/BCECriterion.cu"
#else

void THNN_(BCECriterion_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *target,
           THCTensor *output,
           int64_t reduction,
           THCTensor *weights)
{
  THCUNN_check_nElement(state, input, target);
  THCUNN_check_nElement(state, input, weights);
  THCUNN_assertSameGPU(state, 3, input, target, weights);

  if (reduction == Reduction::None) {
    THCTensor_(resizeAs)(state, output, input);
    THC_pointwiseApply3<scalar_t, scalar_t, scalar_t>(state, input, target, output,
        bce_updateOutput_no_reduce_functor<scalar_t, accreal>());
    if (weights) {
      THCTensor_(cmul)(state, output, output, weights);
    }
    return;
  }

  THCTensor_(resize1d)(state, output, 1);
  ptrdiff_t size = THCTensor_(nElement)(state, input);

  input = THCTensor_(newContiguous)(state, input);
  target = THCTensor_(newContiguous)(state, target);
  THCThrustAllocator thrustAlloc(state);
  thrust::device_ptr<scalar_t> input_data(THCTensor_(data)(state, input));
  thrust::device_ptr<scalar_t> target_data(THCTensor_(data)(state, target));

  accreal sum;
  if (weights) {
    weights = THCTensor_(newContiguous)(state, weights);
    thrust::device_ptr<scalar_t> weights_data(THCTensor_(data)(state, weights));
    sum = thrust::transform_reduce(
      thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
      thrust::make_zip_iterator(thrust::make_tuple(input_data, target_data, weights_data)),
      thrust::make_zip_iterator(thrust::make_tuple(input_data+size, target_data+size, weights_data+size)),
      bce_functor_weights<scalar_t, accreal>(),
      (accreal) 0,
      thrust::plus<accreal>()
    );
    THCTensor_(free)(state, weights);
  } else {
    sum = thrust::transform_reduce(
      thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
      thrust::make_zip_iterator(thrust::make_tuple(input_data, target_data)),
      thrust::make_zip_iterator(thrust::make_tuple(input_data+size, target_data+size)),
      bce_functor<scalar_t, accreal>(),
      (accreal) 0,
      thrust::plus<accreal>()
    );
  }

  if (reduction == Reduction::Mean)
    sum /= size;

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, target);

  THCTensor_(set1d)(state, output, 0, ScalarConvert<accreal, scalar_t>::to(sum));
}

void THNN_(BCECriterion_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *target,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           int64_t reduction,
           THCTensor *weights)
{
  THCUNN_check_nElement(state, input, target);
  THCUNN_check_nElement(state, input, weights);
  THCUNN_assertSameGPU(state, 4, input, target, gradInput, weights);

  THCTensor_(resizeAs)(state, gradInput, input);

  if (reduction == Reduction::None) {
    THCUNN_check_nElement(state, gradOutput, input);
    THC_pointwiseApply3<scalar_t, scalar_t, scalar_t>(state, input, target, gradInput,
        bce_updateGradInput_no_reduce_functor<scalar_t, accreal>());
    THCTensor_(cmul)(state, gradInput, gradInput, gradOutput);
    if (weights) {
      THCTensor_(cmul)(state, gradInput, gradInput, weights);
    }
    return;
  }

  THCUNN_check_dim_size(state, gradOutput, 1, 0, 1);

  ptrdiff_t size = THCTensor_(nElement)(state, input);
  scalar_t norm = ScalarConvert<accreal, scalar_t>::to((reduction == Reduction::Mean ? accreal(1)/size : accreal(1)) * THCTensor_(get1d)(state, gradOutput, 0));

  input = THCTensor_(newContiguous)(state, input);
  target = THCTensor_(newContiguous)(state, target);

  thrust::device_ptr<scalar_t> input_data(THCTensor_(data)(state, input));
  thrust::device_ptr<scalar_t> target_data(THCTensor_(data)(state, target));
  thrust::device_ptr<scalar_t> gradInput_data(THCTensor_(data)(state, gradInput));

  if (weights) {
    weights = THCTensor_(newContiguous)(state, weights);
    thrust::device_ptr<scalar_t> weights_data(THCTensor_(data)(state, weights));
    thrust::transform(
      thrust::make_zip_iterator(thrust::make_tuple(input_data, target_data, weights_data)),
      thrust::make_zip_iterator(thrust::make_tuple(input_data+size, target_data+size, weights_data+size)),
      gradInput_data,
      bce_updateGradInput_functor_weights<scalar_t, accreal>(norm)
    );
    THCTensor_(free)(state, weights);
  } else {
    thrust::transform(
      thrust::make_zip_iterator(thrust::make_tuple(input_data, target_data)),
      thrust::make_zip_iterator(thrust::make_tuple(input_data+size, target_data+size)),
      gradInput_data,
      bce_updateGradInput_functor<scalar_t, accreal>(norm)
    );
  }

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, target);
}

#endif
