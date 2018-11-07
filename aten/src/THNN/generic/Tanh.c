#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Tanh.c"
#else

void THNN_(Tanh_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output)
{
  THTensor_(tanh)(output, input);
}

void THNN_(Tanh_updateGradInput)(
          THNNState *state,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output)
{
  THNN_CHECK_SHAPE(output, gradOutput);
  THTensor_(resizeAs)(gradInput, output);

  if (THTensor_nDimensionLegacyAll(output) == 1 ||
      !THTensor_(isContiguous)(output) ||
      !THTensor_(isContiguous)(gradOutput) ||
      !THTensor_(isContiguous)(gradInput))
  {
    TH_TENSOR_APPLY3(scalar_t, gradInput, scalar_t, gradOutput, scalar_t, output,
      scalar_t z = *output_data;            \
      *gradInput_data = *gradOutput_data * (1. - z*z);
    );
  }
  else
  {
    scalar_t* ptr_gradOutput = gradOutput->data<scalar_t>();
    scalar_t* ptr_gradInput  = gradInput->data<scalar_t>();
    scalar_t* ptr_output     = output->data<scalar_t>();
    int64_t i;

#pragma omp parallel for private(i)
    for (i = 0; i < THTensor_(nElement)(gradInput); i++)
    {
      scalar_t z = ptr_output[i];
      ptr_gradInput[i] = ptr_gradOutput[i] * (1. - z*z);
    }
  }
}

#endif
