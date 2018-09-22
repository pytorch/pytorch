#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Square.c"
#else

void THNN_(Square_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output)
{
  THTensor_(resizeAs)(output, input);
  
  if (THTensor_nDimensionLegacyAll(input) == 1 || !THTensor_(isContiguous)(input) || !THTensor_(isContiguous)(output))
  {
    TH_TENSOR_APPLY2(scalar_t, output, scalar_t, input,
      *output_data = (*input_data) * (*input_data);
    );
  }
  else
  {
    scalar_t *output_data = output->data<scalar_t>();
    scalar_t *input_data  = input->data<scalar_t>();
    int64_t i;
#pragma omp parallel for private(i)
    for (i = 0; i < THTensor_(nElement)(input); i++)
      output_data[i] = input_data[i]*input_data[i];
  }
}

void THNN_(Square_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput)
{
  THNN_CHECK_SHAPE(input, gradOutput);
  THTensor_(resizeAs)(gradInput, input);

  if (THTensor_nDimensionLegacyAll(input) == 1 || 
      !THTensor_(isContiguous)(input) || 
      !THTensor_(isContiguous)(gradOutput) ||
      !THTensor_(isContiguous)(gradInput))
  {
    TH_TENSOR_APPLY3(scalar_t, gradInput, scalar_t, gradOutput, scalar_t, input,
      *gradInput_data  = 2.0 * (*gradOutput_data) * (*input_data);
    );
  }
  else
  {
    scalar_t *gradOutput_data = gradOutput->data<scalar_t>();
    scalar_t *gradInput_data  = gradInput->data<scalar_t>();
    scalar_t *input_data  = input->data<scalar_t>();
    int64_t i;
#pragma omp parallel for private(i)
    for (i = 0; i < THTensor_(nElement)(gradInput); i++)
      gradInput_data[i] = 2.0 * gradOutput_data[i] * input_data[i];
  }
}

#endif
