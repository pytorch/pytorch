#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Square.c"
#else

void THNN_(Square_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output)
{
  THTensor_(resizeAs)(output, input);
  
  if (input->nDimension == 1 || !THTensor_(isContiguous)(input) || !THTensor_(isContiguous)(output))
  {
    TH_TENSOR_APPLY2(real, output, real, input,
      *output_data = (*input_data) * (*input_data);
    );
  }
  else
  {
    real *output_data = THTensor_(data)(output);
    real *input_data  = THTensor_(data)(input);
    long i;
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

  if (input->nDimension == 1 || 
      !THTensor_(isContiguous)(input) || 
      !THTensor_(isContiguous)(gradOutput) ||
      !THTensor_(isContiguous)(gradInput))
  {
    TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, input,
      *gradInput_data  = 2.0 * (*gradOutput_data) * (*input_data);
    );
  }
  else
  {
    real *gradOutput_data = THTensor_(data)(gradOutput);
    real *gradInput_data  = THTensor_(data)(gradInput);
    real *input_data  = THTensor_(data)(input);
    long i;
#pragma omp parallel for private(i)
    for (i = 0; i < THTensor_(nElement)(gradInput); i++)
      gradInput_data[i] = 2.0 * gradOutput_data[i] * input_data[i];
  }
}

#endif
