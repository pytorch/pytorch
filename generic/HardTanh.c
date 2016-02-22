#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/HardTanh.c"
#else

void THNN_(HardTanh_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          real min_val,
          real max_val)
{
  THTensor_(resizeAs)(output, input);
  
  if (input->nDimension == 1 || !THTensor_(isContiguous)(input) || !THTensor_(isContiguous)(output))
  {
    TH_TENSOR_APPLY2(real, output, real, input,
      if (*input_data < min_val)
        *output_data = min_val;
      else if (*input_data <= max_val)
        *output_data = *input_data;
      else
        *output_data = max_val;
    );
  }
  else
  {
    real* ptr_output = THTensor_(data)(output);
    real* ptr_input  = THTensor_(data)(input);
    long i;

#pragma omp parallel for private(i)
    for (i = 0; i < THTensor_(nElement)(input); i++)
    {
      if (ptr_input[i] < min_val)
        ptr_output[i] = min_val;
      else if (ptr_input[i] <= max_val)
        ptr_output[i] = ptr_input[i];
      else
        ptr_output[i] = max_val;
    }
  }
}

void THNN_(HardTanh_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          real min_val,
          real max_val)
{
  THTensor_(resizeAs)(gradInput, input);

  if (input->nDimension == 1 ||
    !THTensor_(isContiguous)(input) ||
    !THTensor_(isContiguous)(gradOutput) ||
    !THTensor_(isContiguous)(gradInput))
  {
    TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, input,
      if (*input_data < min_val || *input_data > max_val)
        *gradInput_data = 0;
      else
        *gradInput_data = *gradOutput_data;
    );
  }
  else
  {
    real* ptr_gradOutput = THTensor_(data)(gradOutput);
    real* ptr_gradInput  = THTensor_(data)(gradInput);
    real* ptr_input      = THTensor_(data)(input);
    long i;

#pragma omp parallel for private(i)
    for (i = 0; i < THTensor_(nElement)(input); i++)
    {
      if (ptr_input[i] < min_val || ptr_input[i] > max_val)
        ptr_gradInput[i] = 0;
      else
        ptr_gradInput[i] = ptr_gradOutput[i];
    }
  }
}

#endif
