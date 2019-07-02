#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "THNN/generic/HardTanh.c"
#else

#include <ATen/Parallel.h>

void THNN_(HardTanh_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          accreal min_val_,
          accreal max_val_,
          bool inplace)
{
  scalar_t min_val = TH_CONVERT_ACCREAL_TO_REAL(min_val_);
  scalar_t max_val = TH_CONVERT_ACCREAL_TO_REAL(max_val_);
  if (inplace)
    THTensor_(set)(output, input);
  else
    THTensor_(resizeAs)(output, input);

  if (THTensor_nDimensionLegacyAll(input) == 1 || !THTensor_(isContiguous)(input) || !THTensor_(isContiguous)(output))
  {
    if (inplace)
    {
      TH_TENSOR_APPLY(scalar_t, input,
        if (*input_data < min_val)
          *input_data = min_val;
        else if (*input_data > max_val)
          *input_data = max_val;
      );
    }
    else
    {
      TH_TENSOR_APPLY2(scalar_t, output, scalar_t, input,
        if (*input_data < min_val)
          *output_data = min_val;
        else if (*input_data > max_val)
          *output_data = max_val;
        else
          *output_data = *input_data;
      );
    }
  }
  else
  {
    scalar_t* ptr_input  = input->data<scalar_t>();
    scalar_t* ptr_output = output->data<scalar_t>();
    ptrdiff_t n = THTensor_(nElement)(input);

    if (inplace) {
      at::parallel_for(0, n, 0, [&](int64_t start, int64_t end) {
        for (auto i = start; i < end; i++)
        {
          if (ptr_input[i] < min_val)
            ptr_input[i] = min_val;
          else if (ptr_input[i] > max_val)
            ptr_input[i] = max_val;
        }
      });
    } else {
      at::parallel_for(0, n, 0, [&](int64_t start, int64_t end) {
        for (auto i = start; i < end; i++)
        {
          if (ptr_input[i] < min_val)
            ptr_output[i] = min_val;
          else if (ptr_input[i] <= max_val)
            ptr_output[i] = ptr_input[i];
          else
            ptr_output[i] = max_val;
        }
      });
    }
  }
}

void THNN_(HardTanh_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          accreal min_val_,
          accreal max_val_,
          bool inplace)
{
  scalar_t min_val = TH_CONVERT_ACCREAL_TO_REAL(min_val_);
  scalar_t max_val = TH_CONVERT_ACCREAL_TO_REAL(max_val_);

  THNN_CHECK_NELEMENT(input, gradOutput);
  if (inplace)
    THTensor_(set)(gradInput, gradOutput);
  else
    THTensor_(resizeAs)(gradInput, input);

  if (THTensor_nDimensionLegacyAll(input) == 1 ||
    !THTensor_(isContiguous)(input) ||
    !THTensor_(isContiguous)(gradOutput) ||
    !THTensor_(isContiguous)(gradInput))
  {
    if (inplace)
    {
      TH_TENSOR_APPLY2(scalar_t, gradOutput, scalar_t, input,
        if (*input_data <= min_val || *input_data >= max_val)
          *gradOutput_data = 0;
      );
    }
    else
      TH_TENSOR_APPLY3(scalar_t, gradInput, scalar_t, gradOutput, scalar_t, input,
        if (*input_data <= min_val || *input_data >= max_val)
          *gradInput_data = 0;
        else
          *gradInput_data = *gradOutput_data;
      );
  }
  else
  {
    scalar_t* ptr_gradOutput = gradOutput->data<scalar_t>();
    scalar_t* ptr_gradInput  = gradInput->data<scalar_t>();
    scalar_t* ptr_input      = input->data<scalar_t>();
    ptrdiff_t n = THTensor_(nElement)(input);

    if (inplace) {
      at::parallel_for(0, n, 0, [&](int64_t start, int64_t end) {
        for (auto i = start; i < end; i++)
        {
          if (ptr_input[i] <= min_val || ptr_input[i] >= max_val)
            ptr_gradInput[i] = 0;
        }
      });
    } else {
      at::parallel_for(0, n, 0, [&](int64_t start, int64_t end) {
        for (auto i = start; i < end; i++)
        {
          if (ptr_input[i] <= min_val || ptr_input[i] >= max_val)
            ptr_gradInput[i] = 0;
          else
            ptr_gradInput[i] = ptr_gradOutput[i];
        }
      });
    }
  }
}

#endif
