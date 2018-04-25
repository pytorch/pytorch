#ifndef THNN_OMP_OVERHEAD_THRESHOLD
#define THNN_OMP_OVERHEAD_THRESHOLD 5000
#endif
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
    int serial_path = 0;
#ifdef _OPENMP
    int inOMP = omp_in_parallel();
    if (inOMP) {
      serial_path = 1;
    } else {
      int64_t output_size = THTensor_(nElement)(output);
      int output_contig = THTensor_(isContiguous)(output);
      int input_contig = THTensor_(isContiguous)(input);
      TH_TENSOR_APPLY2_OMP(output_size, output_contig, input_contig,
        real, output, real, input,
        *output_data = (*input_data) * (*input_data);,
        THNN_OMP_OVERHEAD_THRESHOLD
      );
    }
#else
    serial_path = 1;
#endif
    if (serial_path) {
      TH_TENSOR_APPLY2(real, output, real, input,
        *output_data = (*input_data) * (*input_data);
      );
    }
  }
  else
  {
    real *output_data = THTensor_(data)(output);
    real *input_data  = THTensor_(data)(input);
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

  if (input->nDimension == 1 ||
      !THTensor_(isContiguous)(input) ||
      !THTensor_(isContiguous)(gradOutput) ||
      !THTensor_(isContiguous)(gradInput))
  {
    int serial_path = 0;
#ifdef _OPENMP
    int inOMP = omp_in_parallel();
    if (inOMP) {
      serial_path = 1;
    } else {
      int64_t gradInput_size = THTensor_(nElement)(gradInput);
      int gradInput_contig = THTensor_(isContiguous)(gradInput);
      int gradOutput_contig = THTensor_(isContiguous)(gradOutput);
      int input_contig = THTensor_(isContiguous)(input);
      TH_TENSOR_APPLY3_OMP(gradInput_size, gradInput_contig, gradOutput_contig, input_contig,
        real, gradInput, real, gradOutput, real, input,
        *gradInput_data  = 2.0 * (*gradOutput_data) * (*input_data);,
        THNN_OMP_OVERHEAD_THRESHOLD
      );
    }
#else
    serial_path = 1;
#endif
    if (serial_path) {
      TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, input,
        *gradInput_data  = 2.0 * (*gradOutput_data) * (*input_data);
      );
    }
  }
  else
  {
    real *gradOutput_data = THTensor_(data)(gradOutput);
    real *gradInput_data  = THTensor_(data)(gradInput);
    real *input_data  = THTensor_(data)(input);
    int64_t i;
#pragma omp parallel for private(i)
    for (i = 0; i < THTensor_(nElement)(gradInput); i++)
      gradInput_data[i] = 2.0 * gradOutput_data[i] * input_data[i];
  }
}

#undef THNN_OMP_OVERHEAD_THRESHOLD
#endif
