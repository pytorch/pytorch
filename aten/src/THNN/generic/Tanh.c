#ifndef THNN_OMP_OVERHEAD_THRESHOLD
#define THNN_OMP_OVERHEAD_THRESHOLD 1000
#endif
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

  if (output->nDimension == 1 ||
      !THTensor_(isContiguous)(output) ||
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
      int output_contig = THTensor_(isContiguous)(output);
      TH_TENSOR_APPLY3_OMP(gradInput_size, gradInput_contig, gradOutput_contig, output_contig,
        real, gradInput, real, gradOutput, real, output,
        real z = *output_data;            \
        *gradInput_data = *gradOutput_data * (1. - z*z);,
        THNN_OMP_OVERHEAD_THRESHOLD
      );
    }
#else
    serial_path = 1;
#endif
    if (serial_path) {
      TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, output,
        real z = *output_data;            \
        *gradInput_data = *gradOutput_data * (1. - z*z);
      );
    }
  }
  else
  {
    real* ptr_gradOutput = THTensor_(data)(gradOutput);
    real* ptr_gradInput  = THTensor_(data)(gradInput);
    real* ptr_output     = THTensor_(data)(output);
    int64_t i;

#pragma omp parallel for private(i)
    for (i = 0; i < THTensor_(nElement)(gradInput); i++)
    {
      real z = ptr_output[i];
      ptr_gradInput[i] = ptr_gradOutput[i] * (1. - z*z);
    }
  }
}

#undef THNN_OMP_OVERHEAD_THRESHOLD
#endif
