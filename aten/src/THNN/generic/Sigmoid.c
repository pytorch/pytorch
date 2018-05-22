#ifndef THNN_OMP_OVERHEAD_THRESHOLD
#define THNN_OMP_OVERHEAD_THRESHOLD 1000
#endif
#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Sigmoid.c"
#else

void THNN_(Sigmoid_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output)
{
  THTensor_(sigmoid)(output, input);
}

void THNN_(Sigmoid_updateGradInput)(
          THNNState *state,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output)
{
  THNN_CHECK_NELEMENT(output, gradOutput);
  THTensor_(resizeAs)(gradInput, output);
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
      real z = *output_data;
      *gradInput_data = *gradOutput_data * (1. - z) * z;,
      THNN_OMP_OVERHEAD_THRESHOLD
    );
  }
#else
  serial_path = 1;
#endif
  if (serial_path) {
    TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, output,
      real z = *output_data;
      *gradInput_data = *gradOutput_data * (1. - z) * z;
    );
  }
}

#undef THNN_OMP_OVERHEAD_THRESHOLD
#endif
