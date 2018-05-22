#ifndef THNN_OMP_OVERHEAD_THRESHOLD
#define THNN_OMP_OVERHEAD_THRESHOLD 5000
#endif
#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Abs.c"
#else

void THNN_(Abs_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output)
{
  THTensor_(resizeAs)(output, input);
  THTensor_(abs)(output, input);
}

void THNN_(Abs_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput)
{
  THNN_CHECK_NELEMENT(input, gradOutput);
  THTensor_(resizeAs)(gradInput, input);
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
      real z = *input_data;
      *gradInput_data = *gradOutput_data * (z >= 0 ? 1 : -1);,
      THNN_OMP_OVERHEAD_THRESHOLD
    );
  }
#else
  serial_path = 1;
#endif
  if (serial_path) {
    TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, input,
      real z = *input_data;
      *gradInput_data = *gradOutput_data * (z >= 0 ? 1 : -1);
    );
  }
}

#undef THNN_OMP_OVERHEAD_THRESHOLD
#endif
