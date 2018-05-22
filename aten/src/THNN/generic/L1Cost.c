#ifndef THNN_OMP_OVERHEAD_THRESHOLD
#define THNN_OMP_OVERHEAD_THRESHOLD 5000
#endif
#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/L1Cost.c"
#else

void THNN_(L1Cost_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output)
{
  THNN_CHECK_DIM_SIZE(output, 1, 0, 1);
  accreal sum = 0;
  int serial_path = 0;
#ifdef _OPENMP
  int inOMP = omp_in_parallel();
  if (inOMP) {
    serial_path = 1;
  } else {
    TH_TENSOR_APPLY_REDUCTION_OMP(real, input, +:sum,
      sum += fabs(*input_data);,
      THNN_OMP_OVERHEAD_THRESHOLD);
  }
#else
  serial_path = 1;
#endif
  if (serial_path) {
    TH_TENSOR_APPLY(real, input,
        sum += fabs(*input_data););
  }
  THTensor_(set1d)(output, 0, sum);
}

void THNN_(L1Cost_updateGradInput)(
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
    int input_contig = THTensor_(isContiguous)(input);
    TH_TENSOR_APPLY2_OMP(gradInput_size, gradInput_contig, input_contig,
      real, gradInput, real, input,
      if (*input_data > 0)
        *gradInput_data = 1;
      else if (*input_data < 0)
        *gradInput_data = -1;
      else
        *gradInput_data = 0;,
      THNN_OMP_OVERHEAD_THRESHOLD
    );
  }
#else
  serial_path = 1;
#endif
  if (serial_path) {
    TH_TENSOR_APPLY2(real, gradInput, real, input,
      if (*input_data > 0)
        *gradInput_data = 1;
      else if (*input_data < 0)
        *gradInput_data = -1;
      else
        *gradInput_data = 0;
    );
  }
}

#undef THNN_OMP_OVERHEAD_THRESHOLD
#endif
