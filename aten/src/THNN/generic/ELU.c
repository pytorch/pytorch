#ifndef THNN_OMP_OVERHEAD_THRESHOLD
#define THNN_OMP_OVERHEAD_THRESHOLD 1000
#endif
#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/ELU.c"
#else

void THNN_(ELU_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          accreal alpha_,
          accreal scale,
          bool inplace)
{
  real negcoef = TH_CONVERT_ACCREAL_TO_REAL(alpha_ * scale);
  real poscoef = TH_CONVERT_ACCREAL_TO_REAL(scale);
  if (inplace) {
    TH_TENSOR_APPLY(real, input,
      *input_data = *input_data <= 0 ? (exp(*input_data)-1) * negcoef : *input_data * poscoef;
    );
    THTensor_(set)(output, input);
  } else {
    THTensor_(resizeAs)(output, input);
    int serial_path = 0;
#ifdef _OPENMP
    int inOMP = omp_in_parallel();
    if (inOMP) {
      serial_path = 1;
    } else {
      int64_t input_size = THTensor_(nElement)(input);
      int input_contig = THTensor_(isContiguous)(input);
      int output_contig = THTensor_(isContiguous)(output);
      TH_TENSOR_APPLY2_OMP(input_size, input_contig, output_contig,
        real, input, real, output,
        *output_data = *input_data <= 0 ? (exp(*input_data)-1) * negcoef : *input_data * poscoef;,
        THNN_OMP_OVERHEAD_THRESHOLD
      );
    }
#else
    serial_path = 1;
#endif
    if (serial_path) {
      TH_TENSOR_APPLY2(real, input, real, output,
        *output_data = *input_data <= 0 ? (exp(*input_data)-1) * negcoef : *input_data * poscoef;
      );
    }
  }
}

void THNN_(ELU_updateGradInput)(
          THNNState *state,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output,
          accreal alpha_,
          accreal scale)
{
  real negcoef = TH_CONVERT_ACCREAL_TO_REAL(alpha_ * scale);
  real poscoef = TH_CONVERT_ACCREAL_TO_REAL(scale);
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
      *gradInput_data = *output_data <= 0 ? *gradOutput_data * (*output_data + negcoef) : *gradOutput_data * poscoef;,
      THNN_OMP_OVERHEAD_THRESHOLD
    );
  }
#else
  serial_path = 1;
#endif
  if (serial_path) {
    TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, output,
      *gradInput_data = *output_data <= 0 ? *gradOutput_data * (*output_data + negcoef) : *gradOutput_data * poscoef;
    );
  }
}

#undef THNN_OMP_OVERHEAD_THRESHOLD
#endif
