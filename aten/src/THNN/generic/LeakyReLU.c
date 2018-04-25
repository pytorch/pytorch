#ifndef THNN_OMP_OVERHEAD_THRESHOLD
#define THNN_OMP_OVERHEAD_THRESHOLD 5000
#endif
#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/LeakyReLU.c"
#else

void THNN_(LeakyReLU_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          accreal negval_,
          bool inplace)
{
  real negval = TH_CONVERT_ACCREAL_TO_REAL(negval_);
  if (inplace)
  {
    TH_TENSOR_APPLY(real, input,
      if (*input_data <= 0)
        *input_data *= negval;
    );
    THTensor_(set)(output, input);
  }
  else
  {
    THTensor_(resizeAs)(output, input);
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
        *output_data = *input_data > 0 ? *input_data : *input_data * negval;,
        THNN_OMP_OVERHEAD_THRESHOLD
      );
    }
#else
    serial_path = 1;
#endif
    if (serial_path) {
      TH_TENSOR_APPLY2(real, output, real, input,
        *output_data = *input_data > 0 ? *input_data : *input_data * negval;
      );
    }
  }
}

void THNN_(LeakyReLU_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          accreal negval_,
          bool inplace)
{
  real negval = TH_CONVERT_ACCREAL_TO_REAL(negval_);
  THNN_CHECK_NELEMENT(input, gradOutput);
  if (inplace)
  {
    TH_TENSOR_APPLY2(real, gradOutput, real, input,
      if (*input_data <= 0)
        *gradOutput_data *= negval;
    );
    THTensor_(set)(gradInput, gradOutput);
  }
  else
  {
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
        *gradInput_data = *input_data > 0 ? *gradOutput_data : *gradOutput_data * negval;,
        THNN_OMP_OVERHEAD_THRESHOLD
      );
    }
#else
    serial_path = 1;
#endif
    if (serial_path) {
      TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, input,
         *gradInput_data = *input_data > 0 ? *gradOutput_data : *gradOutput_data * negval;
      );
    }
  }
}

#undef THNN_OMP_OVERHEAD_THRESHOLD
#endif
