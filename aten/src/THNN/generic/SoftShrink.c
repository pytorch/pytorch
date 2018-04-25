#ifndef THNN_OMP_OVERHEAD_THRESHOLD
#define THNN_OMP_OVERHEAD_THRESHOLD 5000
#endif
#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SoftShrink.c"
#else

void THNN_(SoftShrink_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          accreal lambda_)
{
  real lambda = TH_CONVERT_ACCREAL_TO_REAL(lambda_);
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
      if ((*input_data) > lambda)
       *output_data = *input_data - lambda;
      else if ((*input_data) < -lambda)
       *output_data = *input_data + lambda;
      else
       *output_data = 0;,
      THNN_OMP_OVERHEAD_THRESHOLD
    );
  }
#else
  serial_path = 1;
#endif
  if (serial_path) {
    TH_TENSOR_APPLY2(real, output, real, input,
      if ((*input_data) > lambda)
       *output_data = *input_data - lambda;
      else if ((*input_data) < -lambda)
       *output_data = *input_data + lambda;
      else
       *output_data = 0;
    );
  }
}

void THNN_(SoftShrink_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          accreal lambda_)
{
  real lambda = TH_CONVERT_ACCREAL_TO_REAL(lambda_);
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
      if ((*input_data) > lambda || (*input_data) < -lambda)
        *gradInput_data = (*gradOutput_data);
      else
        *gradInput_data = 0;,
      THNN_OMP_OVERHEAD_THRESHOLD
    );
  }
#else
  serial_path = 1;
#endif
  if (serial_path) {
    TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, input,
      if ((*input_data) > lambda || (*input_data) < -lambda)
        *gradInput_data = (*gradOutput_data);
      else
        *gradInput_data = 0;
    );
  }
}

#undef THNN_OMP_OVERHEAD_THRESHOLD
#endif
