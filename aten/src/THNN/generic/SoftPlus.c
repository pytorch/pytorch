#ifndef THNN_OMP_OVERHEAD_THRESHOLD
#define THNN_OMP_OVERHEAD_THRESHOLD 1000
#endif
#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SoftPlus.c"
#else

void THNN_(SoftPlus_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          accreal beta_,
          accreal threshold_)
{
  real beta = TH_CONVERT_ACCREAL_TO_REAL(beta_);
  real threshold = TH_CONVERT_ACCREAL_TO_REAL(threshold_);
  THTensor_(resizeAs)(output, input);

  // f(x) = 1/beta * log(1 + exp(beta * x))
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
      real, output, real, input,               \
      *output_data = (*input_data * beta) > threshold ? *input_data : THLog1p(exp(*input_data * beta)) / beta;,
      THNN_OMP_OVERHEAD_THRESHOLD
    );
  }
#else
  serial_path = 1;
#endif
  if (serial_path) {
    TH_TENSOR_APPLY2(real, output, real, input,               \
      *output_data = (*input_data * beta) > threshold ? *input_data : THLog1p(exp(*input_data * beta)) / beta;
    );
  }
}

void THNN_(SoftPlus_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output,
          accreal beta_,
          accreal threshold_)
{
  real beta = TH_CONVERT_ACCREAL_TO_REAL(beta_);
  real threshold = TH_CONVERT_ACCREAL_TO_REAL(threshold_);
  THNN_CHECK_NELEMENT(input, gradOutput);
  THTensor_(resizeAs)(gradInput, output);

  // d/dx[log(1+exp(k*x))/k] = exp(kx) / (exp(kx) + 1)
  // SINCE
  // y = (1/k)*log(1+exp(k*x)) --> x = (1/k)*log(exp(k*y)-1)
  // THEREFORE:
  // d/dx(f(x)) = (exp(k*y) - 1) / exp(k*y)
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
      real z = exp(*output_data * beta);
      *gradInput_data = (*output_data * beta) > threshold ? *gradOutput_data : *gradOutput_data * (z - 1.)/z;,
      THNN_OMP_OVERHEAD_THRESHOLD
    );
  }
#else
  serial_path = 1;
#endif
  if (serial_path) {
    TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, output,
      real z = exp(*output_data * beta);
      *gradInput_data = (*output_data * beta) > threshold ? *gradOutput_data : *gradOutput_data * (z - 1.)/z;
    );
  }
}

#undef THNN_OMP_OVERHEAD_THRESHOLD
#endif
