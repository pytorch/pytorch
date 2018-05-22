#ifndef THNN_OMP_OVERHEAD_THRESHOLD
#define THNN_OMP_OVERHEAD_THRESHOLD 1000
#endif
#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/LogSigmoid.c"
#else

void THNN_(LogSigmoid_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *buffer)
{
  THTensor_(resizeAs)(output, input);
  THTensor_(resizeAs)(buffer, input);
  //Use the LogSumExp trick to make this stable against overflow
  int serial_path = 0;
#ifdef _OPENMP
  int inOMP = omp_in_parallel();
  if (inOMP) {
    serial_path = 1;
  } else {
    int64_t output_size = THTensor_(nElement)(output);
    int output_contig = THTensor_(isContiguous)(output);
    int input_contig = THTensor_(isContiguous)(input);
    int buffer_contig = THTensor_(isContiguous)(buffer);
    TH_TENSOR_APPLY3_OMP(output_size, output_contig, input_contig, buffer_contig,
      real, output, real, input, real, buffer,
      real max_elem = fmax(0, -*input_data);
      real z = exp(-max_elem) + exp(-*input_data - max_elem);
      *buffer_data = z;
      *output_data = -(max_elem + log(z));,
      THNN_OMP_OVERHEAD_THRESHOLD
    );
  }
#else
  serial_path = 1;
#endif
  if (serial_path) {
    TH_TENSOR_APPLY3(real, output, real, input, real, buffer,
      real max_elem = fmax(0, -*input_data);
      real z = exp(-max_elem) + exp(-*input_data - max_elem);
      *buffer_data = z;
      *output_data = -(max_elem + log(z));
    );
  }
}

void THNN_(LogSigmoid_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *buffer)
{
  THNN_CHECK_NELEMENT(input, gradOutput);
  THTensor_(resizeAs)(gradInput, buffer);
/* deriv of -max(0,-x) - log(e(0 - max(0,-x)) + e(-x - max(0,-x)) is
 * -max_deriv - (-max_deriv*e(0-max(0,-x)) + (-1 - max_deriv)*e(-x - max(0,-x)))/z
 * where z = e(0 - max(0,-x)) + e(-x - max(0,-x))
 * which simplifies to
 *  -max_deriv - (z-1)/z if x is >= 0 or
 *  -max_deriv + (z-1)/z if x is < 0
 */
  int serial_path = 0;
#ifdef _OPENMP
  int inOMP = omp_in_parallel();
  if (inOMP) {
    serial_path = 1;
  } else {
    int64_t input_size = THTensor_(nElement)(input);
    int input_contig = THTensor_(isContiguous)(input);
    int gradInput_contig = THTensor_(isContiguous)(gradInput);
    int buffer_contig = THTensor_(isContiguous)(buffer);
    TH_TENSOR_APPLY3_OMP(input_size, input_contig, gradInput_contig, buffer_contig,
      real, input, real, gradInput, real, buffer,
      real z = *buffer_data;
      real max_deriv = 0.0;
      real sign = -1.0;
      if (*input_data < 0){
          max_deriv = -1.0;
          sign = 1.0;
      }
      *gradInput_data = -max_deriv - sign*((z - 1.0)/ z);,
      THNN_OMP_OVERHEAD_THRESHOLD
    );
  }
#else
  serial_path = 1;
#endif
  if (serial_path) {
    TH_TENSOR_APPLY3(real, input, real, gradInput, real, buffer,
      real z = *buffer_data;
      real max_deriv = 0.0;
      real sign = -1.0;
      if (*input_data < 0){
          max_deriv = -1.0;
          sign = 1.0;
      }
      *gradInput_data = -max_deriv - sign*((z - 1.0)/ z);
    );
  }
  THTensor_(cmul)(gradInput, gradOutput, gradInput);
}

#undef THNN_OMP_OVERHEAD_THRESHOLD
#endif
