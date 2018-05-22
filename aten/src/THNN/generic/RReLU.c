#ifndef THNN_OMP_OVERHEAD_THRESHOLD
#define THNN_OMP_OVERHEAD_THRESHOLD 5000
#endif
#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/RReLU.c"
#else

void THNN_(RReLU_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *noise,
          accreal lower_,
          accreal upper_,
          bool train,
          bool inplace,
          THGenerator *generator)
{
  real lower = TH_CONVERT_ACCREAL_TO_REAL(lower_);
  real upper = TH_CONVERT_ACCREAL_TO_REAL(upper_);
  if (train)
  {
    // get default random generator
    THTensor_(resizeAs)(noise, input);
    if (inplace)
    {
      TH_TENSOR_APPLY2(real, input, real, noise,
        if (*input_data <= 0)
        {
          const real r = (real)THRandom_uniform(generator, lower, upper);
          *input_data = (*input_data) * r;
          *noise_data = r;
        }
        else
        {
          *noise_data = 1;
        }
      );
      THTensor_(set)(output, input);
    }
    else
    {
      THTensor_(resizeAs)(output, input);
      TH_TENSOR_APPLY3(real, input, real, output, real, noise,
        if (*input_data <= 0)
        {
          const real r = (real)THRandom_uniform(generator, lower, upper);
          *output_data = (*input_data) * r;
          *noise_data = r;
        }
        else
        {
          *output_data = *input_data;
          *noise_data = 1;
        }
       );
    }
  }
  else
  {
    const real negSlope = (lower + upper) / 2;
    if (inplace)
    {
      TH_TENSOR_APPLY(real, input,
        if (*input_data <= 0)
        {
          *input_data = *input_data * negSlope;
        }
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
        int64_t input_size = THTensor_(nElement)(input);
        int input_contig = THTensor_(isContiguous)(input);
        int output_contig = THTensor_(isContiguous)(output);
        TH_TENSOR_APPLY2_OMP(input_size, input_contig, output_contig,
          real, input, real, output,
          const real r = (*input_data) <= 0 ? negSlope : 1;
          *output_data = *input_data * r;,
          THNN_OMP_OVERHEAD_THRESHOLD
        );
      }
#else
      serial_path = 1;
#endif
      if (serial_path) {
        TH_TENSOR_APPLY2(real, input, real, output,
          const real r = (*input_data) <= 0 ? negSlope : 1;
          *output_data = *input_data * r;
        );
      }
    }
  }
}

void THNN_(RReLU_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *noise,
          accreal lower_,
          accreal upper_,
          bool train,
          bool inplace)
{
  real lower = TH_CONVERT_ACCREAL_TO_REAL(lower_);
  real upper = TH_CONVERT_ACCREAL_TO_REAL(upper_);
  THNN_CHECK_NELEMENT(input, gradOutput);
  if (train && upper - lower > 1E-6)    // e.g. if upper == lower, RReLU behaves like LeakyReLU
  {
    // multiply the gradient by the noise tensor
    if (inplace)
    {
      THTensor_(cmul)(gradOutput, gradOutput, noise);
      THTensor_(set)(gradInput, gradOutput);
    }
    else
    {
      THTensor_(resizeAs)(gradInput, input);
      THTensor_(cmul)(gradInput, gradOutput, noise);
    }
  }
  else
  {
    // use constant factor for negative input values
    const real negSlope = (lower + upper) / 2;
    if (inplace)
    {
      int serial_path = 0;
#ifdef _OPENMP
      int inOMP = omp_in_parallel();
      if (inOMP) {
        serial_path = 1;
      } else {
        int64_t gradOutput_size = THTensor_(nElement)(gradOutput);
        int gradOutput_contig = THTensor_(isContiguous)(gradOutput);
        int input_contig = THTensor_(isContiguous)(input);
        TH_TENSOR_APPLY2_OMP(gradOutput_size, gradOutput_contig, input_contig,
          real, gradOutput, real, input,
          if (*input_data <= 0)
          {
            *gradOutput_data = (*gradOutput_data) * negSlope;
          },
          THNN_OMP_OVERHEAD_THRESHOLD
        );
      }
#else
      serial_path = 1;
#endif
      if (serial_path) {
        TH_TENSOR_APPLY2(real, gradOutput, real, input,
          if (*input_data <= 0)
          {
            *gradOutput_data = (*gradOutput_data) * negSlope;
          }
        );
      }
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
          *gradInput_data = (*input_data) <= 0 ? (*gradOutput_data) * negSlope : (*gradOutput_data);,
          THNN_OMP_OVERHEAD_THRESHOLD
        );
      }
#else
      serial_path = 1;
#endif
      if (serial_path) {
        TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, input,
          *gradInput_data = (*input_data) <= 0 ? (*gradOutput_data) * negSlope : (*gradOutput_data);
        );
      }
    }
  }
}
#undef THNN_OMP_OVERHEAD_THRESHOLD
#endif
