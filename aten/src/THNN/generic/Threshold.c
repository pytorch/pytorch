#ifndef THNN_OMP_OVERHEAD_THRESHOLD
#define THNN_OMP_OVERHEAD_THRESHOLD 5000
#endif
#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Threshold.c"
#else

void THNN_(Threshold_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          accreal threshold_,
          accreal val_,
          bool inplace)
{
  real threshold = TH_CONVERT_ACCREAL_TO_REAL(threshold_);
  real val = TH_CONVERT_ACCREAL_TO_REAL(val_);
  if (inplace)
  {
    TH_TENSOR_APPLY(real, input,
      if (*input_data <= threshold)
        *input_data = val;
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
        *output_data = (*input_data > threshold) ? *input_data : val;,
        THNN_OMP_OVERHEAD_THRESHOLD
      );
    }
#else
    serial_path = 1;
#endif
    if (serial_path) {
      TH_TENSOR_APPLY2(real, output, real, input,
        *output_data = (*input_data > threshold) ? *input_data : val;
      );
    }
  }
}

void THNN_(Threshold_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          accreal threshold_,
          accreal val_,
          bool inplace)
{
  real threshold = TH_CONVERT_ACCREAL_TO_REAL(threshold_);
  THNN_CHECK_NELEMENT(input, gradOutput);
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
        if ((*input_data) <= threshold)
          *gradOutput_data = 0;,
        THNN_OMP_OVERHEAD_THRESHOLD
      );
    }
#else
    serial_path = 1;
#endif
    if (serial_path) {
      TH_TENSOR_APPLY2(real, gradOutput, real, input,
         if ((*input_data) <= threshold)
          *gradOutput_data = 0;
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
        if ((*input_data) > threshold)
          *gradInput_data = *gradOutput_data;
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
        if ((*input_data) > threshold)
          *gradInput_data = *gradOutput_data;
        else
          *gradInput_data = 0;
      );
    }
  }
}

#undef THNN_OMP_OVERHEAD_THRESHOLD
#endif
