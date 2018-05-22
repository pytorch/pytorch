#ifndef THNN_OMP_OVERHEAD_THRESHOLD
#define THNN_OMP_OVERHEAD_THRESHOLD 5000
#endif
#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Sqrt.c"
#else

void THNN_(Sqrt_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          accreal eps_)
{
  THTensor_(resizeAs)(output, input);
  THTensor_(sqrt)(output, input);
}

void THNN_(Sqrt_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output)
{
  THNN_CHECK_SHAPE(output, gradOutput);
  THTensor_(resizeAs)(gradInput, input);

  if (output->nDimension == 1 ||
      !THTensor_(isContiguous)(output) ||
      !THTensor_(isContiguous)(gradOutput) ||
      !THTensor_(isContiguous)(gradInput))
  {
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
        *gradInput_data = (*output_data == 0.0) ? 0.0 : (0.5 * (*gradOutput_data / *output_data));,
        THNN_OMP_OVERHEAD_THRESHOLD
      );
    }
#else
    serial_path = 1;
#endif
    if (serial_path) {
      TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, output,
        *gradInput_data = (*output_data == 0.0) ? 0.0 : (0.5 * (*gradOutput_data / *output_data));
      );
    }
  }
  else
  {
    real *gradOutput_data = THTensor_(data)(gradOutput);
    real *gradInput_data  = THTensor_(data)(gradInput);
    real *output_data     = THTensor_(data)(output);
    int64_t i;
#pragma omp parallel for private(i)
    for(i = 0; i < THTensor_(nElement)(output); i++)
    {
      if (output_data[i] == 0.0)
        gradInput_data[i] = 0.0;
      else
        gradInput_data[i] = 0.5 * (gradOutput_data[i] / output_data[i]);
    }
  }
}

#undef THNN_OMP_OVERHEAD_THRESHOLD
#endif
