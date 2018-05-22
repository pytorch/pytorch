#ifndef THNN_OMP_OVERHEAD_THRESHOLD
#define THNN_OMP_OVERHEAD_THRESHOLD 1000
#endif
#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SoftMarginCriterion.c"
#else

void THNN_(SoftMarginCriterion_updateOutput)(
  THNNState *state,
  THTensor *input,
  THTensor *target,
  THTensor *output,
  bool sizeAverage,
  bool reduce)
{
  THNN_CHECK_SHAPE(input, target);

  if (!reduce) {
    THTensor_(resizeAs)(output, input);

    TH_TENSOR_APPLY3(real, input, real, target, real, output,
                     *output_data = log(1. + exp(-*input_data * *target_data));)
    return;
  }

  THTensor_(resize1d)(output, 1);

  real sum;

  sum = 0;
  TH_TENSOR_APPLY2(real, input, real, target,
    real z = log(1. + exp(-*input_data* *target_data));
    sum += z;)
  if(sizeAverage)
    sum /= THTensor_(nElement)(input);

  THTensor_(set1d)(output, 0, sum);
}

void THNN_(SoftMarginCriterion_updateGradInput)(
  THNNState *state,
  THTensor *input,
  THTensor *target,
  THTensor *gradOutput,
  THTensor *gradInput,
  bool sizeAverage,
  bool reduce)
{
  THNN_CHECK_SHAPE(input, target);
  THTensor_(resizeAs)(gradInput, input);

  if (!reduce) {
    THNN_CHECK_SHAPE(gradOutput, input);

    TH_TENSOR_APPLY3(real, gradInput, real, input, real, target,
                     real z = exp(-*target_data * *input_data);
                     *gradInput_data = -*target_data * z/(1. + z);)
    THTensor_(cmul)(gradInput, gradInput, gradOutput);
    return;
  }

  real norm = (sizeAverage ? 1./((real)THTensor_(nElement)(input)) : 1.);

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
    int target_contig = THTensor_(isContiguous)(target);
    TH_TENSOR_APPLY3_OMP(gradInput_size, gradInput_contig, input_contig, target_contig,
                     real, gradInput, real, input, real, target,
                     real z = exp(-*target_data * *input_data);
                     *gradInput_data = -norm*(*target_data)*z/(1. + z) * THTensor_fastGet1d(gradOutput, 0);,
                     THNN_OMP_OVERHEAD_THRESHOLD);
  }
#else
  serial_path = 1;
#endif
  if (serial_path) {
    TH_TENSOR_APPLY3(real, gradInput, real, input, real, target,
                     real z = exp(-*target_data * *input_data);
                     *gradInput_data = -norm*(*target_data)*z/(1. + z) * THTensor_fastGet1d(gradOutput, 0);)
  }
}

#undef THNN_OMP_OVERHEAD_THRESHOLD
#endif
