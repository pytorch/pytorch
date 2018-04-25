#ifndef THNN_OMP_OVERHEAD_THRESHOLD
#define THNN_OMP_OVERHEAD_THRESHOLD 5000
#endif
#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SmoothL1Criterion.c"
#else

void THNN_(SmoothL1Criterion_updateOutput)(
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
    int serial_path = 0;
#ifdef _OPENMP
    int inOMP = omp_in_parallel();
    if (inOMP) {
      serial_path = 1;
    } else {
      int64_t input_size = THTensor_(nElement)(input);
      int input_contig = THTensor_(isContiguous)(input);
      int target_contig = THTensor_(isContiguous)(target);
      int output_contig = THTensor_(isContiguous)(output);
      TH_TENSOR_APPLY3_OMP(input_size, input_contig, target_contig, output_contig,
        real, input, real, target, real, output,
        real z = fabs(*input_data - *target_data);
        *output_data = z < 1 ? 0.5 * z * z : z - 0.5;,
        THNN_OMP_OVERHEAD_THRESHOLD
      );
    }
#else
    serial_path = 1;
#endif
    if (serial_path) {
      TH_TENSOR_APPLY3(real, input, real, target, real, output,
        real z = fabs(*input_data - *target_data);
        *output_data = z < 1 ? 0.5 * z * z : z - 0.5;
      );
    }
    return;
  }

  THTensor_(resize1d)(output, 1);

  real sum = 0;
  TH_TENSOR_APPLY2(real, input, real, target,
    real z = fabs(*input_data - *target_data);
    sum += z < 1 ? 0.5*z*z : z - 0.5;
  );

  if (sizeAverage)
    sum /= THTensor_(nElement)(input);

  THTensor_(set1d)(output, 0, sum);
}

void THNN_(SmoothL1Criterion_updateGradInput)(
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
        real x = *input_data - *target_data;
        if (x < -1.) {
          *gradInput_data = -1.;
        } else if (x > 1.) {
          *gradInput_data = 1.;
        } else {
          *gradInput_data = x;
        },
        THNN_OMP_OVERHEAD_THRESHOLD
      );
    }
#else
    serial_path = 1;
#endif
    if (serial_path) {
      TH_TENSOR_APPLY3(real, gradInput, real, input, real, target,
        real x = *input_data - *target_data;
        if (x < -1.) {
          *gradInput_data = -1.;
        } else if (x > 1.) {
          *gradInput_data = 1.;
        } else {
          *gradInput_data = x;
        }
      );
    }
    TH_TENSOR_APPLY2(real, gradInput, real, gradOutput,
      *gradInput_data *= *gradOutput_data;
    );
    return;
  }

  THNN_CHECK_DIM_SIZE(gradOutput, 1, 0, 1);
  real norm = (sizeAverage ? 1./((real)THTensor_(nElement)(input)) : 1.) * THTensor_fastGet1d(gradOutput, 0);

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
      real x = *input_data - *target_data;
      if (x < -1.)
        *gradInput_data = - norm;
      else if (x > 1.)
       *gradInput_data = norm;
      else
       *gradInput_data = norm * x;,
      THNN_OMP_OVERHEAD_THRESHOLD
    );
  }
#else
  serial_path = 1;
#endif
  if (serial_path) {
    TH_TENSOR_APPLY3(real, gradInput, real, input, real, target,
    real x = *input_data - *target_data;
    if (x < -1.)
      *gradInput_data = - norm;
    else if (x > 1.)
      *gradInput_data = norm;
    else
      *gradInput_data = norm * x;
    );
  }
}

#undef THNN_OMP_OVERHEAD_THRESHOLD
#endif
