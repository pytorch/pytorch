#ifndef THNN_OMP_OVERHEAD_THRESHOLD
#define THNN_OMP_OVERHEAD_THRESHOLD 1000
#endif
#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/BCECriterion.c"
#else

#define EPS 1e-12

void THNN_(BCECriterion_updateOutput)(
    THNNState *state,
    THTensor *input,
    THTensor *target,
    THTensor *output,
    bool sizeAverage,
    THTensor *weights,
    bool reduce)
{
  THNN_CHECK_NELEMENT(input, target);
  THNN_CHECK_NELEMENT(input, weights);

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
          real x = *input_data;
          real y = *target_data;
          THAssertMsg(x >= 0. && x <= 1.,
            "input value should be between 0~1, but got %f",
  		      (double) x);
  		    *output_data = -(log(x + EPS) * y + log(1. - x + EPS) * (1. - y));,
          THNN_OMP_OVERHEAD_THRESHOLD
      );
    }
#else
    serial_path = 1;
#endif
    if (serial_path) {
      TH_TENSOR_APPLY3(real, input, real, target, real, output,
          real x = *input_data;
          real y = *target_data;
          THAssertMsg(x >= 0. && x <= 1.,
            "input value should be between 0~1, but got %f",
                      (double) x);
                    *output_data = -(log(x + EPS) * y + log(1. - x + EPS) * (1. - y));
      );
    }
    if (weights) {
      THTensor_(cmul)(output, output, weights);
    }
    return;
  }

	THTensor_(resize1d)(output, 1);
  real sum = 0;

  if (weights) {
    TH_TENSOR_APPLY3(real, input, real, target, real, weights,
      real x = *input_data;
      real y = *target_data;
      real w = *weights_data;
      THAssertMsg(x >= 0. && x <= 1.,
        "input value should be between 0~1, but got %f",
		  (double) x);
      sum -= (log(x + EPS) * y + log(1. - x + EPS) * (1. - y)) * w;
    );
  } else {
    TH_TENSOR_APPLY2(real, input, real, target,
      real x = *input_data;
      real y = *target_data;
      THAssertMsg(x >= 0. && x <= 1.,
        "input value should be between 0~1, but got %f",
                  (double) x);
      sum -= log(x + EPS) * y + log(1. - x + EPS) * (1. - y);
    );
  }


  if (sizeAverage)
    sum /= THTensor_(nElement)(input);

  THTensor_(set1d)(output, 0, sum);
}

void THNN_(BCECriterion_updateGradInput)(
    THNNState *state,
    THTensor *input,
    THTensor *target,
    THTensor *gradOutput,
    THTensor *gradInput,
    bool sizeAverage,
    THTensor *weights,
    bool reduce)
{
  THNN_CHECK_NELEMENT(input, target);
  THNN_CHECK_NELEMENT(input, weights);
  THTensor_(resizeAs)(gradInput, input);

  if (!reduce) {
    THNN_CHECK_NELEMENT(gradOutput, input);
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
        real x = *input_data;
        real y = *target_data;
        *gradInput_data = -(y - x) / ((1. - x + EPS) * (x + EPS));,
        THNN_OMP_OVERHEAD_THRESHOLD
      );
    }
#else
    serial_path = 1;
#endif
    if (serial_path) {
      TH_TENSOR_APPLY3(real, gradInput, real, input, real, target,
        real x = *input_data;
        real y = *target_data;
        *gradInput_data = -(y - x) / ((1. - x + EPS) * (x + EPS));      );
    }
    if (weights) {
      int serial_path = 0;
#ifdef _OPENMP
      int inOMP = omp_in_parallel();
      if (inOMP) {
        serial_path = 1;
      } else {
        int64_t gradInput_size = THTensor_(nElement)(gradInput);
        int gradInput_contig = THTensor_(isContiguous)(gradInput);
        int weights_contig = THTensor_(isContiguous)(weights);
        int gradOutput_contig = THTensor_(isContiguous)(gradOutput);
        TH_TENSOR_APPLY3_OMP(gradInput_size, gradInput_contig, weights_contig, gradOutput_contig,
          real, gradInput, real, weights, real, gradOutput,
          *gradInput_data = *gradInput_data * *weights_data * *gradOutput_data;,
          THNN_OMP_OVERHEAD_THRESHOLD
        );
      }
#else
      serial_path = 1;
#endif
      if (serial_path) {
        TH_TENSOR_APPLY3(real, gradInput, real, weights, real, gradOutput,
          *gradInput_data = *gradInput_data * *weights_data * *gradOutput_data;
        );
      }
    } else {
      THTensor_(cmul)(gradInput, gradInput, gradOutput);
    }
    return;
  }

  THNN_CHECK_DIM_SIZE(gradOutput, 1, 0, 1);
  real norm = (sizeAverage ? 1./((real)THTensor_(nElement)(input)) : 1.);

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
      real x = *input_data;
      real y = *target_data;
      *gradInput_data = - norm * (y - x) / ((1. - x + EPS) * (x + EPS)) * THTensor_fastGet1d(gradOutput, 0);,
      THNN_OMP_OVERHEAD_THRESHOLD
    );
  }
#else
  serial_path = 1;
#endif
  if (serial_path) {
    TH_TENSOR_APPLY3(real, gradInput, real, input, real, target,
      real x = *input_data;
      real y = *target_data;
      *gradInput_data = - norm * (y - x) / ((1. - x + EPS) * (x + EPS)) * THTensor_fastGet1d(gradOutput, 0);
    );
  }
  if(weights)
    THTensor_(cmul)(gradInput, gradInput, weights);
}

#undef EPS
#undef THNN_OMP_OVERHEAD_THRESHOLD
#endif
