#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/BatchNormalization.c"
#else

void THNN_(BatchNormalization_updateOutput)(
  THNNState *state, THTensor *input, THTensor *output,
  THTensor *weight, THTensor *bias,
  THTensor *running_mean, THTensor *running_var,
  THTensor *save_mean, THTensor *save_std,
  bool train, double momentum, double eps)
{
  THTensor_(resizeAs)(output, input);
  int64_t nInput = THTensor_(size)(input, 1);
  int64_t f;
  ptrdiff_t n = THTensor_(nElement)(input) / nInput;

  if (train) {
    THTensor_(resize1d)(save_mean, nInput);
    THTensor_(resize1d)(save_std, nInput);
  }

  #pragma omp parallel for
  for (f = 0; f < nInput; ++f) {
    THTensor *in = THTensor_(newSelect)(input, 1, f);
    THTensor *out = THTensor_(newSelect)(output, 1, f);

    scalar_t mean, invstd;

    if (train) {
      // compute mean per input
      accreal sum = 0;
      TH_TENSOR_APPLY(scalar_t, in, sum += *in_data;);

      mean = (scalar_t) sum / n;
      THTensor_(set1d)(save_mean, f, (scalar_t) mean);

      // compute variance per input
      sum = 0;
      TH_TENSOR_APPLY(scalar_t, in,
        sum += (*in_data - mean) * (*in_data - mean););

      if (sum == 0 && eps == 0.0) {
        invstd = 0;
      } else {
        invstd = (scalar_t) (1 / sqrt(sum/n + eps));
      }
      THTensor_(set1d)(save_std, f, (scalar_t) invstd);

      // update running averages
      if (running_mean) {
        THTensor_(set1d)(running_mean, f,
          (scalar_t) (momentum * mean + (1 - momentum) * THTensor_(get1d)(running_mean, f)));
      }
      if (running_var) {
        accreal unbiased_var = sum / (n - 1);
        THTensor_(set1d)(running_var, f,
          (scalar_t) (momentum * unbiased_var + (1 - momentum) * THTensor_(get1d)(running_var, f)));
      }
    } else {
      mean = THTensor_(get1d)(running_mean, f);
      invstd = 1 / sqrt(THTensor_(get1d)(running_var, f) + eps);
    }

    // compute output
    scalar_t w = weight ? THTensor_(get1d)(weight, f) : 1;
    scalar_t b = bias ? THTensor_(get1d)(bias, f) : 0;

    TH_TENSOR_APPLY2(scalar_t, in, scalar_t, out,
      *out_data = (scalar_t) (((*in_data - mean) * invstd) * w + b););

    c10::raw::intrusive_ptr::decref(out);
    c10::raw::intrusive_ptr::decref(in);
  }
}

void THNN_(BatchNormalization_backward)(
  THNNState *state, THTensor *input, THTensor *gradOutput, THTensor *gradInput,
  THTensor *gradWeight, THTensor *gradBias, THTensor *weight,
  THTensor *running_mean, THTensor *running_var,
  THTensor *save_mean, THTensor *save_std,
  bool train, double scale, double eps)
{
  THNN_CHECK_SHAPE(input, gradOutput);
  int64_t nInput = THTensor_(size)(input, 1);
  int64_t f;
  ptrdiff_t n = THTensor_(nElement)(input) / nInput;

  if (gradInput) {
    THTensor_(resizeAs)(gradInput, input);
  }

  #pragma omp parallel for
  for (f = 0; f < nInput; ++f) {
    THTensor *in = THTensor_(newSelect)(input, 1, f);
    THTensor *gradOut = THTensor_(newSelect)(gradOutput, 1, f);
    scalar_t w = weight ? THTensor_(get1d)(weight, f) : 1;
    scalar_t mean, invstd;
    if (train) {
      mean = THTensor_(get1d)(save_mean, f);
      invstd = THTensor_(get1d)(save_std, f);
    } else {
      mean = THTensor_(get1d)(running_mean, f);
      invstd = 1 / sqrt(THTensor_(get1d)(running_var, f) + eps);
    }

    // sum over all gradOutput in feature plane
    accreal sum = 0;
    TH_TENSOR_APPLY(scalar_t, gradOut, sum += *gradOut_data;);

    // dot product of the Q(X) and gradOuput
    accreal dotp = 0;
    TH_TENSOR_APPLY2(scalar_t, in, scalar_t, gradOut,
      dotp += (*in_data - mean) * (*gradOut_data););

    if (gradInput) {
      THTensor *gradIn = THTensor_(newSelect)(gradInput, 1, f);

      if (train) {
        // when in training mode
        // Q(X) = X - E[x] ; i.e. input centered to zero mean
        // Y = Q(X) / σ    ; i.e. BN output before weight and bias
        // dL/dX = (Q(dL/dY) - dot(Y, dL/dY) * Y) / σ * w

        // projection of gradOutput on to output scaled by std
        scalar_t k = (scalar_t) dotp * invstd * invstd / n;
        TH_TENSOR_APPLY2(scalar_t, gradIn, scalar_t, in,
          *gradIn_data = (*in_data - mean) * k;);

        accreal gradMean = sum / n;
        TH_TENSOR_APPLY2(scalar_t, gradIn, scalar_t, gradOut,
          *gradIn_data = (*gradOut_data - gradMean - *gradIn_data) * invstd * w;);

      } else {
        // when in evaluation mode
        // Q(X) = X - running_mean  ; i.e. input centered to zero mean
        // Y = Q(X) / running_std    ; i.e. BN output before weight and bias
        // dL/dX = w / running_std
        TH_TENSOR_APPLY2(scalar_t, gradIn, scalar_t, gradOut,
          *gradIn_data = *gradOut_data * invstd * w;);
      }

      c10::raw::intrusive_ptr::decref(gradIn);
    }

    if (gradWeight) {
      scalar_t val = THTensor_(get1d)(gradWeight, f);
      THTensor_(set1d)(gradWeight, f, val + scale * dotp * invstd);
    }

    if (gradBias) {
      scalar_t val = THTensor_(get1d)(gradBias, f);
      THTensor_(set1d)(gradBias, f, val + scale * sum);
    }

    c10::raw::intrusive_ptr::decref(gradOut);
    c10::raw::intrusive_ptr::decref(in);
  }
}

#endif
