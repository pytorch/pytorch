#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "THNN/generic/SoftPlus.c"
#else

void THNN_(SoftPlus_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          accreal beta_,
          accreal threshold_)
{
  scalar_t beta = TH_CONVERT_ACCREAL_TO_REAL(beta_);
  scalar_t threshold = TH_CONVERT_ACCREAL_TO_REAL(threshold_);
  THTensor_(resizeAs)(output, input);

  // f(x) = 1/beta * log(1 + exp(beta * x))
  TH_TENSOR_APPLY2(scalar_t, output, scalar_t, input,               \
    *output_data = (*input_data * beta) > threshold ? *input_data : std::log1p(exp(*input_data * beta)) / beta;
  );
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
  scalar_t beta = TH_CONVERT_ACCREAL_TO_REAL(beta_);
  scalar_t threshold = TH_CONVERT_ACCREAL_TO_REAL(threshold_);
  THNN_CHECK_NELEMENT(input, gradOutput);
  THTensor_(resizeAs)(gradInput, output);

  // d/dx[log(1+exp(k*x))/k] = exp(kx) / (exp(kx) + 1)
  // SINCE
  // y = (1/k)*log(1+exp(k*x)) --> x = (1/k)*log(exp(k*y)-1)
  // THEREFORE:
  // d/dx(f(x)) = (exp(k*y) - 1) / exp(k*y)
  TH_TENSOR_APPLY3(scalar_t, gradInput, scalar_t, gradOutput, scalar_t, output,
    scalar_t z = exp(*output_data * beta);
    *gradInput_data = (*output_data * beta) > threshold ? *gradOutput_data : *gradOutput_data * (z - 1.)/z;
  );
}

#endif
