#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "THNN/generic/RReLU.c"
#else

#include <ATen/Utils.h>

void THNN_(RReLU_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *noise,
          accreal lower_,
          accreal upper_,
          bool train,
          bool inplace,
          at::Generator *generator)
{
  auto gen = at::check_generator_with_default<at::CPUGenerator>(generator, at::detail::getDefaultCPUGenerator());
  // See Note [Thread-safety and Generators]
  std::lock_guard<std::mutex> lock(gen->mutex_);

  scalar_t lower = TH_CONVERT_ACCREAL_TO_REAL(lower_);
  scalar_t upper = TH_CONVERT_ACCREAL_TO_REAL(upper_);
  if (train)
  {
    // get default random generator
    THTensor_(resizeAs)(noise, input);
    if (inplace)
    {
      TH_TENSOR_APPLY2(scalar_t, input, scalar_t, noise,
        if (*input_data <= 0)
        {
          at::uniform_real_distribution<double> uniform(lower, upper);
          const scalar_t r = (scalar_t)uniform(gen);
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
      TH_TENSOR_APPLY3(scalar_t, input, scalar_t, output, scalar_t, noise,
        if (*input_data <= 0)
        {
          at::uniform_real_distribution<double> uniform(lower, upper);
          const scalar_t r = (scalar_t)uniform(gen);
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
    const scalar_t negSlope = (lower + upper) / 2;
    if (inplace)
    {
      TH_TENSOR_APPLY(scalar_t, input,
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
      TH_TENSOR_APPLY2(scalar_t, input, scalar_t, output,
        const scalar_t r = (*input_data) <= 0 ? negSlope : 1;
        *output_data = *input_data * r;
      );
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
  scalar_t lower = TH_CONVERT_ACCREAL_TO_REAL(lower_);
  scalar_t upper = TH_CONVERT_ACCREAL_TO_REAL(upper_);
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
    const scalar_t negSlope = (lower + upper) / 2;
    if (inplace)
    {
      TH_TENSOR_APPLY2(scalar_t, gradOutput, scalar_t, input,
        if (*input_data <= 0)
        {
          *gradOutput_data = (*gradOutput_data) * negSlope;
        }
      );
      THTensor_(set)(gradInput, gradOutput);
    }
    else
    {
      THTensor_(resizeAs)(gradInput, input);
      TH_TENSOR_APPLY3(scalar_t, gradInput, scalar_t, gradOutput, scalar_t, input,
        *gradInput_data = (*input_data) <= 0 ? (*gradOutput_data) * negSlope : (*gradOutput_data);
      );
    }
  }
}

#endif
