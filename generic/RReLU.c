#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/RReLU.c"
#else

void THNN_(RReLU_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *noise,
          real lower,
          real upper,
          bool train,
          bool inplace,
          THGenerator *generator)
{
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
      TH_TENSOR_APPLY2(real, input, real, output,
        const real r = (*input_data) <= 0 ? negSlope : 1;
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
          real lower,
          real upper,
          bool train,
          bool inplace)
{
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
      TH_TENSOR_APPLY2(real, gradOutput, real, input,
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
      TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, input,
        *gradInput_data = (*input_data) <= 0 ? (*gradOutput_data) * negSlope : (*gradOutput_data);
      );
    }
  }
}

#endif
