#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/RReLU.cu"
#else

#include "../common.h"

void THNN_(RReLU_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           THCTensor *noise,
           double lower,
           double upper,
           bool train,
           bool inplace,
           void *generator)
{
  THCUNN_assertSameGPU(state, 3, input, output, noise);
  struct curandStateMtgp32* gen_states = THCRandom_generatorStates(state);

  if (train)
  {
    input = THCTensor_(newContiguous)(state, input);
    THCTensor_(resizeAs)(state, noise, input);
    real *input_data = THCTensor_(data)(state, input);
    real *noise_data = THCTensor_(data)(state, noise);
    ptrdiff_t n = THCTensor_(nElement)(state, input);
    if (inplace)
    {
      rreluUpdateOutputTrain<<<NUM_BLOCKS(n), BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
        n, gen_states, input_data, noise_data, input_data, lower, upper);
      THCTensor_(set)(state, output, input);
    }
    else
    {
      THCTensor_(resizeAs)(state, output, input);
      real *output_data = THCTensor_(data)(state, output);
      rreluUpdateOutputTrain<<<NUM_BLOCKS(n), BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
        n, gen_states, input_data, noise_data, output_data, lower, upper);
    }
    THCudaCheck(cudaGetLastError());
    THCTensor_(free)(state, input);
  }
  else
  {
    const real negSlope = ScalarConvert<double, real>::to((lower + upper) / 2);
    if (inplace)
    {
      THC_pointwiseApply1(state, input, RReLUUpdateOutputEvalIP_functor<real>(negSlope));
      THCTensor_(set)(state, output, input);
    }
    else
    {
      THCTensor_(resizeAs)(state, output, input);
      THC_pointwiseApply2(state, output, input, RReLUUpdateOutputEval_functor<real>(negSlope));
    }
  }
}

void THNN_(RReLU_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCTensor *noise,
           double lower,
           double upper,
           bool train,
           bool inplace)
{
  THCUNN_check_nElement(state, input, gradOutput);
  THCUNN_assertSameGPU(state, 4, input, gradOutput, gradInput, noise);

  gradOutput = THCTensor_(newContiguous)(state, gradOutput);

  if (train && upper - lower > 1E-6)    // e.g. if upper == lower, RReLU behaves like LeakyReLU
  {
    // multiply the gradient by the noise tensor
    if (inplace)
    {
      THCTensor_(cmul)(state, gradOutput, gradOutput, noise);
      THCTensor_(set)(state, gradInput, gradOutput);
    }
    else
    {
      THCTensor_(resizeAs)(state, gradInput, input);
      THCTensor_(cmul)(state, gradInput, gradOutput, noise);
    }
  }
  else
  {
    // use constant factor for negative input values
    const real negSlope = ScalarConvert<double, real>::to((lower + upper) / 2);
    if (inplace)
    {
      THC_pointwiseApply2(state, gradOutput, input, RReLUupdateGradInputEvalIP_functor<real>(negSlope));
      THCTensor_(set)(state, gradInput, gradOutput);
    }
    else
    {
      THCTensor_(resizeAs)(state, gradInput, input);
      THC_pointwiseApply3(state, gradInput, gradOutput, input, RReLUupdateGradInputEval_functor<real>(negSlope));
    }
  }

  THCTensor_(free)(state, gradOutput);
}

#endif
