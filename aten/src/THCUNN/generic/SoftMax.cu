#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/SoftMax.cu"
#else

#include "../common.h"

void THNN_(SoftMax_updateOutput)(
          THCState *state,
          THCTensor *input,
          THCTensor *output,
          int dim)
{
  THCUNN_assertSameGPU(state, 2, input, output);
  THArgCheck(dim >= 0 && dim < input->nDimension, 4,
        "dim out of range (got %d, but input has %d dims)", dim, input->nDimension);
  THArgCheck(TensorUtils<THCTensor>::canUse32BitIndexMath(state, input), 4,
        "input tensor is too large (unsupported size. file a feature request)");

  input = THCTensor_(newContiguous)(state, input);
  THCTensor_(resizeAs)(state, output, input);

  int64_t outer_size = 1;
  int64_t dim_size = input->size[dim];
  int64_t inner_size = 1;
  for (int64_t i = 0; i < dim; ++i)
    outer_size *= input->size[i];
  for (int64_t i = dim + 1; i < input->nDimension; ++i)
    inner_size *= input->size[i];

  HostSoftMaxForward<real, accreal, SoftMaxForwardEpilogue>(
      state,
      THCTensor_(data)(state, input), THCTensor_(data)(state, output),
      outer_size, dim_size, inner_size,
      dim);

  THCTensor_(free)(state, input);
}

void THNN_(SoftMax_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCTensor *output,
           int dim)
{
  THArgCheck(dim >= 0 && dim < output->nDimension, 6,
             "dim out of range (got %d, but input has %d dims)", dim, output->nDimension);
        THArgCheck(TensorUtils<THCTensor>::canUse32BitIndexMath(state, output), 6,
             "input tensor is too large (unsupported size. file a feature request)");
  THCUNN_check_nElement(state, output, gradOutput);
  THCUNN_assertSameGPU(state, 3, output, gradOutput, gradInput);

  THCTensor_(resizeAs)(state, gradInput, output);

  int64_t outer_size = 1;
  int64_t dim_size = output->size[dim];
  int64_t inner_size = 1;
  for (int64_t i = 0; i < dim; ++i)
    outer_size *= output->size[i];
  for (int64_t i = dim + 1; i < output->nDimension; ++i)
    inner_size *= output->size[i];

  output = THCTensor_(newContiguous)(state, output);
  // SoftMaxBackward kernels only sum gradOutput, but softmax needs a gradOutput * output sum
  {
    THCTensor *tmp = THCTensor_(new)(state);
    THCTensor_(cmul)(state, tmp, gradOutput, output);
    gradOutput = tmp;
  }

  HostSoftMaxBackward<real, accreal, SoftMaxBackwardEpilogue>(
      state,
      THCTensor_(data)(state, gradOutput),
      THCTensor_(data)(state, gradInput),
      THCTensor_(data)(state, output),
      outer_size, dim_size, inner_size,
      dim);

  THCTensor_(free)(state, gradOutput);
  THCTensor_(free)(state, output);
}

#endif
