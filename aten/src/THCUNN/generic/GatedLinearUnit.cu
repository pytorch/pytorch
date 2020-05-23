#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THCUNN/generic/GatedLinearUnit.cu"
#else

void THNN_(GatedLinear_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           int dim)
{
  THCUNN_assertSameGPU(state, 2, input, output);

  dim = at::maybe_wrap_dim(dim, input);
  // size output to half of input
  const int64_t nIn = THCTensor_(sizeLegacyNoScalars)(state, input, dim);
  THArgCheck(nIn % 2 == 0, 2, "Halving dimension must be even. Dim %d is size %ld",
      dim, nIn);
  const int64_t inputSize = THCTensor_(size)(state, input, dim) / 2;
  std::vector<int64_t> newSizes = THTensor_sizesLegacyNoScalars(input);
  newSizes[dim] = inputSize;
  THCTensor_(resize)(state, output, newSizes, {});

  // halve tensor
  THCTensor *firstHalf = THCTensor_(newNarrow)(state, input, dim, 0, inputSize);
  THCTensor *secondHalf = THCTensor_(newNarrow)(state, input, dim, inputSize, inputSize);

  // x = x1:cmul( sigmoid(x2) )
  THC_pointwiseApply3<scalar_t, scalar_t, scalar_t>(state, output, secondHalf, firstHalf, gatedLinearCSigMul_functor<scalar_t, accreal>());

  THCTensor_(free)(state, firstHalf);
  THCTensor_(free)(state, secondHalf);
}

void THNN_(GatedLinear_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           int dim)
{
  THCUNN_assertSameGPU(state, 2, gradOutput, gradInput);
  dim = at::maybe_wrap_dim(dim, input);
  const int64_t nIn = THCTensor_(size)(state, input, dim);
  THArgCheck(nIn % 2 == 0, 2, "Halving dimension must be even. Dim %d is size %ld",
      dim, nIn);

  THCTensor_(resizeAs)(state, gradInput, input);
  const int64_t inputSize = THCTensor_(size)(state, input, dim) / 2;
  THCTensor *firstHalf = THCTensor_(newNarrow)(state, input, dim, 0, inputSize);
  THCTensor *gradInputfirstHalf = THCTensor_(newNarrow)(state, gradInput, dim, 0, inputSize);
  const int64_t stride_i = THCTensor_(stride)(state, input, dim) * inputSize;
  const int64_t stride_gI = THCTensor_(stride)(state, gradInput, dim) * inputSize;
  THC_pointwiseApply3<scalar_t, scalar_t, scalar_t>(state, gradInputfirstHalf, gradOutput, firstHalf, gatedLinearDerivative<scalar_t,accreal>(stride_i, stride_gI));
  THCTensor_(free)(state, firstHalf);
  THCTensor_(free)(state, gradInputfirstHalf);
}

#endif
