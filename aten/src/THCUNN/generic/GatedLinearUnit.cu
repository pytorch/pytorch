#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THCUNN/generic/GatedLinearUnit.cu"
#else

void THNN_(GatedLinear_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           int dim)
{
  TORCH_INTERNAL_ASSERT(false, "called GatedLinear_updateOutput, but this is just " \
                        "a stub for nn.yaml parsing");
}

void THNN_(GatedLinear_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           int dim)
{
  THCUNN_assertSameGPU(state, 2, gradOutput, gradInput);
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
