#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/GatedLinearUnit.c"
#else

void THNN_(GatedLinear_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int dim)
{
  // size output to half of input
  dim = dim - TH_INDEX_BASE;
  const int64_t nIn = THTensor_(size)(input, dim);
  THArgCheck(nIn % 2 == 0, 2, "Halving dimension must be even. Dim %d is size %ld",
      dim + TH_INDEX_BASE, nIn);

  const int64_t inputSize = THTensor_(size)(input, dim) / 2;
  THLongStorage *newSizes = THTensor_(newSizeOf)(input);
  THLongStorage_set(newSizes, dim, inputSize);
  THTensor_(resize)(output, newSizes, NULL);

  // halve tensor
  THTensor *firstHalf = THTensor_(newNarrow)(input, dim, 0, inputSize);
  THTensor *secondHalf = THTensor_(newNarrow)(input, dim, inputSize, inputSize);

  // x = x1:cmul( sigmoid(x2) )
  THTensor_(sigmoid)(output, secondHalf);
  THTensor_(cmul)(output, output, firstHalf);

  THLongStorage_free(newSizes);
  THTensor_(free)(firstHalf);
  THTensor_(free)(secondHalf);
}

void THNN_(GatedLinear_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          int dim)
{
  // set up tensors
  dim = dim - TH_INDEX_BASE;
  const int64_t nIn = THTensor_(size)(input, dim);
  THArgCheck(nIn % 2 == 0, 2, "Halving dimension must be even. Dim %d is size %ld",
      dim + TH_INDEX_BASE, nIn);

  THTensor_(resizeAs)(gradInput, input);
  const int64_t inputSize = THTensor_(size)(input, dim) / 2;
  THTensor *firstHalf = THTensor_(newNarrow)(input, dim, 0, inputSize);
  THTensor *secondHalf = THTensor_(newNarrow)(input, dim, inputSize, inputSize);
  THTensor *gradInputfirstHalf = THTensor_(newNarrow)(gradInput, dim, 0, inputSize);
  THTensor *gradInputsecondHalf = THTensor_(newNarrow)(gradInput, dim, inputSize, inputSize);

  THTensor_(sigmoid)(gradInputfirstHalf, secondHalf);

  TH_TENSOR_APPLY2(real, gradInputsecondHalf, real, gradInputfirstHalf,
    real z = *gradInputfirstHalf_data;
    *gradInputsecondHalf_data = (1. - z) * z;
  );

  THTensor_(cmul)(gradInputfirstHalf, gradInputfirstHalf, gradOutput);

  THTensor_(cmul)(gradInputsecondHalf, gradInputsecondHalf, gradOutput);
  THTensor_(cmul)(gradInputsecondHalf, gradInputsecondHalf, firstHalf);

  THTensor_(free)(firstHalf);
  THTensor_(free)(secondHalf);
  THTensor_(free)(gradInputfirstHalf);
  THTensor_(free)(gradInputsecondHalf);
}

#endif
