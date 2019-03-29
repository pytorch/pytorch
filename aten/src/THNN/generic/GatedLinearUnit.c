#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "THNN/generic/GatedLinearUnit.c"
#else

void THNN_(GatedLinear_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int dim)
{
  // size output to half of input
  const int64_t nIn = THTensor_sizeLegacyNoScalars(input, dim);
  THArgCheck(nIn % 2 == 0, 2, "Halving dimension must be even. Dim %d is size %ld",
      dim, nIn);

  const int64_t inputSize = THTensor_(size)(input, dim) / 2;
  std::vector<int64_t> newSizes = THTensor_sizesLegacyNoScalars(input);
  newSizes[dim] = inputSize;
  THTensor_(resize)(output, newSizes, {});

  // halve tensor
  THTensor *firstHalf = THTensor_(newNarrow)(input, dim, 0, inputSize);
  THTensor *secondHalf = THTensor_(newNarrow)(input, dim, inputSize, inputSize);

  // x = x1:cmul( sigmoid(x2) )
  THTensor_(sigmoid)(output, secondHalf);
  THTensor_(cmul)(output, output, firstHalf);

  c10::raw::intrusive_ptr::decref(firstHalf);
  c10::raw::intrusive_ptr::decref(secondHalf);
}

void THNN_(GatedLinear_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          int dim)
{
  // set up tensors
  const int64_t nIn = THTensor_(size)(input, dim);
  THArgCheck(nIn % 2 == 0, 2, "Halving dimension must be even. Dim %d is size %ld",
      dim, nIn);

  THTensor_(resizeAs)(gradInput, input);
  const int64_t inputSize = THTensor_(size)(input, dim) / 2;
  THTensor *firstHalf = THTensor_(newNarrow)(input, dim, 0, inputSize);
  THTensor *secondHalf = THTensor_(newNarrow)(input, dim, inputSize, inputSize);
  THTensor *gradInputfirstHalf = THTensor_(newNarrow)(gradInput, dim, 0, inputSize);
  THTensor *gradInputsecondHalf = THTensor_(newNarrow)(gradInput, dim, inputSize, inputSize);

  THTensor_(sigmoid)(gradInputfirstHalf, secondHalf);

  TH_TENSOR_APPLY2(scalar_t, gradInputsecondHalf, scalar_t, gradInputfirstHalf,
    scalar_t z = *gradInputfirstHalf_data;
    *gradInputsecondHalf_data = (1. - z) * z;
  );

  THTensor_(cmul)(gradInputfirstHalf, gradInputfirstHalf, gradOutput);

  THTensor_(cmul)(gradInputsecondHalf, gradInputsecondHalf, gradOutput);
  THTensor_(cmul)(gradInputsecondHalf, gradInputsecondHalf, firstHalf);

  c10::raw::intrusive_ptr::decref(firstHalf);
  c10::raw::intrusive_ptr::decref(secondHalf);
  c10::raw::intrusive_ptr::decref(gradInputfirstHalf);
  c10::raw::intrusive_ptr::decref(gradInputsecondHalf);
}

#endif
