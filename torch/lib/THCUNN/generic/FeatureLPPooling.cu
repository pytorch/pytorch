#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/FeatureLPPooling.cu"
#else

#include "../common.h"

// non-batch mode:
// [feature dim]
// [feature dim][opt dim 1]
// [feature dim][opt dim 1][opt dim 2]
//
// batch mode:
// [batch dim][feature dim]
// [batch dim][feature dim][opt dim 1]
// [batch dim][feature dim][opt dim 1][opt dim 2]
THCDeviceTensor<real, 4>
THNN_(FeatureLPPooling_upcast)(THCState* state, THCTensor* t, bool batchMode) {
  int inputDim = THCTensor_(nDimension)(state, t);

  if (inputDim == 1) {
    // [feature dim]
    return toDeviceTensor<real, 1>(state, t).
      upcastOuter<2>().upcastInner<4>();
  } else if (inputDim == 2) {
    if (batchMode) {
      // [batch dim][feature dim]
      return toDeviceTensor<real, 2>(state, t).
        upcastInner<4>();
    } else {
      // [feature dim][opt dim 1]
      return toDeviceTensor<real, 2>(state, t).
        upcastOuter<3>().upcastInner<4>();
    }
  } else if (inputDim == 3) {
    if (batchMode) {
      // [batch dim][feature dim][opt dim 1]
      return toDeviceTensor<real, 3>(state, t).
        upcastInner<4>();
    } else {
      // [feature dim][opt dim 1][opt dim 2]
      return toDeviceTensor<real, 3>(state, t).
        upcastOuter<4>();
    }
  } else {
    // inputDim == 4
    // [batch dim][feature dim][opt dim 1][opt dim 2]
    THAssert(batchMode);
    return toDeviceTensor<real, 4>(state, t);
  }
}

// Resizes `toResize` based on the output size for `src` as an input
// tensor
void
THNN_(FeatureLPPooling_resizeForOutput)(THCState* state,
                                        THCTensor* toResize,
                                        THCTensor* input,
                                        bool batchMode,
                                        int width,
                                        int stride) {
  int inputDim = THCTensor_(nDimension)(state, input);
  THAssert(inputDim >= 1 && inputDim <= 4);

  int64_t outSize =
    lpPoolingOutputSize(THCTensor_(size)(state, input, 0), width, stride);
  if (batchMode) {
    THAssert(inputDim > 1);
    outSize =
      lpPoolingOutputSize(THCTensor_(size)(state, input, 1), width, stride);
  } else {
    THAssert(inputDim < 4);
  }

  if (inputDim == 1) {
    THCTensor_(resize1d)(state, toResize, outSize);
  } else if (inputDim == 2) {
    if (batchMode) {
      THCTensor_(resize2d)(
        state, toResize, THCTensor_(size)(state, input, 0), outSize);
    } else {
      THCTensor_(resize2d)(
        state, toResize, outSize, THCTensor_(size)(state, input, 1));
    }
  } else if (inputDim == 3) {
    if (batchMode) {
      THCTensor_(resize3d)(
        state,
        toResize,
        THCTensor_(size)(state, input, 0), outSize,
        THCTensor_(size)(state, input, 2));
    } else {
      THCTensor_(resize3d)(
        state,
        toResize,
        outSize, THCTensor_(size)(state, input, 1),
        THCTensor_(size)(state, input, 2));
    }
  } else if (inputDim == 4) {
    THCTensor_(resize4d)(
      state,
      toResize,
      THCTensor_(size)(state, input, 0), outSize,
      THCTensor_(size)(state, input, 2), THCTensor_(size)(state, input, 3));
  }
}

// Makes `toResize` the same size/dimensionality as `src`
void
THNN_(FeatureLPPooling_resize)(THCState* state,
                               THCTensor* toResize,
                               THCTensor* src) {
  int inputDim = THCTensor_(nDimension)(state, src);
  THAssert(inputDim >= 1 && inputDim <= 4);

  if (inputDim == 1) {
    THCTensor_(resize1d)(state,
                         toResize,
                         THCTensor_(size)(state, src, 0));
  } else if (inputDim == 2) {
    THCTensor_(resize2d)(
      state,
      toResize,
      THCTensor_(size)(state, src, 0),
      THCTensor_(size)(state, src, 1));
  } else if (inputDim == 3) {
    THCTensor_(resize3d)(
      state,
      toResize,
      THCTensor_(size)(state, src, 0),
      THCTensor_(size)(state, src, 1),
      THCTensor_(size)(state, src, 2));
  } else if (inputDim == 4) {
    THCTensor_(resize4d)(
      state,
      toResize,
      THCTensor_(size)(state, src, 0),
      THCTensor_(size)(state, src, 1),
      THCTensor_(size)(state, src, 2),
      THCTensor_(size)(state, src, 3));
  }
}

void THNN_(FeatureLPPooling_updateOutput)(THCState* state,
                                          THCTensor* inputTH,
                                          THCTensor* outputTH,
                                          accreal power,
                                          int width,
                                          int stride,
                                          bool batchMode) {
  THCUNN_assertSameGPU(state, 2, inputTH, outputTH);

  int inputDim = THCTensor_(nDimension)(state, inputTH);

  if (batchMode) {
    THArgCheck(inputDim >= 2 && inputDim <= 4, 2,
               "input must be 2-4 dimensions for batch mode");
  } else {
    THArgCheck(inputDim >= 1 && inputDim <= 3, 2,
               "input must be 1-3 dimensions for non-batch mode");
  }

  THArgCheck(TensorUtils<THCTensor>::canUse32BitIndexMath(state, inputTH), 2,
             "input tensor must fit into 32-bit index math");

  THCDeviceTensor<TensorUtils<THCTensor>::DataType, 4> input;
  THCDeviceTensor<TensorUtils<THCTensor>::DataType, 4> output;

  input = THNN_(FeatureLPPooling_upcast)(state, inputTH, batchMode);

  // Make sure the feature dimension is properly sized
  THArgCheck(input.getSize(1) >= width, 2,
             "input: feature dimension must be >= width");

  // Make sure that width and stride are within range
  THArgCheck(width >= 2 && width <= 16, 5,
             "width must be between 2 - 16");

  THArgCheck(stride >= 1 && stride <= 4, 6,
             "stride must be between 1 - 4");

  THNN_(FeatureLPPooling_resizeForOutput)(
    state, outputTH, inputTH, batchMode, width, stride);

  output = THNN_(FeatureLPPooling_upcast)(state, outputTH, batchMode);

  bool found = runFeatureLPPoolingUpdateOutput(state,
                                               input,
                                               output,
                                               power,
                                               width,
                                               stride);
  THAssert(found);
}

void THNN_(FeatureLPPooling_updateGradInput)(THCState* state,
                                             THCTensor* gradOutputTH,
                                             THCTensor* inputTH,
                                             THCTensor* outputTH,
                                             THCTensor* gradInputTH,
                                             accreal power,
                                             int width,
                                             int stride,
                                             bool batchMode) {
  THArgCheck(TensorUtils<THCTensor>::canUse32BitIndexMath(state, gradOutputTH), 2,
                "output gradient tensor must fit into 32-bit index math");
  THArgCheck(TensorUtils<THCTensor>::canUse32BitIndexMath(state, inputTH), 3,
                "input tensor must fit into 32-bit index math");
  THCUNN_assertSameGPU(state, 4, gradOutputTH, inputTH, outputTH, gradInputTH);

  int inputDim = THCTensor_(nDimension)(state, inputTH);

  if (batchMode) {
    THArgCheck(inputDim >= 2 && inputDim <= 4, 2,
               "input must be 2-4 dimensions for batch mode");
  } else {
    THArgCheck(inputDim >= 1 && inputDim <= 3, 2,
               "input must be 1-3 dimensions for non-batch mode");
  }

  THCDeviceTensor<TensorUtils<THCTensor>::DataType, 4> gradOutput;
  THCDeviceTensor<TensorUtils<THCTensor>::DataType, 4> input;
  THCDeviceTensor<TensorUtils<THCTensor>::DataType, 4> output;
  THCDeviceTensor<TensorUtils<THCTensor>::DataType, 4> gradInput;

  input = THNN_(FeatureLPPooling_upcast)(state, inputTH, batchMode);

  // Make sure the feature dimension is properly sized
  THArgCheck(input.getSize(1) >= width, 3,
             "input: feature dimension must be >= width");

  // Make sure that width and stride are within range
  THArgCheck(width >= 2 && width <= 16, 7,
             "width must be between 2 - 16");

  THArgCheck(stride >= 1 && stride <= 4, 8,
             "stride must be between 1 - 4");

  gradOutput = THNN_(FeatureLPPooling_upcast)(state, gradOutputTH, batchMode);
  output = THNN_(FeatureLPPooling_upcast)(state, outputTH, batchMode);

  for (int i = 0; i < 4; ++i) {
    THAssertMsg(output.getSize(i) == gradOutput.getSize(i),
                "output and gradOutput sizes do not match");
  }

  // Make sure that the input sizes produce the output sizes
  THArgCheck(lpPoolingOutputSize(input.getSize(1), width, stride) ==
             output.getSize(1), 3,
             "input and output sizes do not match with respect to "
             "width and stride");

  // Resize `gradInput` based on `input`
  THNN_(FeatureLPPooling_resize)(state, gradInputTH, inputTH);
  gradInput = THNN_(FeatureLPPooling_upcast)(state, gradInputTH, batchMode);

  bool found = runFeatureLPPoolingUpdateGradInput(state,
                                                  gradOutput,
                                                  input,
                                                  output,
                                                  gradInput,
                                                  power,
                                                  width,
                                                  stride);
  THAssert(found);
}

#endif
