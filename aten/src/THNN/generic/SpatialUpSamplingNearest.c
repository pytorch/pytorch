#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialUpSamplingNearest.c"
#else

#include "linear_upsampling.h"

static inline void THNN_(SpatialUpSamplingNearest_shapeCheck)
     (THTensor *input, THTensor *gradOutput,
      int nBatch, int nChannels,
      int inputHeight, int inputWidth,
      int outputHeight, int outputWidth) {
  THArgCheck(inputHeight > 0 && inputWidth > 0
       && outputHeight > 0 && outputWidth > 0, 2,
       "input and output sizes should be greater than 0,"
       " but got input (H: %d, W: %d) output (H: %d, W: %d)",
       inputHeight, inputWidth, outputHeight, outputWidth);
  if (input != NULL) {
    THNN_ARGCHECK(input->_dim() == 4, 2, input,
      "4D input tensor expected but got: %s");
  }

  if (gradOutput != NULL) {
    THNN_CHECK_DIM_SIZE(gradOutput, 4, 0, nBatch);
    THNN_CHECK_DIM_SIZE(gradOutput, 4, 1, nChannels);
    THNN_CHECK_DIM_SIZE(gradOutput, 4, 2, outputHeight);
    THNN_CHECK_DIM_SIZE(gradOutput, 4, 3, outputWidth);
  }
}


void THNN_(SpatialUpSamplingNearest_updateOutput)(
    THNNState *state,
    THTensor *input,
    THTensor *output,
    int outputHeight,
    int outputWidth)
{
  int nbatch = THTensor_(size)(input, 0);
  int channels = THTensor_(size)(input, 1);
  int inputHeight = THTensor_(size)(input, 2);
  int inputWidth = THTensor_(size)(input, 3);
  const float height_scale = (float) inputHeight / (float) outputHeight;
  const float width_scale = (float) inputWidth / (float) outputWidth;

  THNN_(SpatialUpSamplingNearest_shapeCheck)(input, NULL, nbatch, channels,
		  inputHeight, inputWidth, outputHeight, outputWidth);

  THTensor_(resize4d)(output,
                      THTensor_(size)(input, 0),
                      THTensor_(size)(input, 1),
                      outputHeight,
                      outputWidth);
  channels = channels * nbatch;

  THAssert(inputWidth > 0 && outputWidth > 0);

  input = THTensor_(newContiguous)(input);
  THTensor_(zero)(output);
  real *idata = THTensor_(data)(input);
  real *odata = THTensor_(data)(output);

  // special case: just copy
  if (inputHeight == outputHeight && inputWidth == outputWidth) {
    for (int h2 = 0; h2 < outputHeight; ++h2) {
      const int h1 = h2;
      for (int w2 = 0; w2 < outputWidth; ++w2) {
        const int w1 = w2;
        const real* pos1 = &idata[h1 * inputWidth + w1];
        real* pos2 = &odata[h2 * outputWidth + w2];
        for (int c = 0; c < channels; ++c) {
          pos2[0] = pos1[0];
          pos1 += inputHeight * inputWidth;
          pos2 += outputHeight * outputWidth;
        }
      }
    }
    THTensor_(free)(input);
    return;
  }

  for (int h2 = 0; h2 < outputHeight; ++h2) {
    const int h1 = nearest_neighbor_compute_source_index(height_scale, h2, inputHeight);
    for (int w2 = 0; w2 < outputWidth; ++w2) {
      const int w1 = nearest_neighbor_compute_source_index(width_scale, w2, inputWidth);
      const real* pos1 = &idata[h1 * inputWidth + w1];
      real* pos2 = &odata[h2 * outputWidth + w2];
      for (int c = 0; c < channels; ++c) {
        pos2[0] = pos1[0];
        pos1 += inputHeight * inputWidth;
        pos2 += outputHeight * outputWidth;
      }
    }
  }
  THTensor_(free)(input);
}

void THNN_(SpatialUpSamplingNearest_updateGradInput)(
    THNNState *state,
    THTensor *gradOutput,
    THTensor *gradInput,
    int nbatch,
    int channels,
    int inputHeight,
    int inputWidth,
    int outputHeight,
    int outputWidth)
{
  THNN_(SpatialUpSamplingNearest_shapeCheck)(NULL, gradOutput, nbatch, channels,
		  inputHeight, inputWidth, outputHeight, outputWidth);
  THTensor_(resize4d)(gradInput, nbatch, channels, inputHeight, inputWidth);
  THTensor_(zero)(gradInput);
  gradOutput = THTensor_(newContiguous)(gradOutput);
  real *idata = THTensor_(data)(gradInput);
  real *odata = THTensor_(data)(gradOutput);
  channels = nbatch * channels;
  const float height_scale = (float) inputHeight / (float)outputHeight;
  const float width_scale = (float) inputWidth / (float)outputWidth;
  // special case: just copy
  if (inputHeight == outputHeight && inputWidth == outputWidth) {
    for (int h2 = 0; h2 < outputHeight; ++h2) {
      const int h1 = h2;
      for (int w2 = 0; w2 < outputWidth; ++w2) {
        const int w1 = w2;
        real* pos1 = &idata[h1 * inputWidth + w1];
        const real* pos2 = &odata[h2 * outputWidth + w2];
        for (int c = 0; c < channels; ++c) {
          pos1[0] = pos2[0];
          pos1 += inputHeight * inputWidth;
          pos2 += outputHeight * outputWidth;
        }
      }
    }
    THTensor_(free)(gradOutput);
    return;
  }

  for (int h2 = 0; h2 < outputHeight; ++h2) {
    const int h1 = nearest_neighbor_compute_source_index(height_scale, h2, inputHeight);
    for (int w2 = 0; w2 < outputWidth; ++w2) {
      const int w1 = nearest_neighbor_compute_source_index(width_scale, w2, inputWidth);
      real* pos1 = &idata[h1 * inputWidth + w1];
      const real* pos2 = &odata[h2 * outputWidth + w2];
      for (int c = 0; c < channels; ++c) {
        pos1[0] += pos2[0];
        pos1 += inputHeight * inputWidth;
        pos2 += outputHeight * outputWidth;
      }
    }
  }

  THTensor_(free)(gradOutput);
}

#endif
