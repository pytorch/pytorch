#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/TemporalUpSamplingNearest.c"
#else

#include "linear_upsampling.h"

static inline void THNN_(TemporalUpSamplingNearest_shapeCheck)
     (THTensor *input, THTensor *gradOutput,
      int nBatch, int nChannels,
      int inputWidth, int outputWidth) {
  THArgCheck(inputWidth > 0 && outputWidth > 0, 2,
       "input and output sizes should be greater than 0,"
       " but got input (W: %d) output (W: %d)",
       inputWidth, outputWidth);
  if (input != NULL) {
    THNN_ARGCHECK(input->_dim() == 3, 2, input,
      "3D input tensor expected but got: %s");
  }

  if (gradOutput != NULL) {
    THNN_CHECK_DIM_SIZE(gradOutput, 3, 0, nBatch);
    THNN_CHECK_DIM_SIZE(gradOutput, 3, 1, nChannels);
    THNN_CHECK_DIM_SIZE(gradOutput, 3, 2, outputWidth);
  }
}

void THNN_(TemporalUpSamplingNearest_updateOutput)(
    THNNState *state,
    THTensor *input,
    THTensor *output,
    int outputWidth)
{
  int nbatch = THTensor_(size)(input, 0);
  int channels = THTensor_(size)(input, 1);
  int inputWidth = THTensor_(size)(input, 2);
  const float scale = (float) inputWidth / (float)outputWidth;

  THNN_(TemporalUpSamplingNearest_shapeCheck)(input, NULL, nbatch, channels, inputWidth, outputWidth);

    THTensor_(resize3d)(output,
			THTensor_(size)(input, 0),
      THTensor_(size)(input, 1),
      outputWidth);
    channels = channels * nbatch;

  THAssert(inputWidth > 0 && outputWidth > 0);

  input = THTensor_(newContiguous)(input);
  THTensor_(zero)(output);
  real *idata = THTensor_(data)(input);
  real *odata = THTensor_(data)(output);

  // special case: just copy
  if (inputWidth == outputWidth) {
    for (int w2 = 0; w2 < outputWidth; ++w2) {
      const int w1 = w2;
      const real* pos1 = &idata[w1];
      real* pos2 = &odata[w2];
      for (int c = 0; c < channels; ++c) {
        pos2[0] = pos1[0];
        pos1 += inputWidth;
        pos2 += outputWidth;
      }
    }
    THTensor_(free)(input);
    return;
  }

  for (int w2 = 0; w2 < outputWidth; ++w2) {
    const accreal src_x = nearest_neighbor_compute_source_index(scale, w2, inputWidth);
    const int w1 = src_x;
    const real* pos1 = &idata[w1];
    real* pos2 = &odata[w2];
    for (int c = 0; c < channels; ++c) {
      pos2[0] = pos1[0];
      pos1 += inputWidth;
      pos2 += outputWidth;
    }
  }
  THTensor_(free)(input);
}

void THNN_(TemporalUpSamplingNearest_updateGradInput)(
    THNNState *state,
    THTensor *gradOutput,
    THTensor *gradInput,
    int nbatch,
    int channels,
    int inputWidth,
    int outputWidth)
{
  THNN_(TemporalUpSamplingNearest_shapeCheck)(NULL, gradOutput, nbatch, channels, inputWidth, outputWidth);
  THTensor_(resize3d)(gradInput, nbatch, channels, inputWidth);
  THTensor_(zero)(gradInput);
  gradOutput = THTensor_(newContiguous)(gradOutput);
  real *data1 = THTensor_(data)(gradInput);
  real *data2 = THTensor_(data)(gradOutput);
  channels = nbatch * channels;
  const float scale = (float) inputWidth / (float)outputWidth;

  // special case: same-size matching grids
  if (inputWidth == outputWidth) {
    for (int w2 = 0; w2 < outputWidth; ++w2) {
      const int w1 = w2;
      real* pos1 = &data1[w1];
      const real* pos2 = &data2[w2];
      for (int c = 0; c < channels; ++c) {
        pos1[0] += pos2[0];
        pos1 += inputWidth;
        pos2 += outputWidth;
      }
    }
    THTensor_(free)(gradOutput);
    return;
  }

  for (int w2 = 0; w2 < outputWidth; ++w2) {
    const int w1 = nearest_neighbor_compute_source_index(scale, w2, inputWidth);
    real* pos1 = &data1[w1];
    const real* pos2 = &data2[w2];
    for (int c = 0; c < channels; ++c) {
      pos1[0] += pos2[0];
      pos1 += inputWidth;
      pos2 += outputWidth;
    }
  }
  THTensor_(free)(gradOutput);
}

#endif
