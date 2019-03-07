#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "THNN/generic/TemporalUpSamplingNearest.c"
#else

#include <THNN/generic/upsampling.h>

static inline void THNN_(TemporalUpSamplingNearest_shapeCheck)
     (THTensor *input, THTensor *gradOutput,
      int nBatch, int nChannels,
      int inputWidth, int outputWidth) {
  THArgCheck(inputWidth > 0 && outputWidth > 0, 2,
       "input and output sizes should be greater than 0,"
       " but got input (W: %d) output (W: %d)",
       inputWidth, outputWidth);
  if (input != NULL) {
    THNN_ARGCHECK(THTensor_nDimensionLegacyAll(input) == 3, 2, input,
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
  scalar_t *idata = input->data<scalar_t>();
  scalar_t *odata = output->data<scalar_t>();

  // special case: just copy
  if (inputWidth == outputWidth) {
    for (int w2 = 0; w2 < outputWidth; ++w2) {
      const int w1 = w2;
      const scalar_t* pos1 = &idata[w1];
      scalar_t* pos2 = &odata[w2];
      for (int c = 0; c < channels; ++c) {
        pos2[0] = pos1[0];
        pos1 += inputWidth;
        pos2 += outputWidth;
      }
    }
    c10::raw::intrusive_ptr::decref(input);
    return;
  }

  for (int w2 = 0; w2 < outputWidth; ++w2) {
    const accreal src_x = nearest_neighbor_compute_source_index(scale, w2, inputWidth);
    const int w1 = src_x;
    const scalar_t* pos1 = &idata[w1];
    scalar_t* pos2 = &odata[w2];
    for (int c = 0; c < channels; ++c) {
      pos2[0] = pos1[0];
      pos1 += inputWidth;
      pos2 += outputWidth;
    }
  }
  c10::raw::intrusive_ptr::decref(input);
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
  scalar_t *data1 = gradInput->data<scalar_t>();
  scalar_t *data2 = gradOutput->data<scalar_t>();
  channels = nbatch * channels;
  const float scale = (float) inputWidth / (float)outputWidth;

  // special case: same-size matching grids
  if (inputWidth == outputWidth) {
    for (int w2 = 0; w2 < outputWidth; ++w2) {
      const int w1 = w2;
      scalar_t* pos1 = &data1[w1];
      const scalar_t* pos2 = &data2[w2];
      for (int c = 0; c < channels; ++c) {
        pos1[0] += pos2[0];
        pos1 += inputWidth;
        pos2 += outputWidth;
      }
    }
    c10::raw::intrusive_ptr::decref(gradOutput);
    return;
  }

  for (int w2 = 0; w2 < outputWidth; ++w2) {
    const int w1 = nearest_neighbor_compute_source_index(scale, w2, inputWidth);
    scalar_t* pos1 = &data1[w1];
    const scalar_t* pos2 = &data2[w2];
    for (int c = 0; c < channels; ++c) {
      pos1[0] += pos2[0];
      pos1 += inputWidth;
      pos2 += outputWidth;
    }
  }
  c10::raw::intrusive_ptr::decref(gradOutput);
}

#endif
