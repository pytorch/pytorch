#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "THNN/generic/VolumetricUpSamplingNearest.c"
#else

#include <THNN/generic/linear_upsampling.h>

static inline void THNN_(VolumetricUpSamplingNearest_shapeCheck)
     (THTensor *input, THTensor *gradOutput,
      int nBatch, int nChannels,
      int inputDepth, int inputHeight, int inputWidth,
      int outputDepth, int outputHeight, int outputWidth) {
  THArgCheck(inputDepth > 0 && inputHeight > 0 && inputWidth > 0
       && outputDepth > 0 && outputHeight > 0 && outputWidth > 0, 2,
       "input and output sizes should be greater than 0,"
       " but got input (D: %d, H: %d, W: %d) output (D: %d, H: %d, W: %d)",
       inputDepth, inputHeight, inputWidth, outputDepth, outputHeight, outputWidth);
  if (input != NULL) {
    THNN_ARGCHECK(THTensor_nDimensionLegacyAll(input) == 5, 2, input,
      "5D input tensor expected but got: %s");
  }

  if (gradOutput != NULL) {
    THNN_CHECK_DIM_SIZE(gradOutput, 5, 0, nBatch);
    THNN_CHECK_DIM_SIZE(gradOutput, 5, 1, nChannels);
    THNN_CHECK_DIM_SIZE(gradOutput, 5, 2, outputDepth);
    THNN_CHECK_DIM_SIZE(gradOutput, 5, 3, outputHeight);
    THNN_CHECK_DIM_SIZE(gradOutput, 5, 4, outputWidth);
  }
}


void THNN_(VolumetricUpSamplingNearest_updateOutput)(
    THNNState *state,
    THTensor *input,
    THTensor *output,
    int outputDepth,
    int outputHeight,
    int outputWidth)
{
  int nbatch = THTensor_(size)(input, 0);
  int channels = THTensor_(size)(input, 1);
  int inputDepth = THTensor_(size)(input, 2);
  int inputHeight = THTensor_(size)(input, 3);
  int inputWidth = THTensor_(size)(input, 4);
  const float depth_scale = (float) inputDepth / (float) outputDepth;
  const float height_scale = (float) inputHeight / (float)outputHeight;
  const float width_scale = (float) inputWidth / (float)outputWidth;

  THNN_(VolumetricUpSamplingNearest_shapeCheck)(input, NULL, nbatch, channels, inputDepth, inputHeight, inputWidth, outputDepth, outputHeight, outputWidth);

  THTensor_(resize5d)(output,
                      THTensor_(size)(input, 0),
                      THTensor_(size)(input, 1),
                      outputDepth,
                      outputHeight,
                      outputWidth);
  channels = channels * nbatch;

  THAssert(inputDepth > 0 && inputHeight > 0 && inputWidth > 0 && outputDepth > 0 && outputHeight > 0 && outputWidth > 0);

  input = THTensor_(newContiguous)(input);
  THTensor_(zero)(output);
  scalar_t *idata = input->data<scalar_t>();
  scalar_t *odata = output->data<scalar_t>();

  // special case: just copy
  if (inputDepth == outputDepth && inputHeight == outputHeight && inputWidth == outputWidth) {
    for (int d2 = 0; d2 < outputDepth; ++d2) {
      const int d1 = d2;
      for (int h2 = 0; h2 < outputHeight; ++h2) {
        const int h1 = h2;
        for (int w2 = 0; w2 < outputWidth; ++w2) {
          const int w1 = w2;
          const scalar_t* pos1 = &idata[d1 * inputHeight * inputWidth + h1 * inputWidth + w1];
          scalar_t* pos2 = &odata[d2 * outputHeight * outputWidth + h2 * outputWidth + w2];
          for (int c = 0; c < channels; ++c) {
            pos2[0] = pos1[0];
            pos1 += inputDepth * inputHeight * inputWidth;
            pos2 += outputDepth * outputHeight * outputWidth;
          }
        }
      }
    }
    c10::raw::intrusive_ptr::decref(input);
    return;
  }

  for (int d2 = 0; d2 < outputDepth; ++d2) {
    const int d1 = nearest_neighbor_compute_source_index(depth_scale, d2, inputDepth);
    for (int h2 = 0; h2 < outputHeight; ++h2) {
      const int h1 = nearest_neighbor_compute_source_index(height_scale, h2, inputHeight);
      for (int w2 = 0; w2 < outputWidth; ++w2) {
        const int w1 = nearest_neighbor_compute_source_index(width_scale, w2, inputWidth);
        const scalar_t* pos1 = &idata[d1 * inputHeight * inputWidth + h1 * inputWidth + w1];
        scalar_t* pos2 = &odata[d2 * outputHeight * outputWidth + h2 * outputWidth + w2];
        for (int c = 0; c < channels; ++c) {
          pos2[0] = pos1[0];
          pos1 += inputDepth * inputHeight * inputWidth;
          pos2 += outputDepth * outputHeight * outputWidth;
        }
      }
    }
  }
  c10::raw::intrusive_ptr::decref(input);
}

void THNN_(VolumetricUpSamplingNearest_updateGradInput)(
    THNNState *state,
    THTensor *gradOutput,
    THTensor *gradInput,
    int nbatch,
    int channels,
    int inputDepth,
    int inputHeight,
    int inputWidth,
    int outputDepth,
    int outputHeight,
    int outputWidth)
{
  THNN_(VolumetricUpSamplingNearest_shapeCheck)(NULL, gradOutput, nbatch, channels, inputDepth, inputHeight, inputWidth, outputDepth, outputHeight, outputWidth);
  THTensor_(resize5d)(gradInput, nbatch, channels, inputDepth, inputHeight, inputWidth);
  THTensor_(zero)(gradInput);
  gradOutput = THTensor_(newContiguous)(gradOutput);
  scalar_t *idata = gradInput->data<scalar_t>();
  scalar_t *odata = gradOutput->data<scalar_t>();
  channels = nbatch * channels;
  const float depth_scale = (float) inputDepth / (float) outputDepth;
  const float height_scale = (float) inputHeight / (float)outputHeight;
  const float width_scale = (float) inputWidth / (float)outputWidth;

  // special case: just copy
  if (inputDepth == outputDepth && inputHeight == outputHeight && inputWidth == outputWidth) {
    for (int d2 = 0; d2 < outputDepth; ++d2) {
      const int d1 = d2;
      for (int h2 = 0; h2 < outputHeight; ++h2) {
        const int h1 = h2;
        for (int w2 = 0; w2 < outputWidth; ++w2) {
          const int w1 = w2;
          scalar_t* pos1 = &idata[d1 * inputHeight * inputWidth + h1 * inputWidth + w1];
          const scalar_t* pos2 = &odata[d2 * outputHeight * outputWidth + h2 * outputWidth + w2];
          for (int c = 0; c < channels; ++c) {
            pos1[0] += pos2[0];
            pos1 += inputDepth * inputHeight * inputWidth;
            pos2 += outputDepth * outputHeight * outputWidth;
          }
        }
      }
    }
    c10::raw::intrusive_ptr::decref(gradOutput);
    return;
  }

  for (int d2 = 0; d2 < outputDepth; ++d2) {
    const int d1 = nearest_neighbor_compute_source_index(depth_scale, d2, inputDepth);
    for (int h2 = 0; h2 < outputHeight; ++h2) {
      const int h1 = nearest_neighbor_compute_source_index(height_scale, h2, inputHeight);
      for (int w2 = 0; w2 < outputWidth; ++w2) {
        const int w1 = nearest_neighbor_compute_source_index(width_scale, w2, inputWidth);
        scalar_t* pos1 = &idata[d1 * inputHeight * inputWidth + h1 * inputWidth + w1];
        const scalar_t* pos2 = &odata[d2 * outputHeight * outputWidth + h2 * outputWidth + w2];
        for (int c = 0; c < channels; ++c) {
          pos1[0] += pos2[0];
          pos1 += inputDepth * inputHeight * inputWidth;
          pos2 += outputDepth * outputHeight * outputWidth;
        }
      }
    }
  }

  c10::raw::intrusive_ptr::decref(gradOutput);
}

#endif
