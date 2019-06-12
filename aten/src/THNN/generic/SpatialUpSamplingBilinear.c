// Adapted from interp.cpp from Caffe util by Pauline Luc
// Originally developed by George Papandreou

#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "THNN/generic/SpatialUpSamplingBilinear.c"
#else

#include <THNN/generic/upsampling.h>

static inline void THNN_(SpatialUpSamplingBilinear_shapeCheck)
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
    THNN_ARGCHECK(!input->is_empty() && input->dim() == 4, 2, input,
		  "non-empty 4D input tensor expected but got: %s");
  }

  if (gradOutput != NULL) {
    THNN_CHECK_DIM_SIZE(gradOutput, 4, 0, nBatch);
    THNN_CHECK_DIM_SIZE(gradOutput, 4, 1, nChannels);
    THNN_CHECK_DIM_SIZE(gradOutput, 4, 2, outputHeight);
    THNN_CHECK_DIM_SIZE(gradOutput, 4, 3, outputWidth);
  }
}

void THNN_(SpatialUpSamplingBilinear_updateOutput)(
    THNNState *state,
    THTensor *input,
    THTensor *output,
    int outputHeight,
    int outputWidth,
    bool align_corners){

  int nbatch = THTensor_(size)(input, 0);
  int channels = THTensor_(size)(input, 1);
  int inputHeight = THTensor_(size)(input, 2);
  int inputWidth = THTensor_(size)(input, 3);

  THNN_(SpatialUpSamplingBilinear_shapeCheck)
    (input, NULL,
     nbatch, channels,
     inputHeight, inputWidth,
     outputHeight, outputWidth);

  input = THTensor_(newContiguous)(input);
  THTensor_(resize4d)(output,
		      THTensor_(size)(input, 0),
		      THTensor_(size)(input, 1),
		      outputHeight, outputWidth);
  THTensor_(zero)(output);
  scalar_t *idata = input->data<scalar_t>();
  scalar_t *odata = output->data<scalar_t>();
  channels = nbatch * channels;
  THAssert(inputHeight > 0 && inputWidth > 0 && outputHeight > 0 && outputWidth > 0);
  // special case: just copy
  if (inputHeight == outputHeight && inputWidth == outputWidth) {
    for (int h2 = 0; h2 < outputHeight; ++h2) {
      const int h1 = h2;
      for (int w2 = 0; w2 < outputWidth; ++w2) {
        const int w1 = w2;
        const scalar_t* pos1 = &idata[h1 * inputWidth + w1];
        scalar_t* pos2 = &odata[h2 * outputWidth + w2];
        for (int c = 0; c < channels; ++c) {
          pos2[0] = pos1[0];
          pos1 += inputWidth * inputHeight;
          pos2 += outputWidth * outputHeight;
        }
      }
    }
    c10::raw::intrusive_ptr::decref(input);
    return;
  }
  const accreal rheight = linear_upsampling_compute_scale<accreal>(inputHeight, outputHeight, align_corners);
  const accreal rwidth = linear_upsampling_compute_scale<accreal>(inputWidth, outputWidth, align_corners);
  for (int h2 = 0; h2 < outputHeight; ++h2) {
    const accreal h1r = linear_upsampling_compute_source_index<accreal>(rheight, h2, align_corners);
    const int h1 = h1r;
    const int h1p = (h1 < inputHeight - 1) ? 1 : 0;
    const scalar_t h1lambda = h1r - h1;
    const scalar_t h0lambda = (scalar_t)1. - h1lambda;
    for (int w2 = 0; w2 < outputWidth; ++w2) {
      const accreal w1r = linear_upsampling_compute_source_index<accreal>(rwidth, w2, align_corners);
      const int w1 = w1r;
      const int w1p = (w1 < inputWidth - 1) ? 1 : 0;
      const scalar_t w1lambda = w1r - w1;
      const scalar_t w0lambda = (scalar_t)1. - w1lambda;
      const scalar_t* pos1 = &idata[h1 * inputWidth + w1];
      scalar_t* pos2 = &odata[h2 * outputWidth + w2];
      for (int c = 0; c < channels; ++c) {
        pos2[0] = h0lambda * (w0lambda * pos1[0]+ w1lambda * pos1[w1p])
                  + h1lambda * (w0lambda * pos1[h1p * inputWidth]
                  + w1lambda * pos1[h1p * inputWidth + w1p]);
        pos1 += inputWidth * inputHeight;
        pos2 += outputWidth * outputHeight;
      }
    }
  }
  c10::raw::intrusive_ptr::decref(input);
}

void THNN_(SpatialUpSamplingBilinear_updateGradInput)(
    THNNState *state,
    THTensor *gradOutput,
    THTensor *gradInput,
    int nbatch,
    int channels,
    int inputHeight,
    int inputWidth,
    int outputHeight,
    int outputWidth,
    bool align_corners){

  THNN_(SpatialUpSamplingBilinear_shapeCheck)
    (NULL, gradOutput,
     nbatch, channels,
     inputHeight, inputWidth,
     outputHeight, outputWidth);

  THTensor_(resize4d)(gradInput, nbatch, channels, inputHeight, inputWidth);
  THTensor_(zero)(gradInput);
  gradOutput = THTensor_(newContiguous)(gradOutput);
  scalar_t *data1 = gradInput->data<scalar_t>();
  scalar_t *data2 = gradOutput->data<scalar_t>();
  channels = nbatch * channels;

  // special case: same-size matching grids
  if (inputHeight == outputHeight && inputWidth == outputWidth) {
    for (int h2 = 0; h2 < outputHeight; ++h2) {
      const int h1 = h2;
      for (int w2 = 0; w2 < outputWidth; ++w2) {
        const int w1 = w2;
        scalar_t* pos1 = &data1[h1 * inputWidth + w1];
        const scalar_t* pos2 = &data2[h2 * outputWidth + w2];
        for (int c = 0; c < channels; ++c) {
          pos1[0] += pos2[0];
          pos1 += inputWidth * inputHeight;
          pos2 += outputWidth * outputHeight;
        }
      }
    }
    c10::raw::intrusive_ptr::decref(gradOutput);
    return;
  }
  const accreal rheight = linear_upsampling_compute_scale<accreal>(inputHeight, outputHeight, align_corners);
  const accreal rwidth = linear_upsampling_compute_scale<accreal>(inputWidth, outputWidth, align_corners);
  for (int h2 = 0; h2 < outputHeight; ++h2) {
    const accreal h1r = linear_upsampling_compute_source_index<accreal>(rheight, h2, align_corners);
    const int h1 = h1r;
    const int h1p = (h1 < inputHeight - 1) ? 1 : 0;
    const scalar_t h1lambda = h1r - h1;
    const scalar_t h0lambda = (scalar_t)1. - h1lambda;
    for (int w2 = 0; w2 < outputWidth; ++w2) {
      const accreal w1r = linear_upsampling_compute_source_index<accreal>(rwidth, w2, align_corners);
      const int w1 = w1r;
      const int w1p = (w1 < inputWidth - 1) ? 1 : 0;
      const scalar_t w1lambda = w1r - w1;
      const scalar_t w0lambda = (scalar_t)1. - w1lambda;
      scalar_t* pos1 = &data1[h1 * inputWidth + w1];
      const scalar_t* pos2 = &data2[h2 * outputWidth + w2];
      for (int c = 0; c < channels; ++c) {
        pos1[0] += h0lambda * w0lambda * pos2[0];
        pos1[w1p] += h0lambda * w1lambda * pos2[0];
        pos1[h1p * inputWidth] += h1lambda * w0lambda * pos2[0];
        pos1[h1p * inputWidth + w1p] += h1lambda * w1lambda * pos2[0];
        pos1 += inputWidth * inputHeight;
        pos2 += outputWidth * outputHeight;
      }
    }
  }
  c10::raw::intrusive_ptr::decref(gradOutput);
}

#endif
