// Adapted from interp.cpp from Caffe util by Pauline Luc
// Originally developed by George Papandreou

#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/VolumetricUpSamplingTrilinear.c"
#else

static inline void THNN_(VolumetricUpSamplingTrilinear_shapeCheck)
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
    THNN_ARGCHECK(input->nDimension == 5, 2, input,
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

void THNN_(VolumetricUpSamplingTrilinear_updateOutput)(
    THNNState *state,
    THTensor *input,
    THTensor *output,
    int outputDepth,
    int outputHeight,
    int outputWidth,
    bool align_corners){

  int nbatch = THTensor_(size)(input, 0);
  int channels = THTensor_(size)(input, 1);
  int inputDepth = THTensor_(size)(input, 2);
  int inputHeight = THTensor_(size)(input, 3);
  int inputWidth = THTensor_(size)(input, 4);

  THNN_(VolumetricUpSamplingTrilinear_shapeCheck)
    (input, NULL,
     nbatch, channels,
     inputDepth, inputHeight, inputWidth,
     outputDepth, outputHeight, outputWidth);

  input = THTensor_(newContiguous)(input);
  THTensor_(resize5d)(output,
		      THTensor_(size)(input, 0),
		      THTensor_(size)(input, 1),
		      outputDepth, outputHeight, outputWidth);
  THTensor_(zero)(output);
  real *idata = THTensor_(data)(input);
  real *odata = THTensor_(data)(output);
  channels = nbatch * channels;
  THAssert(inputDepth > 0 && inputHeight > 0 && inputWidth > 0 &&
           outputDepth > 0 && outputHeight > 0 && outputWidth > 0);
  // special case: just copy
  if (inputDepth == outputDepth && inputHeight == outputHeight && inputWidth == outputWidth) {
    for (int t2 = 0; t2 < outputDepth; ++t2) {
      const int t1 = t2;
      for (int h2 = 0; h2 < outputHeight; ++h2) {
        const int h1 = h2;
        for (int w2 = 0; w2 < outputWidth; ++w2) {
          const int w1 = w2;
          const real* pos1 = &idata[t1 * inputHeight * inputWidth + h1 * inputWidth + w1];
          real* pos2 = &odata[t2 * outputHeight * outputWidth + h2 * outputWidth + w2];
          for (int c = 0; c < channels; ++c) {
            pos2[0] = pos1[0];
            pos1 += inputWidth * inputHeight * inputDepth;
            pos2 += outputWidth * outputHeight * outputDepth;
          }
        }
      }
    }
    return;
  }
  const float rdepth  = THNN_(linear_upsampling_compute_scale)(inputDepth, outputDepth, align_corners);
  const float rheight = THNN_(linear_upsampling_compute_scale)(inputHeight, outputHeight, align_corners);
  const float rwidth  = THNN_(linear_upsampling_compute_scale)(inputWidth, outputWidth, align_corners);
  for (int t2 = 0; t2 < outputDepth; ++t2) {
    const float t1r = THNN_(linear_upsampling_compute_source_index)(rdepth, t2, align_corners);
    const int t1 = t1r;
    const int t1p = (t1 < inputDepth - 1) ? 1 : 0;
    const real t1lambda = t1r - t1;
    const real t0lambda = (real)1. - t1lambda;
    for (int h2 = 0; h2 < outputHeight; ++h2) {
      const float h1r = THNN_(linear_upsampling_compute_source_index)(rheight, h2, align_corners);
      const int h1 = h1r;
      const int h1p = (h1 < inputHeight - 1) ? 1 : 0;
      const real h1lambda = h1r - h1;
      const real h0lambda = (real)1. - h1lambda;
      for (int w2 = 0; w2 < outputWidth; ++w2) {
        const float w1r = THNN_(linear_upsampling_compute_source_index)(rwidth, w2, align_corners);
        const int w1 = w1r;
        const int w1p = (w1 < inputWidth - 1) ? 1 : 0;
        const real w1lambda = w1r - w1;
        const real w0lambda = (real)1. - w1lambda;
        const real* pos1 = &idata[t1 * inputHeight * inputWidth + h1 * inputWidth + w1];
        real* pos2 = &odata[t2 * outputHeight * outputWidth + h2 * outputWidth + w2];
        for (int c = 0; c < channels; ++c) {
          pos2[0] = t0lambda * (h0lambda * (w0lambda * pos1[0] + w1lambda * pos1[w1p])
                              + h1lambda * (w0lambda * pos1[h1p * inputWidth]
                                          + w1lambda * pos1[h1p * inputWidth + w1p]))
                  + t1lambda * (h0lambda * (w0lambda * pos1[t1p * inputHeight * inputWidth]
                                          + w1lambda * pos1[t1p * inputHeight * inputWidth
                                                            + w1p])
                              + h1lambda * (w0lambda * pos1[t1p * inputHeight * inputWidth
                                                            + h1p * inputWidth]
                                          + w1lambda * pos1[t1p * inputHeight * inputWidth
                                                            + h1p * inputWidth + w1p]));
          pos1 += inputWidth * inputHeight * inputDepth;
          pos2 += outputWidth * outputHeight * outputDepth;
        }
      }
    }
  }
  THTensor_(free)(input);
}

void THNN_(VolumetricUpSamplingTrilinear_updateGradInput)(
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
    int outputWidth,
    bool align_corners){

  THNN_(VolumetricUpSamplingTrilinear_shapeCheck)
    (NULL, gradOutput,
     nbatch, channels,
     inputDepth, inputHeight, inputWidth,
     outputDepth, outputHeight, outputWidth);

  THTensor_(resize5d)(gradInput, nbatch, channels, inputDepth, inputHeight, inputWidth);
  THTensor_(zero)(gradInput);
  gradOutput = THTensor_(newContiguous)(gradOutput);
  real *data1 = THTensor_(data)(gradInput);
  real *data2 = THTensor_(data)(gradOutput);
  channels = nbatch * channels;

  // special case: same-size matching grids
  if (inputDepth == outputDepth && inputHeight == outputHeight && inputWidth == outputWidth) {
    for (int t2 = 0; t2 < outputDepth; ++t2) {
      const int t1 = t2;
      for (int h2 = 0; h2 < outputHeight; ++h2) {
        const int h1 = h2;
        for (int w2 = 0; w2 < outputWidth; ++w2) {
          const int w1 = w2;
          real* pos1 = &data1[t1 * inputHeight * inputWidth + h1 * inputWidth + w1];
          const real* pos2 = &data2[t2 * outputHeight * outputWidth + h2 * outputWidth + w2];
          for (int c = 0; c < channels; ++c) {
            pos1[0] += pos2[0];
            pos1 += inputWidth * inputHeight * inputDepth;
            pos2 += outputWidth * outputHeight * outputDepth;
          }
        }
      }
    }
    return;
  }
  const float rdepth  = THNN_(linear_upsampling_compute_scale)(inputDepth, outputDepth, align_corners);
  const float rheight = THNN_(linear_upsampling_compute_scale)(inputHeight, outputHeight, align_corners);
  const float rwidth  = THNN_(linear_upsampling_compute_scale)(inputWidth, outputWidth, align_corners);
  for (int t2 = 0; t2 < outputDepth; ++t2) {
    const float t1r = THNN_(linear_upsampling_compute_source_index)(rdepth, t2, align_corners);
    const int t1 = t1r;
    const int t1p = (t1 < inputDepth - 1) ? 1 : 0;
    const real t1lambda = t1r - t1;
    const real t0lambda = (real)1. - t1lambda;
    for (int h2 = 0; h2 < outputHeight; ++h2) {
      const float h1r = THNN_(linear_upsampling_compute_source_index)(rheight, h2, align_corners);
      const int h1 = h1r;
      const int h1p = (h1 < inputHeight - 1) ? 1 : 0;
      const real h1lambda = h1r - h1;
      const real h0lambda = (real)1. - h1lambda;
      for (int w2 = 0; w2 < outputWidth; ++w2) {
        const float w1r = THNN_(linear_upsampling_compute_source_index)(rwidth, w2, align_corners);
        const int w1 = w1r;
        const int w1p = (w1 < inputWidth - 1) ? 1 : 0;
        const real w1lambda = w1r - w1;
        const real w0lambda = (real)1. - w1lambda;
        real* pos1 = &data1[t1 * inputHeight * inputWidth + h1 * inputWidth + w1];
        const real* pos2 = &data2[t2 * outputHeight * outputWidth + h2 * outputWidth + w2];
        for (int c = 0; c < channels; ++c) {
          pos1[0] += t0lambda * h0lambda * w0lambda * pos2[0];
          pos1[w1p] += t0lambda * h0lambda * w1lambda * pos2[0];
          pos1[h1p * inputWidth] += t0lambda * h1lambda * w0lambda * pos2[0];
          pos1[h1p * inputWidth + w1p] += t0lambda * h1lambda * w1lambda * pos2[0];
          pos1[t1p * inputHeight * inputWidth] += t1lambda * h0lambda * w0lambda * pos2[0];
          pos1[t1p * inputHeight * inputWidth + w1p] += t1lambda * h0lambda * w1lambda * pos2[0];
          pos1[t1p * inputHeight * inputWidth + h1p * inputWidth] += t1lambda * h1lambda * w0lambda * pos2[0];
          pos1[t1p * inputHeight * inputWidth + h1p * inputWidth + w1p] += t1lambda * h1lambda * w1lambda * pos2[0];
          pos1 += inputWidth * inputHeight * inputDepth;
          pos2 += outputWidth * outputHeight * outputDepth;
        }
      }
    }
  }
  THTensor_(free)(gradOutput);
}

#endif
