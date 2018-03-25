// Adapted from interp.cpp from Caffe util by Pauline Luc
// Originally developed by George Papandreou

#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/TemporalUpSamplingLinear.c"
#else

static inline void THNN_(TemporalUpSamplingLinear_shapeCheck)
     (THTensor *input, THTensor *gradOutput,
      int nBatch, int nChannels,
      int inputWidth, int outputWidth) {
  THArgCheck(inputWidth > 0 && outputWidth > 0, 2,
	     "input and output sizes should be greater than 0,"
	     " but got input (W: %d) output (W: %d)",
	     inputWidth, outputWidth);
  if (input != NULL) {
    THNN_ARGCHECK(input->nDimension == 3, 2, input,
		  "3D input tensor expected but got: %s");
  }

  if (gradOutput != NULL) {
    THNN_CHECK_DIM_SIZE(gradOutput, 3, 0, nBatch);
    THNN_CHECK_DIM_SIZE(gradOutput, 3, 1, nChannels);
    THNN_CHECK_DIM_SIZE(gradOutput, 3, 2, outputWidth);
  }
}

void THNN_(TemporalUpSamplingLinear_updateOutput)(
    THNNState *state,
    THTensor *input,
    THTensor *output,
    int outputWidth,
    bool align_corners){

  int nbatch = THTensor_(size)(input, 0);
  int channels = THTensor_(size)(input, 1);
  int inputWidth = THTensor_(size)(input, 2);

  THNN_(TemporalUpSamplingLinear_shapeCheck)
    (input, NULL,
     nbatch, channels,
     inputWidth, outputWidth);

  input = THTensor_(newContiguous)(input);
  THTensor_(resize3d)(output,
		      THTensor_(size)(input, 0),
		      THTensor_(size)(input, 1),
		      outputWidth);
  THTensor_(zero)(output);
  real *idata = THTensor_(data)(input);
  real *odata = THTensor_(data)(output);
  channels = nbatch * channels;
  THAssert(inputWidth > 0 && outputWidth > 0);
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
    return;
  }
  const float rwidth = THNN_(linear_upsampling_compute_scale)(inputWidth, outputWidth, align_corners);
  for (int w2 = 0; w2 < outputWidth; ++w2) {
    const float w1r = THNN_(linear_upsampling_compute_source_index)(rwidth, w2, align_corners);
    const int w1 = w1r;
    const int w1p = (w1 < inputWidth - 1) ? 1 : 0;
    const real w1lambda = w1r - w1;
    const real w0lambda = (real)1. - w1lambda;
    const real* pos1 = &idata[w1];
    real* pos2 = &odata[w2];
    for (int c = 0; c < channels; ++c) {
      pos2[0] = w0lambda * pos1[0] + w1lambda * pos1[w1p];
      pos1 += inputWidth;
      pos2 += outputWidth;
    }
  }
  THTensor_(free)(input);
}

void THNN_(TemporalUpSamplingLinear_updateGradInput)(
    THNNState *state,
    THTensor *gradOutput,
    THTensor *gradInput,
    int nbatch,
    int channels,
    int inputWidth,
    int outputWidth,
    bool align_corners){

  THNN_(TemporalUpSamplingLinear_shapeCheck)
    (NULL, gradOutput,
     nbatch, channels,
     inputWidth,
     outputWidth);

  THTensor_(resize3d)(gradInput, nbatch, channels, inputWidth);
  THTensor_(zero)(gradInput);
  gradOutput = THTensor_(newContiguous)(gradOutput);
  real *data1 = THTensor_(data)(gradInput);
  real *data2 = THTensor_(data)(gradOutput);
  channels = nbatch * channels;

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
    return;
  }
  const float rwidth = THNN_(linear_upsampling_compute_scale)(inputWidth, outputWidth, align_corners);
  for (int w2 = 0; w2 < outputWidth; ++w2) {
    const float w1r = THNN_(linear_upsampling_compute_source_index)(rwidth, w2, align_corners);
    const int w1 = w1r;
    const int w1p = (w1 < inputWidth - 1) ? 1 : 0;
    const real w1lambda = w1r - w1;
    const real w0lambda = (real)1. - w1lambda;
    real* pos1 = &data1[w1];
    const real* pos2 = &data2[w2];
    for (int c = 0; c < channels; ++c) {
      pos1[0] += w0lambda * pos2[0];
      pos1[w1p] += w1lambda * pos2[0];
      pos1 += inputWidth;
      pos2 += outputWidth;
    }
  }
  THTensor_(free)(gradOutput);
}

#endif
