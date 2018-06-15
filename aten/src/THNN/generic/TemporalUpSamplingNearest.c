#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/TemporalUpSamplingNearest.c"
#else

#include "linear_upsampling.h"
#include <stdio.h>

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
/*
     (THTensor *input, THTensor *gradOutput,
      int outputWidth) {
  if (input != NULL) {
    THArgCheck(input != NULL, 2, "3D input tensor expected but got NULL");
    THNN_ARGCHECK(input->_dim() == 2 || input->_dim() == 3, 2, input,
		  "2D or 3D input tensor expected but got: %s");
  if (input->_dim() == 2) {
    int nChannels    = THTensor_(size)(input, 0);
    int inputWidth   = THTensor_(size)(input, 1);
    if (gradOutput != NULL) {
      THNN_CHECK_DIM_SIZE(gradOutput, 3, 0, nChannels);
      THNN_CHECK_DIM_SIZE(gradOutput, 3, 1, outputWidth);
    }
  } else {
    int nBatch       = THTensor_(size)(input, 0);
    int nChannels    = THTensor_(size)(input, 1);
    int inputWidth   = THTensor_(size)(input, 2);
    if (gradOutput != NULL) {
      THNN_CHECK_DIM_SIZE(gradOutput, 3, 0, nBatch);
      THNN_CHECK_DIM_SIZE(gradOutput, 3, 1, nChannels);
      THNN_CHECK_DIM_SIZE(gradOutput, 3, 2, outputWidth);
    }
  }
}
  */

void THNN_(TemporalUpSamplingNearest_updateOutput)(
    THNNState *state,
    THTensor *input,
    THTensor *output,
    int outputWidth,
    bool align_corners)
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
    return;
  }

  for (int w2 = 0; w2 < outputWidth; ++w2) {
    const accreal src_x = nearest_neighbor_compute_source_index(scale, w2, inputWidth, align_corners);
    const int in_x = src_x;
    printf("scale: %d, %d, src_index : %f, %d\n", inputWidth, outputWidth, src_x, in_x);
    const real* pos1 = &idata[in_x];
    real* pos2 = &odata[w2];
    for (int c = 0; c < channels; ++c) {
      pos2[0] = pos1[0];
      pos1 += inputWidth;
      pos2 += outputWidth;
    }
  }
  THTensor_(free)(input);
}
  /*
  int xDim = input->_dim()-1;

  // dims
  int idim = input->dim();
  int osz0 = output->size[0];
  int osz1 = output->size[1];
  int osz2 = 1;
  if (idim > 2) {
    osz2 = output->size[2];
  }

  // get strides
  int64_t *is = input->stride;
  int64_t *os = output->stride;

  // get raw pointers
  real *pin = THTensor_(data)(input);
  real *pout = THTensor_(data)(output);

  // perform the upsampling
  int i0, i1, i2, isrc, idst;
  int iout[3];  // Output indices
  int iin[3];  // Input indices

  for (i0 = 0; i0 < osz0; i0++) {
    iout[0] = i0;
    iin[0] = i0;
    for (i1 = 0; i1 < osz1; i1++) {
      iout[1] = i1;
      iin[1] = i1;
      for (i2 = 0; i2 < osz2; i2++) {
        iout[2] = i2;
        iin[2] = i2;

        // set the indices for the upsampled dimensions
        iin[xDim] = iout[xDim] / dW;

        idst = i0*os[0] + i1*os[1];
        isrc = iin[0]*is[0] + iin[1]*is[1];
        if (idim > 2) {
          idst += i2*os[2];
          isrc += iin[2]*is[2];
        }

        pout[idst] = pin[isrc];
      }
    }
  }
}
  */

void THNN_(TemporalUpSamplingNearest_updateGradInput)(
    THNNState *state,
    THTensor *gradOutput,
    THTensor *gradInput,
    int nbatch,
    int channels,
    int inputWidth,
    int outputWidth,
    bool align_corners)
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
    return;
  }

  for (int w2 = 0; w2 < outputWidth; ++w2) {
    const int in_x = nearest_neighbor_compute_source_index(scale, w2, inputWidth, align_corners);
    real* pos1 = &data1[in_x];
    const real* pos2 = &data2[w2];
    for (int c = 0; c < channels; ++c) {
      pos1[0] += pos2[0];
      pos1 += inputWidth;
      pos2 += outputWidth;
    }
  }
  THTensor_(free)(gradOutput);
}
 
  /*
  int xDim = gradInput->_dim()-1;

  // dims
  int idim = gradInput->dim();  // Guaranteed to be between 2 and 4
  int isz0 = gradInput->size[0];
  int isz1 = gradInput->size[1];
  int isz2 = 1;
  if (idim > 2) {
    isz2 = gradInput->size[2];
  }

  // get strides
  int64_t *is = gradInput->stride;
  int64_t *os = gradOutput->stride;

  // get raw pointers
  real *pin = THTensor_(data)(gradInput);
  real *pout = THTensor_(data)(gradOutput);

  // perform the upsampling
  int i0, i1, i2, isrc, idst, x;
  int iin[3];  // Input indices
  int iout[3];  // Output indices

  THTensor_(zero)(gradInput);

  for (i0 = 0; i0 < isz0; i0++) {
    iin[0] = i0;
    iout[0] = i0;
    for (i1 = 0; i1 < isz1; i1++) {
      iin[1] = i1;
      iout[1] = i1;
      for (i2 = 0; i2 < isz2; i2++) {
        iin[2] = i2;
        iout[2] = i2;

        idst = i0*is[0] + i1*is[1];
        if (idim > 2) {
          idst += i2*is[2];
        }

        // Now accumulate the gradients from gradOutput
        for (x = 0; x < dW; x++) {
          iout[xDim] = dW * iin[xDim] + x;
          isrc = iout[0]*os[0] + iout[1]*os[1];
          if (idim > 2) {
            isrc += iout[2]*os[2];
          }
          pin[idst] += pout[isrc];
        }
      }
    }
  }
}
  */

#endif
