#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/TemporalUpSamplingNearest.c"
#else


static inline void THNN_(TemporalUpSamplingNearest_shapeCheck)
     (THTensor *input, THTensor *gradOutput,
      int scale_factor) {
  THArgCheck(input != NULL, 2, "3D input tensor expected but got NULL");
  THArgCheck(scale_factor > 1, 4,
	     "scale_factor must be greater than 1, but got: %d", scale_factor);
  THNN_ARGCHECK(input->nDimension == 2 || input->nDimension == 3, 2, input,
		"2D or 3D input tensor expected but got: %s");
  if (input->nDimension == 2) {
    int nChannels    = THTensor_(size)(input, 0);
    int inputWidth   = THTensor_(size)(input, 1);
    int outputWidth  = inputWidth  * scale_factor;
    if (gradOutput != NULL) {
      THNN_CHECK_DIM_SIZE(gradOutput, 3, 0, nChannels);
      THNN_CHECK_DIM_SIZE(gradOutput, 3, 1, outputWidth);
    }
  } else {
    int nBatch       = THTensor_(size)(input, 0);
    int nChannels    = THTensor_(size)(input, 1);
    int inputWidth   = THTensor_(size)(input, 2);
    int outputWidth  = inputWidth  * scale_factor;
    if (gradOutput != NULL) {
      THNN_CHECK_DIM_SIZE(gradOutput, 3, 0, nBatch);
      THNN_CHECK_DIM_SIZE(gradOutput, 3, 1, nChannels);
      THNN_CHECK_DIM_SIZE(gradOutput, 3, 2, outputWidth);
    }
  }
}

void THNN_(TemporalUpSamplingNearest_updateOutput)(
    THNNState *state,
    THTensor *input,
    THTensor *output,
    int scale_factor)
{
  THNN_(TemporalUpSamplingNearest_shapeCheck)(input, NULL, scale_factor);
  int inputWidth  = THTensor_(size)(input,  input->nDimension-1);
  int outputWidth = inputWidth * scale_factor;

  if (input->nDimension == 2) {
    THTensor_(resize2d)(output,
			THTensor_(size)(input, 0),
      outputWidth);
  } else {
    THTensor_(resize3d)(output,
			THTensor_(size)(input, 0),
      THTensor_(size)(input, 1),
      outputWidth);
  }

  int dW = scale_factor;
  int xDim = input->nDimension-1;

  // dims
  int idim = input->nDimension;
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

void THNN_(TemporalUpSamplingNearest_updateGradInput)(
    THNNState *state,
    THTensor *input,
    THTensor *gradOutput,
    THTensor *gradInput,
    int scale_factor)
{
  THNN_(TemporalUpSamplingNearest_shapeCheck)(input, gradOutput, scale_factor);
  THTensor_(resizeAs)(gradInput, input);

  int dW = scale_factor;
  int xDim = gradInput->nDimension-1;

  // dims
  int idim = gradInput->nDimension;  // Guaranteed to be between 2 and 4
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

#endif
