#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/VolumetricUpSamplingNearest.c"
#else


static inline void THNN_(VolumetricUpSamplingNearest_shapeCheck)
     (THTensor *input, THTensor *gradOutput,
      int scale_factor) {
  THArgCheck(input != NULL, 2, "5D input tensor expected but got NULL");
  THArgCheck(scale_factor > 1, 4,
	     "scale_factor must be greater than 1, but got: %d", scale_factor);
  THNN_ARGCHECK(input->nDimension == 4 || input->nDimension == 5, 2, input,
		"4D or 5D input tensor expected but got: %s");
  if (input->nDimension == 4) {
    int nChannels    = THTensor_(size)(input, 0);
    int inputDepth   = THTensor_(size)(input, 1);
    int inputHeight  = THTensor_(size)(input, 2);
    int inputWidth   = THTensor_(size)(input, 3);
    int outputDepth  = inputDepth  * scale_factor;
    int outputHeight = inputHeight * scale_factor;
    int outputWidth  = inputWidth  * scale_factor;
    if (gradOutput != NULL) {
      THNN_CHECK_DIM_SIZE(gradOutput, 4, 0, nChannels);
      THNN_CHECK_DIM_SIZE(gradOutput, 4, 1, outputDepth);
      THNN_CHECK_DIM_SIZE(gradOutput, 4, 2, outputHeight);
      THNN_CHECK_DIM_SIZE(gradOutput, 4, 3, outputWidth);
    }
  } else {
    int nBatch       = THTensor_(size)(input, 0);
    int nChannels    = THTensor_(size)(input, 1);
    int inputDepth   = THTensor_(size)(input, 2);
    int inputHeight  = THTensor_(size)(input, 3);
    int inputWidth   = THTensor_(size)(input, 4);  
    int outputDepth  = inputDepth  * scale_factor;
    int outputHeight = inputHeight * scale_factor;
    int outputWidth  = inputWidth  * scale_factor;
    if (gradOutput != NULL) {
      THNN_CHECK_DIM_SIZE(gradOutput, 5, 0, nBatch);
      THNN_CHECK_DIM_SIZE(gradOutput, 5, 1, nChannels);
      THNN_CHECK_DIM_SIZE(gradOutput, 5, 2, outputDepth);
      THNN_CHECK_DIM_SIZE(gradOutput, 5, 3, outputHeight);
      THNN_CHECK_DIM_SIZE(gradOutput, 5, 4, outputWidth);
    }
  }
}

void THNN_(VolumetricUpSamplingNearest_updateOutput)(
    THNNState *state,
    THTensor *input,
    THTensor *output,
    int scale_factor)
{
  THNN_(VolumetricUpSamplingNearest_shapeCheck)(input, NULL, scale_factor);
  int inputDepth   = THTensor_(size)(input, input->nDimension-3);
  int inputHeight  = THTensor_(size)(input, input->nDimension-2);
  int inputWidth   = THTensor_(size)(input,  input->nDimension-1);
  int outputDepth  = inputDepth * scale_factor;
  int outputHeight = inputHeight * scale_factor;
  int outputWidth  = inputWidth * scale_factor;

  if (input->nDimension == 4) {
    THTensor_(resize4d)(output,
			THTensor_(size)(input, 0),
			outputDepth, outputHeight, outputWidth);    
  } else {
    THTensor_(resize5d)(output,
			THTensor_(size)(input, 0),
			THTensor_(size)(input, 1),
			outputDepth, outputHeight, outputWidth);
  }

  int dT = scale_factor;
  int dW = scale_factor;
  int dH = scale_factor;
  int xDim = input->nDimension-3;
  int yDim = input->nDimension-2;
  int zDim = input->nDimension-1;

  // dims
  int idim = input->nDimension;
  int osz0 = output->size[0];
  int osz1 = output->size[1];
  int osz2 = output->size[2];
  int osz3 = output->size[3];
  int osz4 = 1;
  if (idim > 4) {
    osz4 = output->size[4];
  }

  // get strides
  int64_t *is = input->stride;
  int64_t *os = output->stride;

  // get raw pointers
  real *pin = THTensor_(data)(input);
  real *pout = THTensor_(data)(output);

  // perform the upsampling
  int i0, i1, i2, i3, i4, isrc, idst;
  int iout[5];  // Output indices
  int iin[5];  // Input indices

  for (i0 = 0; i0 < osz0; i0++) {
    iout[0] = i0;
    iin[0] = i0;
    for (i1 = 0; i1 < osz1; i1++) {
      iout[1] = i1;
      iin[1] = i1;
      for (i2 = 0; i2 < osz2; i2++) {
        iout[2] = i2;
        iin[2] = i2;
        for (i3 = 0; i3 < osz3; i3++) {
          iout[3] = i3;
          iin[3] = i3;
          for (i4 = 0; i4 < osz4; i4++) {
            iout[4] = i4;
            iin[4] = i4;

            // set the indices for the upsampled dimensions
            iin[xDim] = iout[xDim] / dW;
            iin[yDim] = iout[yDim] / dH;
            iin[zDim] = iout[zDim] / dT;

            idst = i0*os[0] + i1*os[1] + i2*os[2] + i3*os[3];
            isrc = iin[0]*is[0] + iin[1]*is[1] + iin[2]*is[2] + iin[3]*is[3];
            if (idim > 4) {
              idst += i4*os[4];
              isrc += iin[4]*is[4];
            }

            pout[idst] = pin[isrc];
          }
        }
      }
    }
  }
}

void THNN_(VolumetricUpSamplingNearest_updateGradInput)(
    THNNState *state,
    THTensor *input,
    THTensor *gradOutput,
    THTensor *gradInput,
    int scale_factor)
{
  THNN_(VolumetricUpSamplingNearest_shapeCheck)(input, gradOutput, scale_factor);
  THTensor_(resizeAs)(gradInput, input);

  int dW = scale_factor;
  int dH = scale_factor;
  int dT = scale_factor;
  int xDim = gradInput->nDimension-3;
  int yDim = gradInput->nDimension-2;
  int zDim = gradInput->nDimension-1;

  // dims
  int idim = gradInput->nDimension;  // Guaranteed to be between 3 and 5
  int isz0 = gradInput->size[0];
  int isz1 = gradInput->size[1];
  int isz2 = gradInput->size[2];
  int isz3 = gradInput->size[3];
  int isz4 = 1;
  if (idim > 4) {
    isz4 = gradInput->size[4];
  }

  // get strides
  int64_t *is = gradInput->stride;
  int64_t *os = gradOutput->stride;

  // get raw pointers
  real *pin = THTensor_(data)(gradInput);
  real *pout = THTensor_(data)(gradOutput);

  // perform the upsampling
  int i0, i1, i2, i3, i4, isrc, idst, x, y, z;
  int iin[5];  // Input indices
  int iout[5];  // Output indices

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
        for (i3 = 0; i3 < isz3; i3++) {
          iin[3] = i3;
          iout[3] = i3;

          for (i4 = 0; i4 < isz4; i4++) {
            iin[4] = i4;
            iout[4] = i4;

            idst = i0*is[0] + i1*is[1] + i2*is[2] + i3*is[3];
            if (idim > 4) {
              idst += i4*is[4];
            }

            // Now accumulate the gradients from gradOutput
            for (z = 0; z < dT; z++) {
              for (y = 0; y < dH; y++) {
                for (x = 0; x < dW; x++) {
                  iout[xDim] = dW * iin[xDim] + x;
                  iout[yDim] = dH * iin[yDim] + y;
                  iout[zDim] = dT * iin[zDim] + z;
                  isrc = iout[0]*os[0] + iout[1]*os[1] + iout[2]*os[2] + iout[3]*os[3];
                  if (idim > 4) {
                    isrc += iout[4]*os[4];
                  }
                  pin[idst] += pout[isrc];
                }
              }
            }
          }
        }
      }
    }
  }
}

#endif
