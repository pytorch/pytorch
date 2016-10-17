#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialUpSamplingNearest.c"
#else


static inline void THNN_(SpatialUpSamplingNearest_shapeCheck)
     (THTensor *input, THTensor *gradOutput,
      int scale_factor) {
  THArgCheck(input != NULL, 2, "4D input tensor expected but got NULL");
  THArgCheck(scale_factor > 1, 4,
	     "scale_factor must be greater than 1, but got: %d", scale_factor);
  THNN_ARGCHECK(input->nDimension == 3 || input->nDimension == 4, 2, input,
		"3D or 4D input tensor expected but got: %s");
  if (input->nDimension == 3) {
    int nChannels    = THTensor_(size)(input, 0);
    int inputHeight  = THTensor_(size)(input, 1);
    int inputWidth   = THTensor_(size)(input, 2);
    int outputHeight = inputHeight * scale_factor;
    int outputWidth  = inputWidth  * scale_factor;
    if (gradOutput != NULL) {
      THNN_CHECK_DIM_SIZE(gradOutput, 3, 0, nChannels);
      THNN_CHECK_DIM_SIZE(gradOutput, 3, 1, outputHeight);
      THNN_CHECK_DIM_SIZE(gradOutput, 3, 2, outputWidth);
    }
  } else {
    int nBatch       = THTensor_(size)(input, 0);
    int nChannels    = THTensor_(size)(input, 1);
    int inputHeight  = THTensor_(size)(input, 2);
    int inputWidth   = THTensor_(size)(input, 3);  
    int outputHeight = inputHeight * scale_factor;
    int outputWidth  = inputWidth  * scale_factor;
    if (gradOutput != NULL) {
      THNN_CHECK_DIM_SIZE(gradOutput, 4, 0, nBatch);
      THNN_CHECK_DIM_SIZE(gradOutput, 4, 1, nChannels);
      THNN_CHECK_DIM_SIZE(gradOutput, 4, 2, outputHeight);
      THNN_CHECK_DIM_SIZE(gradOutput, 4, 3, outputWidth);
    }
  }
}

void THNN_(SpatialUpSamplingNearest_updateOutput)(
    THNNState *state,
    THTensor *input,
    THTensor *output,
    int scale_factor)
{
  THNN_(SpatialUpSamplingNearest_shapeCheck)(input, NULL, scale_factor);
  int inputHeight = THTensor_(size)(input, input->nDimension-2);
  int inputWidth  = THTensor_(size)(input,  input->nDimension-1);
  int outputHeight = inputHeight * scale_factor;
  int outputWidth = inputWidth * scale_factor;

  if (input->nDimension == 3) {
    THTensor_(resize3d)(output,
			THTensor_(size)(input, 0),
			outputHeight, outputWidth);    
  } else {
    THTensor_(resize4d)(output,
			THTensor_(size)(input, 0),
			THTensor_(size)(input, 1),
			outputHeight, outputWidth);
  }

  int dW = scale_factor;
  int dH = scale_factor;
  int xDim = input->nDimension-2;
  int yDim = input->nDimension-1;

  // dims
  int idim = input->nDimension;
  int osz0 = output->size[0];
  int osz1 = output->size[1];
  int osz2 = output->size[2];
  int osz3 = 1;  
  if (idim > 3) {
    osz3 = output->size[3];
  }

  // get strides
  long *is = input->stride;
  long *os = output->stride;

  // get raw pointers
  real *pin = THTensor_(data)(input);
  real *pout = THTensor_(data)(output);

  // perform the upsampling
  int i0, i1, i2, i3, isrc, idst;
  int iout[4];  // Output indices
  int iin[4];  // Input indices

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

          // set the indices for the upsampled dimensions
          iin[xDim] = iout[xDim] / dW;
          iin[yDim] = iout[yDim] / dH;

          idst = i0*os[0] + i1*os[1] + i2*os[2];
          isrc = iin[0]*is[0] + iin[1]*is[1] + iin[2]*is[2];
          if (idim > 3) {
            idst += i3*os[3];
            isrc += iin[3]*is[3];
          }

          pout[idst] = pin[isrc];
        }
      }
    }
  }
}

void THNN_(SpatialUpSamplingNearest_updateGradInput)(
    THNNState *state,
    THTensor *input,
    THTensor *gradOutput,
    THTensor *gradInput,
    int scale_factor)
{
  THNN_(SpatialUpSamplingNearest_shapeCheck)(input, gradOutput, scale_factor);
  THTensor_(resizeAs)(gradInput, input);

  int dW = scale_factor;
  int dH = scale_factor;
  int xDim = gradInput->nDimension-2;
  int yDim = gradInput->nDimension-1;

  // dims
  int idim = gradInput->nDimension;  // Guaranteed to be between 3 and 5
  int isz0 = gradInput->size[0];
  int isz1 = gradInput->size[1];
  int isz2 = gradInput->size[2];
  int isz3 = 1;
  if (idim > 3) {
    isz3 = gradInput->size[3];
  }

  // get strides
  long *is = gradInput->stride;
  long *os = gradOutput->stride;

  // get raw pointers
  real *pin = THTensor_(data)(gradInput);
  real *pout = THTensor_(data)(gradOutput);

  // perform the upsampling
  int i0, i1, i2, i3, isrc, idst, x, y;
  int iin[4];  // Input indices
  int iout[4];  // Output indices

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

          idst = i0*is[0] + i1*is[1] + i2*is[2];
          if (idim > 3) {
            idst += i3*is[3];
          }

          // Now accumulate the gradients from gradOutput
          for (y = 0; y < dH; y++) {
            for (x = 0; x < dW; x++) {
              iout[xDim] = dW * iin[xDim] + x;
              iout[yDim] = dH * iin[yDim] + y;
              isrc = iout[0]*os[0] + iout[1]*os[1] + iout[2]*os[2];
              if (idim > 3) {
                isrc += iout[3]*os[3];
              }
              pin[idst] += pout[isrc];
            }
          }
        }
      }
    }
  }
}

#endif
