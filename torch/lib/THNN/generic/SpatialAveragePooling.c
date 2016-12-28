#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialAveragePooling.c"
#else

static inline void THNN_(SpatialAveragePooling_shapeCheck)(
	THTensor *input, THTensor *gradOutput,
	int kH, int kW, int dH, int dW, int padH, int padW,
	bool ceil_mode) {

  THArgCheck(kW > 0 && kH > 0, 5,
             "kernel size should be greater than zero, but got kH: %d kW: %d", kH, kW);
  THArgCheck(dW > 0 && dH > 0, 8,
             "stride should be greater than zero, but got dH: %d dW: %d", dH, dW);

  int ndim = input->nDimension;
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;

  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }

  THNN_ARGCHECK(ndim == 3 || ndim == 4, 2, input,
		"3D or 4D input tensor expected but got: %s");

  THArgCheck(kW/2 >= padW && kH/2 >= padH, 2,
	     "pad should be smaller than half of kernel size, but got "
	     "padW = %d, padH = %d, kW = %d, kH = %d",
	     padW, padH, kW, kH);

  long nInputPlane = input->size[dimh-1];
  long inputHeight = input->size[dimh];
  long inputWidth = input->size[dimw];
  long outputHeight, outputWidth;
  long nOutputPlane = nInputPlane;

  if(ceil_mode)
  {
    outputHeight = (long)(ceil((float)(inputHeight - kH + 2*padH) / dH)) + 1;
    outputWidth  = (long)(ceil((float)(inputWidth  - kW + 2*padW) / dW)) + 1;
  }
  else
  {
    outputHeight = (long)(floor((float)(inputHeight - kH + 2*padH) / dH)) + 1;
    outputWidth  = (long)(floor((float)(inputWidth  - kW + 2*padW) / dW)) + 1;
  }

  if (padW || padH)
  {
    // ensure that the last pooling starts inside the image
    // needed to avoid problems in ceil mode
    if ((outputHeight - 1)*dH >= inputHeight + padH)
      --outputHeight;
    if ((outputWidth  - 1)*dW >= inputWidth  + padW)
      --outputWidth;
  }

  if (outputWidth < 1 || outputHeight < 1)
    THError("Given input size: (%dx%dx%d). "
	    "Calculated output size: (%dx%dx%d). Output size is too small",
            nInputPlane,inputHeight,inputWidth,nInputPlane,outputHeight,outputWidth);

  if (gradOutput != NULL) {
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimf, nOutputPlane);
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimh, outputHeight);
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimw, outputWidth);
  }
}

void THNN_(SpatialAveragePooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH,
          bool ceil_mode,
          bool count_include_pad)
{
  real *output_data;
  real *input_data;

  int dimw = 2;
  int dimh = 1;
  int dimc = 0;
  long nbatch = 1;

  long inputWidth;
  long inputHeight;
  long outputWidth;
  long outputHeight;
  long nInputPlane; // number of channels (or colors)

  long k;

  THNN_(SpatialAveragePooling_shapeCheck)
    (input, NULL, kH, kW, dH, dW, padH, padW, ceil_mode);

  if (input->nDimension == 4) {
    nbatch = input->size[0];
    dimw++;
    dimh++;
    dimc++;
  }

  inputWidth = input->size[dimw];
  inputHeight = input->size[dimh];
  nInputPlane = input->size[dimc];

  if(ceil_mode)
  {
    outputWidth  = (long)(ceil((float)(inputWidth  - kW + 2*padW) / dW)) + 1;
    outputHeight = (long)(ceil((float)(inputHeight - kH + 2*padH) / dH)) + 1;
  }
  else
  {
    outputWidth  = (long)(floor((float)(inputWidth  - kW + 2*padW) / dW)) + 1;
    outputHeight = (long)(floor((float)(inputHeight - kH + 2*padH) / dH)) + 1;
  }
  if (padW || padH)
  {
    // ensure that the last pooling starts inside the image
    // needed to avoid problems in ceil mode
    if ((outputHeight - 1)*dH >= inputHeight + padH)
      --outputHeight;
    if ((outputWidth  - 1)*dW >= inputWidth  + padW)
      --outputWidth;
  }

  if (input->nDimension == 3)
    THTensor_(resize3d)(output, nInputPlane, outputHeight, outputWidth);
  else
    THTensor_(resize4d)(output, input->size[0], nInputPlane, outputHeight, outputWidth);

  input = THTensor_(newContiguous)(input);
  THArgCheck(THTensor_(isContiguous)(output), 3, "output must be contiguous");
  input_data = THTensor_(data)(input);
  output_data = THTensor_(data)(output);

#pragma omp parallel for private(k)
  for(k = 0; k < nInputPlane; k++)
  {
    long p;
    for(p = 0; p < nbatch; p++)
    {
      long xx, yy;
      /* For all output pixels... */
      real *ptr_output = output_data + p*nInputPlane*outputWidth*outputHeight + k*outputWidth*outputHeight;
      real *ptr_input = input_data + p*nInputPlane*inputWidth*inputHeight + k*inputWidth*inputHeight;
      long i;
      for(i = 0; i < outputWidth*outputHeight; i++)
        ptr_output[i] = 0;

      for(yy = 0; yy < outputHeight; yy++)
      {
        for(xx = 0; xx < outputWidth; xx++)
        {
          /* Compute the mean of the input image... */
          long hstart = yy * dH - padH;
          long wstart = xx * dW - padW;
          long hend = fminf(hstart + kH, inputHeight + padH);
          long wend = fminf(wstart + kW, inputWidth + padW);
          int pool_size = (hend - hstart) * (wend - wstart);
          hstart = fmaxf(hstart, 0);
          wstart = fmaxf(wstart, 0);
          hend = fminf(hend, inputHeight);
          wend = fminf(wend, inputWidth);

          real sum = 0;

          int divide_factor;
          if(count_include_pad)
            divide_factor = pool_size;
          else
            divide_factor = (hend - hstart) * (wend - wstart);

          long kx, ky;

          for(ky = hstart; ky < hend; ky++)
          {
            for(kx = wstart; kx < wend; kx++)
              sum += ptr_input[ky*inputWidth + kx];
          }
          /* Update output */
          *ptr_output++ += sum/divide_factor;
        }
      }
    }
  }
  THTensor_(free)(input);
}

void THNN_(SpatialAveragePooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH,
          bool ceil_mode,
          bool count_include_pad)
{
  int dimw = 2;
  int dimh = 1;
  int dimc = 0;
  long nbatch = 1;
  long ndim = 3;

  long inputWidth;
  long inputHeight;
  long outputWidth;
  long outputHeight;
  long nInputPlane; // number of channels (or colors)

  real *gradOutput_data;
  real *input_data, *gradInput_data;

  long k;

  THNN_(SpatialAveragePooling_shapeCheck)
    (input, gradOutput, kH, kW, dH, dW, padH, padW, ceil_mode);


  if (input->nDimension == 4) {
    nbatch = input->size[0];
    dimw++;
    dimh++;
    dimc++;
    ndim = 4;
  }

  inputWidth = input->size[dimw];
  inputHeight = input->size[dimh];
  nInputPlane = input->size[dimc];

  if(ceil_mode)
  {
    outputWidth  = (long)(ceil((float)(inputWidth  - kW + 2*padW) / dW)) + 1;
    outputHeight = (long)(ceil((float)(inputHeight - kH + 2*padH) / dH)) + 1;
  }
  else
  {
    outputWidth  = (long)(floor((float)(inputWidth  - kW + 2*padW) / dW)) + 1;
    outputHeight = (long)(floor((float)(inputHeight - kH + 2*padH) / dH)) + 1;
  }
  if (padW || padH)
  {
    // ensure that the last pooling starts inside the image
    // needed to avoid problems in ceil mode
    if ((outputHeight - 1)*dH >= inputHeight + padH)
      --outputHeight;
    if ((outputWidth  - 1)*dW >= inputWidth  + padW)
      --outputWidth;
  }

  THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimh, outputHeight);
  THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimw, outputWidth);

  THTensor_(resizeAs)(gradInput, input);

  gradOutput = THTensor_(newContiguous)(gradOutput);
  THArgCheck(THTensor_(isContiguous)(gradInput), 4, "gradInput must be contiguous");

  gradInput_data = THTensor_(data)(gradInput);
  gradOutput_data = THTensor_(data)(gradOutput);

#pragma omp parallel for private(k)
  for(k = 0; k < nInputPlane; k++)
  {
    long p;
    for(p = 0; p < nbatch; p++)
    {
      real *ptr_gradOutput = gradOutput_data + p*nInputPlane*outputHeight*outputWidth + k*outputWidth*outputHeight;
      long xx, yy;

      real* ptr_gi = gradInput_data + p*nInputPlane*inputWidth*inputHeight + k*inputWidth*inputHeight;
      real *ptr_gradInput = gradInput_data + p*nInputPlane*inputWidth*inputHeight + k*inputWidth*inputHeight;

      long i;
      for(i=0; i<inputWidth*inputHeight; i++)
        ptr_gi[i] = 0.0;

      for(yy = 0; yy < outputHeight; yy++)
      {
        for(xx = 0; xx < outputWidth; xx++)
        {
          long hstart = yy * dH - padH;
          long wstart = xx * dW - padW;
          long hend = fminf(hstart + kH, inputHeight + padH);
          long wend = fminf(wstart + kW, inputWidth + padW);
          int pool_size = (hend - hstart) * (wend - wstart);
          hstart = fmaxf(hstart, 0);
          wstart = fmaxf(wstart, 0);
          hend = fminf(hend, inputHeight);
          wend = fminf(wend, inputWidth);

          real z = *ptr_gradOutput++;

          int divide_factor;
          if(count_include_pad)
            divide_factor = pool_size;
          else
            divide_factor = (hend - hstart) * (wend - wstart);

          long kx, ky;
          for(ky = hstart ; ky < hend; ky++)
          {
            for(kx = wstart; kx < wend; kx++)
              ptr_gradInput[ky*inputWidth + kx] += z/divide_factor;
          }
        }
      }
    }
  }

  THTensor_(free)(gradOutput);
}

#endif
