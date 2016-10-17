#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialDilatedMaxPooling.c"
#else

static inline void THNN_(SpatialDilatedMaxPooling_shapeCheck)(
	THTensor *input, THTensor *gradOutput, THTensor *indices,
	int kH, int kW, int dH, int dW, int padH, int padW,
	int dilationH, int dilationW, bool ceil_mode) {

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

  THArgCheck(input->size[dimw] >= kW - padW && input->size[dimh] >= kH - padH, 2,
	     "input image (H: %d, W: %d) smaller than kernel "
	     "size - padding( kH: %d padH: %d kW: %d padW: %d",
	     input->size[dimh], input->size[dimw], kH, padH, kW, padW);
  THArgCheck(kW/2 >= padW && kH/2 >= padH, 2,
	     "pad should be smaller than half of kernel size, but got "
	     "padW = %d, padH = %d, kW = %d, kH = %d",
	     padW, padH, kW, kH);
  
  long nInputPlane = input->size[dimh-1];
  long inputHeight = input->size[dimh];
  long inputWidth = input->size[dimw];
  long outputHeight, outputWidth;
  long nOutputPlane = nInputPlane;

  if (ceil_mode)
  {
    outputHeight = (long)(ceil((float)(inputHeight - (dilationH * (kH - 1) + 1) + 2*padH) / dH)) + 1;
    outputWidth  = (long)(ceil((float)(inputWidth  - (dilationW * (kW - 1) + 1) + 2*padW) / dW)) + 1;
  }
  else
  {
    outputHeight = (long)(floor((float)(inputHeight - (dilationH * (kH - 1) + 1) + 2*padH) / dH)) + 1;
    outputWidth  = (long)(floor((float)(inputWidth  - (dilationW * (kW - 1) + 1) + 2*padW) / dW)) + 1;
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
  if (indices != NULL) {
    THNN_CHECK_DIM_SIZE(indices, ndim, dimf, nOutputPlane);
    THNN_CHECK_DIM_SIZE(indices, ndim, dimh, outputHeight);
    THNN_CHECK_DIM_SIZE(indices, ndim, dimw, outputWidth);
  }
}

static void THNN_(SpatialDilatedMaxPooling_updateOutput_frame)(
          real *input_p,
          real *output_p,
          real *ind_p,
          long nslices,
          long iwidth,
          long iheight,
          long owidth,
          long oheight,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH,
          int dilationW,
          int dilationH
          )
{
  long k;
#pragma omp parallel for private(k)
  for (k = 0; k < nslices; k++)
  {
    /* loop over output */
    long i, j;
    real *ip = input_p   + k*iwidth*iheight;
    for(i = 0; i < oheight; i++)
    {
      for(j = 0; j < owidth; j++)
      {
        long hstart = i * dH - padH;
        long wstart = j * dW - padW;
        long hend = fminf(hstart + (kH - 1) * dilationH + 1, iheight);
        long wend = fminf(wstart + (kW - 1) * dilationW + 1, iwidth);
        while(hstart < 0)
          hstart += dilationH;
        while(wstart < 0)
          wstart += dilationW;

        /* local pointers */
        real *op = output_p  + k*owidth*oheight + i*owidth + j;
        real *indp = ind_p   + k*owidth*oheight + i*owidth + j;

        /* compute local max: */
        long maxindex = -1;
        real maxval = -THInf;
        long tcntr = 0;
        long x,y;
        for(y = hstart; y < hend; y += dilationH)
        {
          for(x = wstart; x < wend; x += dilationW)
          {
            tcntr = y*iwidth + x;
            real val = *(ip + tcntr);
            if (val > maxval)
            {
              maxval = val;
              maxindex = tcntr;
            }
          }
        }

        /* set output to local max */
        *op = maxval;

        /* store location of max */
        *indp = maxindex + TH_INDEX_BASE;
      }
    }
  }
}

void THNN_(SpatialDilatedMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *indices,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH,
          int dilationW,
          int dilationH,
          bool ceil_mode)
{

  int dimw = 2;
  int dimh = 1;
  long nbatch = 1;
  long nInputPlane;
  long inputHeight;
  long inputWidth;
  long outputHeight;
  long outputWidth;
  real *input_data;
  real *output_data;
  real *indices_data;

  THNN_(SpatialDilatedMaxPooling_shapeCheck)
    (input, NULL, NULL, kH, kW, dH, dW,
     padH, padW, dilationH, dilationW, ceil_mode);

  if (input->nDimension == 4)
  {
    nbatch = input->size[0];
    dimw++;
    dimh++;
  }

  /* sizes */
  nInputPlane = input->size[dimh-1];
  inputHeight = input->size[dimh];
  inputWidth = input->size[dimw];
  if (ceil_mode)
  {
    outputHeight = (long)(ceil((float)(inputHeight - (dilationH * (kH - 1) + 1) + 2*padH) / dH)) + 1;
    outputWidth  = (long)(ceil((float)(inputWidth  - (dilationW * (kW - 1) + 1) + 2*padW) / dW)) + 1;
  }
  else
  {
    outputHeight = (long)(floor((float)(inputHeight - (dilationH * (kH - 1) + 1) + 2*padH) / dH)) + 1;
    outputWidth  = (long)(floor((float)(inputWidth  - (dilationW * (kW - 1) + 1) + 2*padW) / dW)) + 1;
  }

  if (padW || padH)
  {
    // ensure that the last pooling starts inside the image
    if ((outputHeight - 1)*dH >= inputHeight + padH)
      --outputHeight;
    if ((outputWidth  - 1)*dW >= inputWidth  + padW)
      --outputWidth;
  }

  /* get contiguous input */
  input = THTensor_(newContiguous)(input);

  /* resize output */
  if (input->nDimension == 3)
  {
    THTensor_(resize3d)(output, nInputPlane, outputHeight, outputWidth);
    /* indices will contain the locations for each output point */
    THTensor_(resize3d)(indices,  nInputPlane, outputHeight, outputWidth);

    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);
    indices_data = THTensor_(data)(indices);

    THNN_(SpatialDilatedMaxPooling_updateOutput_frame)
      (input_data, output_data,
       indices_data,
       nInputPlane,
       inputWidth, inputHeight,
       outputWidth, outputHeight,
       kW, kH, dW, dH,
       padW, padH,
       dilationW, dilationH
       );
  }
  else
  {
    long p;

    THTensor_(resize4d)(output, nbatch, nInputPlane, outputHeight, outputWidth);
    /* indices will contain the locations for each output point */
    THTensor_(resize4d)(indices, nbatch, nInputPlane, outputHeight, outputWidth);

    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);
    indices_data = THTensor_(data)(indices);

#pragma omp parallel for private(p)
    for (p = 0; p < nbatch; p++)
    {
      THNN_(SpatialDilatedMaxPooling_updateOutput_frame)
	(input_data+p*nInputPlane*inputWidth*inputHeight,
	 output_data+p*nInputPlane*outputWidth*outputHeight,
	 indices_data+p*nInputPlane*outputWidth*outputHeight,
	 nInputPlane,
	 inputWidth, inputHeight,
	 outputWidth, outputHeight,
	 kW, kH, dW, dH,
	 padW, padH,
	 dilationW, dilationH
	 );
    }
  }

  /* cleanup */
  THTensor_(free)(input);
}

static void THNN_(SpatialDilatedMaxPooling_updateGradInput_frame)(
          real *gradInput_p,
          real *gradOutput_p,
          real *ind_p,
          long nInputPlane,
          long inputWidth,
          long inputHeight,
          long outputWidth,
          long outputHeight,
          int dW,
          int dH)
{
  long k;
#pragma omp parallel for private(k)
  for (k = 0; k < nInputPlane; k++)
  {
    real *gradInput_p_k = gradInput_p + k*inputWidth*inputHeight;
    real *gradOutput_p_k = gradOutput_p + k*outputWidth*outputHeight;
    real *ind_p_k = ind_p + k*outputWidth*outputHeight;

    /* calculate max points */
    long i, j;
    for(i = 0; i < outputHeight; i++)
    {
      for(j = 0; j < outputWidth; j++)
      {
        /* retrieve position of max */
        long maxp = ind_p_k[i*outputWidth + j] - TH_INDEX_BASE;
        /* update gradient */
        gradInput_p_k[maxp] += gradOutput_p_k[i*outputWidth + j];
      }
    }
  }
}

void THNN_(SpatialDilatedMaxPooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *indices,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH,
          int dilationW,
          int dilationH,
          bool ceil_mode)
{
  int dimw = 2;
  int dimh = 1;
  long nbatch = 1;
  int nInputPlane;
  int inputHeight;
  int inputWidth;
  int outputHeight;
  int outputWidth;
  real *gradInput_data;
  real *gradOutput_data;
  real *indices_data;

  THNN_(SpatialDilatedMaxPooling_shapeCheck)
    (input, gradOutput, indices, kH, kW, dH, dW,
     padH, padW, dilationH, dilationW, ceil_mode);

  /* get contiguous gradOutput */
  gradOutput = THTensor_(newContiguous)(gradOutput);

  /* resize */
  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  if (input->nDimension == 4) {
    nbatch = input->size[0];
    dimw++;
    dimh++;
  }

  /* sizes */
  nInputPlane = input->size[dimh-1];
  inputHeight = input->size[dimh];
  inputWidth = input->size[dimw];
  outputHeight = gradOutput->size[dimh];
  outputWidth = gradOutput->size[dimw];

  /* get raw pointers */
  gradInput_data = THTensor_(data)(gradInput);
  gradOutput_data = THTensor_(data)(gradOutput);
  indices_data = THTensor_(data)(indices);

  /* backprop */
  if (input->nDimension == 3)
  {
    THNN_(SpatialDilatedMaxPooling_updateGradInput_frame)
      (gradInput_data, gradOutput_data,
       indices_data,
       nInputPlane,
       inputWidth, inputHeight,
       outputWidth, outputHeight,
       dW, dH);
  }
  else
  {
    long p;
#pragma omp parallel for private(p)
    for (p = 0; p < nbatch; p++)
    {
      THNN_(SpatialDilatedMaxPooling_updateGradInput_frame)
	(gradInput_data+p*nInputPlane*inputWidth*inputHeight,
	 gradOutput_data+p*nInputPlane*outputWidth*outputHeight,
	 indices_data+p*nInputPlane*outputWidth*outputHeight,
	 nInputPlane,
	 inputWidth, inputHeight,
	 outputWidth, outputHeight,
	 dW, dH);
    }
  }

  /* cleanup */
  THTensor_(free)(gradOutput);
}

#endif
