#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialConvolutionLocal.c"
#else

static inline void THNN_(SpatialConvolutionLocal_shapeCheck)(
	THTensor *input, THTensor *gradOutput,
	THTensor *weight, THTensor *bias, 
	int kH, int kW, int dH, 
	int dW, int padH, int padW,
	long inputHeight, long inputWidth,
	long outputHeight, long outputWidth) {

  THArgCheck(kW > 0 && kH > 0, 9,
	       "kernel size should be greater than zero, but got kH: %d kW: %d", kH, kW);
  THArgCheck(dW > 0 && dH > 0, 11,
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

  long nInputPlane = weight->size[2] / (kH * kW);
  long nOutputPlane = weight->size[1];

  if (bias != NULL) {
    THNN_CHECK_DIM_SIZE(bias, 3, 0, nOutputPlane);
    THNN_CHECK_DIM_SIZE(bias, 3, 1, outputHeight);
    THNN_CHECK_DIM_SIZE(bias, 3, 2, outputWidth);
  }

  THNN_CHECK_DIM_SIZE(input, ndim, dimf, nInputPlane);
  
  if (gradOutput != NULL) {
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimf, nOutputPlane);
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimh, outputHeight);
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimw, outputWidth);
  }
}

static int THNN_(view_weight_local)(THTensor **_weight)
{
  THTensor *weight = *_weight;
  THArgCheck(weight->nDimension == 3 || weight->nDimension == 6, 4,
          "weight tensor should be 3D or 6D - got %dD", weight->nDimension);
  if (weight->nDimension == 6) {
    long s1 = weight->size[0] * weight->size[1];
    long s2 = weight->size[2];
    long s3 = weight->size[3] * weight->size[4] * weight->size[5];
    *_weight = THTensor_(newWithStorage3d)(weight->storage, 
					   weight->storageOffset, 
					   s1, -1, s2, -1, s3, -1);
    return 1;
  }
  return 0;
}

static void THNN_(SpatialConvolutionLocal_updateOutput_frame)
     (
      THTensor *input, THTensor *output,
      THTensor *weight, THTensor *bias, THTensor *finput,
      int kW, int kH, int dW, int dH, int padW, int padH,
      long nInputPlane, long inputWidth, long inputHeight,
      long nOutputPlane, long outputWidth, long outputHeight)
{
  long i;
  THTensor *output3d, *finput3d;

  THNN_(unfolded_copy)(finput, input, kW, kH, dW, dH, padW, padH, 
		       nInputPlane, inputWidth, inputHeight, 
		       outputWidth, outputHeight);

  THTensor_(copy)(output, bias);

  output3d = THTensor_(newWithStorage3d)
    (output->storage, output->storageOffset,
     outputHeight * outputWidth, 1,
     nOutputPlane, outputHeight * outputWidth,
     1, nOutputPlane * outputHeight * outputWidth);
  
  finput3d = THTensor_(newWithStorage3d)
    (finput->storage, finput->storageOffset,
     outputHeight * outputWidth, 1,
     kW * kH * nInputPlane, outputHeight * outputWidth,
     1, kW * kH * nInputPlane * outputHeight * outputWidth);

  // weight:    oH*oW x nOutputPlane x nInputPlane*kH*kW
  // finput3d:  oH*oW x nInputPlane*kH*kW x 1  
  THTensor_(baddbmm)(output3d, 1.0, output3d, 1.0, weight, finput3d);
  // output3d:  oH*oW x nOutputPlane x 1
  
  THTensor_(free)(output3d);
  THTensor_(free)(finput3d);
}

void THNN_(SpatialConvolutionLocal_updateOutput)(
    THNNState *state,
    THTensor *input,
    THTensor *output,
    THTensor *weight,
    THTensor *bias,
    THTensor *finput,
    THTensor *fgradInput,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    long inputWidth, long inputHeight,
    long outputWidth, long outputHeight)
{
  int freeWeight = THNN_(view_weight_local)(&weight);

  THNN_(SpatialConvolutionLocal_shapeCheck)
    (input, NULL, weight, bias, kH, kW, dH, dW, padH, padW,
     inputHeight, inputWidth, outputHeight, outputWidth);					    

  input = THTensor_(newContiguous)(input);
  
  long nInputPlane = THTensor_(size)(weight, 2)/ (kW * kH);
  long nOutputPlane = THTensor_(size)(weight, 1);

  if(input->nDimension == 3)
  {
    THTensor_(resize2d)(finput, kW*kH*nInputPlane, outputHeight*outputWidth);
    THTensor_(resize3d)(output, nOutputPlane, outputHeight, outputWidth);

    THNN_(SpatialConvolutionLocal_updateOutput_frame)
      (input, output, weight, bias, finput,
       kW, kH, dW, dH, padW, padH,
       nInputPlane, inputWidth, inputHeight,
       nOutputPlane, outputWidth, outputHeight);
  }
  else
  {
    long T = input->size[0];
    long t;

    THTensor_(resize3d)(finput, T, kW*kH*nInputPlane, outputHeight*outputWidth);
    THTensor_(resize4d)(output, T, nOutputPlane, outputHeight, outputWidth);

#pragma omp parallel for private(t)
    for(t = 0; t < T; t++)
    {
      THTensor *input_t = THTensor_(newSelect)(input, 0, t);
      THTensor *output_t = THTensor_(newSelect)(output, 0, t);
      THTensor *finput_t = THTensor_(newSelect)(finput, 0, t);

      THNN_(SpatialConvolutionLocal_updateOutput_frame)
	(input_t, output_t, weight, bias, finput_t,
	 kW, kH, dW, dH, padW, padH,
	 nInputPlane, inputWidth, inputHeight,
	 nOutputPlane, outputWidth, outputHeight);

      THTensor_(free)(input_t);
      THTensor_(free)(output_t);
      THTensor_(free)(finput_t);
    }
  }

  THTensor_(free)(input);
  if (freeWeight)
    THTensor_(free)(weight);
}


static void THNN_(SpatialConvolutionLocal_updateGradInput_frame)
     (THTensor *gradInput, THTensor *gradOutput,
      THTensor *weight, THTensor *fgradInput,
      int kW, int kH, int dW, int dH, int padW, int padH, 
      long nInputPlane, long inputWidth, long inputHeight,
      long nOutputPlane, long outputWidth, long outputHeight)
{
  THTensor *gradOutput3d, *fgradInput3d;
  gradOutput3d = THTensor_(newWithStorage3d)(gradOutput->storage, gradOutput->storageOffset,
                                             outputHeight*outputWidth, 1,
                                             nOutputPlane, outputHeight*outputWidth,
                                             1, nOutputPlane*outputHeight*outputWidth);
  fgradInput3d = THTensor_(newWithStorage3d)(fgradInput->storage, fgradInput->storageOffset,
                                             outputHeight*outputWidth, 1,
                                             kW*kH*nInputPlane, outputHeight*outputWidth,
                                             1, kW*kH*nInputPlane*outputHeight*outputWidth);
  // weight:        oH*oW x nInputPlane*kH*kW x nOutputPlane
  // gradOutput3d:  oH*oW x nOutputPlane x 1         
  THTensor_(baddbmm)(fgradInput3d, 0.0, fgradInput3d, 1.0, weight, gradOutput3d);
  // fgradInput3d:  oH*oW x nInputPlane*kH*kW x 1  
  
  THTensor_(free)(gradOutput3d);
  THTensor_(free)(fgradInput3d);
  
  THTensor_(zero)(gradInput);
  
  THNN_(unfolded_acc)(fgradInput, gradInput, kW, kH, dW, dH, padW, padH, 
		      nInputPlane, inputWidth, inputHeight, 
		      outputWidth, outputHeight);

}

void THNN_(SpatialConvolutionLocal_updateGradInput)(
    THNNState *state,
    THTensor *input,
    THTensor *gradOutput,
    THTensor *gradInput,
    THTensor *weight,
    THTensor *finput,
    THTensor *fgradInput,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    long inputWidth, long inputHeight,
    long outputWidth, long outputHeight)
{
  int freeWeight = THNN_(view_weight_local)(&weight);

  THNN_(SpatialConvolutionLocal_shapeCheck)
    (input, gradOutput, weight, NULL, kH, kW, dH, dW, padH, padW,
     inputHeight, inputWidth, outputHeight, outputWidth);

  input = THTensor_(newContiguous)(input);
  gradOutput = THTensor_(newContiguous)(gradOutput);
  long nInputPlane = THTensor_(size)(weight,2)/(kW*kH);
  long nOutputPlane = THTensor_(size)(weight,1);

  THTensor_(resizeAs)(gradInput, input);
  THTensor_(resizeAs)(fgradInput, finput);
  THTensor_(transpose)(weight, weight, 1, 2);

  if(input->nDimension == 3)
  {
    THNN_(SpatialConvolutionLocal_updateGradInput_frame)
      (gradInput, gradOutput, weight, 
       fgradInput, kW, kH, dW, dH, padW, padH, 
       nInputPlane, inputWidth, inputHeight,
       nOutputPlane, outputWidth, outputHeight);
  }
  else
  {
    long T = input->size[0];
    long t;

#pragma omp parallel for private(t)
    for(t = 0; t < T; t++)
    {
      THTensor *gradInput_t = THTensor_(newSelect)(gradInput, 0, t);
      THTensor *gradOutput_t = THTensor_(newSelect)(gradOutput, 0, t);
      THTensor *fgradInput_t = THTensor_(newSelect)(fgradInput, 0, t);

      THNN_(SpatialConvolutionLocal_updateGradInput_frame)
	(gradInput_t, gradOutput_t, weight, fgradInput_t, 
	 kW, kH, dW, dH, padW, padH, 
	 nInputPlane, inputWidth, inputHeight,
	 nOutputPlane, outputWidth, outputHeight);

      THTensor_(free)(gradInput_t);
      THTensor_(free)(gradOutput_t);
      THTensor_(free)(fgradInput_t);
    }
  }

  THTensor_(transpose)(weight, weight, 1, 2);

  THTensor_(free)(input);
  THTensor_(free)(gradOutput);
  if (freeWeight)
    THTensor_(free)(weight);

}

static void THNN_(SpatialConvolutionLocal_accGradParameters_frame)
     (THTensor *gradOutput, THTensor *gradWeight, THTensor *gradBias,
      THTensor *finput, real scale, 
      int kW, int kH, int dW, int dH, int padW, int padH, 
      long nInputPlane, long inputWidth, long inputHeight,
      long nOutputPlane, long outputWidth, long outputHeight)
{
   
  THTensor *gradOutput3d, *finput3d;
  gradOutput3d = THTensor_(newWithStorage3d)(gradOutput->storage, gradOutput->storageOffset,
                                             outputHeight*outputWidth, 1,
                                             nOutputPlane, outputHeight*outputWidth,
                                             1, nOutputPlane*outputHeight*outputWidth);
  finput3d = THTensor_(newWithStorage3d)(finput->storage, finput->storageOffset,
                                         outputHeight*outputWidth, 1,
                                         1, kW*kH*nInputPlane*outputHeight*outputWidth,
                                         kW*kH*nInputPlane, outputHeight*outputWidth);
  // gradOutput3d:  oH*oW x nOutputPlane x 1  
  // finput3d:      oH*oW x 1 x kW*kH*nInputPlane
  THTensor_(baddbmm)(gradWeight, 1.0, gradWeight, scale, gradOutput3d, finput3d);
  // gradWeight:    oH*oW x nOutputPlane x kW*kH*nInputPlane

  THTensor_(cadd)(gradBias, gradBias, scale, gradOutput);

  THTensor_(free)(gradOutput3d);
  THTensor_(free)(finput3d);
}

void THNN_(SpatialConvolutionLocal_accGradParameters)(
    THNNState *state,
    THTensor *input,
    THTensor *gradOutput,
    THTensor *gradWeight,
    THTensor *gradBias,
    THTensor *finput,
    THTensor *fgradInput,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    long inputWidth, long inputHeight,
    long outputWidth, long outputHeight,
    real scale)
{

  int freeWeight = THNN_(view_weight_local)(&gradWeight);

  THNN_(SpatialConvolutionLocal_shapeCheck)
    (input, gradOutput, gradWeight, gradBias, kH, kW, dH, dW, padH, padW,
     inputHeight, inputWidth, outputHeight, outputWidth);

  input = THTensor_(newContiguous)(input);
  gradOutput = THTensor_(newContiguous)(gradOutput);

  long nInputPlane = THTensor_(size)(gradWeight,2)/(kW*kH);
  long nOutputPlane = THTensor_(size)(gradWeight,1);

  if(input->nDimension == 3)
  {
    THNN_(SpatialConvolutionLocal_accGradParameters_frame)
      (gradOutput, gradWeight, gradBias, finput, scale, 
       kW, kH, dW, dH, padW, padH,
       nInputPlane, inputWidth, inputHeight,
       nOutputPlane, outputWidth, outputHeight);
  }
  else
  {
    long T = input->size[0];
    long t;

    for(t = 0; t < T; t++)
    {
      THTensor *gradOutput_t = THTensor_(newSelect)(gradOutput, 0, t);
      THTensor *finput_t = THTensor_(newSelect)(finput, 0, t);

      THNN_(SpatialConvolutionLocal_accGradParameters_frame)
	(gradOutput_t, gradWeight, gradBias, finput_t, scale, 
	 kW, kH, dW, dH, padW, padH,
	 nInputPlane, inputWidth, inputHeight,
	 nOutputPlane, outputWidth, outputHeight);

      THTensor_(free)(gradOutput_t);
      THTensor_(free)(finput_t);
    }
  }

  THTensor_(free)(input);
  THTensor_(free)(gradOutput);

  if (freeWeight)
    THTensor_(free)(gradWeight);

}

#endif
