#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialConvolutionLocal.c"
#else

#ifdef _WIN32
# include <windows.h>
#endif

#include "unfold.h"


static void nn_(SpatialConvolutionLocal_updateOutput_frame)(THTensor *input, THTensor *output, THTensor *weight, THTensor *bias, THTensor *finput,
                                                         int kW, int kH, int dW, int dH, int padW, int padH,
                                                         long nInputPlane, long inputWidth, long inputHeight,
                                                         long nOutputPlane, long outputWidth, long outputHeight)
{
  long i;
  THTensor *output3d, *finput3d;

  nn_(unfolded_copy)(finput, input, kW, kH, dW, dH, padW, padH, nInputPlane, inputWidth, inputHeight, outputWidth, outputHeight);

  THTensor_(copy)(output, bias);

  output3d = THTensor_(newWithStorage3d)(output->storage, output->storageOffset,
                                         outputHeight*outputWidth, 1,
                                         nOutputPlane, outputHeight*outputWidth,
                                         1, nOutputPlane*outputHeight*outputWidth);
 
  finput3d = THTensor_(newWithStorage3d)(finput->storage, finput->storageOffset,
                                         outputHeight*outputWidth, 1,
                                         kW*kH*nInputPlane, outputHeight*outputWidth,
                                         1, kW*kH*nInputPlane*outputHeight*outputWidth);
  // weight:    oH*oW x nOutputPlane x nInputPlane*kH*kW
  // finput3d:  oH*oW x nInputPlane*kH*kW x 1  
  THTensor_(baddbmm)(output3d, 1.0, output3d, 1.0, weight, finput3d);
  // output3d:  oH*oW x nOutputPlane x 1
  
  THTensor_(free)(output3d);
  THTensor_(free)(finput3d);
}

static int nn_(SpatialConvolutionLocal_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  long inputWidth = luaT_getfieldcheckint(L, 1, "iW");
  long inputHeight = luaT_getfieldcheckint(L, 1, "iH");
  long outputWidth = luaT_getfieldcheckint(L, 1, "oW");
  long outputHeight = luaT_getfieldcheckint(L, 1, "oH");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int padW = luaT_getfieldcheckint(L, 1, "padW");
  int padH = luaT_getfieldcheckint(L, 1, "padH");
  long nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  long nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane"); 

  THTensor *finput = luaT_getfieldcheckudata(L, 1, "finput", torch_Tensor);
  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor); 
  THTensor *bias = luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

  if(input->nDimension == 3)
  {
    THTensor_(resize2d)(finput, kW*kH*nInputPlane, outputHeight*outputWidth);
    THTensor_(resize3d)(output, nOutputPlane, outputHeight, outputWidth);

    nn_(SpatialConvolutionLocal_updateOutput_frame)(input, output, weight, bias, finput,
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

      nn_(SpatialConvolutionLocal_updateOutput_frame)(input_t, output_t, weight, bias, finput_t,
                                                   kW, kH, dW, dH, padW, padH,
                                                   nInputPlane, inputWidth, inputHeight,
                                                   nOutputPlane, outputWidth, outputHeight);

      THTensor_(free)(input_t);
      THTensor_(free)(output_t);
      THTensor_(free)(finput_t);
    }
  }

  return 1;
}


static void nn_(SpatialConvolutionLocal_updateGradInput_frame)(THTensor *gradInput, THTensor *gradOutput, THTensor *weight, THTensor *fgradInput,
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

  nn_(unfolded_acc)(fgradInput, gradInput, kW, kH, dW, dH, padW, padH, 
                                            nInputPlane, inputWidth, inputHeight, outputWidth, outputHeight);
}

static int nn_(SpatialConvolutionLocal_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  long inputWidth = luaT_getfieldcheckint(L, 1, "iW");
  long inputHeight = luaT_getfieldcheckint(L, 1, "iH");
  long outputWidth = luaT_getfieldcheckint(L, 1, "oW");
  long outputHeight = luaT_getfieldcheckint(L, 1, "oH");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int padW = luaT_getfieldcheckint(L, 1, "padW");
  int padH = luaT_getfieldcheckint(L, 1, "padH");
  long nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  long nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");

  THTensor *finput = luaT_getfieldcheckudata(L, 1, "finput", torch_Tensor);
  THTensor *fgradInput = luaT_getfieldcheckudata(L, 1, "fgradInput", torch_Tensor);
  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);

  THTensor_(resizeAs)(gradInput, input);
  THTensor_(resizeAs)(fgradInput, finput);
  THTensor_(transpose)(weight, weight, 1, 2);

  if(input->nDimension == 3)
  {
    nn_(SpatialConvolutionLocal_updateGradInput_frame)(gradInput, gradOutput, weight, fgradInput, kW, kH, dW, dH, padW, padH, 
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

      nn_(SpatialConvolutionLocal_updateGradInput_frame)(gradInput_t, gradOutput_t, weight, fgradInput_t, kW, kH, dW, dH, padW, padH, 
                                                         nInputPlane, inputWidth, inputHeight,
                                                         nOutputPlane, outputWidth, outputHeight);

      THTensor_(free)(gradInput_t);
      THTensor_(free)(gradOutput_t);
      THTensor_(free)(fgradInput_t);
    }
  }

  THTensor_(transpose)(weight, weight, 1, 2);

  return 1;
}

static void nn_(SpatialConvolutionLocal_accGradParameters_frame)(THTensor *gradOutput, THTensor *gradWeight, THTensor *gradBias, THTensor *finput, real scale, 
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

static int nn_(SpatialConvolutionLocal_accGradParameters)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  real scale = luaL_optnumber(L, 4, 1);
  long inputWidth = luaT_getfieldcheckint(L, 1, "iW");
  long inputHeight = luaT_getfieldcheckint(L, 1, "iH");
  long outputWidth = luaT_getfieldcheckint(L, 1, "oW");
  long outputHeight = luaT_getfieldcheckint(L, 1, "oH");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int padW = luaT_getfieldcheckint(L, 1, "padW");
  int padH = luaT_getfieldcheckint(L, 1, "padH");
  long nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  long nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");

  THTensor *finput = luaT_getfieldcheckudata(L, 1, "finput", torch_Tensor);
  THTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_Tensor);
  THTensor *gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_Tensor);

  if(input->nDimension == 3)
  {
    nn_(SpatialConvolutionLocal_accGradParameters_frame)(gradOutput, gradWeight, gradBias, finput, scale, kW, kH, dW, dH, padW, padH,
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

      nn_(SpatialConvolutionLocal_accGradParameters_frame)(gradOutput_t, gradWeight, gradBias, finput_t, scale, kW, kH, dW, dH, padW, padH,
                                                           nInputPlane, inputWidth, inputHeight,
                                                           nOutputPlane, outputWidth, outputHeight);

      THTensor_(free)(gradOutput_t);
      THTensor_(free)(finput_t);
    }
  }

  return 0;
}

static const struct luaL_Reg nn_(SpatialConvolutionLocal__) [] = {
  {"SpatialConvolutionLocal_updateOutput", nn_(SpatialConvolutionLocal_updateOutput)},
  {"SpatialConvolutionLocal_updateGradInput", nn_(SpatialConvolutionLocal_updateGradInput)},
  {"SpatialConvolutionLocal_accGradParameters", nn_(SpatialConvolutionLocal_accGradParameters)},
  {NULL, NULL}
};

static void nn_(SpatialConvolutionLocal_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(SpatialConvolutionLocal__), "nn");
  lua_pop(L,1);
}

#endif
