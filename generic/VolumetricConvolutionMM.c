#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/VolumetricConvolutionMM.c"
#else

/* note: due to write issues, this one cannot be parallelized as well as unfolded_copy */
static void nn_(unfolded_acc_vol)(THTensor *finput, THTensor *input,
                               int kT, int kW, int kH,
                               int dT, int dW, int dH,
                               int padT, int padW, int padH,
                               int nInputPlane,
                               int inputDepth, int inputWidth, int inputHeight,
                               int outputDepth, int outputWidth, int outputHeight)
{
  int nip;
  real *input_data = THTensor_(data)(input);
  real *finput_data = THTensor_(data)(finput);

//#pragma omp parallel for private(nip)
  for(nip = 0; nip < nInputPlane; nip++)
  {
    int kt, kw, kh, t, y, x, it, ix, iy;
    for(kt = 0; kt < kT; kt++)
    {
      for(kh = 0; kh < kH; kh++)
      {
        for(kw = 0; kw < kW; kw++)
        {
          real *src = finput_data + nip*(kT*kH*kW*outputDepth*outputHeight*outputWidth) + kt*(kH*kW*outputDepth*outputHeight*outputWidth)+ kh*(kW*outputDepth*outputHeight*outputWidth) + kw*(outputDepth*outputHeight*outputWidth);
          real *dst = input_data + nip*(inputDepth*inputHeight*inputWidth);
          if (padT > 0 || padH > 0 || padW > 0)
          {
            for(t = 0; t < outputDepth; t++)
            {
              it = t*dT - padT + kt;
              for(y = 0; y < outputHeight; y++) 
              {
                iy = y*dH - padH + kh;
                for(x = 0; x < outputWidth; x++)
                {
                  ix = x*dW - padW + kw;
                  if (it < 0 || it >= inputDepth || iy < 0 || iy >= inputHeight || ix < 0 || ix >= inputWidth)
                    {}
                  else
                    THVector_(add)(dst+it*inputHeight*inputWidth+iy*inputWidth+ix, src+t*outputHeight*outputWidth+y*outputWidth+x, 1, 1);
                }
              }
            }
          }
          else {
            for(t = 0; t < outputDepth; t++) {
              it = t*dT + kt;
              for(y = 0; y < outputHeight; y++) {
                iy = y*dH + kh;
                for(x = 0; x < outputWidth; x++) {
                  ix = x*dW + kw;
                  THVector_(add)(dst+it*inputHeight*inputWidth+iy*inputWidth+ix, src+t*outputHeight*outputWidth+y*outputWidth+x, 1, 1);
                }
              }
            }
          }
        }
      }
    }
  }
}
static void nn_(unfolded_copy_vol)(THTensor *finput, THTensor *input,
                               int kT, int kW, int kH,
                               int dT, int dW, int dH,
                               int padT, int padW, int padH,
                               int nInputPlane,
                               int inputDepth, int inputWidth, int inputHeight,
                               int outputDepth, int outputWidth, int outputHeight)
{
  long k;
  real *input_data = THTensor_(data)(input);
  real *finput_data = THTensor_(data)(finput);
// #pragma omp parallel for private(k)
  for(k = 0; k < nInputPlane*kT*kH*kW; k++) {
    int nip = k / (kT*kH*kW);
    int rest = k % (kT*kH*kW);
    int kt = rest / (kH*kW);
    rest = rest % (kH*kW);
    int kh = rest / kW;
    int kw = rest % kW;
    int t,x,y,it,ix,iy;
    real *dst = finput_data + nip*(kT*kH*kW*outputDepth*outputHeight*outputWidth) + kt*(kH*kW*outputDepth*outputHeight*outputWidth) + kh*(kW*outputDepth*outputHeight*outputWidth) + kw*(outputDepth*outputHeight*outputWidth);
    real *src = input_data + nip*(inputDepth*inputHeight*inputWidth);
    
    if (padT > 0 || padH > 0 || padW > 0)
    {
      for(t = 0; t < outputDepth; t++)
      {
        it = t*dT - padT + kt;
        for(y = 0; y < outputHeight; y++) 
        {
          iy = y*dH - padH + kh;
          for(x = 0; x < outputWidth; x++)
          {
            ix = x*dW - padW + kw;
            if (it < 0 || it >= inputDepth || iy < 0 || iy >= inputHeight || ix < 0 || ix >= inputWidth)
              memset(dst+t*outputHeight*outputWidth+y*outputWidth+x, 0, sizeof(real)*(1));
            else
              memcpy(dst+t*outputHeight*outputWidth+y*outputWidth+x, src+it*inputHeight*inputWidth+iy*inputWidth+ix, sizeof(real)*(1));
          }
        }
      }
    }
     else {
      for(t = 0; t < outputDepth; t++) {
        it = t*dT + kt;
        for(y = 0; y < outputHeight; y++) {
          iy = y*dH + kh;
          for(x = 0; x < outputWidth; x++) {
            ix = x*dW + kw;
            memcpy(dst+t*outputHeight*outputWidth+y*outputWidth+x, src+it*inputHeight*inputWidth+iy*inputWidth+ix, sizeof(real)*(1));
          }
        }
      }
    }
  }
}

static void nn_(VolumetricConvolutionMM_updateOutput_frame)(THTensor *input, THTensor *output, THTensor *weight, THTensor *bias, THTensor *finput,
                                                    int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH,
                                                 		long nInputPlane, long inputDepth, long inputWidth, long inputHeight,
                                                 		long nOutputPlane, long outputDepth, long outputWidth, long outputHeight)
{
  long i;
  THTensor *output2d;

  nn_(unfolded_copy_vol)(finput, input, kT, kW, kH, dT, dW, dH, padT, padW, padH, nInputPlane, inputDepth, inputWidth, inputHeight, outputDepth, outputWidth, outputHeight);

  output2d = THTensor_(newWithStorage2d)(output->storage, output->storageOffset,
                                         nOutputPlane, -1,
                                         outputDepth*outputHeight*outputWidth, -1);

  for(i = 0; i < nOutputPlane; i++)
    THVector_(fill)(output->storage->data+output->storageOffset+output->stride[0]*i, THTensor_(get1d)(bias, i), outputDepth*outputHeight*outputWidth);

  THTensor_(addmm)(output2d, 1, output2d, 1, weight, finput);

  THTensor_(free)(output2d);
}

static int nn_(VolumetricConvolutionMM_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  int kT = luaT_getfieldcheckint(L, 1, "kT");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dT = luaT_getfieldcheckint(L, 1, "dT");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int padT = luaT_getfieldcheckint(L, 1, "padT");
  int padW = luaT_getfieldcheckint(L, 1, "padW");
  int padH = luaT_getfieldcheckint(L, 1, "padH");

  THTensor *finput = luaT_getfieldcheckudata(L, 1, "finput", torch_Tensor);
  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *bias = luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

  int dimf = 0;
  int dimt = 1;
  int dimh = 2;
  int dimw = 3;

  long nInputPlane;
  long inputDepth;
  long inputHeight;
  long inputWidth;
  long nOutputPlane;
  long outputDepth;
  long outputHeight;
  long outputWidth;
  
  luaL_argcheck(L, input->nDimension == 4 || input->nDimension == 5, 2, "4D or 5D(batch mode) tensor expected");


  if (input->nDimension == 5) {
    dimf++;
    dimt++;
    dimh++;
    dimw++;
  }

  nInputPlane = input->size[dimf];
  inputDepth = input->size[dimt];
  inputHeight  = input->size[dimh];
  inputWidth   = input->size[dimw];
  nOutputPlane = weight->size[0];
  outputDepth  = (inputDepth + 2*padT - kT) / dT + 1;
  outputHeight = (inputHeight + 2*padH - kH) / dH + 1;
  outputWidth  = (inputWidth + 2*padW - kW) / dW + 1;
  

  if (outputWidth < 1 || outputHeight < 1)
    THError("Given input size: (%dx%dx%dx%d). Calculated output size: (%dx%dx%dx%d). Output size is too small",
        nInputPlane,inputDepth,inputHeight,inputWidth,nInputPlane,outputDepth,outputHeight,outputWidth);

  if(input->nDimension == 4)
  {
    THTensor_(resize2d)(finput, kT*kW*kH*nInputPlane, outputDepth*outputHeight*outputWidth);
    THTensor_(resize4d)(output, nOutputPlane, outputDepth, outputHeight, outputWidth);

    nn_(VolumetricConvolutionMM_updateOutput_frame)(input, output, weight, bias, finput,
                                                 kT, kW, kH, dT, dW, dH, padT, padW, padH,
                                                 nInputPlane, inputDepth, inputWidth, inputHeight,
                                                 nOutputPlane, outputDepth, outputWidth, outputHeight);
  }
  else
  {
    long T = input->size[0];
    long t;

    THTensor_(resize3d)(finput, T, kT*kW*kH*nInputPlane, outputDepth*outputHeight*outputWidth);
    THTensor_(resize5d)(output, T, nOutputPlane, outputDepth, outputHeight, outputWidth);

// #pragma omp parallel for private(t)
    for(t = 0; t < T; t++)
    {
      THTensor *input_t = THTensor_(newSelect)(input, 0, t);
      THTensor *output_t = THTensor_(newSelect)(output, 0, t);
      THTensor *finput_t = THTensor_(newSelect)(finput, 0, t);

      nn_(VolumetricConvolutionMM_updateOutput_frame)(input_t, output_t, weight, bias, finput_t,
                                                 kT, kW, kH, dT, dW, dH, padT, padW, padH,
                                                 nInputPlane, inputDepth, inputWidth, inputHeight,
                                                 nOutputPlane, outputDepth, outputWidth, outputHeight);

      THTensor_(free)(input_t);
      THTensor_(free)(output_t);
      THTensor_(free)(finput_t);
    }
  }

  return 1;
}


static void nn_(VolumetricConvolutionMM_updateGradInput_frame)(THTensor *gradInput, THTensor *gradOutput, THTensor *weight, THTensor *fgradInput,
                                                            int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH)
{
  THTensor *gradOutput2d = THTensor_(newWithStorage2d)(gradOutput->storage, gradOutput->storageOffset,
                                                       gradOutput->size[0], -1,
                                                       gradOutput->size[1]*gradOutput->size[2]*gradOutput->size[3], -1);
  THTensor_(addmm)(fgradInput, 0, fgradInput, 1, weight, gradOutput2d);
  THTensor_(free)(gradOutput2d);

  THTensor_(zero)(gradInput);

  nn_(unfolded_acc_vol)(fgradInput, gradInput, kT, kW, kH, dT, dW, dH, padT, padW, padH, gradInput->size[0], gradInput->size[1], gradInput->size[3], gradInput->size[2], gradOutput->size[1], gradOutput->size[3], gradOutput->size[2]);
}

static int nn_(VolumetricConvolutionMM_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  int kT = luaT_getfieldcheckint(L, 1, "kT");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dT = luaT_getfieldcheckint(L, 1, "dT");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int padT = luaT_getfieldcheckint(L, 1, "padT");
  int padW = luaT_getfieldcheckint(L, 1, "padW");
  int padH = luaT_getfieldcheckint(L, 1, "padH");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");

  THTensor *finput = luaT_getfieldcheckudata(L, 1, "finput", torch_Tensor);
  THTensor *fgradInput = luaT_getfieldcheckudata(L, 1, "fgradInput", torch_Tensor);
  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);

  THArgCheck( nOutputPlane == gradOutput->size[input->nDimension == 5 ? 1 : 0], 1, "Number of output features is not equal to nOutputPlane" );

  THTensor_(resizeAs)(gradInput, input);
  THTensor_(resizeAs)(fgradInput, finput);
  THTensor_(transpose)(weight, weight, 0, 1);

  if(input->nDimension == 4)
  {
    nn_(VolumetricConvolutionMM_updateGradInput_frame)(gradInput, gradOutput, weight, fgradInput, kT, kW, kH, dT, dW, dH, padT, padW, padH);
  }
  else
  {
    long T = input->size[0];
    long t;

//#pragma omp parallel for private(t)
    for(t = 0; t < T; t++)
    {
      THTensor *gradInput_t = THTensor_(newSelect)(gradInput, 0, t);
      THTensor *gradOutput_t = THTensor_(newSelect)(gradOutput, 0, t);
      THTensor *fgradInput_t = THTensor_(newSelect)(fgradInput, 0, t);

      nn_(VolumetricConvolutionMM_updateGradInput_frame)(gradInput_t, gradOutput_t, weight, fgradInput_t, kT, kW, kH, dT, dW, dH, padT, padW, padH);

      THTensor_(free)(gradInput_t);
      THTensor_(free)(gradOutput_t);
      THTensor_(free)(fgradInput_t);
    }
  }

  THTensor_(transpose)(weight, weight, 0, 1);

  return 1;
}

static void nn_(VolumetricConvolutionMM_accGradParameters_frame)(THTensor *gradOutput, THTensor *gradWeight, THTensor *gradBias, THTensor *finput,
                                                              real scale)
{
  long i;
  THTensor *gradOutput2d = THTensor_(newWithStorage2d)(gradOutput->storage, gradOutput->storageOffset,
                                                       gradOutput->size[0], -1,
                                                       gradOutput->size[1]*gradOutput->size[2]*gradOutput->size[3], -1);

  THTensor_(transpose)(finput, finput, 0, 1);
  THTensor_(addmm)(gradWeight, 1, gradWeight, scale, gradOutput2d, finput);
  THTensor_(transpose)(finput, finput, 0, 1);

  for(i = 0; i < gradBias->size[0]; i++)
  {
    long k;
    real sum = 0;
    real *data = gradOutput2d->storage->data + gradOutput2d->storageOffset + i*gradOutput2d->stride[0];
    for(k = 0; k < gradOutput2d->size[1]; k++)
      sum += data[k];
    (gradBias->storage->data + gradBias->storageOffset)[i] += scale*sum;
  }

  THTensor_(free)(gradOutput2d);
}

static int nn_(VolumetricConvolutionMM_accGradParameters)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  real scale = luaL_optnumber(L, 4, 1);
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");

  THTensor *finput = luaT_getfieldcheckudata(L, 1, "finput", torch_Tensor);
  THTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_Tensor);
  THTensor *gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_Tensor);

  THArgCheck( nOutputPlane == gradOutput->size[input->nDimension == 5 ? 1 : 0], 1, "Number of output features is not equal to nOutputPlane" );

  if(input->nDimension == 4)
  {
    nn_(VolumetricConvolutionMM_accGradParameters_frame)(gradOutput, gradWeight, gradBias, finput, scale);
  }
  else
  {
    long T = input->size[0];
    long t;

    for(t = 0; t < T; t++)
    {
      THTensor *gradOutput_t = THTensor_(newSelect)(gradOutput, 0, t);
      THTensor *finput_t = THTensor_(newSelect)(finput, 0, t);

      nn_(VolumetricConvolutionMM_accGradParameters_frame)(gradOutput_t, gradWeight, gradBias, finput_t, scale);

      THTensor_(free)(gradOutput_t);
      THTensor_(free)(finput_t);
    }
  }

  return 0;
}

static const struct luaL_Reg nn_(VolumetricConvolutionMM__) [] = {
  {"VolumetricConvolutionMM_updateOutput", nn_(VolumetricConvolutionMM_updateOutput)},
  {"VolumetricConvolutionMM_updateGradInput", nn_(VolumetricConvolutionMM_updateGradInput)},
  {"VolumetricConvolutionMM_accGradParameters", nn_(VolumetricConvolutionMM_accGradParameters)},
  {NULL, NULL}
};

static void nn_(VolumetricConvolutionMM_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(VolumetricConvolutionMM__), "nn");
  lua_pop(L,1);
}

#endif
