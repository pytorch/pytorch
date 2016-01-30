#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/VolumetricConvolution.c"
#else

static int nn_(VolumetricConvolution_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);  
  int dT = luaT_getfieldcheckint(L, 1, "dT");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *bias = luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

  luaL_argcheck(L, input->nDimension == 4 || input->nDimension == 5, 
		2, "4D or 5D (batch-mode) tensor expected");
  int dimt = 1;
  int dimh = 2;
  int dimw = 3;

  if (input->nDimension == 5) {
    dimt++;
    dimh++;
    dimw++;
  }

  long nOutputPlane = weight->size[0];
  long kT           = weight->size[2];
  long kH           = weight->size[3];
  long kW           = weight->size[4];
  long inputDepth   = input->size[dimt];
  long inputHeight  = input->size[dimh];
  long inputWidth   = input->size[dimw];
  long outputDepth  = (inputDepth - kT) / dT + 1;
  long outputWidth  = (inputWidth - kW) / dW + 1;
  long outputHeight = (inputHeight - kH) / dH + 1;
  THTensor *outn = THTensor_(new)();
  long i,j;
  if (input->nDimension == 4) { /* non-batch mode */
    THTensor_(resize4d)(output, nOutputPlane, outputDepth, outputHeight, outputWidth);
  
    /* add bias */
    for (i=0; i<bias->size[0]; i++) {
      THTensor_(select)(outn,output,0,i);
      THTensor_(fill)(outn, THTensor_(get1d)(bias, i));
    }

    /* do convolutions */
    THTensor_(conv3Dmv)(output, 1.0, 1.0, input, weight, dT, dH, dW, "V", "X");
  } else { /* batch mode */
    long nBatch = input->size[0];
    THTensor_(resize5d)(output, nBatch, nOutputPlane, 
			outputDepth, outputHeight, outputWidth);
    THTensor *inb = THTensor_(new)();
    THTensor *outb = THTensor_(new)();

    for (j=0; j<nBatch; j++) { /* loop over batches */
      THTensor_(select)(inb,input,0,j);
      THTensor_(select)(outb,output,0,j);
	
      /* add bias */
      for (i=0; i<bias->size[0]; i++) {
	THTensor_(select)(outn,outb,0,i);
	THTensor_(fill)(outn, THTensor_(get1d)(bias, i));
      }

      /* do convolutions */
      THTensor_(conv3Dmv)(outb, 1.0, 1.0, inb, weight, dT, dH, dW, "V", "X");
    }

    THTensor_(free)(inb);
    THTensor_(free)(outb);
  }
  THTensor_(free)(outn);
  
  return 1;
}


static int nn_(VolumetricConvolution_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);  
  int dT = luaT_getfieldcheckint(L, 1, "dT");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
  THTensor *tweight;

  luaL_argcheck(L, gradOutput->nDimension == 4 || gradOutput->nDimension == 5, 
		3, "4D or 5D (batch-mode) tensor expected");
  int dimPlane = 0;
  if (gradOutput->nDimension == 5) {
    dimPlane++;
  }
  THArgCheck( nOutputPlane == gradOutput->size[dimPlane], 1, 
	      "Number of output features is not equal to nOutputPlane" );

  /* gradient to input */
  tweight = THTensor_(newTranspose)(weight,0,1);
  if (gradOutput->nDimension == 4) { /* non-batch mode */
    THTensor_(conv3Dmv)(gradInput, 0.0, 1.0, gradOutput, tweight, dT, dH, dW, "F", "C");
  } else { /* batch mode */
    long nBatch = gradOutput->size[0];
    THTensor *ginpb = THTensor_(new)();
    THTensor *goutb = THTensor_(new)();
    long j;
    THTensor_(resize5d)(gradInput, input->size[0], input->size[1], input->size[2],
			input->size[3], input->size[4]);

    for (j=0; j<nBatch; j++) { /* loop over batches */
      THTensor_(select)(ginpb,gradInput,0,j);
      THTensor_(select)(goutb,gradOutput,0,j);
      THTensor_(conv3Dmv)(ginpb, 0.0, 1.0, goutb, tweight, dT, dH, dW, "F", "C");
    }
    THTensor_(free)(ginpb);
    THTensor_(free)(goutb);
  }

  THTensor_(free)(tweight);

  return 1;
}

static int nn_(VolumetricConvolution_accGradParameters)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);  
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  real scale = luaL_optnumber(L, 4, 1);
  int dT = luaT_getfieldcheckint(L, 1, "dT");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");

  THTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_Tensor);
  THTensor *gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_Tensor);

  long k;
  real *gradBias_data;
  THTensor* gradOutSlice;
  int dimPlane = 0;
  if (gradOutput->nDimension == 5) {
    dimPlane++;
  }
  
  THArgCheck( nOutputPlane == gradOutput->size[dimPlane], 1, 
	      "Number of output features is not equal to nOutputPlane" );

  
  if (gradOutput->nDimension == 4) { /* non-batch mode */
    /* gradient to bias */
    gradBias_data = THTensor_(data)(gradBias);
    gradOutSlice = THTensor_(new)();
    for(k = 0; k < nOutputPlane; k++)
      {
	THTensor_(select)(gradOutSlice, gradOutput, 0, k);
	gradBias_data[k] += scale*THTensor_(sumall)(gradOutSlice);
      }
    THTensor_(free)(gradOutSlice);
    
    /* gradient to kernels */
    THTensor_(conv3DRevger)(gradWeight, 1.0, scale, input, gradOutput, dT, dH, dW);
  } else { /* batch mode */
    long nBatch = gradOutput->size[0];
    THTensor *inpb = THTensor_(new)();
    THTensor *goutb = THTensor_(new)();
    long j;

    for (j=0; j<nBatch; j++) { /* loop over batches */
      THTensor_(select)(inpb,input,0,j);
      THTensor_(select)(goutb,gradOutput,0,j);
      
      /* gradient to bias */
      gradBias_data = THTensor_(data)(gradBias);
      gradOutSlice = THTensor_(new)();
      for(k = 0; k < nOutputPlane; k++)
	{
	  THTensor_(select)(gradOutSlice, goutb, 0, k);
	  gradBias_data[k] += scale*THTensor_(sumall)(gradOutSlice);
	}
      THTensor_(free)(gradOutSlice);
      
      /* gradient to kernels */
      THTensor_(conv3DRevger)(gradWeight, 1.0, scale, inpb, goutb, dT, dH, dW);
    }
    THTensor_(free)(inpb);
    THTensor_(free)(goutb);
  }

  return 0;
}

static const struct luaL_Reg nn_(VolumetricConvolution__) [] = {
  {"VolumetricConvolution_updateOutput", nn_(VolumetricConvolution_updateOutput)},
  {"VolumetricConvolution_updateGradInput", nn_(VolumetricConvolution_updateGradInput)},
  {"VolumetricConvolution_accGradParameters", nn_(VolumetricConvolution_accGradParameters)},
  {NULL, NULL}
};

static void nn_(VolumetricConvolution_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(VolumetricConvolution__), "nn");
  lua_pop(L,1);
}

#endif
