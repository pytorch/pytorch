#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialFullConvolutionMap.c"
#else

static int nn_(SpatialFullConvolutionMap_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");


  THTensor *connTable = luaT_getfieldcheckudata(L, 1, "connTable", torch_Tensor);
  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *bias = luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

  real *input_data;
  real *output_data;
  real *weight_data;
  real *bias_data;
  real *connTable_data;

  long input_h;
  long input_w;
  long output_h;
  long output_w;
  long weight_h;
  long weight_w;

  long p;

  luaL_argcheck(L, input->nDimension == 3, 2, "3D tensor expected");
  luaL_argcheck(L, input->size[0] >= nInputPlane, 2, "invalid number of input planes");


  THTensor_(resize3d)(output, nOutputPlane,
                      (input->size[1] - 1) * dH + kH,
                      (input->size[2] - 1) * dW + kW);

  /* contiguous */
  input = THTensor_(newContiguous)(input);
  output = THTensor_(newContiguous)(output);

  /* get raw pointers */
  input_data = THTensor_(data)(input);
  output_data = THTensor_(data)(output);
  weight_data = THTensor_(data)(weight);
  bias_data = THTensor_(data)(bias);
  connTable_data = THTensor_(data)(connTable);

  /* and dims */
  input_h = input->size[1];
  input_w = input->size[2];
  output_h = output->size[1];
  output_w = output->size[2];
  weight_h = weight->size[1];
  weight_w = weight->size[2];

#pragma omp parallel for private(p)
  for (p = 0; p < nOutputPlane; p++) {
    /* add bias */
    real *ptr_output = output_data + p*output_w*output_h;
    long j;
    int nweight;
    long k;

    for(j = 0; j < output_h*output_w; j++)
      ptr_output[j] = bias_data[p];

    /* convolve all maps */
    nweight = connTable->size[0];
    for (k = 0; k < nweight; k++) {
      /* get offsets for input/output */
      int o = (int)connTable_data[k*2+1]-1;
      int i = (int)connTable_data[k*2+0]-1;

      if (o == p)
        {
          THTensor_(fullConv2Dptr)(output_data + o*output_w*output_h,
				   1.0,
				   input_data + i*input_w*input_h, input_h, input_w,
				   weight_data + k*weight_w*weight_h, weight_h, weight_w,
				   dH, dW);
        }
    }
  }

  /* clean up */
  THTensor_(free)(input);
  THTensor_(free)(output);

  return 1;
}

static int nn_(SpatialFullConvolutionMap_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");

  THTensor *connTable = luaT_getfieldcheckudata(L, 1, "connTable", torch_Tensor);
  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);

  real *gradInput_data;
  real *gradOutput_data;
  real *weight_data;
  real *connTable_data;

  long input_h;
  long input_w;
  long output_h;
  long output_w;
  long weight_h;
  long weight_w;

  long p;

  /* contiguous */
  gradInput = THTensor_(newContiguous)(gradInput);
  gradOutput = THTensor_(newContiguous)(gradOutput);

  /* Resize/Zero */
  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  /* get raw pointers */
  gradInput_data = THTensor_(data)(gradInput);
  gradOutput_data = THTensor_(data)(gradOutput);
  weight_data = THTensor_(data)(weight);
  connTable_data = THTensor_(data)(connTable);

  /* and dims */
  input_h = input->size[1];
  input_w = input->size[2];
  output_h = gradOutput->size[1];
  output_w = gradOutput->size[2];
  weight_h = weight->size[1];
  weight_w = weight->size[2];

#pragma omp parallel for private(p)
  for(p = 0; p < nInputPlane; p++)
    {
      long k;
      /* backward all */
      int nkernel = connTable->size[0];
      for(k = 0; k < nkernel; k++)
        {
          int o = (int)connTable_data[k*2+1]-1;
          int i = (int)connTable_data[k*2+0]-1;
          if (i == p)
            {
              /* gradient to input */
              THTensor_(validXCorr2Dptr)(gradInput_data + i*input_w*input_h,
					 1.0,
					 gradOutput_data + o*output_w*output_h,  output_h,  output_w,
					 weight_data + k*weight_w*weight_h, weight_h, weight_w,
					 dH, dW);
            }
        }
    }

  /* clean up */
  THTensor_(free)(gradInput);
  THTensor_(free)(gradOutput);

  return 1;
}

static int nn_(SpatialFullConvolutionMap_accGradParameters)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  real scale = luaL_optnumber(L, 4, 1);

  THTensor *connTable = luaT_getfieldcheckudata(L, 1, "connTable", torch_Tensor);
  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_Tensor);
  THTensor *gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_Tensor);

  real *input_data;
  real *gradOutput_data;
  real *gradWeight_data;
  real *gradBias_data;

  long input_h;
  long input_w;
  long output_h;
  long output_w;
  long weight_h;
  long weight_w;

  long k;
  int nkernel;

  /* contiguous */
  input = THTensor_(newContiguous)(input);
  gradOutput = THTensor_(newContiguous)(gradOutput);

  /* get raw pointers */
  input_data = THTensor_(data)(input);
  gradOutput_data = THTensor_(data)(gradOutput);
  gradWeight_data = THTensor_(data)(gradWeight);
  gradBias_data = THTensor_(data)(gradBias);

  /* and dims */
  input_h = input->size[1];
  input_w = input->size[2];
  output_h = gradOutput->size[1];
  output_w = gradOutput->size[2];
  weight_h = weight->size[1];
  weight_w = weight->size[2];

  /* gradients wrt bias */
#pragma omp parallel for private(k)
  for(k = 0; k < nOutputPlane; k++) {
    real *ptr_gradOutput = gradOutput_data + k*output_w*output_h;
    long l;
    for(l = 0; l < output_h*output_w; l++)
      gradBias_data[k] += scale*ptr_gradOutput[l];
  }

  /* gradients wrt weight */
  nkernel = connTable->size[0];
#pragma omp parallel for private(k)
  for(k = 0; k < nkernel; k++)
    {
      int o = (int)THTensor_(get2d)(connTable,k,1)-1;
      int i = (int)THTensor_(get2d)(connTable,k,0)-1;

      /* gradient to kernel */
      THTensor_(validXCorr2DRevptr)(gradWeight_data + k*weight_w*weight_h,
                                 scale,
                                 gradOutput_data + o*output_w*output_h, output_h, output_w,
                                 input_data + i*input_w*input_h, input_h, input_w,
                                 dH, dW);
    }

  /* clean up */
  THTensor_(free)(input);
  THTensor_(free)(gradOutput);
  return 0;
}

static const struct luaL_Reg nn_(SpatialFullConvolutionMapStuff__) [] = {
  {"SpatialFullConvolutionMap_updateOutput", nn_(SpatialFullConvolutionMap_updateOutput)},
  {"SpatialFullConvolutionMap_updateGradInput", nn_(SpatialFullConvolutionMap_updateGradInput)},
  {"SpatialFullConvolutionMap_accGradParameters", nn_(SpatialFullConvolutionMap_accGradParameters)},
  {NULL, NULL}
};

static void nn_(SpatialFullConvolutionMap_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(SpatialFullConvolutionMapStuff__), "nn");
  lua_pop(L,1);
}

#endif
