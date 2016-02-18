#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialConvolutionMap.c"
#else

static int nn_(SpatialConvolutionMap_updateOutput)(lua_State *L)
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
  real *weight_data = THTensor_(data)(weight);
  real *bias_data = THTensor_(data)(bias);
  real *connTable_data = THTensor_(data)(connTable);


  long p;

  int dimw = 2;
  int dimh = 1;
  int dimc = 0;
  long nbatch = 1;
  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D(batch mode) tensor expected");

  if (input->nDimension == 4) {
    nbatch = input->size[0];
    dimc++;
    dimw++;
    dimh++;
  }
  luaL_argcheck(L, input->size[dimc] >= nInputPlane, 2, "invalid number of input planes");
  luaL_argcheck(L, input->size[dimw] >= kW && input->size[dimh] >= kH, 2, "input image smaller than kernel size");

  long input_w   = input->size[dimw];
  long input_h   = input->size[dimh];
  long output_w  = (input_w - kW) / dW + 1;
  long output_h  = (input_h - kH) / dH + 1;


  if (input->nDimension == 3)
    THTensor_(resize3d)(output, nOutputPlane, output_h, output_w);
  else
    THTensor_(resize4d)(output, input->size[0], nOutputPlane, output_h, output_w);

  /* contiguous */
  input = THTensor_(newContiguous)(input);
  output = THTensor_(newContiguous)(output);

  /* get raw pointers */
  input_data = THTensor_(data)(input);
  output_data = THTensor_(data)(output);


#pragma omp parallel for private(p)
  for (p = 0; p < nOutputPlane; p++) {

    long m;
    for(m = 0; m < nbatch; m++){
      /* add bias */
      real *ptr_output = output_data + p*output_w*output_h + m*nOutputPlane*output_w*output_h;
      long j,k;
      real z= bias_data[p];
      for(j = 0; j < output_h*output_w; j++)
        ptr_output[j] = z;

      /* convolve all maps */
      int nweight = connTable->size[0];
      for (k = 0; k < nweight; k++) {
        /* get offsets for input/output */
        int o = (int)connTable_data[k*2+1]-1;
        int i = (int)connTable_data[k*2+0]-1;

        if (o == p){
          THTensor_(validXCorr2Dptr)(output_data + o*output_w*output_h + m*nOutputPlane*output_w*output_h,
                                    1.0,
                                    input_data + i*input_w*input_h + m*nInputPlane*input_w*input_h, input_h, input_w,
                                    weight_data + k*kW*kH, kH, kW,
                                    dH, dW);

        }
      }
    }
  }

  /* clean up */
  THTensor_(free)(input);
  THTensor_(free)(output);

  return 1;
}

static int nn_(SpatialConvolutionMap_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");

  THTensor *connTable = luaT_getfieldcheckudata(L, 1, "connTable", torch_Tensor);
  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);

  real *gradInput_data;
  real *gradOutput_data;
  real *weight_data = THTensor_(data)(weight);
  real *connTable_data = THTensor_(data)(connTable);

    /* and dims */
  int dimw = 2;
  int dimh = 1;
  long nbatch = 1;
  if (input->nDimension == 4) {
    nbatch = input->size[0];
    dimw++;
    dimh++;
  }

  long input_w   = input->size[dimw];
  long input_h   = input->size[dimh];
  long weight_h = weight->size[1];
  long weight_w = weight->size[2];
  long output_h = gradOutput->size[dimh];
  long output_w = gradOutput->size[dimw];

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


#pragma omp parallel for private(p)
  for(p = 0; p < nInputPlane; p++){
    long m;
    for(m = 0; m < nbatch; m++){
      long k;
      /* backward all */
      int nkernel = connTable->size[0];
      for(k = 0; k < nkernel; k++)
      {
        int o = (int)connTable_data[k*2+1]-1;
        int i = (int)connTable_data[k*2+0]-1;
        if (i == p){
          /* gradient to input */
          THTensor_(fullConv2Dptr)(gradInput_data + i*input_w*input_h + m*nInputPlane*input_w*input_h, 1.0,
                gradOutput_data + o*output_w*output_h + m*nOutputPlane*output_w*output_h,  output_h,  output_w,
                weight_data + k*weight_w*weight_h, weight_h, weight_w, dH, dW);
        }
      }
    }
  }

  /* clean up */
  THTensor_(free)(gradInput);
  THTensor_(free)(gradOutput);

  return 1;
}

static int nn_(SpatialConvolutionMap_accGradParameters)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  real scale = luaL_optnumber(L, 4, 1);

  THTensor *connTable = luaT_getfieldcheckudata(L, 1, "connTable", torch_Tensor);
  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_Tensor);
  THTensor *gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_Tensor);

  real *input_data;
  real *gradOutput_data;
  real *gradWeight_data = THTensor_(data)(gradWeight);
  real *gradBias_data = THTensor_(data)(gradBias);

    /* and dims */
  int dimw = 2;
  int dimh = 1;
  long nbatch = 1;
  if (input->nDimension == 4) {
    nbatch = input->size[0];
    dimw++;
    dimh++;
  }

  long input_w   = input->size[dimw];
  long input_h   = input->size[dimh];
  long output_h = gradOutput->size[dimh];
  long output_w = gradOutput->size[dimw];
  long weight_h  = weight->size[1];
  long weight_w  = weight->size[2];

  int nkernel;

  /* contiguous */
  input = THTensor_(newContiguous)(input);
  gradOutput = THTensor_(newContiguous)(gradOutput);

  /* get raw pointers */
  input_data = THTensor_(data)(input);
  gradOutput_data = THTensor_(data)(gradOutput);

  long k;
    /* gradients wrt bias */
#pragma omp parallel for private(k)
  for(k = 0; k < nOutputPlane; k++) {
    long m;
    for(m = 0; m < nbatch; m++){
      real *ptr_gradOutput = gradOutput_data + k*output_w*output_h + m*nOutputPlane*output_w*output_h;
      long l;
      for(l = 0; l < output_h*output_w; l++)
        gradBias_data[k] += scale*ptr_gradOutput[l];
    }
  }

  /* gradients wrt weight */
  nkernel = connTable->size[0];
#pragma omp parallel for private(k)
  for(k = 0; k < nkernel; k++){
    long m;
    for(m = 0; m < nbatch; m++){
      int o = (int)THTensor_(get2d)(connTable,k,1)-1;
      int i = (int)THTensor_(get2d)(connTable,k,0)-1;

      /* gradient to kernel */
      THTensor_(validXCorr2DRevptr)(gradWeight_data + k*weight_w*weight_h,
                                   scale,
                                   input_data + i*input_w*input_h + m*nInputPlane*input_w*input_h, input_h, input_w,
                                   gradOutput_data + o*output_w*output_h + m*nOutputPlane*output_w*output_h , output_h, output_w,
                                   dH, dW);
    }
  }


  /* clean up */
  THTensor_(free)(input);
  THTensor_(free)(gradOutput);
  return 0;
}

static const struct luaL_Reg nn_(SpatialConvolutionMap__) [] = {
  {"SpatialConvolutionMap_updateOutput", nn_(SpatialConvolutionMap_updateOutput)},
  {"SpatialConvolutionMap_updateGradInput", nn_(SpatialConvolutionMap_updateGradInput)},
  {"SpatialConvolutionMap_accGradParameters", nn_(SpatialConvolutionMap_accGradParameters)},
  {NULL, NULL}
};

static void nn_(SpatialConvolutionMap_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(SpatialConvolutionMap__), "nn");
  lua_pop(L,1);
}

#endif
