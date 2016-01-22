#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/PReLU.c"
#else


static int nn_(PReLU_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  long nOutputPlane = luaT_getfieldchecknumber(L, 1, "nOutputPlane");

  THTensor_(resizeAs)(output, input);

  if (nOutputPlane == 0)
  {
    // handle shared parameter case
    real w = *THTensor_(data)(weight);
    TH_TENSOR_APPLY2(real, output, real, input, \
		     *output_data = (*input_data > 0) ? *input_data : w*(*input_data););
  }
  else
  {
    long bs, ks;
    {
      long input_ndim = THTensor_(nDimension)(input);
      switch (input_ndim)
      {
	case 1:
	  bs = 1;
	  ks = 1;
	  break;
	case 2:
	  bs = input->size[0];
	  ks = 1;
	  break;
	case 3:
	  bs = 1;
	  ks = input->size[1] * input->size[2];
	  break;
	case 4:
	  bs = input->size[0];
	  ks = input->size[2] * input->size[3];
	  break;
      }

      if(input->size[(input_ndim + 1) % 2] != nOutputPlane)
	THError("wrong number of input planes");
    }

    real* output_data = THTensor_(data)(output);
    real* input_data = THTensor_(data)(input);
    real* weight_data = THTensor_(data)(weight);
    long i,j,k;
#pragma omp parallel for private(j,k)
    for (i=0; i < bs; ++i)
    {
      real* n_input_data = input_data + i*nOutputPlane*ks;
      real* n_output_data = output_data + i*nOutputPlane*ks;
      for (j=0; j < nOutputPlane; ++j)
      {
	for (k=0; k < ks; ++k)
	  n_output_data[k] = (n_input_data[k] > 0) ? n_input_data[k] : weight_data[j] * n_input_data[k];
	n_input_data += ks;
	n_output_data += ks;
      }
    }
  }

  return 1;
}

static int nn_(PReLU_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
  long nOutputPlane = luaT_getfieldchecknumber(L, 1, "nOutputPlane");

  THTensor_(resizeAs)(gradInput, input);

  if (nOutputPlane == 0)
  {
    real w = THTensor_(data)(weight)[0];
    TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, input,	\
       if ((*input_data) > 0) *gradInput_data = *gradOutput_data;	\
       else *gradInput_data = w* *gradOutput_data;);     		\
  }
  else
  {
    const real* input_data = THTensor_(data)(input);
    const real* gradOutput_data = THTensor_(data)(gradOutput);
    const real* weight_data = THTensor_(data)(weight);
    real* gradInput_data = THTensor_(data)(gradInput);

    long bs, ks;
    {
      long input_ndim = THTensor_(nDimension)(input);
      switch (input_ndim)
      {
	case 1:
	  bs = 1;
	  ks = 1;
	  break;
	case 2:
	  bs = input->size[0];
	  ks = 1;
	  break;
	case 3:
	  bs = 1;
	  ks = input->size[1] * input->size[2];
	  break;
	case 4:
	  bs = input->size[0];
	  ks = input->size[2] * input->size[3];
	  break;
      }

      if(input->size[(input_ndim + 1) % 2] != nOutputPlane)
	THError("wrong number of input planes");
    }

    long i,j,k;
#pragma omp parallel for private(j,k)
    for (i = 0; i < bs; ++i)
    {
      const real* n_input_data = input_data + i*nOutputPlane*ks;
      const real* n_gradOutput_data = gradOutput_data + i*nOutputPlane*ks;
      real* n_gradInput_data = gradInput_data + i*nOutputPlane*ks;

      for (j=0; j < nOutputPlane; ++j)
      {
	real w = weight_data[j];
	for (k=0; k < ks; ++k)
	  if (n_input_data[k] > 0)
	    n_gradInput_data[k] = n_gradOutput_data[k];
	  else
	    n_gradInput_data[k] = n_gradOutput_data[k] * w;
	n_input_data += ks;
	n_gradInput_data += ks;
	n_gradOutput_data += ks;
      }
    }
  }

  return 1;
}

static int nn_(PReLU_accGradParameters)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_Tensor);
  long nOutputPlane = luaT_getfieldchecknumber(L, 1, "nOutputPlane");
  real scale = luaL_optnumber(L, 4, 1);

  real* gradWeight_data = THTensor_(data)(gradWeight);

  if (nOutputPlane == 0)
  {
    real sum = 0;
    TH_TENSOR_APPLY2(real, input, real, gradOutput,  \
	if ((*input_data) <= 0) sum += *input_data* *gradOutput_data;);
    gradWeight_data[0] += scale*sum;
  }
  else
  {
    long bs, ks;
    {
      long input_ndim = THTensor_(nDimension)(input);
      switch (input_ndim)
      {
	case 1:
	  bs = 1;
	  ks = 1;
	  break;
	case 2:
	  bs = input->size[0];
	  ks = 1;
	  break;
	case 3:
	  bs = 1;
	  ks = input->size[1] * input->size[2];
	  break;
	case 4:
	  bs = input->size[0];
	  ks = input->size[2] * input->size[3];
	  break;
      }

      if(input->size[(input_ndim + 1) % 2] != nOutputPlane)
	THError("wrong number of input planes");
    }

    const real* input_data = THTensor_(data)(input);
    const real* gradOutput_data = THTensor_(data)(gradOutput);
    const real* weight_data = THTensor_(data)(weight);
    real* gradWeight_data = THTensor_(data)(gradWeight);

    long i,j,k;
    for (i = 0; i < bs; ++i)
    {
      const real* n_input_data = input_data + i*nOutputPlane*ks;
      const real* n_gradOutput_data = gradOutput_data + i*nOutputPlane*ks;

      for (j=0; j < nOutputPlane; ++j)
      {
	real sum = 0;
	for (k=0; k < ks; ++k)
	  if (n_input_data[k] <= 0)
	    sum += n_gradOutput_data[k] * n_input_data[k];
	gradWeight_data[j] += scale * sum;
	n_input_data += ks;
	n_gradOutput_data += ks;
      }
    }
  }
  return 1;
}


static const struct luaL_Reg nn_(PReLU__) [] = {
  {"PReLU_updateOutput", nn_(PReLU_updateOutput)},
  {"PReLU_updateGradInput", nn_(PReLU_updateGradInput)},
  {"PReLU_accGradParameters", nn_(PReLU_accGradParameters)},
  {NULL, NULL}
};

static void nn_(PReLU_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(PReLU__), "nn");
  lua_pop(L,1);
}

#endif
