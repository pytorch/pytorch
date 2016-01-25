#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/RReLU.c"
#else

static int nn_(RReLU_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  THTensor *noise = luaT_getfieldcheckudata(L, 1, "noise", torch_Tensor);
  real lower = luaT_getfieldchecknumber(L, 1, "lower");
  real upper = luaT_getfieldchecknumber(L, 1, "upper");
  int train = luaT_getfieldcheckboolean(L, 1, "train");
  int inplace = luaT_getfieldcheckboolean(L, 1, "inplace");
  
  if (train)
  {
    // get default random generator
    lua_getglobal(L, "torch");
    THGenerator *generator = luaT_getfieldcheckudata(L, -1, "_gen", torch_Generator);
    lua_pop(L, 2);

    THTensor_(resizeAs)(noise, input);
    if (inplace)
    {
      TH_TENSOR_APPLY2(real, input, real, noise, \
        if (*input_data <= 0) { \
          const real r = (real)THRandom_uniform(generator, lower, upper); \
          *input_data = (*input_data) * r; \
          *noise_data = r; \
        } \
        else { \
          *noise_data = 1; \
        }
      );
      THTensor_(set)(output, input);
    }
    else
    {
      THTensor_(resizeAs)(output, input);
      TH_TENSOR_APPLY3(real, input, real, output, real, noise, \
        if (*input_data <= 0) { \
          const real r = (real)THRandom_uniform(generator, lower, upper); \
          *output_data = (*input_data) * r; \
          *noise_data = r; \
        } \
        else { \
          *output_data = *input_data;
          *noise_data = 1; \
        }
      );
    }
  }
  else
  {
    const real negSlope = (lower + upper) / 2;
    if (inplace)
    {
      TH_TENSOR_APPLY(real, input, \
        if (*input_data <= 0) { \
          *input_data = *input_data * negSlope; \
        }
      );
      THTensor_(set)(output, input);
    }
    else
    {
      THTensor_(resizeAs)(output, input);
      TH_TENSOR_APPLY2(real, input, real, output, \
        const real r = (*input_data) <= 0 ? negSlope : 1; \
        *output_data = *input_data * r; \
      );
    }
  }  
  return 1;
}

static int nn_(RReLU_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
  THTensor *noise = luaT_getfieldcheckudata(L, 1, "noise", torch_Tensor);
  real lower = luaT_getfieldchecknumber(L, 1, "lower");
  real upper = luaT_getfieldchecknumber(L, 1, "upper");
  int train = luaT_getfieldcheckboolean(L, 1, "train");
  int inplace = luaT_getfieldcheckboolean(L, 1, "inplace");
  
  if (train && upper - lower > 1E-6)    // e.g. if upper == lower, RReLU behaves like LeakyReLU
  {
    // multiply the gradient by the noise tensor
    if (inplace)
    {
      THTensor_(cmul)(gradOutput, gradOutput, noise);
      THTensor_(set)(gradInput, gradOutput);
    }
    else
    {
      THTensor_(resizeAs)(gradInput, input);
      THTensor_(cmul)(gradInput, gradOutput, noise);
    }    
  }
  else
  { 
    // use constant factor for negative input values
    const real negSlope = (lower + upper) / 2;
    if (inplace)
    {
      TH_TENSOR_APPLY2(real, gradOutput, real, input, \
        if (*input_data <= 0) { \
         *gradOutput_data = (*gradOutput_data) * negSlope; \
        } \
      );
      THTensor_(set)(gradInput, gradOutput);
    }
    else
    {
      THTensor_(resizeAs)(gradInput, input);
      TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, input, \
        *gradInput_data = (*input_data) <= 0 ? (*gradOutput_data) * negSlope : (*gradOutput_data); \
      );
    }
  }
  return 1;
}

static const struct luaL_Reg nn_(RReLU__) [] = {
  { "RReLU_updateOutput", nn_(RReLU_updateOutput) },
  { "RReLU_updateGradInput", nn_(RReLU_updateGradInput) },
  { NULL, NULL }
};

static void nn_(RReLU_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(RReLU__), "nn");
  lua_pop(L, 1);
}

#endif
