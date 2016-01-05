#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/LeakyReLU.c"
#else

static int nn_(LeakyReLU_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  real negval = luaT_getfieldchecknumber(L, 1, "negval");
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  int inPlace = luaT_getfieldcheckboolean(L, 1, "inplace");

  if (inPlace) {
    TH_TENSOR_APPLY(real, input,                   \
                    if (*input_data <= 0) { \
                      *input_data *=  negval ;           \
                    });
    THTensor_(set)(output, input);
  } else {
    THTensor_(resizeAs)(output, input);
    TH_TENSOR_APPLY2(real, output, real, input,                         \
                     *output_data = (*input_data > 0) ? *input_data : negval * (*input_data););

  }

  return 1;
}

static int nn_(LeakyReLU_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  real negval = luaT_getfieldchecknumber(L, 1, "negval");
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
  int inPlace = luaT_getfieldcheckboolean(L, 1, "inplace");

  if (inPlace) {
    TH_TENSOR_APPLY2(real, gradOutput, real, input,    \
                     if ((*input_data) <= 0) { \
                       *gradOutput_data *= negval;           \
                         });
    THTensor_(set)(gradInput, gradOutput);
  } else {
    THTensor_(resizeAs)(gradInput, input);
    TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, input,    \
                     if ((*input_data) > 0) *gradInput_data = *gradOutput_data; \
                     else *gradInput_data = *gradOutput_data * negval;);                        \
  }

  return 1;
}

static const struct luaL_Reg nn_(LeakyReLU__) [] = {
  {"LeakyReLU_updateOutput", nn_(LeakyReLU_updateOutput)},
  {"LeakyReLU_updateGradInput", nn_(LeakyReLU_updateGradInput)},
  {NULL, NULL}
};

static void nn_(LeakyReLU_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(LeakyReLU__), "nn");
  lua_pop(L,1);
}

#endif
