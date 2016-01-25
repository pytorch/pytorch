#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Sigmoid.c"
#else

static int nn_(Sigmoid_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

  THTensor_(resizeAs)(output, input);

  TH_TENSOR_APPLY2(real, output, real, input, \
                   *output_data = 1./(1.+ exp(- *input_data));)

  return 1;
}

static int nn_(Sigmoid_updateGradInput)(lua_State *L)
{
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);

  THTensor_(resizeAs)(gradInput, output);
  TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, output, \
                   real z = *output_data; \
                   *gradInput_data = *gradOutput_data * (1. - z) * z;)
  return 1;
}

static const struct luaL_Reg nn_(Sigmoid__) [] = {
  {"Sigmoid_updateOutput", nn_(Sigmoid_updateOutput)},
  {"Sigmoid_updateGradInput", nn_(Sigmoid_updateGradInput)},
  {NULL, NULL}
};

static void nn_(Sigmoid_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(Sigmoid__), "nn");
  lua_pop(L,1);
}

#endif
