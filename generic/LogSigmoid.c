#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/LogSigmoid.c"
#else

static int nn_(LogSigmoid_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *buffer = luaT_getfieldcheckudata(L, 1, "buffer", torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

  THTensor_(resizeAs)(output, input);
  THTensor_(resizeAs)(buffer, input);

  TH_TENSOR_APPLY3(real, output, real, input, real, buffer,    \
                   real z = exp(-*input_data);                 \
                   *buffer_data = z;                           \
                   *output_data = -log(1. + z);)

  return 1;
}

static int nn_(LogSigmoid_updateGradInput)(lua_State *L)
{
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *buffer = luaT_getfieldcheckudata(L, 1, "buffer", torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);

  THTensor_(resizeAs)(gradInput, buffer);
  TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, buffer,    \
                   real z = *buffer_data;                              \
                   *gradInput_data = *gradOutput_data * z / (1. + z);)

  return 1;
}

static const struct luaL_Reg nn_(LogSigmoid__) [] = {
  {"LogSigmoid_updateOutput", nn_(LogSigmoid_updateOutput)},
  {"LogSigmoid_updateGradInput", nn_(LogSigmoid_updateGradInput)},
  {NULL, NULL}
};

static void nn_(LogSigmoid_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(LogSigmoid__), "nn");
  lua_pop(L,1);
}

#endif
